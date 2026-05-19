"""Evaluation of predicted velocity fields against ground-truth simulations.

For every sample the model output is:

1. divided by ``SCALING_FACTOR`` (undo the training scale),
2. squared (undo the ``sqrt`` applied to targets),
3. multiplied by ``VEL_SCALE`` to convert lattice units to physical m/s
   (``VEL_SCALE = dx / dt = 0.5`` m/s per lattice unit),

and metrics are computed only over the fracture (open) voxels:

* MAE   -- mean absolute error
* RMSE  -- root mean squared error
* RRMSE -- RMSE divided by RMS of the truth (percent)
* sMAPE -- symmetric mean absolute percentage error (percent)

Per-sample results are written to a CSV.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import FractureDataset, build_dataframe, train_val_split
from models import MODELS

# Must match the value used during training (see dataset.SCALING_FACTOR).
SCALING_FACTOR: float = 1000.0

# Lattice-to-physical conversion from the paper:
#   dx = 2.74e-5 m, dt = 5.48e-5 s  ->  VEL_SCALE = dx / dt = 0.5 m/s per LU.
DX: float = 2.74e-5
DT: float = 5.48e-5
VEL_SCALE: float = DX / DT


def load_trained_model(
    model_class, checkpoint_path: str, device: torch.device, in_channels: int = 2
) -> torch.nn.Module:
    """Instantiate ``model_class`` and load weights from a checkpoint.

    Args:
        model_class: Model class (e.g. ``UNet3D`` or ``AttResUNet``).
        checkpoint_path: Path to a ``.pth`` checkpoint dictionary.
        device: Device to map the weights onto.
        in_channels: Input channel count the checkpoint was trained with.

    Returns:
        The loaded model, moved to ``device`` (still in train mode; the
        caller is responsible for calling ``.eval()``).
    """
    model = model_class(in_channels=in_channels, out_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Model loaded from: {checkpoint_path}")
    return model


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    results_save_path: str,
) -> pd.DataFrame:
    """Evaluate ``model`` on ``loader`` and write per-sample metrics to CSV.

    Metrics are computed only on the fracture voxels (where the input
    geometry channel is > 0), in physical units (m/s).

    Args:
        model: Trained model.
        loader: Data loader (``shuffle=False`` so names line up by index).
        device: Compute device.
        results_save_path: Output CSV path for per-sample metrics.

    Returns:
        The per-sample results as a DataFrame.
    """
    model.eval()
    results = []
    epsilon = 1e-10

    print(f"VEL_SCALE = {VEL_SCALE} m/s per lattice unit")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(
            tqdm(loader, desc="Evaluating")
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            batch_size = loader.batch_size

            outputs_unscaled = outputs / SCALING_FACTOR
            targets_unscaled = targets / SCALING_FACTOR

            for i in range(inputs.size(0)):
                sample_idx = (batch_idx * batch_size) + i
                name = loader.dataset.df.iloc[sample_idx]["name"]

                geometry_mask = inputs[i, 0, :, :, :] > 0

                pred_velocity = torch.square(outputs_unscaled[i].squeeze(0))
                true_velocity = torch.square(targets_unscaled[i].squeeze(0))

                pred_ms = pred_velocity * VEL_SCALE
                true_ms = true_velocity * VEL_SCALE

                filtered_pred = pred_ms[geometry_mask]
                filtered_true = true_ms[geometry_mask]

                if filtered_true.numel() == 0:
                    print(f"Warning: Skipping sample {name}, no fracture voxels found.")
                    continue

                mae = torch.mean(torch.abs(filtered_pred - filtered_true)).item()

                mse = torch.mean((filtered_pred - filtered_true) ** 2)
                rmse_tensor = torch.sqrt(mse)
                rmse = rmse_tensor.item()

                rms_true = torch.sqrt(torch.mean(filtered_true ** 2))
                rrmse = (rmse_tensor / (rms_true + epsilon) * 100).item()

                smape_numerator = 2 * torch.abs(filtered_pred - filtered_true)
                smape_denominator = (
                    torch.abs(filtered_true) + torch.abs(filtered_pred) + epsilon
                )
                smape = (100.0 * torch.mean(smape_numerator / smape_denominator)).item()

                results.append(
                    {
                        "name": name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "RRMSE": rrmse,
                        "sMAPE": smape,
                    }
                )

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_save_path) or ".", exist_ok=True)
    results_df.to_csv(results_save_path, index=False)
    print(f"\nPer-sample evaluation results saved to: {results_save_path}")

    print("\n--- Average Evaluation Metrics (fracture voxels) ---")
    print(f"Average MAE:    {results_df['MAE'].mean():.6e}")
    print(f"Average RMSE:   {results_df['RMSE'].mean():.6e}")
    print(f"Average RRMSE:  {results_df['RRMSE'].mean():.6e}")
    print(f"Average sMAPE:  {results_df['sMAPE'].mean():.6e} %")

    print("\n--- Metric Summary (fracture voxels) ---")
    print(f"{'Metric':<12} {'Mean':>12} {'Median':>12} {'Min':>12} {'Max':>12}")
    print("-" * 52)
    for col, label in [
        ("MAE", "MAE"),
        ("RMSE", "RMSE"),
        ("RRMSE", "RRMSE (%)"),
        ("sMAPE", "sMAPE (%)"),
    ]:
        print(
            f"{label:<12} "
            f"{results_df[col].mean():>12.6e} "
            f"{results_df[col].median():>12.6e} "
            f"{results_df[col].min():>12.6e} "
            f"{results_df[col].max():>12.6e}"
        )

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--input-folder", required=True, help="Folder of .mat files")
    parser.add_argument("--mask-folder", required=True, help="Folder of .csv files")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model", choices=tuple(MODELS), default="unet3d")
    parser.add_argument("--in-channels", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--split",
        choices=("val", "test", "all"),
        default="val",
        help="'val'/'all' use the stratified split; 'test' treats all "
        "matched pairs as one held-out set",
    )
    parser.add_argument(
        "--output",
        default="results/eval_results.csv",
        help="Per-sample metrics CSV output path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = build_dataframe(args.input_folder, args.mask_folder)

    if args.split == "val":
        _, eval_df = train_val_split(df)
    elif args.split == "test":
        eval_df = df
    else:
        eval_df = df

    loader = DataLoader(
        FractureDataset(eval_df, in_channels=args.in_channels),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = load_trained_model(
        MODELS[args.model], args.checkpoint, device, in_channels=args.in_channels
    )
    evaluate_model(model, loader, device, args.output)


if __name__ == "__main__":
    main()
