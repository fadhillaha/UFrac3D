"""Evaluation script to calculate voxel-wise metrics (RMSE, SMAPE, AAE) against LBM ground truth."""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import FractureDataset, SCALING_FACTOR, build_dataframe, inverse_transform_torch, train_val_split
from models import MODELS

# Lattice-to-physical conversion from the paper:
#   dx = 2.74e-5 m, dt = 5.48e-5 s  ->  VEL_SCALE = dx / dt = 0.5 m/s per LU.
DX: float = 2.74e-5
DT: float = 5.48e-5
VEL_SCALE: float = DX / DT

COMPONENT_NAMES = ("ux", "uy", "uz")


def load_trained_model(
    model_class,
    checkpoint_path: str,
    device: torch.device,
    in_channels: int = 2,
    out_channels: "int | None" = None,
):
    """Instantiate ``model_class`` and load weights from a checkpoint.

    Reads ``out_channels``/``in_channels``/``target_mode`` from the
    checkpoint (written by the updated ``train.py``) when present, so a
    vector checkpoint can't silently be loaded as if it were a
    magnitude-only model (or vice versa) -- this used to fail with a
    confusing tensor-shape error deep inside ``load_state_dict``; now it
    fails with an explicit message. Checkpoints from the *original* script
    (no metadata) fall back to ``out_channels=1`` with a warning.

    Args:
        model_class: Model class (e.g. ``UNet3D`` or ``AttResUNet``).
        checkpoint_path: Path to a ``.pth`` checkpoint dictionary.
        device: Device to map the weights onto.
        in_channels: Input channel count to build the model with.
        out_channels: Expected output channel count. If ``None``, taken
            from the checkpoint (or defaults to 1 with a warning).

    Returns:
        ``(model, out_channels, target_mode)``.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_out_channels = checkpoint.get("out_channels")
    ckpt_target_mode = checkpoint.get("target_mode")

    if ckpt_out_channels is None:
        print("[evaluate] checkpoint has no 'out_channels' metadata "
              "(looks like an original/legacy checkpoint) -- assuming "
              "out_channels=1, target_mode='magnitude'")
        ckpt_out_channels = 1
        ckpt_target_mode = ckpt_target_mode or "magnitude"

    if out_channels is not None and out_channels != ckpt_out_channels:
        raise ValueError(
            f"--out-channels {out_channels} was requested but the "
            f"checkpoint at {checkpoint_path} was trained with "
            f"out_channels={ckpt_out_channels} (target_mode="
            f"{ckpt_target_mode!r}). Use --target-mode "
            f"{'vector' if ckpt_out_channels == 3 else 'magnitude'} to match it."
        )
    out_channels = ckpt_out_channels

    model = model_class(in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Model loaded from: {checkpoint_path} "
          f"(out_channels={out_channels}, target_mode={ckpt_target_mode})")
    return model, out_channels, (ckpt_target_mode or "vector")


def _basic_metrics(pred: torch.Tensor, true: torch.Tensor, epsilon: float = 1e-10) -> dict:
    """MAE/RMSE/RRMSE/sMAPE on two flat 1-D tensors (already masked)."""
    mae = torch.mean(torch.abs(pred - true)).item()
    mse = torch.mean((pred - true) ** 2)
    rmse_tensor = torch.sqrt(mse)
    rmse = rmse_tensor.item()
    rms_true = torch.sqrt(torch.mean(true ** 2))
    rrmse = (rmse_tensor / (rms_true + epsilon) * 100).item()
    smape_num = 2 * torch.abs(pred - true)
    smape_den = torch.abs(true) + torch.abs(pred) + epsilon
    smape = (100.0 * torch.mean(smape_num / smape_den)).item()
    return {"MAE": mae, "RMSE": rmse, "RRMSE": rrmse, "sMAPE": smape}


def _directional_metrics(
    pred_vec: torch.Tensor, true_vec: torch.Tensor, rel_threshold: float = 0.01
) -> dict:
    """Cosine similarity / angular error between predicted and true vectors.

    Args:
        pred_vec: ``(3, N)`` predicted vectors at N void voxels (m/s).
        true_vec: ``(3, N)`` true vectors at the same voxels (m/s).
        rel_threshold: Voxels with true speed below
            ``rel_threshold * max(true_speed)`` are excluded (direction is
            not meaningfully defined near-stagnant flow).

    Returns:
        A dict of directional-accuracy summary statistics.
    """
    true_speed = torch.linalg.vector_norm(true_vec, dim=0)
    max_speed = true_speed.max()
    if max_speed <= 0:
        return {
            "n_voxels_for_angle": 0, "frac_voxels_excluded_angle": 1.0,
            "cos_sim_mean": float("nan"), "cos_sim_median": float("nan"),
            "angerr_mean_deg": float("nan"), "angerr_median_deg": float("nan"),
            "frac_within_10deg": float("nan"), "frac_within_30deg": float("nan"),
        }
    thresh = rel_threshold * max_speed
    valid = true_speed > thresh
    frac_excluded = 1.0 - valid.float().mean().item()

    pv, tv = pred_vec[:, valid], true_vec[:, valid]
    pred_speed = torch.linalg.vector_norm(pv, dim=0)
    tv_speed = torch.linalg.vector_norm(tv, dim=0)
    cos_sim = torch.sum(pv * tv, dim=0) / (pred_speed * tv_speed + 1e-12)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    ang_err_deg = torch.acos(cos_sim) * (180.0 / math.pi)

    return {
        "n_voxels_for_angle": int(valid.sum().item()),
        "frac_voxels_excluded_angle": frac_excluded,
        "cos_sim_mean": cos_sim.mean().item(),
        "cos_sim_median": cos_sim.median().item(),
        "angerr_mean_deg": ang_err_deg.mean().item(),
        "angerr_median_deg": ang_err_deg.median().item(),
        "frac_within_10deg": (ang_err_deg <= 10).float().mean().item(),
        "frac_within_30deg": (ang_err_deg <= 30).float().mean().item(),
    }


def _divergence_error(vec_field: np.ndarray, mask: np.ndarray) -> float:
    """Mean |div(u)| (voxel-unit finite differences) over interior void voxels.

    Args:
        vec_field: ``(3, D, H, W)`` array.
        mask: ``(D, H, W)`` boolean void mask.

    Returns:
        Mean absolute divergence over void voxels that are not on the
        volume's outer 1-voxel border (to avoid one-sided-difference
        artifacts there).
    """
    grads = [np.gradient(vec_field[k], axis=k) for k in range(3)]
    div = grads[0] + grads[1] + grads[2]
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1, 1:-1] = True
    m = mask & interior
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(div[m])))


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    results_save_path: str,
    out_channels: int,
    angle_rel_threshold: float = 0.01,
    compute_divergence: bool = False,
) -> pd.DataFrame:
    """Evaluate ``model`` on ``loader`` and write per-sample metrics to CSV.

    Args:
        model: Trained model.
        loader: Data loader (``shuffle=False`` so names line up by index).
        device: Compute device.
        results_save_path: Output CSV path for per-sample metrics.
        out_channels: 3 for vector, 1 for legacy magnitude-only.
        angle_rel_threshold: See :func:`_directional_metrics`.
        compute_divergence: If True, also report the divergence diagnostic
            (slower -- pulls fields to CPU/numpy per sample).

    Returns:
        The per-sample results as a DataFrame.
    """
    model.eval()
    results = []
    print(f"VEL_SCALE = {VEL_SCALE} m/s per lattice unit | out_channels={out_channels}")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Evaluating")):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            batch_size = loader.batch_size

            pred_lu = inverse_transform_torch(outputs)  # sign-aware inverse
            true_lu = inverse_transform_torch(targets)
            pred_ms = pred_lu * VEL_SCALE
            true_ms = true_lu * VEL_SCALE

            for i in range(inputs.size(0)):
                sample_idx = batch_idx * batch_size + i
                name = loader.dataset.df.iloc[sample_idx]["name"]
                geometry_mask = inputs[i, 0] > 0

                if geometry_mask.sum() == 0:
                    print(f"Warning: Skipping sample {name}, no fracture voxels found.")
                    continue

                row = {"name": name}

                if out_channels == 3:
                    pred_field = pred_ms[i]     # (3,D,H,W)
                    true_field = true_ms[i]

                    for k, comp in enumerate(COMPONENT_NAMES):
                        pk = pred_field[k][geometry_mask]
                        tk = true_field[k][geometry_mask]
                        comp_metrics = _basic_metrics(pk, tk)
                        row.update({f"{m}_{comp}": v for m, v in comp_metrics.items()})

                    pred_mag = torch.linalg.vector_norm(pred_field, dim=0)[geometry_mask]
                    true_mag = torch.linalg.vector_norm(true_field, dim=0)[geometry_mask]
                    row.update(_basic_metrics(pred_mag, true_mag))

                    pred_vec_flat = pred_field[:, geometry_mask]
                    true_vec_flat = true_field[:, geometry_mask]
                    row.update(_directional_metrics(
                        pred_vec_flat, true_vec_flat, rel_threshold=angle_rel_threshold
                    ))

                    if compute_divergence:
                        mask_np = geometry_mask.cpu().numpy()
                        row["div_pred_mean_abs"] = _divergence_error(
                            pred_field.cpu().numpy(), mask_np)
                        row["div_true_mean_abs"] = _divergence_error(
                            true_field.cpu().numpy(), mask_np)
                else:
                    pred_mag = pred_ms[i].squeeze(0)[geometry_mask]
                    true_mag = true_ms[i].squeeze(0)[geometry_mask]
                    row.update(_basic_metrics(pred_mag, true_mag))

                results.append(row)

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_save_path) or ".", exist_ok=True)
    results_df.to_csv(results_save_path, index=False)
    print(f"\nPer-sample evaluation results saved to: {results_save_path}")

    print("\n--- Metric Summary (fracture voxels) ---")
    numeric_cols = [c for c in results_df.columns if c != "name"]
    print(f"{'Metric':<24} {'Mean':>12} {'Median':>12} {'Min':>12} {'Max':>12}")
    print("-" * 68)
    for col in numeric_cols:
        s = results_df[col].dropna()
        if s.empty or not np.issubdtype(s.dtype, np.number):
            continue
        print(f"{col:<24} {s.mean():>12.6g} {s.median():>12.6g} "
              f"{s.min():>12.6g} {s.max():>12.6g}")

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--input-folder", required=True, help="Folder of .mat files")
    parser.add_argument("--mask-folder", required=True, help="Folder of .npy/.csv target files")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model", choices=tuple(MODELS), default="attresunet")
    parser.add_argument("--in-channels", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--target-mode", choices=("vector", "magnitude", "auto"), default="auto",
        help="Must match how the checkpoint was trained; 'auto' (default) "
             "trusts the checkpoint's own metadata",
    )
    parser.add_argument(
        "--split", choices=("val", "test", "all"), default="val",
        help="'val'/'all' use the stratified split; 'test' treats all "
             "matched pairs as one held-out set (use this for the real-rock "
             "test folders)",
    )
    parser.add_argument("--angle-rel-threshold", type=float, default=0.01)
    parser.add_argument("--divergence", action="store_true",
                         help="Also compute the divergence diagnostic (slower)")
    parser.add_argument(
        "--output", default="results/eval_results.csv",
        help="Per-sample metrics CSV output path",
    )
    parser.add_argument(
        "--manifest-path", default=None,
        help="Path to manifest CSV to filter for perc=1",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_peek = torch.load(args.checkpoint, map_location="cpu")
    ckpt_target_mode = checkpoint_peek.get("target_mode", "magnitude")
    target_mode = ckpt_target_mode if args.target_mode == "auto" else args.target_mode
    del checkpoint_peek

    df = build_dataframe(args.input_folder, args.mask_folder, manifest_path=args.manifest_path)
    if args.split == "val":
        _, eval_df = train_val_split(df)
    else:
        eval_df = df

    loader = DataLoader(
        FractureDataset(eval_df, in_channels=args.in_channels, target_mode=target_mode),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
    )

    model, out_channels, _ = load_trained_model(
        MODELS[args.model], args.checkpoint, device, in_channels=args.in_channels,
    )
    evaluate_model(
        model, loader, device, args.output, out_channels=out_channels,
        angle_rel_threshold=args.angle_rel_threshold,
        compute_divergence=args.divergence,
    )


if __name__ == "__main__":
    main()
