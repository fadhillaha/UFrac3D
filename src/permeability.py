"""Permeability estimation from predicted velocity fields.

For each sample the void-fraction-weighted mean lattice-unit velocity over
the open (fracture) voxels is converted into a Darcy permeability using the
lattice viscosity and a pressure gradient, then optionally compared against
lattice-Boltzmann (LBM/Palabos) reference values from a CSV.

The Palabos-equivalent permeability is

    k = nu_lu * <U>_lu / gradP

with ``nu_lu = (1/omega - 0.5) / 3`` and ``<U>`` the void-fraction-weighted
mean velocity over the open voxels.
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

SCALING_FACTOR: float = 1000.0


def get_gradP(deltaP: float, nx: int = 128) -> float:
    """Return the pressure gradient from a pressure drop and domain length.

    Args:
        deltaP: Pressure drop across the domain (lattice units).
        nx: Domain edge length in lattice units.

    Returns:
        The pressure gradient ``deltaP / (nx - 1)`` in lattice units.
    """
    return deltaP / (nx - 1)


def evaluate_permeability(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    results_save_path: str,
    deltaP: float,
    lbm_csv_path: str | None = None,
    omega: float = 1.0,
    nx: int = 128,
) -> pd.DataFrame:
    """Estimate permeability per sample and compare to LBM reference values.

    Args:
        model: Trained model.
        loader: Data loader (``shuffle=False``).
        device: Compute device.
        results_save_path: Output CSV path.
        deltaP: Pressure drop across the domain (lattice units).
        lbm_csv_path: Optional CSV of reference permeabilities. The first
            column is treated as the sample id; ``Average Velocity`` and
            ``Permeability`` columns are read.
        omega: BGK relaxation parameter (sets the lattice viscosity).
        nx: Domain edge length in lattice units.

    Returns:
        The per-sample permeability results as a DataFrame.
    """
    invCs2 = 3.0
    nu_lu = (1.0 / omega - 0.5) / invCs2
    gradP = get_gradP(deltaP, nx=nx)

    print("Palabos-equivalent permeability evaluation")
    print(f"  omega   = {omega}")
    print(f"  nu_lu   = {nu_lu:.6f}")
    print(f"  deltaP  = {deltaP}")
    print(f"  gradP   = {gradP}")

    model.eval()
    results = []
    batch_size = loader.batch_size

    lbm_lookup: dict = {}
    if lbm_csv_path is not None:
        lbm_df = pd.read_csv(lbm_csv_path)
        lbm_df.columns = lbm_df.columns.str.strip()
        lbm_df = lbm_df.rename(columns={lbm_df.columns[0]: "sample_id"})
        lbm_df["sample_id"] = lbm_df["sample_id"].str.strip()
        lbm_lookup = lbm_df.set_index("sample_id")[
            ["Average Velocity", "Permeability"]
        ].to_dict("index")
        print(f"  LBM CSV : {len(lbm_df)} samples loaded")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(
            tqdm(loader, desc="Evaluating permeability")
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            outputs_sqrt = outputs / SCALING_FACTOR
            targets_sqrt = targets / SCALING_FACTOR

            pred_vel_lu = torch.square(outputs_sqrt)
            true_vel_lu = torch.square(targets_sqrt)

            for i in range(inputs.size(0)):
                sample_idx = batch_idx * batch_size + i
                name = loader.dataset.df.iloc[sample_idx]["name"]

                geometry_mask = inputs[i, 0] > 0
                void_fraction = geometry_mask.float().mean().item()

                pred_field = pred_vel_lu[i].squeeze(0)
                true_field = true_vel_lu[i].squeeze(0)

                void_pred = pred_field[geometry_mask]
                void_true = true_field[geometry_mask]

                if void_true.numel() == 0:
                    print(f"  Warning: skipping '{name}' - no open voxels.")
                    continue

                meanU_pred_lu = (void_pred.mean() * void_fraction).item()
                meanU_true_lu = (void_true.mean() * void_fraction).item()

                k_ML_lu = nu_lu * meanU_pred_lu / gradP
                k_label_lu = nu_lu * meanU_true_lu / gradP

                if name in lbm_lookup:
                    k_csv_lu = float(lbm_lookup[name]["Permeability"])
                else:
                    k_csv_lu = float("nan")

                if k_csv_lu and k_csv_lu > 0:
                    err_ml = abs(k_ML_lu - k_csv_lu) / k_csv_lu * 100
                    err_label = abs(k_label_lu - k_csv_lu) / k_csv_lu * 100
                else:
                    err_ml = err_label = float("nan")

                results.append(
                    {
                        "name": name,
                        "k_ML": k_ML_lu,
                        "k_label": k_label_lu,
                        "k_csv": k_csv_lu,
                        "k_err_ML_vs_csv_%": err_ml,
                        "k_err_label_vs_csv_%": err_label,
                        "gradP_used": gradP,
                        "deltaP_used": deltaP,
                    }
                )

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_save_path) or ".", exist_ok=True)
    results_df.to_csv(results_save_path, index=False)
    print(f"\nSaved: {results_save_path}  ({len(results_df)} samples)")

    print("\n-- Summary by deltaP " + "-" * 40)
    print(
        results_df.groupby("deltaP_used")
        .agg(
            count=("k_csv", "count"),
            k_ML_mean=("k_ML", "mean"),
            k_label_mean=("k_label", "mean"),
            k_csv_mean=("k_csv", "mean"),
            err_ML_mean=("k_err_ML_vs_csv_%", "mean"),
            err_label_mean=("k_err_label_vs_csv_%", "mean"),
        )
        .round(4)
    )

    summary_cols = [
        ("k_ML", "k_ML    [lu^2]"),
        ("k_label", "k_label [lu^2]"),
        ("k_csv", "k_csv   [lu^2]"),
        ("k_err_ML_vs_csv_%", "ML  err vs CSV (%)"),
        ("k_err_label_vs_csv_%", "label err vs CSV (%)"),
    ]
    print(f"\n{'':30s} {'Mean':>12} {'Median':>12} {'Min':>12} {'Max':>12}")
    print("-" * 82)
    for col, label in summary_cols:
        s = results_df[col].dropna()
        if s.empty:
            continue
        print(
            f"  {label:<28} {s.mean():>12.4e} {s.median():>12.4e} "
            f"{s.min():>12.4e} {s.max():>12.4e}"
        )

    missing = results_df[results_df["k_csv"].isna()]
    if len(missing):
        print(f"\n  !! {len(missing)} samples not found in LBM CSV:")
        print(missing["name"].tolist())

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate permeability from predicted velocity fields."
    )
    parser.add_argument("--input-folder", required=True, help="Folder of .mat files")
    parser.add_argument("--mask-folder", required=True, help="Folder of .csv files")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model", choices=tuple(MODELS), default="unet3d")
    parser.add_argument("--in-channels", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--split", choices=("val", "test", "all"), default="val"
    )
    parser.add_argument(
        "--lbm-csv", default=None, help="Optional reference permeability CSV"
    )
    parser.add_argument(
        "--delta-p",
        type=float,
        required=True,
        help="Pressure drop across the domain (lattice units)",
    )
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument(
        "--output",
        default="results/permeability_results.csv",
        help="Per-sample permeability CSV output path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = build_dataframe(args.input_folder, args.mask_folder)

    if args.split == "val":
        _, eval_df = train_val_split(df)
    else:
        eval_df = df

    loader = DataLoader(
        FractureDataset(eval_df, in_channels=args.in_channels),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MODELS[args.model](in_channels=args.in_channels, out_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    evaluate_permeability(
        model,
        loader,
        device,
        results_save_path=args.output,
        deltaP=args.delta_p,
        lbm_csv_path=args.lbm_csv,
        omega=args.omega,
        nx=args.nx,
    )


if __name__ == "__main__":
    main()
