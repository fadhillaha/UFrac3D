"""Estimates macroscopic fracture permeability from predicted flow fields using Darcys law."""

from __future__ import annotations


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))

import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import FLOW_AXIS, FractureDataset, build_dataframe, inverse_transform_torch, train_val_split
from evaluate import load_trained_model
from models import MODELS


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
    out_channels: int,
    lbm_csv_path: "str | None" = None,
    omega: float = 1.0,
    nx: int = 128,
    use_magnitude: bool = False,
    flow_axis: int = FLOW_AXIS,
) -> pd.DataFrame:
    """Estimate permeability per sample and compare to LBM reference values.

    Args:
        model: Trained model.
        loader: Data loader (``shuffle=False``).
        device: Compute device.
        results_save_path: Output CSV path.
        deltaP: Pressure drop across the domain (lattice units).
        out_channels: 3 for a vector checkpoint, 1 for a legacy magnitude one.
        lbm_csv_path: Optional CSV of reference permeabilities. The first
            column is treated as the sample id; ``Average Velocity`` and
            ``Permeability`` columns are read.
        omega: BGK relaxation parameter (sets the lattice viscosity).
        nx: Domain edge length in lattice units.
        use_magnitude: If True (and ``out_channels == 3``), use ``||u||``
            in place of ``u_{flow_axis}`` -- reproduces the original,
            approximate method for comparison.
        flow_axis: Which of the 3 predicted components is the streamwise
            one (see ``dataset.FLOW_AXIS`` -- verify with
            ``dataset.sanity_check_alignment`` before trusting this).

    Returns:
        The per-sample permeability results as a DataFrame.
    """
    invCs2 = 3.0
    nu_lu = (1.0 / omega - 0.5) / invCs2
    gradP = get_gradP(deltaP, nx=nx)

    method = "magnitude (approximate, legacy)" if (use_magnitude or out_channels == 1) else \
        f"streamwise component u[axis={flow_axis}] (direct)"
    print("Palabos-equivalent permeability evaluation")
    print(f"  omega   = {omega}")
    print(f"  nu_lu   = {nu_lu:.6f}")
    print(f"  deltaP  = {deltaP}")
    print(f"  gradP   = {gradP}")
    print(f"  method  = {method}")

    model.eval()
    results = []
    batch_size = loader.batch_size

    lbm_lookup: dict = {}
    if lbm_csv_path is not None:
        lbm_df = pd.read_csv(lbm_csv_path)
        lbm_df.columns = lbm_df.columns.str.strip()
        lbm_df = lbm_df.rename(columns={lbm_df.columns[0]: "sample_id"})
        lbm_df["sample_id"] = lbm_df["sample_id"].str.strip()
        
        # Support both old Palabos output names and new LBM simulator names
        perm_col = "Permeability" if "Permeability" in lbm_df.columns else "k_lattice"
        
        lbm_lookup = lbm_df.set_index("sample_id")[
            [perm_col]
        ].to_dict("index")
        
        # Rename the column inside the dict so the rest of the code works unchanged
        for d in lbm_lookup.values():
            d["Permeability"] = d.pop(perm_col)
            
        print(f"  LBM CSV : {len(lbm_df)} samples loaded (using '{perm_col}')")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(
            tqdm(loader, desc="Evaluating permeability")
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            pred_lu = inverse_transform_torch(outputs)  # sign-aware inverse
            true_lu = inverse_transform_torch(targets)

            for i in range(inputs.size(0)):
                sample_idx = batch_idx * batch_size + i
                name = loader.dataset.df.iloc[sample_idx]["name"]

                geometry_mask = inputs[i, 0] > 0
                void_fraction = geometry_mask.float().mean().item()

                if geometry_mask.sum() == 0:
                    print(f"  Warning: skipping '{name}' - no open voxels.")
                    continue

                if out_channels == 3:
                    pred_vec = pred_lu[i]   # (3,D,H,W)
                    true_vec = true_lu[i]
                    if use_magnitude:
                        pred_scalar = torch.linalg.vector_norm(pred_vec, dim=0)
                        true_scalar = torch.linalg.vector_norm(true_vec, dim=0)
                    else:
                        pred_scalar = pred_vec[flow_axis]
                        true_scalar = true_vec[flow_axis]
                else:
                    pred_scalar = pred_lu[i].squeeze(0)
                    true_scalar = true_lu[i].squeeze(0)

                void_pred = pred_scalar[geometry_mask]
                void_true = true_scalar[geometry_mask]

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
                        "method": "magnitude" if (use_magnitude or out_channels == 1) else "streamwise",
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
    parser.add_argument("--mask-folder", required=True, help="Folder of .npy/.csv target files")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model", choices=tuple(MODELS), default="attresunet")
    parser.add_argument("--in-channels", type=int, default=2, choices=(1, 2))
    parser.add_argument("--split", choices=("val", "test", "all"), default="val")
    parser.add_argument("--lbm-csv", default=None, help="Optional reference permeability CSV")
    parser.add_argument(
        "--delta-p", type=float, required=True,
        help="Pressure drop across the domain (lattice units) -- see the "
             "module docstring's note on the rescaled reference pressure",
    )
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument(
        "--use-magnitude", action="store_true",
        help="Use ||u|| instead of the streamwise component (reproduces "
             "the original approximate method, for comparison)",
    )
    parser.add_argument("--flow-axis", type=int, default=FLOW_AXIS,
                         help="Which predicted component is streamwise (0/1/2)")
    parser.add_argument(
        "--output", default="results/permeability_results.csv",
        help="Per-sample permeability CSV output path",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = build_dataframe(args.input_folder, args.mask_folder)

    if args.split == "val":
        _, eval_df = train_val_split(df)
    else:
        eval_df = df

    checkpoint_peek = torch.load(args.checkpoint, map_location="cpu")
    target_mode = checkpoint_peek.get("target_mode", "magnitude")
    del checkpoint_peek

    loader = DataLoader(
        FractureDataset(eval_df, in_channels=args.in_channels, target_mode=target_mode),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
    )

    model, out_channels, _ = load_trained_model(
        MODELS[args.model], args.checkpoint, device, in_channels=args.in_channels,
    )

    evaluate_permeability(
        model, loader, device,
        results_save_path=args.output,
        deltaP=args.delta_p,
        out_channels=out_channels,
        lbm_csv_path=args.lbm_csv,
        omega=args.omega,
        nx=args.nx,
        use_magnitude=args.use_magnitude,
        flow_axis=args.flow_axis,
    )


if __name__ == "__main__":
    main()
