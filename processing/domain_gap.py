"""Computes statistical distances (Kolmogorov-Smirnov and Wasserstein) to quantify structural domain gaps."""

from __future__ import annotations


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))

import argparse
import glob
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from dataset import APERTURE_AXIS, load_mat_file
from geometry_utils import (
    compute_aperture_field,
    compute_contact_fraction,
    compute_edt,
    estimate_hurst_via_psd,
    mean_wall_surface,
)


def sample_descriptors(geom: np.ndarray, aperture_axis: int = APERTURE_AXIS) -> dict:
    """Per-sample descriptor values for the domain-gap comparison.

    Args:
        geom: Binary geometry volume.
        aperture_axis: Wall-normal / gap axis.

    Returns:
        Dict with ``aperture`` (flat array, one value per column),
        ``edt`` (flat array, one value per void voxel), ``contact_fraction``
        (scalar) and ``hurst`` (scalar).
    """
    aperture = compute_aperture_field(geom, aperture_axis=aperture_axis)
    edt = compute_edt(geom)
    void = geom > 0
    surf = mean_wall_surface(geom, aperture_axis=aperture_axis)
    return {
        "aperture": aperture.flatten(),
        "edt": edt[void].flatten(),
        "contact_fraction": compute_contact_fraction(aperture),
        "hurst": estimate_hurst_via_psd(surf),
    }


def collect_group_descriptors(
    folder: str, aperture_axis: int = APERTURE_AXIS, max_samples: "int | None" = None
) -> dict:
    """Pool descriptors across every ``.mat`` file in a folder.

    Args:
        folder: Folder of geometry ``.mat`` files.
        aperture_axis: Wall-normal / gap axis.
        max_samples: Optional cap (useful for a quick look on a huge folder).

    Returns:
        Dict with pooled ``aperture``/``edt`` arrays and per-sample lists
        ``contact_fractions``/``hursts``.
    """
    if folder.endswith(".mat"):
        paths = sorted(glob.glob(folder))
    else:
        paths = sorted(glob.glob(os.path.join(folder, "*.mat")))
        
    if max_samples:
        paths = paths[:max_samples]
    if not paths:
        raise ValueError(f"No .mat files found for {folder}")

    apertures, edts, contacts, hursts = [], [], [], []
    for path in paths:
        geom = load_mat_file(path)
        d = sample_descriptors(geom, aperture_axis=aperture_axis)
        apertures.append(d["aperture"])
        edts.append(d["edt"])
        contacts.append(d["contact_fraction"])
        hursts.append(d["hurst"])

    aperture_arr = np.concatenate(apertures)
    edt_arr = np.concatenate(edts)

    # Subsample to prevent Out-Of-Memory errors in ks_2samp for large datasets
    MAX_POINTS = 1_000_000
    if len(aperture_arr) > MAX_POINTS:
        aperture_arr = np.random.choice(aperture_arr, size=MAX_POINTS, replace=False)
    if len(edt_arr) > MAX_POINTS:
        edt_arr = np.random.choice(edt_arr, size=MAX_POINTS, replace=False)

    return {
        "n_samples": len(paths),
        "aperture": aperture_arr,
        "edt": edt_arr,
        "contact_fractions": np.array(contacts),
        "hursts": np.array(hursts),
    }


def compare_to_reference(ref: dict, group: dict) -> dict:
    """KS statistic + normalised Wasserstein distance, ``group`` vs ``ref``.

    Args:
        ref: Output of :func:`collect_group_descriptors` for the synthetic
            training set.
        group: Same, for one lithology.

    Returns:
        Dict with ``aperture_KS``, ``aperture_W1``, ``edt_KS``, ``edt_W1``
        (W1 normalised by the *reference* pooled standard deviation, per
        the paper), plus ``contact_fraction`` and ``hurst_exponent``
        (means over the group).
    """
    out = {"n_samples": group["n_samples"]}
    for desc in ("aperture", "edt"):
        ref_vals, grp_vals = ref[desc], group[desc]
        ks_stat, _p = ks_2samp(grp_vals, ref_vals)
        w1_raw = wasserstein_distance(grp_vals, ref_vals)
        ref_std = np.std(ref_vals)
        w1_norm = w1_raw / ref_std if ref_std > 0 else float("nan")
        out[f"{desc}_KS"] = float(ks_stat)
        out[f"{desc}_W1"] = float(w1_norm)
    out["contact_fraction"] = float(np.mean(group["contact_fractions"]))
    out["hurst_exponent_mean"] = float(np.mean(group["hursts"]))
    out["hurst_exponent_std"] = float(np.std(group["hursts"]))
    return out


def make_distribution_figure(
    ref: dict, groups: Dict[str, dict], out_path: str
) -> str:
    """Write the synthetic-vs-lithology distribution comparison figure
    (paper Figure 25): one panel per descriptor (aperture, EDT).

    Args:
        ref: Synthetic training set descriptors.
        groups: ``{lithology_name: descriptors}``.
        out_path: PNG output path.

    Returns:
        ``out_path``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    descriptors = [("aperture", "Column aperture (voxels)"),
                   ("edt", "Euclidean distance transform (voxels)")]

    for ax, (key, xlabel) in zip(axes, descriptors):
        ref_vals = ref[key]
        bins = np.linspace(
            min(ref_vals.min(), *(g[key].min() for g in groups.values())),
            max(ref_vals.max(), *(g[key].max() for g in groups.values())),
            40,
        )
        ax.hist(ref_vals, bins=bins, density=True, histtype="step",
                lw=2, label="Synthetic (training)", color="black")
        for name, g in groups.items():
            ax.hist(g[key], bins=bins, density=True, histtype="step",
                     lw=1.5, label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle("Synthetic vs. real-rock distributions of the network's input descriptors")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_domain_gap(
    synthetic_folder: str,
    lithology_folders: Dict[str, str],
    aperture_axis: int = APERTURE_AXIS,
    max_samples: "int | None" = None,
) -> pd.DataFrame:
    """Full computation across every configured lithology.

    Args:
        synthetic_folder: Folder of synthetic training ``.mat`` files.
        lithology_folders: ``{lithology_name: folder_of_mat_files}``.
        aperture_axis: Wall-normal / gap axis.
        max_samples: Optional per-group cap.

    Returns:
        A DataFrame with one row per lithology plus a ``Synthetic`` row
        (self-comparison, included as a sanity check -- its KS/W1 should
        come out ~0).
    """
    print(f"Collecting synthetic descriptors from {synthetic_folder} ...")
    ref = collect_group_descriptors(synthetic_folder, aperture_axis=aperture_axis,
                                     max_samples=max_samples)
    print(f"  {ref['n_samples']} synthetic samples, "
          f"{len(ref['aperture'])} aperture values, {len(ref['edt'])} EDT values")

    rows = []
    self_row = compare_to_reference(ref, ref)
    self_row["group"] = "Synthetic (self-check)"
    rows.append(self_row)

    groups_for_figure = {}
    for name, folder in lithology_folders.items():
        print(f"Collecting '{name}' descriptors from {folder} ...")
        g = collect_group_descriptors(folder, aperture_axis=aperture_axis,
                                       max_samples=max_samples)
        print(f"  {g['n_samples']} samples")
        row = compare_to_reference(ref, g)
        row["group"] = name
        rows.append(row)
        groups_for_figure[name] = g

    df = pd.DataFrame(rows).set_index("group")
    df.attrs["_ref"] = ref
    df.attrs["_groups"] = groups_for_figure
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify the synthetic-vs-real-rock domain gap (statistical tables)."
    )
    parser.add_argument("--synthetic-folder", required=True)
    parser.add_argument(
        "--lithology", action="append", default=[],
        metavar="NAME=FOLDER",
        help="Repeatable, e.g. --lithology Andesite=/path/to/andesite/mat",
    )
    parser.add_argument("--aperture-axis", type=int, default=APERTURE_AXIS)
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Optional cap per group, for a quick look")
    parser.add_argument("--output", default="results/domain_gap.csv")
    parser.add_argument("--figure", default="results/figure25_domain_gap.png")
    args = parser.parse_args()

    lithology_folders = {}
    for item in args.lithology:
        if "=" not in item:
            raise ValueError(f"--lithology expects NAME=FOLDER, got: {item}")
        name, folder = item.split("=", 1)
        lithology_folders[name] = folder
    if not lithology_folders:
        raise ValueError("Pass at least one --lithology NAME=FOLDER")

    df = run_domain_gap(
        args.synthetic_folder, lithology_folders,
        aperture_axis=args.aperture_axis, max_samples=args.max_samples,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.drop(columns=[], errors="ignore").to_csv(args.output)
    print(f"\nWrote {args.output}")
    print(df.to_string(float_format="%.4f"))

    os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)
    make_distribution_figure(df.attrs["_ref"], df.attrs["_groups"], args.figure)
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
