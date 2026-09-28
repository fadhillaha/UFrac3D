"""Aggregates error metrics across samples to generate statistical tables and error distribution plots."""

from __future__ import annotations


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))

import argparse
import os
import re
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def _boxplot(ax, data, labels):
    """``ax.boxplot`` wrapper tolerant of the matplotlib 3.9+ rename of the
    ``labels`` kwarg to ``tick_labels`` (avoids a deprecation warning on new
    matplotlib while still working on older versions)."""
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=True)

BASE_METRICS: List[str] = ["MAE", "RMSE", "RRMSE", "sMAPE"]
COMPONENT_METRICS: List[str] = [
    f"{m}_{c}" for c in ("ux", "uy", "uz") for m in BASE_METRICS
]
DIRECTIONAL_METRICS: List[str] = [
    "cos_sim_mean", "cos_sim_median", "angerr_mean_deg", "angerr_median_deg",
    "frac_within_10deg", "frac_within_30deg",
    "div_pred_mean_abs", "div_true_mean_abs",
]
METRICS: List[str] = BASE_METRICS + COMPONENT_METRICS + DIRECTIONAL_METRICS


def parse_name(name_str: str) -> Tuple[object, object, str]:
    """Parse a synthetic-dataset file name into (roughness, aperture, variation).

    Args:
        name_str: Sample base name, e.g. ``H75a20_G_03``.

    Returns:
        ``(roughness, aperture, variation)``. Roughness is the Hurst
        exponent (e.g. ``0.75``); aperture is an int; variation is one of
        ``Identical``/``Shifted``/``Different`` (or ``Other``/``Unknown``
        for non-matching, e.g. real-rock, names).
    """
    name_str = str(name_str)
    pattern = re.compile(r"^H(\d+)a(\d+)(_G|_S)?_(\d+)$")
    match = pattern.match(name_str)
    if not match:
        return "Unknown", "Unknown", "Other"

    roughness = float(match.group(1)) / 100.0
    aperture = int(match.group(2))
    variation_code = match.group(3)

    if variation_code == "_G":
        variation = "Shifted"
    elif variation_code == "_S":
        variation = "Different"
    elif variation_code is None:
        variation = "Identical"
    else:
        variation = "Other"

    return roughness, aperture, variation


def parse_rocktype(name_str: str) -> str:
    """Guess the lithology from a real-rock sample's file name.

    Args:
        name_str: Sample base name, e.g. ``AndesiteFrac_15_004``.

    Returns:
        ``"Andesite"``, ``"Granite"``, ``"Shale"``, or ``"Unknown"``
        (synthetic ``H..a..`` names also come back ``"Unknown"`` here --
        use :func:`parse_name` for those).
    """
    s = str(name_str).lower()
    if "andesite" in s:
        return "Andesite"
    if "granite" in s or "ag" in s:
        return "Granite"
    if "shale" in s or "inducedfracture" in s:
        return "Shale"
    return "Unknown"


def load_results(csv_paths: List[str]) -> pd.DataFrame:
    """Load and concatenate evaluation CSVs, adding parsed feature columns.

    Args:
        csv_paths: One or more per-sample metric CSV paths.

    Returns:
        A single DataFrame with an added ``source`` column plus
        ``roughness``, ``aperture``, ``variation`` and ``rocktype``.
    """
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df["source"] = os.path.splitext(os.path.basename(path))[0]
        frames.append(df)
    master = pd.concat(frames, ignore_index=True)
    master[["roughness", "aperture", "variation"]] = master["name"].apply(
        lambda x: pd.Series(parse_name(x))
    )
    master["rocktype"] = master["name"].apply(parse_rocktype)
    return master


def _present_metrics(df: pd.DataFrame) -> List[str]:
    return [m for m in METRICS if m in df.columns]


def overall_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/median/P5/P95 per source CSV for every available metric."""
    metrics = _present_metrics(df)
    agg = {
        m: [
            "mean",
            "std",
            "median",
            ("P5", lambda x: x.quantile(0.05)),
            ("P95", lambda x: x.quantile(0.95)),
        ]
        for m in metrics
    }
    return df.groupby("source").agg(agg)


def grouped_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean of each metric grouped by ``source`` and ``group_col``.

    Rows where the grouping key is ``Unknown``/``Other`` are dropped.
    """
    metrics = _present_metrics(df)
    sub = df[~df[group_col].isin(["Unknown", "Other"])]
    return (
        sub.groupby(["source", group_col])
        .agg({m: "mean" for m in metrics})
        .sort_index()
    )


def rocktype_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-source, per-rock-type mean of each metric ."""
    return grouped_table(df, "rocktype")


def make_boxplots(df: pd.DataFrame, out_dir: str, prefix: str = "boxplot") -> List[str]:
    """Write one box plot per metric (grouped by source) to ``out_dir``.

    Returns:
        The list of written PNG paths.
    """
    written = []
    for metric in _present_metrics(df):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sources = sorted(df["source"].unique())
        data = [df.loc[df["source"] == s, metric].dropna().values for s in sources]
        if all(len(d) == 0 for d in data):
            plt.close(fig)
            continue
        _boxplot(ax, data, sources)
        ax.set_xlabel("Result set")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by result set")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        path = os.path.join(out_dir, f"{prefix}_{metric}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def make_rocktype_boxplots(df: pd.DataFrame, out_dir: str) -> List[str]:
    """Box plots grouped by rock type, one panel per rock type per metric
    (paper Figures 19-21 shape): for each metric, one figure with a subplot
    per rock type, each subplot showing all sources side by side.

    Returns:
        The list of written PNG paths.
    """
    written = []
    sub = df[df["rocktype"] != "Unknown"]
    if sub.empty:
        return written
    rock_types = sorted(sub["rocktype"].unique())
    sources = sorted(sub["source"].unique())

    for metric in _present_metrics(sub):
        fig, axes = plt.subplots(1, len(rock_types), figsize=(5 * len(rock_types), 5),
                                  sharey=True)
        if len(rock_types) == 1:
            axes = [axes]
        for ax, rt in zip(axes, rock_types):
            rt_df = sub[sub["rocktype"] == rt]
            data = [rt_df.loc[rt_df["source"] == s, metric].dropna().values for s in sources]
            _boxplot(ax, data, sources)
            ax.set_title(rt)
            ax.tick_params(axis="x", rotation=20)
        axes[0].set_ylabel(metric)
        fig.suptitle(f"{metric} by rock type and result set")
        plt.tight_layout()
        path = os.path.join(out_dir, f"boxplot_rocktype_{metric}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def parity_plot(
    perm_csv_path: str,
    out_path: str,
    pred_col: str = "k_ML",
    true_col: str = "k_label",
    rocktype_col: Optional[str] = None,
    band_pct: float = 20.0,
    log_scale: bool = True,
    title: Optional[str] = None,
) -> str:
    """Draw a k_ML vs k_LBM parity plot (paper Figures 23/24 shape).

    Args:
        perm_csv_path: Output CSV of ``permeability.py``.
        out_path: Where to write the PNG.
        pred_col: Column with the predicted permeability.
        true_col: Column with the reference permeability (use ``"k_csv"``
            for the actual LBM reference if an ``--lbm-csv`` was supplied
            to ``permeability.py``; ``"k_label"`` compares against the
            *ground-truth field's own* estimate instead, which is always
            available even without a separate LBM summary CSV).
        rocktype_col: If given (or if a ``name`` column is present), colour
            points by lithology, guessed via :func:`parse_rocktype`.
        band_pct: Half-width, in percent, of the shaded +/- band around
            the 1:1 line (paper uses +/-20%).
        log_scale: Log-log axes (paper convention; permeability spans
            orders of magnitude).
        title: Optional plot title (defaults to the file name).

    Returns:
        The path the figure was written to.
    """
    df = pd.read_csv(perm_csv_path)
    df = df.dropna(subset=[pred_col, true_col])
    if log_scale:
        df = df[(df[pred_col] > 0) & (df[true_col] > 0)]
    if df.empty:
        raise ValueError(f"No valid rows to plot in {perm_csv_path} "
                          f"(columns {pred_col}, {true_col})")

    if rocktype_col is None and "name" in df.columns:
        df = df.copy()
        df["rocktype"] = df["name"].apply(parse_rocktype)
        rocktype_col = "rocktype"

    fig, ax = plt.subplots(figsize=(6, 6))
    lo = min(df[pred_col].min(), df[true_col].min())
    hi = max(df[pred_col].max(), df[true_col].max())
    pad = (hi - lo) * 0.05 if hi > lo else max(abs(hi), 1.0) * 0.1
    lo, hi = lo - pad, hi + pad

    line = np.array([max(lo, 1e-30), hi])
    ax.plot(line, line, "k--", lw=1, label="1:1")
    if band_pct > 0:
        ax.fill_between(line, line * (1 - band_pct / 100), line * (1 + band_pct / 100),
                         color="gray", alpha=0.2, label=f"+/-{band_pct:.0f}%")

    if rocktype_col is not None and rocktype_col in df.columns:
        for rt, sub in df.groupby(rocktype_col):
            ax.scatter(sub[true_col], sub[pred_col], s=18, alpha=0.75, label=str(rt))
    else:
        ax.scatter(df[true_col], df[pred_col], s=18, alpha=0.75, label="samples")

    r2 = r2_score(df[true_col], df[pred_col])
    ax.text(0.05, 0.92, f"R$^2$ = {r2:.3f}", transform=ax.transAxes,
            fontsize=11, va="top")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(f"{true_col} (reference)")
    ax.set_ylabel(f"{pred_col} (predicted)")
    ax.set_title(title or os.path.splitext(os.path.basename(perm_csv_path))[0])
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_analysis(csv_paths: List[str], out_dir: str) -> None:
    """Load result CSVs and write summary tables and plots to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    master = load_results(csv_paths)

    overall = overall_table(master)
    overall.to_csv(os.path.join(out_dir, "summary_overall.csv"))
    print("\n--- Overall summary ---")
    print(overall.to_string(float_format="%.4e"))

    for group_col, fname in [
        ("roughness", "summary_by_roughness.csv"),
        ("aperture", "summary_by_aperture.csv"),
        ("variation", "summary_by_variation.csv"),
    ]:
        table = grouped_table(master, group_col)
        if not table.empty:
            table.to_csv(os.path.join(out_dir, fname))
            print(f"\n--- Summary by {group_col} ---")
            print(table.to_string(float_format="%.4e"))

    rt_table = rocktype_table(master)
    if not rt_table.empty:
        rt_table.to_csv(os.path.join(out_dir, "summary_by_rocktype.csv"))
        print("\n--- Summary by rock type ---")
        print(rt_table.to_string(float_format="%.4e"))

    plots = make_boxplots(master, out_dir)
    plots += make_rocktype_boxplots(master, out_dir)
    print("\nWrote plots:")
    for p in plots:
        print(f"  {p}")
    print(f"\nAll outputs written to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile tables and plots from evaluation CSVs."
    )
    parser.add_argument("csv", nargs="+", help="One or more per-sample metric CSVs")
    parser.add_argument("--out-dir", default="results", help="Directory for tables and plots")
    parser.add_argument(
        "--parity", default=None,
        help="Optional: also draw a parity plot from a permeability.py CSV "
             "(pass its path here)",
    )
    args = parser.parse_args()
    run_analysis(args.csv, args.out_dir)

    if args.parity:
        out_path = os.path.join(args.out_dir, "parity_plot.png")
        parity_plot(args.parity, out_path)
        print(f"\nParity plot -> {out_path}")


if __name__ == "__main__":
    main()
