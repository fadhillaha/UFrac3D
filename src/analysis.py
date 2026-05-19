"""Compile summary tables and plots from evaluation result CSVs.

Reads one or more per-sample metric CSVs produced by ``evaluate.py`` and
writes, into the output directory:

* ``summary_overall.csv``    -- mean/std/median/P5/P95 per input CSV
* ``summary_by_roughness.csv``
* ``summary_by_aperture.csv``
* ``summary_by_variation.csv``
* one box-plot PNG per metric

Sample names follow the dataset convention
``H<roughness>a<aperture>[_G|_S]_<index>``:
``_G`` -> ``Shifted``, ``_S`` -> ``Different``, none -> ``Identical``.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: write files, no display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METRICS: List[str] = ["MAE", "RMSE", "RRMSE", "sMAPE"]


def parse_name(name_str: str) -> Tuple[object, object, str]:
    """Parse a dataset file name into (roughness, aperture, variation).

    Args:
        name_str: Sample base name, e.g. ``H75a20_G_03``.

    Returns:
        ``(roughness, aperture, variation)``. Roughness is the Hurst exponent
        (e.g. ``0.75``); aperture is an int; variation is one of
        ``Identical``/``Shifted``/``Different`` (or ``Other``/``Unknown``).
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


def load_results(csv_paths: List[str]) -> pd.DataFrame:
    """Load and concatenate evaluation CSVs, adding parsed feature columns.

    Args:
        csv_paths: One or more per-sample metric CSV paths.

    Returns:
        A single DataFrame with an added ``source`` column plus
        ``roughness``, ``aperture`` and ``variation``.
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


def make_boxplots(df: pd.DataFrame, out_dir: str) -> List[str]:
    """Write one box plot per metric (grouped by source) to ``out_dir``.

    Returns:
        The list of written PNG paths.
    """
    written = []
    for metric in _present_metrics(df):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sources = sorted(df["source"].unique())
        data = [df.loc[df["source"] == s, metric].dropna().values for s in sources]
        ax.boxplot(data, labels=sources, showfliers=True)
        ax.set_xlabel("Result set")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by result set")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        path = os.path.join(out_dir, f"boxplot_{metric}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


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
        table.to_csv(os.path.join(out_dir, fname))
        print(f"\n--- Summary by {group_col} ---")
        print(table.to_string(float_format="%.4e"))

    plots = make_boxplots(master, out_dir)
    print("\nWrote plots:")
    for p in plots:
        print(f"  {p}")
    print(f"\nAll outputs written to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile tables and plots from evaluation CSVs."
    )
    parser.add_argument(
        "csv", nargs="+", help="One or more per-sample metric CSVs"
    )
    parser.add_argument(
        "--out-dir", default="results", help="Directory for tables and plots"
    )
    args = parser.parse_args()
    run_analysis(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
