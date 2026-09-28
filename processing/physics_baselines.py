"""Computes analytical permeability approximations (standard and local cubic laws) for comparison."""

from __future__ import annotations


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training')))

import argparse
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from dataset import APERTURE_AXIS, FLOW_AXIS, load_mat_file
from geometry_utils import (
    compute_aperture_field,
    compute_contact_fraction,
    estimate_hurst_via_psd,
    mean_wall_surface,
)

# Corrected cubic law calibration constants (Eq. 16, Dharmawan [22]).
ALPHA = 2.57
BETA = 1.57
GAMMA = 0.022
H0 = 4.57
R_OVER_A_RANGE = (0.05, 0.30)
HURST_RANGE = (0.3, 0.9)


def cubic_law_permeability(
    geom: np.ndarray, aperture_axis: int = APERTURE_AXIS
) -> dict:
    """Parallel-plate cubic law permeability.

    Reports permeability in **two conventions**, because it was not
    possible to determine from the paper alone 
    placeholders) which one your actual LBM/CNN ``k`` values use, and the
    two differ by a factor of ``H_full / a_h`` (the full padded voxel
    extent along the aperture axis, divided by the aperture) -- easily
    1-2 orders of magnitude here:

    * ``k`` = ``a_h^2 / 12`` -- the *intrinsic* / pore-velocity cubic law
      exactly as written in the paper (Section 2.10): permeability of the
      fracture's own internal flow, independent of how much solid rock
      pads the voxel cube around it.
    * ``k_domain_avg`` = ``a_h^3 / (12 * H_full)`` -- the same physics
      expressed in the **domain-averaged** convention that
      ``permeability.py`` / Eq. 15 of the paper actually use for the LBM
      and CNN permeabilities (``k = nu * <U> / gradP`` with ``<U>``
      averaged over the *whole* padded voxel volume, solid included).

    Compare a known-aperture sample's ``k_lbm`` against both columns in
    the output CSV to see which one your reference values match, then use
    ``--k-convention`` on :func:`run_baselines_on_folder` /
    ``physics_baselines.py``'s CLI accordingly.

    Args:
        geom: Binary geometry volume (1 = void, 0 = solid).
        aperture_axis: Wall-normal / gap axis.

    Returns:
        Dict with ``k`` (intrinsic, voxel^2), ``k_domain_avg`` (voxel^2),
        ``a_h`` (cubic-mean aperture, voxels), ``H_full`` (voxels) and
        ``mean_aperture`` (arithmetic mean, voxels).
    """
    aperture = compute_aperture_field(geom, aperture_axis=aperture_axis)
    a_h = float(np.mean(aperture ** 3)) ** (1.0 / 3.0)
    k = (a_h ** 2) / 12.0
    h_full = geom.shape[aperture_axis]
    k_domain_avg = (a_h ** 3) / (12.0 * h_full) if h_full > 0 else float("nan")
    return {
        "k": k, "k_domain_avg": k_domain_avg, "a_h": a_h, "H_full": h_full,
        "mean_aperture": float(aperture.mean()),
    }


def corrected_cubic_law_permeability(
    geom: np.ndarray, aperture_axis: int = APERTURE_AXIS
) -> dict:
    """LBM-calibrated corrected cubic law (paper Eq. 16).

    Same two-convention reporting as :func:`cubic_law_permeability`
    (``k`` intrinsic, ``k_domain_avg`` domain-averaged) -- see that
    function's docstring.

    Args:
        geom: Binary geometry volume.
        aperture_axis: Wall-normal / gap axis.

    Returns:
        Dict with ``k``, ``k_domain_avg``, the correction factor ``T``,
        the estimated roughness ``r`` (voxels), Hurst exponent ``H``,
        ``r_over_a``, and ``in_calibration_range`` (``r/a`` and ``H`` both
        inside the ranges the correlation was calibrated for -- see the
        module-level constants).
    """
    base = cubic_law_permeability(geom, aperture_axis=aperture_axis)
    a_h = base["a_h"]

    surf = mean_wall_surface(geom, aperture_axis=aperture_axis)
    r = float(np.std(surf))
    H = estimate_hurst_via_psd(surf)
    r_over_a = r / a_h if a_h > 0 else float("nan")

    T = np.exp(-ALPHA * (r_over_a ** BETA)) * (1.0 - GAMMA * (H0 - H))
    k = base["k"] * T
    k_domain_avg = base["k_domain_avg"] * T

    in_range = (
        R_OVER_A_RANGE[0] <= r_over_a <= R_OVER_A_RANGE[1]
        and HURST_RANGE[0] <= H <= HURST_RANGE[1]
    )
    return {
        "k": float(k), "k_domain_avg": float(k_domain_avg), "a_h": a_h,
        "T": float(T), "r": r, "H": H, "r_over_a": r_over_a,
        "in_calibration_range": bool(in_range),
    }


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise harmonic mean, 0 where either input is 0."""
    out = np.zeros_like(a, dtype=np.float64)
    both_open = (a > 0) & (b > 0)
    out[both_open] = 2.0 * a[both_open] * b[both_open] / (a[both_open] + b[both_open])
    return out


def local_cubic_law(
    geom: np.ndarray,
    aperture_axis: int = APERTURE_AXIS,
    flow_axis: int = FLOW_AXIS,
    delta_p: float = 1.0,
    mu: float = 1.0,
) -> dict:
    """Reynolds-lubrication (local cubic law) permeability.

    Solves ``div( a^3/(12 mu) grad p ) = 0`` on the 2-D aperture map with a
    cell-centred finite-volume scheme, harmonic-averaged ``a^3`` at faces,
    Dirichlet pressure at the first/last column along ``flow_axis``
    (matching the LBM inlet/outlet), no-flow (natural) lateral boundaries,
    and zero transmissivity through contact (zero-aperture) columns.

    Args:
        geom: Binary geometry volume.
        aperture_axis: Wall-normal / gap axis (removed to form the 2-D map).
        flow_axis: Which of the two remaining axes pressure is dropped
            across (must not equal ``aperture_axis``).
        delta_p: Pressure drop, inlet (``flow_axis`` index 0) minus outlet
            (last index), lattice units.
        mu: Dynamic viscosity (lattice units; use ``1.0`` and interpret
            the output ``k`` as ``k/mu`` if unsure -- viscosity only
            rescales the flux, not the pressure field).

    Returns:
        Dict with ``k`` (domain-averaged permeability, voxel^2,
        consistent with ``permeability.py``'s convention), the pressure
        field ``p`` (2-D array), total flow rate ``Q``, and the aperture
        field used.
    """
    if aperture_axis == flow_axis:
        raise ValueError("aperture_axis and flow_axis must differ")

    remaining_axes = [ax for ax in range(3) if ax != aperture_axis]
    # Put flow_axis first in the 2-D map so row index 0 / -1 are inlet/outlet.
    if remaining_axes[0] != flow_axis:
        remaining_axes = remaining_axes[::-1]
    lateral_axis = remaining_axes[1]

    aperture_full = compute_aperture_field(geom, aperture_axis=aperture_axis)
    # compute_aperture_field returns the array with the two remaining axes
    # in their *original relative order*; reorder to (flow, lateral) here.
    orig_order = [ax for ax in range(3) if ax != aperture_axis]
    if orig_order[0] != flow_axis:
        aperture = aperture_full.T
    else:
        aperture = aperture_full

    n0, n1 = aperture.shape
    a3 = aperture.astype(np.float64) ** 3
    open_cell = aperture > 0

    p_inlet, p_outlet = delta_p, 0.0
    n = n0 * n1

    def idx(i, j):
        return i * n1 + j

    rows, cols, vals = [], [], []
    b = np.zeros(n, dtype=np.float64)

    for i in range(n0):
        for j in range(n1):
            k = idx(i, j)
            is_boundary = i == 0 or i == n0 - 1

            if not open_cell[i, j]:
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = 0.0
                continue

            if is_boundary:
                rows.append(k); cols.append(k); vals.append(1.0)
                b[k] = p_inlet if i == 0 else p_outlet
                continue

            diag = 0.0
            neighbors = [(i - 1, j), (i + 1, j)]
            if j > 0:
                neighbors.append((i, j - 1))
            if j < n1 - 1:
                neighbors.append((i, j + 1))

            for (ni, nj) in neighbors:
                t = _harmonic(a3[i, j:j + 1], a3[ni, nj:nj + 1])[0] / (12.0 * mu)
                if t == 0.0:
                    continue
                diag -= t
                rows.append(k); cols.append(idx(ni, nj)); vals.append(t)

            rows.append(k); cols.append(k); vals.append(diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    p_flat = spla.spsolve(A, b)
    p = p_flat.reshape(n0, n1)

    # Total flow rate: sum of face fluxes leaving the prescribed inlet row.
    Q = 0.0
    for j in range(n1):
        if not (open_cell[0, j] and open_cell[1, j]):
            continue
        t = _harmonic(a3[0, j:j + 1], a3[1, j:j + 1])[0] / (12.0 * mu)
        Q += t * (p[0, j] - p[1, j])

    gradP = delta_p / (n0 - 1)
    # Darcy velocity = total flow rate / cross-sectional area (n1 x H_full),
    # NOT / full 3D volume (n0 x n1 x H_full) -- Q already integrates over
    # the cross-section, so dividing by the volume double-counts the
    # flow-direction extent n0. This matches permeability.py's own
    # convention: `void_pred.mean() * void_fraction` is a simple average
    # over void voxels (not a sum across the flow direction) times a
    # *volume* fraction that, for a translationally-uniform fracture,
    # equals the cross-sectional aperture fraction -- see the module
    # docstring / README for the full derivation and the self-test below.
    cross_section_area = n1 * geom.shape[aperture_axis]
    meanU = Q / cross_section_area if cross_section_area > 0 else float("nan")
    # nu = mu (kinematic == dynamic here since rho=1 in lattice units);
    # k = nu * <U> / gradP, matching permeability.py's convention exactly.
    k_domain_avg = mu * meanU / gradP if gradP != 0 else float("nan")

    # Cross-check value in the *intrinsic* (pore-velocity) convention used
    # by cubic_law_permeability()'s plain `k`: average velocity over the
    # OPEN part of the domain only, instead of the full padded volume.
    open_cross_section = float(np.sum(aperture[0, :] > 0)) or 1.0
    # use the inlet row's open width as a representative open aperture sum
    q_per_open_width = Q / open_cross_section if open_cross_section > 0 else float("nan")
    aperture_mean_open = float(aperture[aperture > 0].mean()) if np.any(aperture > 0) else float("nan")
    k_intrinsic = (
        mu * (q_per_open_width / aperture_mean_open) / gradP
        if gradP != 0 and aperture_mean_open and np.isfinite(aperture_mean_open)
        else float("nan")
    )

    return {"k": float(k_intrinsic), "k_domain_avg": float(k_domain_avg),
            "Q": float(Q), "p": p, "aperture": aperture,
            "mean_aperture": float(aperture.mean())}


def _self_test_uniform_slab() -> None:
    """Sanity check: a uniform-aperture slab has no roughness, so the local
    cubic law's finite-volume solve must exactly reproduce the closed-form
    cubic law -- in *both* conventions (intrinsic and domain-averaged),
    which is a stronger check than matching just one number."""
    D, H, W = 8, 16, 8
    geom = np.zeros((D, H, W), dtype=np.uint8)
    geom[:, 4:12, :] = 1  # uniform 8-voxel aperture everywhere
    cubic = cubic_law_permeability(geom, aperture_axis=1)
    local = local_cubic_law(geom, aperture_axis=1, flow_axis=0, delta_p=1.0)

    err_intrinsic = abs(local["k"] - cubic["k"]) / cubic["k"]
    err_domain = abs(local["k_domain_avg"] - cubic["k_domain_avg"]) / cubic["k_domain_avg"]
    print(f"[physics_baselines self-test] uniform slab:")
    print(f"  intrinsic:     cubic k={cubic['k']:.6f}   local k={local['k']:.6f}   "
          f"rel. err={err_intrinsic:.2e}")
    print(f"  domain-avg:    cubic k={cubic['k_domain_avg']:.6f}   "
          f"local k={local['k_domain_avg']:.6f}   rel. err={err_domain:.2e}")
    assert err_intrinsic < 1e-6, "local_cubic_law (intrinsic) should reproduce a_h^2/12 on a uniform slab"
    assert err_domain < 1e-6, "local_cubic_law (domain-avg) should reproduce a_h^3/(12*H_full) on a uniform slab"
    print("  OK: both conventions match the closed-form solution")


def run_baselines_on_folder(
    input_folder: str,
    lbm_csv: Optional[str] = None,
    lbm_id_col: str = "id",
    lbm_k_col: str = "k",
    aperture_axis: int = APERTURE_AXIS,
    flow_axis: int = FLOW_AXIS,
    delta_p: float = 1.0,
    k_convention: str = "domain_avg",
) -> pd.DataFrame:
    """Run all three baselines on every ``.mat`` file in a folder.

    Args:
        input_folder: Folder of geometry ``.mat`` files.
        lbm_csv: Optional CSV with an LBM reference permeability per sample
            (for the sMAPE columns). Matched on ``name``
            (file base name) vs ``lbm_id_col``.
        lbm_id_col: Column in ``lbm_csv`` holding the sample id.
        lbm_k_col: Column in ``lbm_csv`` holding the reference permeability
            (must be in the same voxel^2 units as ``k`` here -- convert
            first if your CSV stores mD).
        aperture_axis: Wall-normal axis.
        flow_axis: Streamwise axis.
        delta_p: Pressure drop used for the local cubic law solve.
        k_convention: ``"domain_avg"`` (default, matches
            ``permeability.py``'s ``k_ML``/``k_label``) or ``"intrinsic"``
            (the plain textbook ``a_h^2/12`` form) -- selects which
            permeability column is used for the sMAPE-vs-LBM comparison.
            Both conventions are always written to the output CSV
            regardless of this choice; see :func:`cubic_law_permeability`.

    Returns:
        Per-sample DataFrame with a column per baseline's ``k`` (both
        conventions) plus diagnostics (contact fraction, r/a, H,
        calibration-range flag).
    """
    if k_convention not in ("domain_avg", "intrinsic"):
        raise ValueError("k_convention must be 'domain_avg' or 'intrinsic'")
    k_key = "k_domain_avg" if k_convention == "domain_avg" else "k"

    lbm_lookup = {}
    if lbm_csv is not None:
        lbm_df = pd.read_csv(lbm_csv)
        lbm_lookup = dict(zip(lbm_df[lbm_id_col].astype(str), lbm_df[lbm_k_col]))

    rows = []
    for path in sorted(glob.glob(os.path.join(input_folder, "*.mat"))):
        name = os.path.splitext(os.path.basename(path))[0]
        geom = load_mat_file(path)

        cubic = cubic_law_permeability(geom, aperture_axis=aperture_axis)
        corrected = corrected_cubic_law_permeability(geom, aperture_axis=aperture_axis)
        contact_frac = compute_contact_fraction(
            compute_aperture_field(geom, aperture_axis=aperture_axis)
        )
        try:
            local = local_cubic_law(geom, aperture_axis=aperture_axis,
                                     flow_axis=flow_axis, delta_p=delta_p)
        except Exception as e:  # pragma: no cover - defensive, e.g. singular volume
            print(f"[physics_baselines] local cubic law failed for {name}: {e}")
            local = {"k": float("nan"), "k_domain_avg": float("nan")}

        row = {
            "name": name,
            "k_cubic": cubic[k_key],
            "k_cubic_intrinsic": cubic["k"],
            "k_cubic_domain_avg": cubic["k_domain_avg"],
            "k_corrected": corrected[k_key],
            "k_corrected_intrinsic": corrected["k"],
            "k_corrected_domain_avg": corrected["k_domain_avg"],
            "k_local": local[k_key],
            "k_local_intrinsic": local["k"],
            "k_local_domain_avg": local["k_domain_avg"],
            "mean_aperture": cubic["mean_aperture"],
            "H_full": cubic["H_full"],
            "contact_fraction": contact_frac,
            "r_over_a": corrected["r_over_a"],
            "hurst_estimate": corrected["H"],
            "in_calibration_range": corrected["in_calibration_range"],
        }
        if lbm_lookup:
            k_ref = lbm_lookup.get(name, float("nan"))
            row["k_lbm"] = k_ref
            for base_name in ("cubic", "corrected", "local"):
                k_pred = row[f"k_{base_name}"]
                if k_ref and k_ref > 0 and np.isfinite(k_pred):
                    row[f"sMAPE_{base_name}_vs_lbm"] = (
                        200.0 * abs(k_pred - k_ref) / (abs(k_pred) + abs(k_ref) + 1e-30)
                    )
                else:
                    row[f"sMAPE_{base_name}_vs_lbm"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute physics baselines for a folder of geometries."
    )
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--lbm-csv", default=None,
                         help="Optional CSV with a reference permeability per sample")
    parser.add_argument("--lbm-id-col", default="id")
    parser.add_argument("--lbm-k-col", default="k")
    parser.add_argument("--aperture-axis", type=int, default=APERTURE_AXIS)
    parser.add_argument("--flow-axis", type=int, default=FLOW_AXIS)
    parser.add_argument("--delta-p", type=float, default=1.0)
    parser.add_argument("--k-convention", choices=("domain_avg", "intrinsic"), default="domain_avg")
    parser.add_argument("--self-test", action="store_true",
                         help="Run the uniform-slab consistency check and exit")
    parser.add_argument("--output", default="results/physics_baselines.csv")
    args = parser.parse_args()

    if args.self_test:
        _self_test_uniform_slab()
        return

    df = run_baselines_on_folder(
        args.input_folder, lbm_csv=args.lbm_csv, lbm_id_col=args.lbm_id_col,
        lbm_k_col=args.lbm_k_col, aperture_axis=args.aperture_axis,
        flow_axis=args.flow_axis, delta_p=args.delta_p, k_convention=args.k_convention,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows -> {args.output}")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
