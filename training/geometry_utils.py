"""Utilities for processing 3D fracture geometry, including Euclidean Distance Transform (EDT)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_edt(void_mask: np.ndarray) -> np.ndarray:
    """Euclidean distance transform of a binary void mask.

    Args:
        void_mask: Array where nonzero/True marks void (fracture) voxels.

    Returns:
        Float32 array, same shape, distance in voxels from each void voxel
        to the nearest solid voxel (0 at solid voxels), matching Section 2.4
        of the paper (computed on void voxels only, no normalisation).
    """
    void_mask = np.asarray(void_mask) > 0
    if not np.any(void_mask):
        return np.zeros(void_mask.shape, dtype=np.float32)
    return distance_transform_edt(void_mask).astype(np.float32)


def compute_aperture_field(
    geom: np.ndarray, aperture_axis: int = 1
) -> np.ndarray:
    """Local (column-wise) aperture, in voxels, along ``aperture_axis``.

    For every column parallel to ``aperture_axis`` the aperture is the
    number of void voxels in that column. This is exact for a single,
    simply-connected gap (no vertically-stacked separate cavities) and is
    the standard "count the open cells in the column" definition used for
    the cubic-law baselines (Section 2.10) and the contact-fraction / EDT
    domain-gap descriptors (Section 3.5, Table 14).

    Args:
        geom: Binary geometry volume, ``1`` = void, ``0`` = solid.
        aperture_axis: Axis along which the two fracture walls face each
            other (default ``1``, i.e. the H axis -- see module docstring).

    Returns:
        A 2-D float32 array (the two remaining axes, in their original
        order) with the aperture, in voxels, of each column.
    """
    geom = np.asarray(geom) > 0
    return geom.sum(axis=aperture_axis).astype(np.float32)


def compute_contact_fraction(aperture_field: np.ndarray) -> float:
    """Fraction of columns with zero aperture (opposing walls touching).

    Args:
        aperture_field: Output of :func:`compute_aperture_field`.

    Returns:
        A value in ``[0, 1]``.
    """
    aperture_field = np.asarray(aperture_field)
    if aperture_field.size == 0:
        return float("nan")
    return float(np.mean(aperture_field <= 0))


def extract_wall_heights(
    geom: np.ndarray, aperture_axis: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Top- and bottom-wall height maps along ``aperture_axis``.

    For every column, the "bottom wall" height is the index of the first
    void voxel and the "top wall" height is one past the index of the last
    void voxel, i.e. the two surfaces bounding the local gap. Columns with
    no void voxels (full contact) get ``NaN`` in both maps.

    Args:
        geom: Binary geometry volume, ``1`` = void, ``0`` = solid.
        aperture_axis: Axis along which the walls face each other.

    Returns:
        ``(bottom_height, top_height)``, each a 2-D float array.
    """
    geom = np.asarray(geom) > 0
    moved = np.moveaxis(geom, aperture_axis, 0)  # (gap_axis, a1, a2)
    n = moved.shape[0]
    idx = np.arange(n).reshape((n,) + (1,) * (moved.ndim - 1))
    has_void = moved.any(axis=0)

    big = n + 1
    first_idx = np.where(moved, idx, big).min(axis=0).astype(np.float64)
    last_idx = np.where(moved, idx, -1).max(axis=0).astype(np.float64)

    bottom = np.where(has_void, first_idx, np.nan)
    top = np.where(has_void, last_idx + 1, np.nan)
    return bottom.astype(np.float32), top.astype(np.float32)


def mean_wall_surface(geom: np.ndarray, aperture_axis: int = 1) -> np.ndarray:
    """A single representative rough-surface height map for PSD/Hurst work.

    Averages the (mid-plane-centred) top and bottom wall height maps into
    one 2-D roughness signal. Using the average of both walls rather than
    just one keeps the estimate meaningful for the "different roughness
    profile on each wall" geometry category described in Section 2.1.

    Args:
        geom: Binary geometry volume.
        aperture_axis: Axis along which the walls face each other.

    Returns:
        A 2-D float array with the mean removed (zero-mean roughness
        signal), NaNs (fully-closed columns) replaced by the local median
        so the FFT below has no missing values.
    """
    bottom, top = extract_wall_heights(geom, aperture_axis=aperture_axis)
    surf = np.where(np.isnan(bottom), np.nan, (top - bottom) / 2.0 + bottom)
    if np.all(np.isnan(surf)):
        return np.zeros_like(surf)
    fill = np.nanmedian(surf)
    surf = np.where(np.isnan(surf), fill, surf)
    return surf - np.nanmean(surf)


def estimate_hurst_via_psd(
    height_map: np.ndarray, min_freq_bins: int = 4
) -> float:
    """Hurst exponent from the radially-averaged power spectral density.

    For a self-affine surface with Hurst exponent ``H``, the 2-D PSD falls
    off as ``P(k) ~ k^-(2H+2)``. This fits that power law to the radially
    (isotropically) averaged PSD of ``height_map`` in log-log space, i.e.
    exactly "the Hurst exponent recovered from the radially averaged power
    spectral density of the wall height map" described for the corrected
    cubic-law baseline (Section 2.10).

    Args:
        height_map: 2-D roughness signal, e.g. from :func:`mean_wall_surface`.
        min_freq_bins: Minimum number of radial bins required to fit; if
            the map is too small, ``0.5`` (a neutral default in the
            0.5-0.75 training range) is returned with a warning.

    Returns:
        The estimated Hurst exponent (not clipped -- values outside
        ``[0, 1]`` indicate the fit is unreliable, e.g. because the volume
        is too small or the surface is not well-resolved).
    """
    h = np.asarray(height_map, dtype=np.float64)
    ny, nx = h.shape
    if min(ny, nx) < 8:
        print("[geometry_utils] height map too small for a PSD fit; "
              "returning default H=0.5")
        return 0.5

    window = np.outer(np.hanning(ny), np.hanning(nx))
    hw = (h - h.mean()) * window
    fft = np.fft.fftshift(np.fft.fft2(hw))
    psd2d = np.abs(fft) ** 2

    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64)

    r_max = min(cy, cx)
    radial_psd = np.array(
        [psd2d[(r == rad)].mean() for rad in range(1, r_max)]
    )
    freqs = np.arange(1, r_max)

    if len(freqs) < min_freq_bins:
        print("[geometry_utils] not enough radial bins for a PSD fit; "
              "returning default H=0.5")
        return 0.5

    valid = radial_psd > 0
    if valid.sum() < min_freq_bins:
        print("[geometry_utils] radially-averaged PSD has too few nonzero "
              "bins for a reliable fit (e.g. a perfectly flat/unresolved "
              "surface); returning default H=0.5")
        return 0.5
    log_f = np.log(freqs[valid])
    log_p = np.log(radial_psd[valid])
    slope, _intercept = np.polyfit(log_f, log_p, 1)
    hurst = (-slope - 2.0) / 2.0
    return float(hurst)


def describe_axes(geom: np.ndarray, aperture_axis: int = 1) -> dict:
    """Print + return quick diagnostics to sanity-check the axis convention.

    Run this on one or two samples with a known/expected aperture (e.g. a
    synthetic sample named ``H70a25_...`` should show a mean aperture near
    25 voxels) before trusting ``aperture_axis`` for a whole dataset.

    Args:
        geom: Binary geometry volume.
        aperture_axis: Candidate aperture axis to test.

    Returns:
        A dict with void fraction and per-axis "column occupancy" stats,
        also printed to stdout.
    """
    geom = np.asarray(geom) > 0
    void_frac = float(geom.mean())
    out = {"void_fraction": void_frac, "shape": geom.shape}
    for axis in range(geom.ndim):
        col = geom.sum(axis=axis)
        out[f"axis{axis}_column_mean"] = float(col.mean())
        out[f"axis{axis}_column_std"] = float(col.std())
    aperture = compute_aperture_field(geom, aperture_axis=aperture_axis)
    out["assumed_aperture_axis"] = aperture_axis
    out["mean_aperture_voxels"] = float(aperture.mean())
    out["contact_fraction"] = compute_contact_fraction(aperture)

    print(f"  shape={out['shape']}  void_fraction={void_frac:.4f}")
    if void_frac > 0.5:
        print("  !! void_fraction > 0.5 -- for a single fracture this "
              "usually means the void/solid convention is INVERTED. "
              "Check whether 1 means void or solid in this .mat file "
              "(paper convention: 1=void, 0=solid -- see README).")
    for axis in range(geom.ndim):
        print(f"  axis={axis}: mean column occupancy "
              f"{out[f'axis{axis}_column_mean']:.2f} "
              f"(std {out[f'axis{axis}_column_std']:.2f})"
              + ("   <- lowest std, likely the aperture axis"
                 if axis == int(np.argmin([out[f'axis{a}_column_std']
                                            for a in range(geom.ndim)]))
                 else ""))
    print(f"  with aperture_axis={aperture_axis}: mean aperture "
          f"{out['mean_aperture_voxels']:.2f} vox, "
          f"contact fraction {out['contact_fraction']:.4f}")
    return out
