"""Dataset utilities and PyTorch dataloaders for the fracture flow-prediction models."""

from __future__ import annotations

import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
import re
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Target components are signed-sqrt compressed then multiplied by this
# factor before training (see forward_transform_np / inverse_transform_torch
# below). Unchanged from the original pipeline's SCALING_FACTOR.
SCALING_FACTOR: float = 1000.0

# Component order convention: index k of the target = velocity along grid
# axis k of the (D, H, W) volume. Purely documentary here (nothing branches
# on it) but every downstream script (evaluate.py, permeability.py,
# physics_baselines.py) imports FLOW_AXIS from here so they can't disagree.
FLOW_AXIS: int = 0
APERTURE_AXIS: int = 1


# ---------------------------------------------------------------------------
# Signed range-compression transform (the critical fix for vector targets)
# ---------------------------------------------------------------------------


def forward_transform_np(x: np.ndarray, scale: float = SCALING_FACTOR) -> np.ndarray:
    """Signed sqrt-and-scale transform: ``sign(x) * sqrt(|x|) * scale``.

    Applied per-component to the raw simulation velocity before training.
    Reduces to the original pipeline's plain ``sqrt(x) * scale`` whenever
    ``x >= 0`` everywhere (i.e. it is a strict generalisation, not a
    behaviour change, for the legacy scalar-magnitude case).

    Args:
        x: Raw velocity array, any shape, any sign.
        scale: Post-sqrt multiplier (default ``SCALING_FACTOR``).

    Returns:
        The transformed array, same shape, dtype ``float32``.
    """
    x = np.asarray(x, dtype=np.float64)
    return (np.sign(x) * np.sqrt(np.abs(x)) * scale).astype(np.float32)


def inverse_transform_torch(x: torch.Tensor, scale: float = SCALING_FACTOR) -> torch.Tensor:
    """Exact inverse of :func:`forward_transform_np`, for torch tensors.

    ``inverse_transform_torch(forward_transform_np(x)) == x`` (up to
    float32 rounding), for any sign of ``x``.

    Args:
        x: Transformed tensor (model output or transformed target).
        scale: Must match the ``scale`` used in the forward pass.

    Returns:
        The recovered raw-velocity tensor, same shape.
    """
    x = x / scale
    return torch.sign(x) * torch.square(x)


# ---------------------------------------------------------------------------
# Geometry loading (unchanged from the original pipeline)
# ---------------------------------------------------------------------------


def load_mat_file(file_path: str) -> np.ndarray:
    """Load a ``.mat`` geometry file and return a binary ``float32`` volume.

    The geometry is read from the key ``sub_volume``; if absent, the key
    ``wadah`` is used. The volume is min-max normalised and thresholded at
    0.5 to produce a binary occupancy field, ``1`` = void (fracture),
    ``0`` = solid -- this is the paper's convention (Section 2.1), which is
    the OPPOSITE of the convention documented inside ``lbm_core.py``
    ("pore = 0, solid = 1"). Those two conventions are not actually in
    conflict: ``make_frac_manifest.py`` explicitly flips the array
    (``arr = (arr == 0)...  #flip``) when it repacks the *source* .mat into
    the LBM solver's internal cubes, so as long as this loader keeps
    reading the original, unflipped ``.mat`` files (not the packed
    ``cubes/*.npz``), the convention here is unchanged. If
    :func:`sanity_check_alignment` reports a void fraction above ~0.5,
    that is the first thing to check.

    Args:
        file_path: Path to the ``.mat`` file.

    Returns:
        A ``(D, H, W)`` ``float32`` array with values in ``{0.0, 1.0}``.
    """
    mat_data = scio.loadmat(file_path)
    try:
        data = mat_data["sub_volume"]
    except KeyError:
        data = mat_data["wadah"]

    min_val = np.min(data)
    max_val = np.max(data)
    if max_val > min_val:
        normalized_data = (data - min_val) / (max_val - min_val)
    else:
        normalized_data = data

    threshold = 0.5
    binary_data = (normalized_data > threshold).astype(np.float32)
    return binary_data


# ---------------------------------------------------------------------------
# Vector target loading (new)
# ---------------------------------------------------------------------------


def _reshape_csv_column(col: np.ndarray, dim: Tuple[int, int, int]) -> np.ndarray:
    """Reshape one flat CSV column the same way the old scalar loader did."""
    vol = col.reshape(dim)                 # (h, w, d)
    return np.transpose(vol, (2, 0, 1))    # (d, h, w)


def load_vector_field(file_path: str, dim: Tuple[int, int, int]) -> np.ndarray:
    """Load a 3-component velocity field, dispatching on file extension.

    Args:
        file_path: Path to a ``.npy`` or ``.csv`` velocity file.
        dim: ``(H, W, D)`` shape used for the CSV reshape path (matches the
            geometry's own ``(D, H, W)`` via ``D, H, W = geom.shape`` then
            ``dim=(H, W, D)`` at the call site, exactly as the original
            scalar loader expected).

    Returns:
        A ``(3, D, H, W)`` ``float32`` array, raw (untransformed) velocity,
        component ``k`` = velocity along grid axis ``k``.

    Raises:
        ValueError: If the file's shape can't be unambiguously interpreted
            as a 3-component field, or doesn't match ``dim``.
    """
    ext = os.path.splitext(file_path)[1].lower()
    d_expected, h_expected, w_expected = dim[2], dim[0], dim[1]

    if ext in [".npy", ".npz"]:
        # Handle compressed npz
        if ext == ".npz":
            data = np.load(file_path)
            key = 'velocity_vector' if 'velocity_vector' in data else list(data.keys())[0]
            arr = data[key]
        else:
            arr = np.load(file_path)
            
        if arr.ndim != 4:
            raise ValueError(
                f"{file_path}: expected a 4-D array (3, D, H, W) or "
                f"(D, H, W, 3), got shape {arr.shape}"
            )
            
        if arr.shape[0] == 3:
            vec = arr
        elif arr.shape[-1] == 3:
            vec = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(
                f"{file_path}: can't find a length-3 component axis in "
                f"shape {arr.shape} (expected (3,D,H,W) or (D,H,W,3))"
            )
        vec = vec.astype(np.float32)

    elif ext == ".csv":
        raw = np.loadtxt(file_path, delimiter=",", dtype=np.float32,
                          skiprows=1, ndmin=2)
        if raw.shape[1] != 3:
            raise ValueError(
                f"{file_path}: expected 3 columns (u_x, u_y, u_z), found "
                f"{raw.shape[1]}. If this is an old scalar-magnitude file, "
                f"load it with FractureDataset(..., target_mode='magnitude')."
            )
        vec = np.stack(
            [_reshape_csv_column(raw[:, k], dim) for k in range(3)], axis=0
        ).astype(np.float32)

    else:
        raise ValueError(f"Unsupported target file extension: {file_path}")

    if vec.shape[1:] != (d_expected, h_expected, w_expected):
        raise ValueError(
            f"{file_path}: velocity field shape {vec.shape[1:]} does not "
            f"match the paired geometry shape "
            f"{(d_expected, h_expected, w_expected)}. Double check the "
            f"axis order your simulation code wrote this file in."
        )
    return vec



def load_csv_mask(file_path: str, dim: Tuple[int, int, int]) -> np.ndarray:
    """Legacy loader: single-column ``velocityNorm`` CSV -> ``(D, H, W)``.

    Kept for backwards compatibility (``target_mode="magnitude"``) and for
    reproducing the original paper's scalar-target results on old data.

    Args:
        file_path: Path to the ``.csv`` file (one header row).
        dim: ``(H, W, D)`` shape used to reshape before transposing.

    Returns:
        A ``(D, H, W)`` ``float32`` array.
    """
    mask = np.loadtxt(file_path, delimiter=",", dtype=np.float32, skiprows=1).flatten()
    return _reshape_csv_column(mask, dim)


# ---------------------------------------------------------------------------
# File pairing
# ---------------------------------------------------------------------------


def find_matching_pairs(
    input_folder: str, mask_folder: str
) -> Tuple[List[str], List[str]]:
    """Return paired ``.mat`` / target paths sharing a base filename.

    Targets may be ``.npy`` or ``.csv``; if both exist for the same base
    name, ``.npy`` is preferred (and a warning is printed) since it is the
    richer, unambiguous format.

    Args:
        input_folder: Folder containing ``.mat`` geometry files.
        mask_folder: Folder containing ``.npy``/``.csv`` velocity files.

    Returns:
        A tuple ``(input_files, mask_files)`` of equal length, sorted by name.
    """
    input_basenames = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(input_folder, "*.mat"))
    }
    
    # --- ADDED NPZ SUPPORT ---
    npz_map = {
        os.path.basename(f).replace("_velocity.npz", ""): f
        for f in glob.glob(os.path.join(mask_folder, "*_velocity.npz"))
    }
    
    npy_map = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(mask_folder, "*.npy"))
    }
    csv_map = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(mask_folder, "*.csv"))
    }
    
    # Merge all found targets
    mask_basenames = set(npz_map) | set(npy_map) | set(csv_map)
    both = set(npy_map) & set(csv_map)
    if both:
        print(f"[dataset] {len(both)} sample(s) have both .npy and .csv "
              f"targets; preferring .npy (e.g. {sorted(both)[:3]})")

    matching_basenames = sorted(input_basenames & mask_basenames)
    missing_targets = sorted(input_basenames - mask_basenames)
    if missing_targets:
        print(f"[dataset] {len(missing_targets)} geometry file(s) have no "
              f"matching target and will be skipped (e.g. {missing_targets[:3]})")

    final_input_files = [
        os.path.join(input_folder, f"{name}.mat") for name in matching_basenames
    ]
    # Grab the target file prioritizing npz, then npy, then csv
    final_mask_files = [
        npz_map.get(name, npy_map.get(name, csv_map.get(name))) for name in matching_basenames
    ]
    return final_input_files, final_mask_files


def build_dataframe(input_folder: str, mask_folder: str, size: int = 128, manifest_path: str = None) -> pd.DataFrame:
    """Build the file table consumed by :class:`FractureDataset`.

    Args:
        input_folder: Folder containing ``.mat`` geometry files.
        mask_folder: Folder containing ``.npy``/``.csv`` velocity files.
        size: Edge length of the cubic volumes (used as a stratification key).

    Returns:
        A DataFrame with columns ``input_path``, ``mask_path``, ``size`` and
        ``name``.
    """
    input_files, mask_files = find_matching_pairs(input_folder, mask_folder)
    df = pd.DataFrame({"input_path": input_files, "mask_path": mask_files})
    df["size"] = size
    df["name"] = df["input_path"].apply(
        lambda path: os.path.splitext(os.path.basename(path))[0]
    )
    
    # --- FILTER NON-PERCOLATING DATA ---
    if manifest_path and os.path.exists(manifest_path):
        manifest_df = pd.read_csv(manifest_path)
        id_col = 'id' if 'id' in manifest_df.columns else manifest_df.columns[0]
        
        # Check for 'perc' or 'percolating' column
        perc_col = 'perc' if 'perc' in manifest_df.columns else 'percolating'
        
        if perc_col in manifest_df.columns:
            valid_ids = manifest_df[manifest_df[perc_col] != 0][id_col].astype(str).tolist()
            initial_count = len(df)
            df = df[df["name"].isin(valid_ids)].reset_index(drop=True)
            print(f"[dataset] Filtered out {initial_count - len(df)} non-percolating samples using {manifest_path}.")
        else:
            print(f"[dataset] Warning: '{perc_col}' column not found in {manifest_path}.")

    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FractureDataset(Dataset):
    """Whole-volume dataset of rock-fracture geometries and velocity fields.

    Each item is a ``(input_tensor, target_tensor)`` pair where:

    * ``input_tensor`` has shape ``(C_in, D, H, W)`` with
      ``C_in == in_channels``. Channel 0 is the binary geometry; channel 1
      (if present) is its Euclidean Distance Transform.
    * ``target_tensor`` has shape ``(3, D, H, W)`` in ``target_mode="vector"``
      (default) -- the signed-sqrt-scaled ``(u_x, u_y, u_z)`` -- or
      ``(1, D, H, W)`` in ``target_mode="magnitude"`` for the original
      scalar-``|u|`` behaviour (unsigned sqrt, single-column CSV only).

    Args:
        dataframe: Table produced by :func:`build_dataframe` (or a split).
        in_channels: ``1`` (geometry only) or ``2`` (geometry + EDT).
        target_mode: ``"vector"`` (default, 3 channels) or ``"magnitude"``
            (legacy, 1 channel).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        in_channels: int = 2,
        target_mode: str = "vector",
    ) -> None:
        if in_channels not in (1, 2):
            raise ValueError("in_channels must be 1 (geometry) or 2 (geometry + EDT)")
        if target_mode not in ("vector", "magnitude"):
            raise ValueError("target_mode must be 'vector' or 'magnitude'")
        self.df = dataframe.reset_index(drop=True)
        self.in_channels = in_channels
        self.target_mode = target_mode

    @property
    def target_channels(self) -> int:
        """Number of target channels this dataset yields (3 or 1)."""
        return 3 if self.target_mode == "vector" else 1

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_info = self.df.iloc[index]
        input_path = sample_info["input_path"]
        mask_path = sample_info["mask_path"]

        input_vol_numpy = load_mat_file(input_path)
        d, h, w = input_vol_numpy.shape

        # Close the first/last H slices (unchanged from the original pipeline).
        input_vol_numpy[:, 0, :], input_vol_numpy[:, -1, :] = 0, 0

        edt_volume = (
            distance_transform_edt(input_vol_numpy)
            if np.any(input_vol_numpy)
            else np.zeros_like(input_vol_numpy)
        )

        if self.in_channels == 2:
            channels = [input_vol_numpy, edt_volume]
        else:
            channels = [input_vol_numpy]
        input_channels = np.stack(channels, axis=0).astype(np.float32)
        input_tensor = torch.tensor(input_channels, dtype=torch.float32)

        if self.target_mode == "vector":
            raw_vec = load_vector_field(mask_path, dim=(h, w, d))       # (3,D,H,W)
            transformed = forward_transform_np(raw_vec)                 # signed sqrt * SCALING_FACTOR
            target_tensor = torch.tensor(transformed, dtype=torch.float32)
        else:
            if os.path.splitext(mask_path)[1].lower() != ".csv":
                raise ValueError(
                    f"target_mode='magnitude' requires a single-column "
                    f".csv target, got {mask_path}"
                )
            raw_mag = load_csv_mask(mask_path, dim=(h, w, d))            # (D,H,W), >= 0
            transformed = forward_transform_np(raw_mag)                  # sqrt * SCALING_FACTOR
            target_tensor = torch.tensor(
                transformed, dtype=torch.float32
            ).unsqueeze(0)

        # NOTE: forward_transform_np already applies SCALING_FACTOR -- do
        # not scale again here (this used to be a separate `* SCALING_FACTOR`
        # line applied on top, which silently double-scaled every vector
        # target; caught by the smoke test's transform round-trip check).
        return input_tensor, target_tensor


def train_val_split(
    df_all_files: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the file table into train/validation sets (stratified by size).

    Unchanged from the original pipeline: fixed seed, stratified on the
    ``size`` column.

    Args:
        df_all_files: Table produced by :func:`build_dataframe`.
        test_size: Validation fraction.
        random_state: Seed for the split.

    Returns:
        A ``(train_df, val_df)`` tuple.
    """
    train_df, val_df = train_test_split(
        df_all_files,
        test_size=test_size,
        random_state=random_state,
        stratify=df_all_files["size"],
    )
    return train_df, val_df


# ---------------------------------------------------------------------------
# Alignment sanity check -- run this before a long training job
# ---------------------------------------------------------------------------


def sanity_check_alignment(
    df: pd.DataFrame, in_channels: int = 2, n_samples: int = 3
) -> pd.DataFrame:
    """Load a few samples and print diagnostics that catch the two most
    likely silent bugs in a new vector-target pipeline: an inverted
    void/solid mask, and a geometry/velocity axis mismatch.

    For each sample this checks:

    * void fraction of the geometry (flags if > 0.5 -- see
      :func:`load_mat_file`'s docstring);
    * fraction of "leakage" -- solid voxels where the raw target speed is
      not (numerically) zero -- which should be ~0;
    * which of the 3 target components has the largest mean |speed| over
      void voxels, i.e. the empirically-inferred flow axis, to compare
      against ``FLOW_AXIS`` in this module.

    Args:
        df: Table produced by :func:`build_dataframe`.
        in_channels: Passed through to :class:`FractureDataset`.
        n_samples: How many rows of ``df`` to check (first N).

    Returns:
        A small per-sample DataFrame of the diagnostics (also printed).
    """
    ds = FractureDataset(df.iloc[:n_samples], in_channels=in_channels, target_mode="vector")
    rows = []
    for i in range(len(ds)):
        name = ds.df.iloc[i]["name"]
        input_t, target_t = ds[i]
        geom = input_t[0].numpy() > 0
        raw_vec = inverse_transform_torch(target_t).numpy()   # (3,D,H,W), physical units
        speed = np.linalg.norm(raw_vec, axis=0)

        void_frac = float(geom.mean())
        solid = ~geom
        leakage = float((speed[solid] > 1e-12).mean()) if solid.any() else float("nan")
        mean_abs_per_axis = [float(np.mean(np.abs(raw_vec[k][geom]))) if geom.any() else float("nan")
                              for k in range(3)]
        inferred_flow_axis = int(np.argmax(mean_abs_per_axis))

        row = {
            "name": name,
            "void_fraction": round(void_frac, 4),
            "leakage_frac_solid_nonzero": round(leakage, 6),
            "mean_|u|_axis0": mean_abs_per_axis[0],
            "mean_|u|_axis1": mean_abs_per_axis[1],
            "mean_|u|_axis2": mean_abs_per_axis[2],
            "inferred_flow_axis": inferred_flow_axis,
            "matches_FLOW_AXIS": inferred_flow_axis == FLOW_AXIS,
        }
        rows.append(row)

        print(f"[{name}]")
        print(f"  void_fraction = {void_frac:.4f}"
              + ("   !! expected a thin fracture (<0.5); check the "
                 "void/solid convention" if void_frac > 0.5 else ""))
        print(f"  leakage (solid voxels with nonzero target speed) = "
              f"{leakage:.6f}"
              + ("   !! should be ~0; geometry and velocity field are "
                 "probably misaligned" if leakage > 0.01 else ""))
        print(f"  mean |u| by axis: {['%.4g' % v for v in mean_abs_per_axis]}"
              f"  -> inferred flow axis = {inferred_flow_axis} "
              f"(module default FLOW_AXIS = {FLOW_AXIS})")
        if inferred_flow_axis != FLOW_AXIS:
            print("  !! inferred flow axis does not match FLOW_AXIS at the "
                  "top of dataset.py -- update it (and APERTURE_AXIS) "
                  "before trusting permeability.py / physics_baselines.py")
        print()

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a FractureDataset.")
    parser.add_argument("--input-folder", required=True, help="Folder of .mat files")
    parser.add_argument("--mask-folder", required=True, help="Folder of .npy/.csv files")
    parser.add_argument("--in-channels", type=int, default=2, choices=(1, 2))
    parser.add_argument("--target-mode", choices=("vector", "magnitude"), default="vector")
    parser.add_argument(
        "--check", action="store_true",
        help="Run sanity_check_alignment on a few samples instead of just printing shapes",
    )
    parser.add_argument("--check-n", type=int, default=3)
    args = parser.parse_args()

    df = build_dataframe(args.input_folder, args.mask_folder)
    train_df, val_df = train_val_split(df)
    print(f"Total pairs: {len(df)} | train: {len(train_df)} | val: {len(val_df)}")

    if args.check:
        sanity_check_alignment(df, in_channels=args.in_channels, n_samples=args.check_n)
    elif len(df) > 0:
        ds = FractureDataset(df, in_channels=args.in_channels, target_mode=args.target_mode)
        x, y = ds[0]
        print(f"First sample '{df.iloc[0]['name']}': input {tuple(x.shape)}, "
              f"target {tuple(y.shape)}")
