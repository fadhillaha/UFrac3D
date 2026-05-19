"""Dataset utilities for the rock-fracture flow-prediction models.

Loads paired ``.mat`` geometry sub-volumes and ``.csv`` velocity fields,
computes the Euclidean Distance Transform, and provides a
``torch.utils.data.Dataset`` plus a fixed, stratified train/validation
split.

The geometry input is either a single channel (binary geometry) or two
channels (binary geometry + EDT), controlled by the ``in_channels``
argument of :class:`FractureDataset` (default 2).
"""

from __future__ import annotations

import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Target velocities are square-rooted then multiplied by this factor before
# training (the inverse transform is applied during evaluation).
SCALING_FACTOR: float = 1000.0


def load_mat_file(file_path: str) -> np.ndarray:
    """Load a ``.mat`` geometry file and return a binary ``float32`` volume.

    The geometry is read from the key ``sub_volume``; if absent, the key
    ``wadah`` is used (the variable name in the dataset used in this
    research). The volume is min-max normalised and thresholded at 0.5 to
    produce a binary occupancy field.

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


def load_csv_mask(file_path: str, dim: Tuple[int, int, int]) -> np.ndarray:
    """Load a ``.csv`` velocity field and reshape it to ``(D, H, W)``.

    Args:
        file_path: Path to the ``.csv`` file (one header row, comma-separated).
        dim: The ``(H, W, D)`` shape used to reshape the flat array before it
            is transposed to ``(D, H, W)``.

    Returns:
        A ``(D, H, W)`` ``float32`` array.
    """
    mask = np.loadtxt(file_path, delimiter=",", dtype=np.float32, skiprows=1).flatten()
    mask_3d = mask.reshape(dim)
    mask_3d = np.transpose(mask_3d, (2, 0, 1))  # (depth, height, width)
    return mask_3d


def find_matching_pairs(
    input_folder: str, mask_folder: str
) -> Tuple[List[str], List[str]]:
    """Return paired ``.mat`` / ``.csv`` paths sharing a base filename.

    Args:
        input_folder: Folder containing ``.mat`` geometry files.
        mask_folder: Folder containing ``.csv`` velocity files.

    Returns:
        A tuple ``(input_files, mask_files)`` of equal length, sorted by name.
    """
    input_basenames = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(input_folder, "*.mat"))
    }
    mask_basenames = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(mask_folder, "*.csv"))
    }

    matching_basenames = sorted(input_basenames.intersection(mask_basenames))

    final_input_files = [
        os.path.join(input_folder, f"{name}.mat") for name in matching_basenames
    ]
    final_mask_files = [
        os.path.join(mask_folder, f"{name}.csv") for name in matching_basenames
    ]
    return final_input_files, final_mask_files


def build_dataframe(input_folder: str, mask_folder: str, size: int = 128) -> pd.DataFrame:
    """Build the file table consumed by :class:`FractureDataset`.

    Args:
        input_folder: Folder containing ``.mat`` geometry files.
        mask_folder: Folder containing ``.csv`` velocity files.
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
    return df


class FractureDataset(Dataset):
    """Whole-volume dataset of rock-fracture geometries and velocity fields.

    Each item is a ``(input_tensor, target_tensor)`` pair where:

    * ``input_tensor`` has shape ``(C, D, H, W)`` with ``C == in_channels``.
      Channel 0 is the binary geometry; channel 1 (if present) is the
      Euclidean Distance Transform of the geometry.
    * ``target_tensor`` has shape ``(1, D, H, W)`` and stores
      ``sqrt(velocity) * SCALING_FACTOR``.

    Args:
        dataframe: Table produced by :func:`build_dataframe` (or a split of
            it). Must contain ``input_path``, ``mask_path`` and ``size``.
        in_channels: ``1`` for geometry only, ``2`` for geometry + EDT
            (default ``2``).
    """

    def __init__(self, dataframe: pd.DataFrame, in_channels: int = 2) -> None:
        if in_channels not in (1, 2):
            raise ValueError("in_channels must be 1 (geometry) or 2 (geometry + EDT)")
        self.df = dataframe
        self.in_channels = in_channels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Get sample information from the DataFrame
        sample_info = self.df.iloc[index]
        input_path = sample_info["input_path"]
        mask_path = sample_info["mask_path"]

        # 2. Load the full-sized volumes
        input_vol_numpy = load_mat_file(input_path)
        d, h, w = input_vol_numpy.shape
        target_vol_numpy = load_csv_mask(mask_path, dim=(h, w, d))

        # 4. Apply fixes to the 128^3 geometry (close the first/last H slices)
        input_vol_numpy[:, 0, :], input_vol_numpy[:, -1, :] = 0, 0

        # 5. Feature engineering on the FULL volume
        edt_volume = (
            distance_transform_edt(input_vol_numpy)
            if np.any(input_vol_numpy)
            else np.zeros_like(input_vol_numpy)
        )

        if self.in_channels == 2:
            channels = [input_vol_numpy, edt_volume]  # geometry + EDT
        else:
            channels = [input_vol_numpy]  # geometry only
        input_channels = np.stack(channels, axis=0).astype(np.float32)

        # 6. Finalise and convert to tensors
        target_volume = np.sqrt(target_vol_numpy)
        input_tensor = torch.tensor(input_channels, dtype=torch.float32)
        target_tensor = torch.tensor(target_volume, dtype=torch.float32)
        target_tensor = target_tensor * SCALING_FACTOR

        return input_tensor, target_tensor.unsqueeze(0)


def train_val_split(
    df_all_files: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the file table into train/validation sets.

    The split is stratified by the ``size`` column and uses a fixed random
    state for reproducibility.

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a FractureDataset.")
    parser.add_argument("--input-folder", required=True, help="Folder of .mat files")
    parser.add_argument("--mask-folder", required=True, help="Folder of .csv files")
    parser.add_argument("--in-channels", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()

    df = build_dataframe(args.input_folder, args.mask_folder)
    train_df, val_df = train_val_split(df)
    print(f"Total pairs: {len(df)} | train: {len(train_df)} | val: {len(val_df)}")

    if len(df) > 0:
        ds = FractureDataset(df, in_channels=args.in_channels)
        x, y = ds[0]
        print(f"First sample '{df.iloc[0]['name']}': input {tuple(x.shape)}, "
              f"target {tuple(y.shape)}")
