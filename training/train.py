"""Main training loop for the fracture flow-prediction surrogate models."""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import FractureDataset, build_dataframe, train_val_split
from models import MODELS


class WeightedMAELoss(nn.Module):
    """Mean absolute error weighting non-zero-speed voxels more heavily.

    The weight mask is computed once per voxel from
    ``||target||`` (the L2 norm across the channel dimension -- for a
    single-channel magnitude target this is just ``|target|``, so
    ``out_channels=1`` reproduces the original ``targets > 0`` behaviour
    exactly) and then broadcast to every output channel, so all components
    at a "flowing" voxel get the same emphasis rather than each channel
    being judged on its own sign.

    Args:
        high_weight: Weight applied where the target speed is > 0.
        low_weight: Weight applied elsewhere.
        component_weights: Optional per-channel multiplier on top of the
            high/low mask, e.g. ``[1.0, 3.0, 3.0]`` to upweight the two
            transverse components relative to the (usually dominant) axial
            one. ``None`` (default) applies no extra per-channel weighting.
    """

    def __init__(
        self,
        high_weight: float = 10.0,
        low_weight: float = 1.0,
        component_weights: "list[float] | None" = None,
    ) -> None:
        super().__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.component_weights = component_weights
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_voxel_loss = self.mae(outputs, targets)  # (N, C, D, H, W)

        speed = torch.linalg.vector_norm(targets, dim=1, keepdim=True)  # (N, 1, D, H, W)
        weights = torch.where(
            speed > 0,
            torch.as_tensor(self.high_weight, device=targets.device, dtype=targets.dtype),
            torch.as_tensor(self.low_weight, device=targets.device, dtype=targets.dtype),
        )
        weighted_loss = per_voxel_loss * weights  # broadcasts (N,1,D,H,W) over channels

        if self.component_weights is not None:
            c = outputs.shape[1]
            if len(self.component_weights) != c:
                raise ValueError(
                    f"component_weights has {len(self.component_weights)} "
                    f"entries but the model has {c} output channels"
                )
            cw = torch.as_tensor(self.component_weights, device=targets.device,
                                  dtype=targets.dtype).view(1, c, 1, 1, 1)
            weighted_loss = weighted_loss * cw

        return torch.mean(weighted_loss)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    loss_fn: nn.Module,
    patience: int = 10,
    num_epochs: int = 75,
    save_path: str = "best_model.pth",
    history_save_path: str = "history.csv",
    out_channels: int = 3,
    target_mode: str = "vector",
    in_channels: int = 2,
    model_name: str = "unet3d",
) -> None:
    """Train ``model`` with early stopping and checkpointing.

    Args:
        model: The network to train.
        train_loader: Training data loader (batch size 1).
        val_loader: Validation data loader.
        optimizer: AdamW optimizer.
        scheduler: ``ReduceLROnPlateau`` scheduler stepped on the val loss.
        loss_fn: Loss function (e.g. :class:`WeightedMAELoss`).
        patience: Epochs of no improvement before early stopping.
        num_epochs: Maximum number of epochs (paper: 75).
        save_path: Where to write the best-checkpoint dictionary.
        history_save_path: Where to write the per-epoch loss CSV.
        out_channels: Recorded in the checkpoint for later shape-checking.
        target_mode: Recorded in the checkpoint ("vector" or "magnitude").
        in_channels: Recorded in the checkpoint.
        model_name: Recorded in the checkpoint ("unet3d" or "attresunet").
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_loss = float("inf")
    model.to(device)
    early_stopping_counter = 0
    history = {"train_loss": [], "val_loss": []}

    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(
            train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(
            val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                val_pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{num_epochs} -> "
            f"Training Loss: {avg_train_loss:.10f}, "
            f"Validation Loss: {avg_val_loss:.10f}"
        )
        if new_lr < old_lr:
            print(f"Learning rate reduced from {old_lr} to {new_lr}")

        if avg_val_loss < best_val_loss:
            print(
                f"Validation loss improved from {best_val_loss:.10f} to "
                f"{avg_val_loss:.10f}. Saving model to {save_path}"
            )
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "out_channels": out_channels,
                    "in_channels": in_channels,
                    "target_mode": target_mode,
                    "model_name": model_name,
                },
                save_path,
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Validation loss did not improve from {best_val_loss:.10f}.")

        pd.DataFrame(history).to_csv(history_save_path, index=False)

        if early_stopping_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fracture flow model.")
    parser.add_argument("--input-folder", required=True, help="Folder of .mat files")
    parser.add_argument("--mask-folder", required=True, help="Folder of .npy/.csv target files")
    parser.add_argument("--model", choices=tuple(MODELS), default="attresunet")
    parser.add_argument("--in-channels", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--target-mode", choices=("vector", "magnitude"), default="vector",
        help="'vector' predicts (u_x,u_y,u_z) [3 channels]; 'magnitude' "
             "reproduces the original |u|-only model [1 channel]",
    )
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size for training and validation",
)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--high-weight", type=float, default=10.0)
    parser.add_argument(
        "--component-weights", type=float, nargs=3, default=None,
        metavar=("W_X", "W_Y", "W_Z"),
        help="Optional extra per-component loss weights, only used when "
             "--target-mode vector, e.g. --component-weights 1 3 3 to "
             "upweight the transverse components",
    )
    parser.add_argument(
        "--save-path", default="weights/best_model.pth", help="Checkpoint output path"
    )
    parser.add_argument(
        "--history-path",
        default="results/train_history.csv",
        help="Per-epoch loss CSV output path",
    )
    parser.add_argument("--manifest-csv", default=None, help="Path to manifest or results CSV to filter perc=0")
    args = parser.parse_args()

    df = build_dataframe(args.input_folder, args.mask_folder, manifest_path=args.manifest_csv)
    train_df, val_df = train_val_split(df)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_ds = FractureDataset(train_df, in_channels=args.in_channels, target_mode=args.target_mode)
    val_ds = FractureDataset(val_df, in_channels=args.in_channels, target_mode=args.target_mode)
    out_channels = train_ds.target_channels
    print(f"target_mode={args.target_mode} -> out_channels={out_channels}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS[args.model](in_channels=args.in_channels, out_channels=out_channels)
    model.to(device)

    if args.component_weights is not None and out_channels != 3:
        raise ValueError("--component-weights only applies to --target-mode vector (3 channels)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=5)
    loss_fn = WeightedMAELoss(high_weight=args.high_weight, component_weights=args.component_weights)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.history_path) or ".", exist_ok=True)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        patience=args.patience,
        num_epochs=args.epochs,
        save_path=args.save_path,
        history_save_path=args.history_path,
        out_channels=out_channels,
        target_mode=args.target_mode,
        in_channels=args.in_channels,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
