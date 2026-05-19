"""Training loop for the fracture flow-prediction models.

Training uses:

* the AdamW optimizer (``weight_decay=1e-5``),
* a weighted MAE loss (higher weight on non-zero target voxels),
* a ``ReduceLROnPlateau`` scheduler on the validation loss,
* automatic mixed precision (AMP) on CUDA,
* early stopping (default ``patience=10``),
* a batch size of 1 (whole 128^3 volumes).

The best checkpoint (by validation loss) is saved as a dictionary
containing the model, optimizer and scheduler states; the per-epoch loss
history is written to a CSV.
"""

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
    """Mean absolute error weighting non-zero target voxels more heavily.

    Args:
        high_weight: Weight applied where ``target > 0``.
        low_weight: Weight applied elsewhere.
    """

    def __init__(self, high_weight: float = 10.0, low_weight: float = 1.0) -> None:
        super().__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_voxel_loss = self.mae(outputs, targets)
        weights = torch.where(targets > 0, self.high_weight, self.low_weight)
        weights = weights.to(per_voxel_loss.device)
        weighted_loss = per_voxel_loss * weights
        return torch.mean(weighted_loss)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    loss_fn: nn.Module,
    patience: int = 10,
    num_epochs: int = 50,
    save_path: str = "best_model.pth",
    history_save_path: str = "history.csv",
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
        num_epochs: Maximum number of epochs.
        save_path: Where to write the best-checkpoint dictionary.
        history_save_path: Where to write the per-epoch loss CSV.
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
    parser.add_argument("--mask-folder", required=True, help="Folder of .csv files")
    parser.add_argument("--model", choices=tuple(MODELS), default="unet3d")
    parser.add_argument("--in-channels", type=int, default=2, choices=(1, 2))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--high-weight", type=float, default=10.0)
    parser.add_argument(
        "--save-path", default="weights/best_model.pth", help="Checkpoint output path"
    )
    parser.add_argument(
        "--history-path",
        default="results/train_history.csv",
        help="Per-epoch loss CSV output path",
    )
    args = parser.parse_args()

    df = build_dataframe(args.input_folder, args.mask_folder)
    train_df, val_df = train_val_split(df)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_loader = DataLoader(
        FractureDataset(train_df, in_channels=args.in_channels),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        FractureDataset(val_df, in_channels=args.in_channels),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS[args.model](in_channels=args.in_channels, out_channels=1)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=5)
    loss_fn = WeightedMAELoss(high_weight=args.high_weight)

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
    )


if __name__ == "__main__":
    main()
