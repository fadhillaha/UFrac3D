"""3D U-Net architectures for fracture flow-field prediction.

Provides two models:

* :class:`UNet3D` -- a plain 3D U-Net (no residual connections, no attention).
* :class:`AttResUNet` -- a 3D U-Net with residual convolution blocks and
  attention gates on the skip connections.

Both take an input of shape ``(N, in_channels, D, H, W)`` and produce an
output of shape ``(N, out_channels, D, H, W)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Plain 3D U-Net
# ---------------------------------------------------------------------------


class SimpleConvBlock(nn.Module):
    """Two Conv3d -> GroupNorm -> GELU layers (no residual connection)."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same")
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding="same")
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.act2(self.gn2(self.conv2(out)))
        return out


class SimpleEncoderBlock(nn.Module):
    """A :class:`SimpleConvBlock` followed by 2x max pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = SimpleConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x_conv = self.conv(x)
        x_pooled = self.pool(x_conv)
        return x_conv, x_pooled


class SimpleDecoderBlock(nn.Module):
    """Trilinear upsample, concatenate with skip, then a conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = SimpleConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_upsampled = self.up(x)
        x_concat = torch.cat((x_upsampled, skip), dim=1)
        return self.conv(x_concat)


class UNet3D(nn.Module):
    """A plain 3D U-Net with four encoder/decoder stages.

    Args:
        in_channels: Number of input channels (1 = geometry, 2 = geometry+EDT).
        out_channels: Number of output channels (1 for the velocity field).
        dropout_rate: Dropout probability applied at the bottleneck during
            training.
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        self.enc1 = SimpleEncoderBlock(in_channels, 32)
        self.enc2 = SimpleEncoderBlock(32, 64)
        self.enc3 = SimpleEncoderBlock(64, 128)
        self.enc4 = SimpleEncoderBlock(128, 256)

        self.bottleneck = SimpleConvBlock(256, 256)
        self.dropout = nn.Dropout3d(p=dropout_rate)

        self.dec1 = SimpleDecoderBlock(256, 256, 256)
        self.dec2 = SimpleDecoderBlock(256, 128, 128)
        self.dec3 = SimpleDecoderBlock(128, 64, 64)
        self.dec4 = SimpleDecoderBlock(64, 32, 32)

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)
        if self.training:
            b = self.dropout(b)

        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        return self.final_conv(d4)


# ---------------------------------------------------------------------------
# Attention Residual 3D U-Net
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Residual conv block: two Conv3d-GN layers with a projected skip."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same")
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding="same")
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act2 = nn.GELU()

        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act1(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.residual_conv is not None:
            identity = self.residual_conv(identity)
        out += identity
        out = self.act2(out)
        return out


class EncoderBlock(nn.Module):
    """A residual :class:`ConvBlock` followed by 2x max pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x_conv = self.conv(x)
        x_pooled = self.pool(x_conv)
        return x_conv, x_pooled


class AttentionGate(nn.Module):
    """Additive attention gate filtering a skip connection by a gating signal."""

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.W_g = nn.Conv3d(g_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv3d(x_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv3d(inter_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_out = self.W_g(g)
        x_out = self.W_x(x)
        gx_sum = g_out + x_out
        activated_gx = self.relu(gx_sum)
        psi_out = self.psi(activated_gx)
        attention_map = self.sigmoid(psi_out)
        attended_x = x * attention_map
        return attended_x


class DecoderBlock(nn.Module):
    """Upsample, project, attention-gate the skip, concatenate, conv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv_after_up = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding="same", bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.GELU(),
        )
        self.attention_gate = AttentionGate(
            g_channels=out_channels,
            x_channels=skip_channels,
            inter_channels=skip_channels // 2,
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_upsampled = self.up(x)
        x_processed_after_up = self.conv_after_up(x_upsampled)

        if x_processed_after_up.shape[2:] != skip.shape[2:]:
            print(
                "Warning: Spatial dimension mismatch in DecoderBlock. "
                f"Upsampled: {x_processed_after_up.shape}, Skip: {skip.shape}. "
                "Attempting center crop of skip."
            )
            target_d, target_h, target_w = x_processed_after_up.shape[2:]
            diff_d = skip.size(2) - target_d
            diff_h = skip.size(3) - target_h
            diff_w = skip.size(4) - target_w
            skip = skip[
                :,
                :,
                diff_d // 2 : skip.size(2) - (diff_d - diff_d // 2),
                diff_h // 2 : skip.size(3) - (diff_h - diff_h // 2),
                diff_w // 2 : skip.size(4) - (diff_w - diff_w // 2),
            ]
            if x_processed_after_up.shape[2:] != skip.shape[2:]:
                raise ValueError(
                    "Spatial dimension mismatch persists after crop. "
                    f"Upsampled: {x_processed_after_up.shape}, Skip: {skip.shape}. "
                    "Check U-Net input dimensions and padding strategies."
                )

        attended_skip = self.attention_gate(g=x_processed_after_up, x=skip)
        x_concat = torch.cat((x_processed_after_up, attended_skip), dim=1)
        return self.conv(x_concat)


class AttResUNet(nn.Module):
    """A 3D U-Net with residual conv blocks and attention-gated skips.

    Args:
        in_channels: Number of input channels (1 = geometry, 2 = geometry+EDT).
        out_channels: Number of output channels (1 for the velocity field).
        dropout_rate: Dropout probability applied at the bottleneck.
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, dropout_rate: float = 0.2
    ) -> None:
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        self.bottleneck = ConvBlock(256, 256)
        self.dropout = nn.Dropout3d(p=dropout_rate)

        self.dec1 = DecoderBlock(256, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec3 = DecoderBlock(128, 64, 64)
        self.dec4 = DecoderBlock(64, 32, 32)

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)
        b = self.dropout(b)

        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        return self.final_conv(d4)


# Registry so scripts can select an architecture by name.
MODELS = {"unet3d": UNet3D, "attresunet": AttResUNet}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print a model parameter count.")
    parser.add_argument("--model", choices=tuple(MODELS), default="unet3d")
    parser.add_argument("--in-channels", type=int, default=2)
    args = parser.parse_args()

    net = MODELS[args.model](in_channels=args.in_channels, out_channels=1)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"{args.model}: {n_params:,} parameters (in_channels={args.in_channels})")
