"""
Building Blocks for the U-Net Architecture

This module contains the fundamental building blocks used in the U-Net:
1. ResBlock: Residual blocks with time embedding injection
2. AttentionBlock: Self-attention for capturing global dependencies

These blocks are designed to be simple and educational while still being
effective for our toy diffusion model.

Key Concepts:
- Residual connections help with gradient flow in deep networks
- Time embeddings are injected to condition the network on the timestep
- Self-attention allows the model to capture long-range spatial dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual Block with Time Embedding.

    This is the workhorse of the U-Net. Each ResBlock:
    1. Applies convolutions to extract features
    2. Injects time information via an additive embedding
    3. Uses a residual (skip) connection to preserve information

    Architecture:
        x ─────────────────────────────────────────┐
        │                                          │
        └─→ GroupNorm → SiLU → Conv3x3 ─────────┐  │
                                                │  │
            time_emb → Linear → SiLU ──────────→ + │
                                                │  │
                    GroupNorm → SiLU → Conv3x3 ←┘  │
                                │                  │
                                └───────── + ←─────┘ (+ optional 1x1 conv if channels change)
                                           │
                                           └→ output

    Why Group Normalization?
    - Works well with small batch sizes (common in diffusion training)
    - More stable than BatchNorm for generative models

    Why SiLU (Swish)?
    - Smooth, non-monotonic activation that works well in practice
    - Used in many state-of-the-art diffusion models
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
    ):
        """
        Initialize the ResBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_emb_dim: Dimension of time embeddings
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First convolution block
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        # Projects the time embedding to match the number of channels
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # Second convolution block
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection
        # If input and output channels differ, we need a 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlock.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            time_emb: Time embedding of shape (batch, time_emb_dim)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        # Store input for residual connection
        residual = x

        # First conv block: norm -> activation -> conv
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        # time_emb shape: (batch, out_channels)
        # We need to broadcast it to (batch, out_channels, 1, 1) for addition
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.residual_conv(residual)


class AttentionBlock(nn.Module):
    """
    Self-Attention Block for capturing global spatial dependencies.

    While convolutions are great for local features, they have limited
    receptive fields. Self-attention allows every position to attend to
    every other position, capturing global relationships.

    This is particularly important for diffusion models because:
    1. Global context helps with coherent image generation
    2. It allows the model to understand spatial relationships between shapes
    3. It helps maintain consistency across the image during denoising

    Architecture:
        x ───────────────────────────────────────┐
        │                                        │
        └─→ GroupNorm → QKV projection ─────────┐│
                                                ││
                        Attention(Q, K, V) ←────┘│
                                │                │
                        Output projection        │
                                │                │
                                └───────── + ←───┘
                                           │
                                           └→ output

    We use scaled dot-product attention:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        """
        Initialize the AttentionBlock.

        Args:
            channels: Number of input/output channels
            num_heads: Number of attention heads
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"

        self.norm = nn.GroupNorm(num_groups, channels)

        # QKV projection: project to 3x channels, then split
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)

        # Output projection
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        # Scale factor for attention
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionBlock.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Output tensor of shape (batch, channels, height, width)
        """
        batch, channels, height, width = x.shape

        # Store for residual
        residual = x

        # Normalize
        h = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(h)  # (batch, 3*channels, height, width)

        # Reshape for multi-head attention
        # (batch, 3*channels, height, width) -> (batch, 3, num_heads, head_dim, height*width)
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, height * width)

        # Split into Q, K, V
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: (batch, num_heads, head_dim, height*width)

        # Transpose for attention: (batch, num_heads, height*width, head_dim)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        # Compute attention scores
        # (batch, num_heads, height*width, head_dim) @ (batch, num_heads, head_dim, height*width)
        # -> (batch, num_heads, height*width, height*width)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        # (batch, num_heads, height*width, height*width) @ (batch, num_heads, height*width, head_dim)
        # -> (batch, num_heads, height*width, head_dim)
        out = torch.matmul(attn, v)

        # Reshape back to image format
        # (batch, num_heads, height*width, head_dim) -> (batch, channels, height, width)
        out = out.permute(0, 1, 3, 2)  # (batch, num_heads, head_dim, height*width)
        out = out.reshape(batch, channels, height, width)

        # Project and add residual
        out = self.proj_out(out)
        return out + residual


class Downsample(nn.Module):
    """
    Downsampling layer using strided convolution.

    Reduces spatial dimensions by 2x while potentially changing channels.
    We use strided convolution instead of pooling because it's learnable
    and tends to work better in generative models.
    """

    def __init__(self, in_channels: int, out_channels: int = None):
        """
        Initialize the Downsample layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (defaults to in_channels)
        """
        super().__init__()
        out_channels = out_channels or in_channels
        # Strided convolution: kernel=3, stride=2, padding=1 -> halves spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling layer using nearest-neighbor interpolation + convolution.

    Increases spatial dimensions by 2x. We use interpolation + conv instead
    of transposed convolution to avoid checkerboard artifacts.
    """

    def __init__(self, in_channels: int, out_channels: int = None):
        """
        Initialize the Upsample layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (defaults to in_channels)
        """
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nearest-neighbor upsampling followed by convolution
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


if __name__ == "__main__":
    # Test the blocks
    batch_size = 2
    channels = 64
    height, width = 32, 32
    time_emb_dim = 128

    # Test ResBlock
    x = torch.randn(batch_size, channels, height, width)
    time_emb = torch.randn(batch_size, time_emb_dim)

    resblock = ResBlock(channels, channels * 2, time_emb_dim)
    out = resblock(x, time_emb)
    print(f"ResBlock: {x.shape} -> {out.shape}")

    # Test AttentionBlock
    x = torch.randn(batch_size, channels, height, width)
    attn = AttentionBlock(channels)
    out = attn(x)
    print(f"AttentionBlock: {x.shape} -> {out.shape}")

    # Test Downsample
    x = torch.randn(batch_size, channels, height, width)
    down = Downsample(channels, channels * 2)
    out = down(x)
    print(f"Downsample: {x.shape} -> {out.shape}")

    # Test Upsample
    x = torch.randn(batch_size, channels * 2, height // 2, width // 2)
    up = Upsample(channels * 2, channels)
    out = up(x)
    print(f"Upsample: {x.shape} -> {out.shape}")
