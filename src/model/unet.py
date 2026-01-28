"""
U-Net Architecture for Diffusion Model

The U-Net is the core neural network in our diffusion model. It learns to
predict the noise that was added to an image at a given timestep.

Why U-Net?
1. The encoder-decoder structure captures both local and global features
2. Skip connections preserve fine details that might be lost in downsampling
3. It's proven effective in image-to-image tasks (originally for segmentation)
4. The symmetric structure is natural for denoising: corrupt -> denoise

Architecture Overview:
```
Input (noisy image + time embedding)
    │
    ├── Encoder: progressively downsample, increase channels
    │   └── ResBlock + Attention at each scale
    │
    ├── Bottleneck: process at lowest resolution
    │   └── ResBlock + Attention + ResBlock
    │
    └── Decoder: progressively upsample, decrease channels
        └── ResBlock + Attention at each scale (with skip connections)
    │
Output (predicted noise, same size as input)
```

The model takes two inputs:
1. Noisy image x_t: The image with noise added at timestep t
2. Timestep t: Which step in the diffusion process

And outputs:
- Predicted noise ε: The noise that was added to create x_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResBlock, AttentionBlock, Downsample, Upsample
from .embeddings import TimeEmbedding


class UNet(nn.Module):
    """
    U-Net architecture for noise prediction in diffusion models.

    This is a simplified but functional U-Net with:
    - 3 encoder stages (32->16->8->4 resolution)
    - 1 bottleneck stage
    - 3 decoder stages with skip connections
    - Attention at every stage for global context

    The model is conditioned on the timestep via additive time embeddings
    that are injected into every ResBlock.

    Architecture for 32x32 input with channel_mults=(1, 2, 4, 8):

    Encoder:
        32x32, 64ch  -> ResBlock -> store skip -> Downsample
        16x16, 128ch -> ResBlock -> store skip -> Downsample
        8x8, 256ch   -> ResBlock -> store skip -> Downsample

    Bottleneck:
        4x4, 512ch   -> ResBlock -> Attention -> ResBlock

    Decoder:
        4x4 -> Upsample -> 8x8, concat skip -> ResBlock (256ch out)
        8x8 -> Upsample -> 16x16, concat skip -> ResBlock (128ch out)
        16x16 -> Upsample -> 32x32, concat skip -> ResBlock (64ch out)

    Output: 32x32, 1ch
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        time_emb_dim: int = 128,
        num_groups: int = 8,
        num_heads: int = 4,
        use_attention: bool = True,
    ):
        """
        Initialize the U-Net.

        Args:
            in_channels: Number of input image channels (1 for grayscale)
            out_channels: Number of output channels (1 for noise prediction)
            base_channels: Base number of channels (multiplied at each stage)
            channel_mults: Channel multipliers for each stage (e.g., (1, 2, 4, 8))
            time_emb_dim: Dimension of time embeddings
            num_groups: Number of groups for GroupNorm
            num_heads: Number of attention heads
            use_attention: Whether to use attention blocks
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.use_attention = use_attention

        # Compute channel counts at each level
        # For mults (1, 2, 4, 8): channels = [64, 128, 256, 512]
        self.channels = [base_channels * m for m in channel_mults]
        self.num_levels = len(channel_mults)

        # Time embedding network
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # Initial convolution: project input to base_channels
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ==================== ENCODER ====================
        # Each level: ResBlock -> (optional Attention) -> Downsample
        # We store the output of ResBlock as skip connection
        self.encoder_resblocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()

        for i in range(self.num_levels - 1):  # Don't downsample at last level (that's bottleneck)
            in_ch = self.channels[i] if i > 0 else base_channels
            out_ch = self.channels[i]

            self.encoder_resblocks.append(
                ResBlock(in_ch, out_ch, time_emb_dim, num_groups)
            )

            if use_attention:
                self.encoder_attns.append(AttentionBlock(out_ch, num_heads, num_groups))
            else:
                self.encoder_attns.append(nn.Identity())

            # Downsample to next level's channels
            next_ch = self.channels[i + 1]
            self.encoder_downsamples.append(Downsample(out_ch, next_ch))

        # ==================== BOTTLENECK ====================
        # Process at the lowest resolution with maximum channels
        bottleneck_ch = self.channels[-1]

        self.bottleneck_res1 = ResBlock(bottleneck_ch, bottleneck_ch, time_emb_dim, num_groups)
        self.bottleneck_attn = AttentionBlock(bottleneck_ch, num_heads, num_groups)
        self.bottleneck_res2 = ResBlock(bottleneck_ch, bottleneck_ch, time_emb_dim, num_groups)

        # ==================== DECODER ====================
        # Each level: Upsample -> Concat skip -> ResBlock -> (optional Attention)
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_resblocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()

        # Go backwards through levels (excluding bottleneck)
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample from current channels to this level's channels
            in_ch = self.channels[i + 1] if i == self.num_levels - 2 else self.channels[i + 1]
            out_ch = self.channels[i]

            self.decoder_upsamples.append(Upsample(in_ch, out_ch))

            # ResBlock receives concatenated features (upsampled + skip)
            # Skip has out_ch channels, upsampled has out_ch channels
            self.decoder_resblocks.append(
                ResBlock(out_ch * 2, out_ch, time_emb_dim, num_groups)
            )

            if use_attention:
                self.decoder_attns.append(AttentionBlock(out_ch, num_heads, num_groups))
            else:
                self.decoder_attns.append(nn.Identity())

        # ==================== OUTPUT ====================
        # Final convolution to project back to output channels
        self.final_norm = nn.GroupNorm(num_groups, base_channels)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net.

        Args:
            x: Noisy input image of shape (batch, in_channels, height, width)
            t: Timesteps of shape (batch,) containing integer timesteps

        Returns:
            Predicted noise of shape (batch, out_channels, height, width)
        """
        # Compute time embeddings
        time_emb = self.time_embedding(t)

        # Initial convolution
        h = self.init_conv(x)

        # Store activations for skip connections
        skips = []

        # ==================== ENCODER ====================
        for resblock, attn, downsample in zip(
            self.encoder_resblocks, self.encoder_attns, self.encoder_downsamples
        ):
            h = resblock(h, time_emb)
            h = attn(h)
            skips.append(h)  # Store for skip connection
            h = downsample(h)

        # ==================== BOTTLENECK ====================
        h = self.bottleneck_res1(h, time_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, time_emb)

        # ==================== DECODER ====================
        for upsample, resblock, attn in zip(
            self.decoder_upsamples, self.decoder_resblocks, self.decoder_attns
        ):
            h = upsample(h)
            skip = skips.pop()  # Get corresponding skip connection
            h = torch.cat([h, skip], dim=1)  # Concatenate along channel dim
            h = resblock(h, time_emb)
            h = attn(h)

        # ==================== OUTPUT ====================
        h = self.final_norm(h)
        h = F.silu(h)
        h = self.final_conv(h)

        return h

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the U-Net
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
    )

    # Print model summary
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))

    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Timesteps: {t}")

    # Verify output shape matches input
    assert out.shape == x.shape, "Output shape must match input shape!"
    print("U-Net test passed!")
