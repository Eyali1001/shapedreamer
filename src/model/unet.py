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

    Total parameters: ~2-4M (varies with channel configuration)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 1,
        attention_resolutions: tuple = (32, 16, 8, 4),
        time_emb_dim: int = 128,
        num_groups: int = 8,
        num_heads: int = 4,
    ):
        """
        Initialize the U-Net.

        Args:
            in_channels: Number of input image channels (1 for grayscale)
            out_channels: Number of output channels (1 for noise prediction)
            base_channels: Base number of channels (multiplied at each stage)
            channel_mults: Channel multipliers for each stage
            num_res_blocks: Number of ResBlocks per stage
            attention_resolutions: Resolutions at which to apply attention
            time_emb_dim: Dimension of time embeddings
            num_groups: Number of groups for GroupNorm
            num_heads: Number of attention heads
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions

        # Time embedding network
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # Initial convolution: project input to base_channels
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ==================== ENCODER ====================
        # The encoder progressively downsamples the image while increasing channels
        # At each stage: ResBlock(s) -> Attention -> Downsample
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()

        # Track channels at each resolution for skip connections
        self.skip_channels = [base_channels]  # After init_conv

        current_channels = base_channels
        current_resolution = 32  # Starting resolution

        for level, mult in enumerate(channel_mults[:-1]):  # All but last (that's bottleneck)
            out_ch = base_channels * mult

            # ResBlocks at this level
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlock(current_channels, out_ch, time_emb_dim, num_groups)
                )
                current_channels = out_ch
                self.skip_channels.append(current_channels)

            self.encoder_blocks.append(level_blocks)

            # Attention at this level (if resolution is in attention_resolutions)
            if current_resolution in attention_resolutions:
                self.encoder_attns.append(AttentionBlock(current_channels, num_heads, num_groups))
            else:
                self.encoder_attns.append(nn.Identity())

            # Downsample (except at last encoder level before bottleneck)
            if level < len(channel_mults) - 2:
                self.encoder_downsamples.append(Downsample(current_channels))
                current_resolution //= 2
            else:
                self.encoder_downsamples.append(nn.Identity())

        # ==================== BOTTLENECK ====================
        # Process at the lowest resolution with maximum channels
        bottleneck_channels = base_channels * channel_mults[-1]

        self.bottleneck_res1 = ResBlock(current_channels, bottleneck_channels, time_emb_dim, num_groups)
        self.bottleneck_attn = AttentionBlock(bottleneck_channels, num_heads, num_groups)
        self.bottleneck_res2 = ResBlock(bottleneck_channels, bottleneck_channels, time_emb_dim, num_groups)

        current_channels = bottleneck_channels

        # ==================== DECODER ====================
        # The decoder progressively upsamples while decreasing channels
        # Skip connections from encoder are concatenated before ResBlocks
        # At each stage: Upsample -> Concat skip -> ResBlock(s) -> Attention
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()

        # Reverse the channel multipliers for decoder
        decoder_mults = list(reversed(channel_mults[:-1]))

        for level, mult in enumerate(decoder_mults):
            out_ch = base_channels * mult

            # Upsample first (increases resolution)
            self.decoder_upsamples.append(Upsample(current_channels))
            current_resolution *= 2

            # ResBlocks with skip connections
            # The first ResBlock receives concatenated features (current + skip)
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                # Pop the corresponding skip connection channels
                skip_ch = self.skip_channels.pop() if self.skip_channels else 0
                in_ch = current_channels + skip_ch if i == 0 else current_channels

                level_blocks.append(
                    ResBlock(in_ch, out_ch, time_emb_dim, num_groups)
                )
                current_channels = out_ch

            self.decoder_blocks.append(level_blocks)

            # Attention at this level
            if current_resolution in attention_resolutions:
                self.decoder_attns.append(AttentionBlock(current_channels, num_heads, num_groups))
            else:
                self.decoder_attns.append(nn.Identity())

        # ==================== OUTPUT ====================
        # Final convolution to project back to output channels
        self.final_norm = nn.GroupNorm(num_groups, current_channels)
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

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
        skips = [h]

        # ==================== ENCODER ====================
        for blocks, attn, downsample in zip(
            self.encoder_blocks, self.encoder_attns, self.encoder_downsamples
        ):
            # Apply ResBlocks
            for block in blocks:
                h = block(h, time_emb)
                skips.append(h)

            # Apply attention
            h = attn(h)

            # Downsample
            h = downsample(h)

        # ==================== BOTTLENECK ====================
        h = self.bottleneck_res1(h, time_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, time_emb)

        # ==================== DECODER ====================
        for upsample, blocks, attn in zip(
            self.decoder_upsamples, self.decoder_blocks, self.decoder_attns
        ):
            # Upsample
            h = upsample(h)

            # Apply ResBlocks with skip connections
            for i, block in enumerate(blocks):
                if i == 0 and skips:
                    # Concatenate skip connection
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb)

            # Apply attention
            h = attn(h)

        # ==================== OUTPUT ====================
        h = self.final_norm(h)
        h = nn.functional.silu(h)
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
        num_res_blocks=1,
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
