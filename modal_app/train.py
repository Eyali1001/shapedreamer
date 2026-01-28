"""
Modal Cloud Training Script

Train the diffusion model on Modal's A100 GPUs with automatic checkpointing.

Prerequisites:
1. Install Modal: pip install modal
2. Authenticate: modal token new
3. Create a volume for checkpoints:
   modal volume create diffusion-checkpoints

Usage:
    # Run training on Modal A100
    modal run modal_app/train.py

    # Run with custom parameters
    modal run modal_app/train.py --epochs 200 --batch-size 128

    # Download checkpoints after training
    modal volume get diffusion-checkpoints best.pt ./checkpoints/

The training will automatically:
- Use an A100 GPU for fast training
- Save checkpoints to a Modal Volume (persistent storage)
- Resume from the latest checkpoint if one exists
- Generate sample images periodically
"""

import modal

# Create Modal app
app = modal.App("toy-diffusion")

# Create a persistent volume for checkpoints
# This survives between runs, so you can resume training
volume = modal.Volume.from_name("diffusion-checkpoints", create_if_missing=True)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.12.0",
        "matplotlib>=3.7.0",
    )
)


@app.function(
    image=image,
    gpu="A100",  # Use A100 for fast training
    timeout=3600 * 4,  # 4 hour timeout
    volumes={"/checkpoints": volume},  # Mount the volume
)
def train(
    num_epochs: int = 100,
    num_samples: int = 10000,
    batch_size: int = 64,
    lr: float = 2e-4,
    resume: bool = True,
):
    """
    Train the diffusion model on Modal.

    Args:
        num_epochs: Number of training epochs
        num_samples: Number of training samples to generate
        batch_size: Training batch size (can be larger on A100)
        lr: Learning rate
        resume: Whether to resume from latest checkpoint
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    import random
    import numpy as np
    from PIL import Image, ImageDraw
    import math
    import torch.nn.functional as F
    from pathlib import Path

    device = "cuda"
    checkpoint_dir = Path("/checkpoints")
    log_dir = checkpoint_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ==================== DATASET ====================
    class GeometricShapesDataset(Dataset):
        def __init__(self, num_samples=10000, image_size=32, seed=42):
            self.num_samples = num_samples
            self.image_size = image_size
            rng = random.Random(seed)
            self.image_seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_samples)]

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            rng = random.Random(self.image_seeds[idx])
            img = Image.new('L', (self.image_size, self.image_size), color=0)
            draw = ImageDraw.Draw(img)
            num_shapes = rng.randint(1, 3)

            for _ in range(num_shapes):
                shape_type = rng.choice(['circle', 'rectangle', 'triangle', 'line'])
                size = self.image_size

                if shape_type == 'circle':
                    radius = rng.randint(3, 12)
                    cx = rng.randint(radius, size - radius - 1)
                    cy = rng.randint(radius, size - radius - 1)
                    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
                elif shape_type == 'rectangle':
                    w, h = rng.randint(4, 16), rng.randint(4, 16)
                    x, y = rng.randint(0, size - w - 1), rng.randint(0, size - h - 1)
                    draw.rectangle([x, y, x + w, y + h], fill=255)
                elif shape_type == 'triangle':
                    base, height = rng.randint(6, 14), rng.randint(6, 14)
                    x = rng.randint(0, size - base - 1)
                    y = rng.randint(height, size - 1)
                    points = [(x, y), (x + base, y), (x + base // 2, y - height)]
                    draw.polygon(points, fill=255)
                elif shape_type == 'line':
                    length = rng.randint(5, 20)
                    x1, y1 = rng.randint(0, size - 1), rng.randint(0, size - 1)
                    angle = rng.uniform(0, 2 * np.pi)
                    x2 = max(0, min(size - 1, int(x1 + length * np.cos(angle))))
                    y2 = max(0, min(size - 1, int(y1 + length * np.sin(angle))))
                    draw.line([x1, y1, x2, y2], fill=255, width=2)

            img_array = np.array(img, dtype=np.float32) / 255.0 * 2 - 1
            return torch.from_numpy(img_array).unsqueeze(0)

    # ==================== MODEL COMPONENTS ====================
    class SinusoidalPositionEmbeddings(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, time):
            device = time.device
            half_dim = self.dim // 2
            embeddings = math.log(10000) / (half_dim - 1)
            embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
            embeddings = time[:, None] * embeddings[None, :]
            return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

    class TimeEmbedding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.sinusoidal = SinusoidalPositionEmbeddings(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

        def forward(self, time):
            return self.mlp(self.sinusoidal(time))

    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=8):
            super().__init__()
            self.norm1 = nn.GroupNorm(num_groups, in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        def forward(self, x, time_emb):
            h = self.conv1(F.silu(self.norm1(x)))
            h = h + self.time_mlp(time_emb)[:, :, None, None]
            h = self.conv2(F.silu(self.norm2(h)))
            return h + self.residual_conv(x)

    class AttentionBlock(nn.Module):
        def __init__(self, channels, num_heads=4, num_groups=8):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = channels // num_heads
            self.norm = nn.GroupNorm(num_groups, channels)
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
            self.proj_out = nn.Conv2d(channels, channels, 1)
            self.scale = self.head_dim ** -0.5

        def forward(self, x):
            b, c, h, w = x.shape
            residual = x
            qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, self.head_dim, h * w)
            q, k, v = [qkv[:, i].permute(0, 1, 3, 2) for i in range(3)]
            attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
            out = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(b, c, h, w)
            return self.proj_out(out) + residual

    class Downsample(nn.Module):
        def __init__(self, in_ch, out_ch=None):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch or in_ch, 3, stride=2, padding=1)
        def forward(self, x):
            return self.conv(x)

    class Upsample(nn.Module):
        def __init__(self, in_ch, out_ch=None):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch or in_ch, 3, padding=1)
        def forward(self, x):
            return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

    class UNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                     channel_mults=(1, 2, 4, 8), time_emb_dim=128, num_groups=8, num_heads=4):
            super().__init__()

            self.channels = [base_channels * m for m in channel_mults]
            self.num_levels = len(channel_mults)

            self.time_embedding = TimeEmbedding(time_emb_dim)
            self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

            # Encoder
            self.encoder_resblocks = nn.ModuleList()
            self.encoder_attns = nn.ModuleList()
            self.encoder_downsamples = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = self.channels[i] if i > 0 else base_channels
                out_ch = self.channels[i]
                self.encoder_resblocks.append(ResBlock(in_ch, out_ch, time_emb_dim, num_groups))
                self.encoder_attns.append(AttentionBlock(out_ch, num_heads, num_groups))
                next_ch = self.channels[i + 1]
                self.encoder_downsamples.append(Downsample(out_ch, next_ch))

            # Bottleneck
            bottleneck_ch = self.channels[-1]
            self.bottleneck_res1 = ResBlock(bottleneck_ch, bottleneck_ch, time_emb_dim, num_groups)
            self.bottleneck_attn = AttentionBlock(bottleneck_ch, num_heads, num_groups)
            self.bottleneck_res2 = ResBlock(bottleneck_ch, bottleneck_ch, time_emb_dim, num_groups)

            # Decoder
            self.decoder_upsamples = nn.ModuleList()
            self.decoder_resblocks = nn.ModuleList()
            self.decoder_attns = nn.ModuleList()

            for i in range(self.num_levels - 2, -1, -1):
                in_ch = self.channels[i + 1]
                out_ch = self.channels[i]
                self.decoder_upsamples.append(Upsample(in_ch, out_ch))
                self.decoder_resblocks.append(ResBlock(out_ch * 2, out_ch, time_emb_dim, num_groups))
                self.decoder_attns.append(AttentionBlock(out_ch, num_heads, num_groups))

            self.final_norm = nn.GroupNorm(num_groups, base_channels)
            self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        def forward(self, x, t):
            time_emb = self.time_embedding(t)
            h = self.init_conv(x)
            skips = []

            for resblock, attn, down in zip(self.encoder_resblocks, self.encoder_attns, self.encoder_downsamples):
                h = resblock(h, time_emb)
                h = attn(h)
                skips.append(h)
                h = down(h)

            h = self.bottleneck_res1(h, time_emb)
            h = self.bottleneck_attn(h)
            h = self.bottleneck_res2(h, time_emb)

            for up, resblock, attn in zip(self.decoder_upsamples, self.decoder_resblocks, self.decoder_attns):
                h = up(h)
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = resblock(h, time_emb)
                h = attn(h)

            return self.final_conv(F.silu(self.final_norm(h)))

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ==================== DIFFUSION ====================
    class GaussianDiffusion:
        def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
            self.num_timesteps = num_timesteps
            self.device = device
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

            self.betas = betas.to(device)
            self.alphas = alphas.to(device)
            self.alphas_cumprod = alphas_cumprod.to(device)
            self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
            self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
            posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            self.posterior_variance = posterior_variance.to(device)
            self.posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20)).to(device)

        def _extract(self, a, t, shape):
            return a.gather(-1, t).reshape(t.shape[0], *((1,) * (len(shape) - 1)))

        def q_sample(self, x_start, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x_start)
            return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                    self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise), noise

        def training_loss(self, model, x_start, t=None):
            if t is None:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device)
            x_noisy, noise = self.q_sample(x_start, t)
            return F.mse_loss(model(x_noisy, t), noise)

        @torch.no_grad()
        def p_sample(self, model, x_t, t):
            predicted_noise = model(x_t, t)
            sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
            betas_t = self._extract(self.betas, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            mean = sqrt_recip_alphas_t * (x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            noise = torch.randn_like(x_t)
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
            return mean + nonzero_mask * torch.exp(0.5 * self._extract(self.posterior_log_variance, t, x_t.shape)) * noise

        @torch.no_grad()
        def sample(self, model, shape):
            model.eval()
            x = torch.randn(shape, device=self.device)
            for t in reversed(range(self.num_timesteps)):
                x = self.p_sample(model, x, torch.full((shape[0],), t, device=self.device, dtype=torch.long))
            return x

    # ==================== TRAINING ====================
    print(f"\nCreating dataset with {num_samples} samples...")
    dataset = GeometricShapesDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("Creating model...")
    model = UNet().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    diffusion = GaussianDiffusion(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    writer = SummaryWriter(log_dir=str(log_dir))

    start_epoch = 0
    best_loss = float('inf')
    global_step = 0

    # Resume from checkpoint
    latest_ckpt = checkpoint_dir / "latest.pt"
    if resume and latest_ckpt.exists():
        print(f"Loading checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_loss = ckpt['best_loss']
        print(f"Resumed from epoch {ckpt['epoch']}")

    print(f"\nStarting training from epoch {start_epoch}...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            x = batch.to(device)
            optimizer.zero_grad()
            loss = diffusion.training_loss(model, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            if global_step % 10 == 0:
                writer.add_scalar('Loss/step', loss.item(), global_step)

        epoch_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.6f}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        torch.save(ckpt, checkpoint_dir / "latest.pt")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, checkpoint_dir / "best.pt")

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, checkpoint_dir / f"epoch_{epoch + 1}.pt")

            # Generate samples
            print("Generating samples...")
            samples = diffusion.sample(model, (16, 1, 32, 32))
            samples = (samples + 1) / 2
            from torchvision.utils import make_grid, save_image
            grid = make_grid(samples, nrow=4)
            save_image(grid, checkpoint_dir / f"samples_epoch_{epoch + 1}.png")

        # Commit volume changes
        volume.commit()

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to Modal volume: diffusion-checkpoints")


@app.local_entrypoint()
def main(
    epochs: int = 100,
    samples: int = 10000,
    batch_size: int = 64,
    lr: float = 2e-4,
    no_resume: bool = False,
):
    """
    Entry point for modal run command.

    Args:
        epochs: Number of training epochs
        samples: Number of training samples
        batch_size: Training batch size
        lr: Learning rate
        no_resume: If set, don't resume from checkpoint
    """
    print("=" * 60)
    print("TOY DIFFUSION MODEL - MODAL CLOUD TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Epochs:      {epochs}")
    print(f"  Samples:     {samples}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Resume:      {not no_resume}")
    print()

    train.remote(
        num_epochs=epochs,
        num_samples=samples,
        batch_size=batch_size,
        lr=lr,
        resume=not no_resume,
    )

    print("\nTo download checkpoints:")
    print("  modal volume get diffusion-checkpoints best.pt ./checkpoints/")
    print("  modal volume get diffusion-checkpoints latest.pt ./checkpoints/")
