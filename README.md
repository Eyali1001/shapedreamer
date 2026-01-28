# Toy Diffusion Model

An educational implementation of DDPM (Denoising Diffusion Probabilistic Models) that generates 32x32 binary pixel art of geometric shapes.

This project is designed for **learning** - the code is heavily commented to explain diffusion concepts, and the simple dataset (geometric shapes) makes it easy to verify that the model is working correctly.

## What You'll Learn

- **Forward diffusion process**: How noise is gradually added to images
- **Reverse diffusion process**: How a neural network learns to denoise
- **U-Net architecture**: The standard backbone for diffusion models
- **Time embeddings**: How models condition on the timestep
- **Attention mechanisms**: Capturing global spatial relationships
- **Training dynamics**: What loss curves and samples look like during training

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd toy_diffusion

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Local Training (CPU/MPS/CUDA)

```bash
# Quick test (5 epochs, 1000 samples)
uv run scripts/train_local.py --epochs 5 --samples 1000 --batch-size 32

# Full training (will be slow on CPU, ~30 min on M1 Mac, ~5 min on GPU)
uv run scripts/train_local.py --epochs 100 --samples 10000

# Resume from checkpoint
uv run scripts/train_local.py --resume --epochs 100
```

### Generate Samples

```bash
# Generate samples from a trained model
uv run scripts/sample.py --checkpoint checkpoints/best.pt

# Show the diffusion process (noise → image)
uv run scripts/sample.py --checkpoint checkpoints/best.pt --show-process

# Save samples to file
uv run scripts/sample.py --checkpoint checkpoints/best.pt --output samples.png
```

### Monitor Training

```bash
# Start TensorBoard to view loss curves and generated samples
tensorboard --logdir logs
```

## Cloud Training with Modal

For faster training on A100 GPUs:

```bash
# Install Modal
pip install modal

# Authenticate (one-time)
modal token new

# Run training on Modal (automatically uses A100)
modal run modal_app/train.py

# With custom parameters
modal run modal_app/train.py --epochs 200 --batch-size 128

# Download checkpoints after training
modal volume get diffusion-checkpoints best.pt ./checkpoints/
modal volume get diffusion-checkpoints latest.pt ./checkpoints/
```

## Project Structure

```
toy_diffusion/
├── pyproject.toml              # Dependencies (uv-compatible)
├── src/
│   ├── model/
│   │   ├── unet.py             # U-Net with attention (~2-4M parameters)
│   │   ├── blocks.py           # ResBlock, AttentionBlock, Up/Downsample
│   │   ├── embeddings.py       # Sinusoidal time embeddings
│   │   └── diffusion.py        # Forward/reverse diffusion processes
│   ├── data/
│   │   └── dataset.py          # Synthetic geometric shapes generator
│   └── training/
│       └── trainer.py          # Training loop with checkpointing
├── scripts/
│   ├── train_local.py          # Local training script
│   └── sample.py               # Sample generation script
└── modal_app/
    └── train.py                # Modal A100 training
```

## How It Works

### The Dataset

We generate simple binary images (32x32 pixels) containing:
- Circles
- Rectangles
- Triangles
- Lines

Each image has 1-3 shapes randomly placed. This simple dataset:
- Is trivial to generate (no downloads needed)
- Is easy to visually verify (you can see if shapes look right)
- Trains quickly (simple patterns to learn)

### The Diffusion Process

**Forward Process** (training): Add noise to a clean image
```
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
```

**Reverse Process** (sampling): Start from noise, iteratively denoise
```
x_{t-1} = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-ᾱ_t)) * ε_θ(x_t, t)) + σ_t * z
```

The neural network learns to predict the noise ε that was added at timestep t.

### Architecture

The U-Net predicts noise with:
- **Encoder**: 32→16→8→4 resolution with increasing channels
- **Bottleneck**: Processing at 4x4 with 512 channels
- **Decoder**: 4→8→16→32 resolution with skip connections
- **Time conditioning**: Sinusoidal embeddings added to every ResBlock
- **Attention**: Self-attention at every resolution for global context

## Expected Results

### Training
- Loss should decrease from ~1.0 to ~0.01-0.05 over 100 epochs
- Early epochs: noisy/blurry samples
- Middle epochs: vague shapes emerging
- Final epochs: clean geometric shapes

### Samples
After training, you should see clear circles, rectangles, triangles, and lines similar to the training data.

## Configuration

Key hyperparameters (defaults work well):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--samples` | 10000 | Number of training images |
| `--batch-size` | 64 | Training batch size |
| `--lr` | 2e-4 | Learning rate |
| Timesteps | 1000 | Diffusion steps (hardcoded) |
| β schedule | 1e-4 → 0.02 | Linear noise schedule |

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` to 32 or 16
- On CPU, use `--batch-size 8`

### Slow Training
- Use Modal for cloud GPU training
- On Mac, ensure MPS is being used (should auto-detect)
- Reduce `--samples` for faster iterations

### Poor Sample Quality
- Train for more epochs
- Check TensorBoard for loss curves
- Ensure loss is decreasing (should reach <0.1)

## Further Reading

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - The original DDPM paper
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) - Great tutorial paper
- [Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion) - HuggingFace blog post

## License

MIT
