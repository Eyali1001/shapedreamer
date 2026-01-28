#!/usr/bin/env python3
"""
Sample Generation Script

Generate samples from a trained diffusion model checkpoint.

Usage:
    # Generate 16 samples from best checkpoint
    uv run scripts/sample.py --checkpoint checkpoints/best.pt

    # Generate more samples with visualization of the diffusion process
    uv run scripts/sample.py --checkpoint checkpoints/best.pt --num-samples 36 --show-process

    # Save samples without display
    uv run scripts/sample.py --checkpoint checkpoints/best.pt --output samples.png --no-display
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.model.unet import UNet
from src.model.diffusion import GaussianDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from trained model")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-samples", type=int, default=16,
        help="Number of samples to generate (default: 16)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for samples (default: display only)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (auto-detected if not specified)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Don't display the samples (useful for headless servers)"
    )
    parser.add_argument(
        "--show-process", action="store_true",
        help="Show the diffusion process (intermediate steps)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=1,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def visualize_samples(samples: torch.Tensor, output_path: str = None, show: bool = True):
    """Create and optionally save a grid of samples."""
    # Convert to displayable format
    samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
    samples = samples.clamp(0, 1)

    # Also create binary version
    samples_binary = (samples > 0.5).float()

    # Calculate grid size
    n = samples.shape[0]
    nrow = int(n ** 0.5)
    if nrow * nrow < n:
        nrow += 1

    # Create grids
    grid_cont = make_grid(samples, nrow=nrow, padding=2)
    grid_binary = make_grid(samples_binary, nrow=nrow, padding=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(grid_cont.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    ax1.set_title('Continuous Output')
    ax1.axis('off')

    ax2.imshow(grid_binary.permute(1, 2, 0).cpu().numpy(), cmap='gray')
    ax2.set_title('Binary (Thresholded)')
    ax2.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved samples to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_diffusion_process(model, diffusion, device, num_samples=4, num_steps_to_show=10):
    """Visualize the reverse diffusion process."""
    print("Generating samples with intermediate steps...")

    # Sample with intermediates
    samples, intermediates = diffusion.sample(
        model,
        shape=(num_samples, 1, 32, 32),
        return_intermediates=True,
    )

    # Select timesteps to show (evenly spaced)
    total_steps = len(intermediates)
    step_indices = [int(i * (total_steps - 1) / (num_steps_to_show - 1)) for i in range(num_steps_to_show)]

    # Create figure
    fig, axes = plt.subplots(num_samples, num_steps_to_show, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, step_idx in enumerate(step_indices):
        t = diffusion.num_timesteps - 1 - step_idx  # Convert to actual timestep
        x = intermediates[step_idx]

        for j in range(num_samples):
            img = (x[j, 0].cpu().numpy() + 1) / 2  # [-1, 1] -> [0, 1]
            img = img.clip(0, 1)

            axes[j, i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[j, i].axis('off')

            if j == 0:
                axes[j, i].set_title(f't={t}')

    plt.suptitle('Reverse Diffusion Process (noise → image)', fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    # Set seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Create diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    # Show diffusion process if requested
    if args.show_process:
        visualize_diffusion_process(model, diffusion, device, num_samples=min(4, args.num_samples))

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        samples = diffusion.sample(
            model,
            shape=(args.num_samples, 1, 32, 32),
        )

    # Visualize
    visualize_samples(
        samples,
        output_path=args.output,
        show=not args.no_display
    )

    print("Done!")


if __name__ == "__main__":
    main()
