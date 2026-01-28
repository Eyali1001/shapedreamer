#!/usr/bin/env python3
"""
Live Continuous Diffusion

Creates an endlessly evolving image using the diffusion model.
The image continuously morphs between different geometric shapes.

How it works:
1. Start with a generated image
2. Add partial noise (jump back to timestep t)
3. Denoise back to a clean image
4. Repeat - creating smooth morphing between shapes

The 'creativity' parameter controls how much noise is added:
- Low (50-100): subtle variations, shapes stay similar
- Medium (200-400): moderate morphing between shapes
- High (500-800): dramatic transformations

Usage:
    uv run scripts/live_diffusion.py --checkpoint checkpoints/best.pt
    uv run scripts/live_diffusion.py --checkpoint checkpoints/best.pt --creativity 300
    uv run scripts/live_diffusion.py --checkpoint checkpoints/best.pt --save-video output.mp4
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from src.model.unet import UNet
from src.model.diffusion import GaussianDiffusion


class LiveDiffusion:
    """
    Continuous diffusion process that evolves images over time.

    Instead of generating single images, this runs an endless loop:
    clean image → add noise → denoise → repeat

    This creates a mesmerizing, ever-changing display of shapes.
    """

    def __init__(
        self,
        model: UNet,
        diffusion: GaussianDiffusion,
        device: str = "mps",
        creativity: int = 300,
    ):
        """
        Initialize live diffusion.

        Args:
            model: Trained U-Net model
            diffusion: Diffusion process
            device: Device to run on
            creativity: How much noise to add each cycle (timestep to jump back to)
                       Higher = more dramatic changes between frames
        """
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.creativity = creativity

        # Current image state
        self.current_image = None

    def initialize(self, batch_size: int = 1):
        """Generate initial image from pure noise."""
        print("Generating initial image...")
        self.current_image = self.diffusion.sample(
            self.model,
            shape=(batch_size, 1, 32, 32),
        )
        return self.current_image

    @torch.no_grad()
    def step(self, steps_per_frame: int = 50):
        """
        Evolve the image by one cycle.

        1. Add noise to current image (jump to timestep=creativity)
        2. Denoise back to clean image

        Args:
            steps_per_frame: How many denoising steps per evolution
                            (fewer = faster but choppier)

        Returns:
            Evolved image tensor
        """
        if self.current_image is None:
            self.initialize()

        batch_size = self.current_image.shape[0]

        # Add noise - jump back to timestep 'creativity'
        t = torch.tensor([self.creativity], device=self.device)
        noisy, _ = self.diffusion.q_sample(self.current_image, t)

        # Denoise from creativity back to 0
        # We skip some steps for speed (step by steps_per_frame intervals)
        x = noisy
        step_size = max(1, self.creativity // steps_per_frame)

        for timestep in range(self.creativity - 1, -1, -step_size):
            t_batch = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
            x = self.diffusion.p_sample(self.model, x, t_batch)

        self.current_image = x
        return x

    def get_display_image(self) -> np.ndarray:
        """Convert current image to displayable numpy array."""
        if self.current_image is None:
            return np.zeros((32, 32))

        img = self.current_image[0, 0].cpu().numpy()
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        img = np.clip(img, 0, 1)
        return img


def load_model(checkpoint_path: str, device: str):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def run_live_display(
    checkpoint_path: str,
    device: str = "mps",
    creativity: int = 300,
    fps: int = 10,
    save_video: str = None,
    num_frames: int = None,
):
    """
    Run the live diffusion display.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to use
        creativity: Noise level for evolution (50-800)
        fps: Frames per second
        save_video: If provided, save to video file instead of display
        num_frames: Number of frames (None = infinite for display, 300 for video)
    """
    # Load model
    model = load_model(checkpoint_path, device)

    # Create diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    # Create live diffusion
    live = LiveDiffusion(
        model=model,
        diffusion=diffusion,
        device=device,
        creativity=creativity,
    )

    # Initialize
    live.initialize()

    if save_video:
        # Save to video file
        num_frames = num_frames or 300
        print(f"Recording {num_frames} frames to {save_video}...")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')

        frames = []
        for i in range(num_frames):
            live.step(steps_per_frame=30)
            img = live.get_display_image()

            ax.clear()
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.axis('off')
            ax.set_title(f'Frame {i+1}/{num_frames}')

            # Convert to image
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)

            if (i + 1) % 10 == 0:
                print(f"  Frame {i+1}/{num_frames}")

        # Save as video
        import imageio
        imageio.mimsave(save_video, frames, fps=fps)
        print(f"Saved video to {save_video}")
        plt.close()

    else:
        # Live display
        print(f"\nStarting live diffusion (creativity={creativity}, fps={fps})")
        print("Close the window to stop.\n")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')

        img_display = ax.imshow(
            live.get_display_image(),
            cmap='gray',
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )
        ax.set_title('Live Diffusion - Evolving Shapes')

        frame_count = [0]
        start_time = [time.time()]

        def update(frame):
            live.step(steps_per_frame=30)
            img_display.set_data(live.get_display_image())

            frame_count[0] += 1
            elapsed = time.time() - start_time[0]
            actual_fps = frame_count[0] / elapsed if elapsed > 0 else 0
            ax.set_title(f'Live Diffusion | Frame {frame_count[0]} | {actual_fps:.1f} FPS')

            return [img_display]

        ani = animation.FuncAnimation(
            fig,
            update,
            interval=1000 // fps,  # milliseconds between frames
            blit=True,
            cache_frame_data=False,
        )

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Live evolving diffusion display")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detects MPS/CUDA/CPU if not specified)"
    )
    parser.add_argument(
        "--creativity", type=int, default=300,
        help="How much the image changes each frame (50-800, default: 300)"
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Target frames per second (default: 10)"
    )
    parser.add_argument(
        "--save-video", type=str, default=None,
        help="Save to video file instead of live display"
    )
    parser.add_argument(
        "--num-frames", type=int, default=None,
        help="Number of frames for video (default: 300)"
    )

    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    run_live_display(
        checkpoint_path=args.checkpoint,
        device=device,
        creativity=args.creativity,
        fps=args.fps,
        save_video=args.save_video,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
