"""
Training Loop for Diffusion Model

This module provides a clean, well-documented training loop with:
1. Automatic checkpointing
2. TensorBoard logging
3. Sample generation during training
4. Resume capability

The trainer is designed to work both locally and on Modal (cloud GPU).
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..model.diffusion import GaussianDiffusion


class Trainer:
    """
    Trainer for the diffusion model.

    Handles the full training loop including:
    - Forward pass through diffusion process
    - Backpropagation and optimizer steps
    - Learning rate scheduling
    - Checkpointing and resuming
    - Logging metrics and sample images
    """

    def __init__(
        self,
        model: nn.Module,
        diffusion: GaussianDiffusion,
        dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 2e-4,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        device: str = "cpu",
        sample_every_n_epochs: int = 10,
        save_every_n_epochs: int = 10,
    ):
        """
        Initialize the trainer.

        Args:
            model: The U-Net model to train
            diffusion: GaussianDiffusion instance
            dataloader: Training data loader
            optimizer: Optional optimizer (creates AdamW if None)
            lr: Learning rate
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            device: Device to train on ('cpu', 'cuda', 'mps')
            sample_every_n_epochs: Generate samples every N epochs
            save_every_n_epochs: Save checkpoint every N epochs
        """
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.dataloader = dataloader
        self.device = device
        self.lr = lr

        # Create optimizer if not provided
        # AdamW is the go-to optimizer for diffusion models
        # - Good for training stability
        # - Weight decay helps regularization
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),  # Standard Adam betas
            weight_decay=1e-4,   # Mild weight decay
        )

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # Logging intervals
        self.sample_every_n_epochs = sample_every_n_epochs
        self.save_every_n_epochs = save_every_n_epochs

    def train(self, num_epochs: int) -> None:
        """
        Run the training loop.

        Args:
            num_epochs: Total number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Dataset size: {len(self.dataloader.dataset)}")
        print(f"Batch size: {self.dataloader.batch_size}")
        print(f"Batches per epoch: {len(self.dataloader)}")
        print("-" * 50)

        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()

            # Log epoch metrics
            self.writer.add_scalar('Loss/epoch', epoch_loss, epoch)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.6f}")

            # Save best model
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_checkpoint("best.pt")

            # Periodic checkpointing
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self._save_checkpoint("latest.pt")
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Generate samples
            if (epoch + 1) % self.sample_every_n_epochs == 0:
                self._generate_samples(epoch + 1)

        # Final save
        self._save_checkpoint("final.pt")
        print("Training complete!")

    def _train_epoch(self) -> float:
        """
        Train for a single epoch.

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.epoch + 1}", leave=False)

        for batch in pbar:
            # Move data to device
            x = batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            loss = self.diffusion.training_loss(self.model, x)

            # Backpropagation
            loss.backward()

            # Gradient clipping (helps stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

            # Log to TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('Loss/step', loss.item(), self.global_step)

        return total_loss / num_batches

    @torch.no_grad()
    def _generate_samples(self, epoch: int, num_samples: int = 16) -> None:
        """
        Generate and save sample images.

        Args:
            epoch: Current epoch number (for naming)
            num_samples: Number of samples to generate
        """
        print(f"Generating {num_samples} samples...")

        # Generate samples
        samples = self.diffusion.sample(
            self.model,
            shape=(num_samples, 1, 32, 32),
        )

        # Convert to displayable format
        samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
        samples = samples.clamp(0, 1)

        # Threshold to binary for clean shapes
        samples_binary = (samples > 0.5).float()

        # Create grid
        from torchvision.utils import make_grid
        grid = make_grid(samples, nrow=4, normalize=False, padding=2)
        grid_binary = make_grid(samples_binary, nrow=4, normalize=False, padding=2)

        # Log to TensorBoard
        self.writer.add_image('Samples/continuous', grid, epoch)
        self.writer.add_image('Samples/binary', grid_binary, epoch)

        # Also save to file
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax1.set_title(f'Epoch {epoch} - Continuous')
        ax1.axis('off')

        ax2.imshow(grid_binary.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax2.set_title(f'Epoch {epoch} - Binary')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / f"samples_epoch_{epoch}.png", dpi=150)
        plt.close()

        print(f"Saved samples to {self.checkpoint_dir / f'samples_epoch_{epoch}.png'}")

    def _save_checkpoint(self, filename: str) -> None:
        """
        Save a training checkpoint.

        Args:
            filename: Name of checkpoint file
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str = "latest.pt") -> bool:
        """
        Load a training checkpoint.

        Args:
            filename: Name of checkpoint file

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"No checkpoint found at {path}")
            return False

        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        print(f"Resumed from epoch {checkpoint['epoch']}, step {self.global_step}")
        return True


def create_trainer(
    checkpoint_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    num_samples: int = 10000,
    batch_size: int = 64,
    lr: float = 2e-4,
    device: Optional[str] = None,
) -> Trainer:
    """
    Factory function to create a Trainer with default settings.

    This is a convenience function that sets up all the components
    (dataset, model, diffusion, trainer) with sensible defaults.

    Args:
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for TensorBoard logs
        num_samples: Number of training samples
        batch_size: Training batch size
        lr: Learning rate
        device: Device to use (auto-detected if None)

    Returns:
        Configured Trainer instance
    """
    from ..data.dataset import GeometricShapesDataset
    from ..model.unet import UNet

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = GeometricShapesDataset(num_samples=num_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep it simple for compatibility
        pin_memory=(device == "cuda"),
    )

    # Create model
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
    )

    # Create diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device,
    )

    return trainer


if __name__ == "__main__":
    # Quick test of the trainer
    trainer = create_trainer(
        num_samples=100,
        batch_size=8,
    )

    # Train for a few epochs
    trainer.train(num_epochs=2)
