#!/usr/bin/env python3
"""
Local Training Script for Toy Diffusion Model

This script runs training on your local machine. It's designed to:
1. Work on CPU, CUDA, or Apple Silicon (MPS)
2. Support small-scale testing before cloud training
3. Be easy to customize via command-line arguments

Usage:
    # Quick test run
    uv run scripts/train_local.py --epochs 5 --samples 1000 --batch-size 32

    # Full local training (will be slow on CPU)
    uv run scripts/train_local.py --epochs 100 --samples 10000

    # Resume from checkpoint
    uv run scripts/train_local.py --resume --epochs 100
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.training.trainer import create_trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the toy diffusion model locally"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--samples", type=int, default=10000,
        help="Number of training samples (default: 10000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (auto-detected if not specified)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Directory for checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs",
        help="Directory for TensorBoard logs (default: ./logs)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--sample-every", type=int, default=10,
        help="Generate samples every N epochs (default: 10)"
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="Save checkpoint every N epochs (default: 10)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("TOY DIFFUSION MODEL - LOCAL TRAINING")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Samples:     {args.samples}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Log dir:     {args.log_dir}")
    print()

    # Create trainer
    trainer = create_trainer(
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_samples=args.samples,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # Update logging intervals
    trainer.sample_every_n_epochs = args.sample_every
    trainer.save_every_n_epochs = args.save_every

    # Resume if requested
    if args.resume:
        if trainer.load_checkpoint("latest.pt"):
            print(f"Resuming training from epoch {trainer.epoch}")
        else:
            print("No checkpoint found, starting fresh")

    # Run training
    try:
        trainer.train(num_epochs=args.epochs)
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        trainer._save_checkpoint("interrupted.pt")
        print("Checkpoint saved. You can resume with --resume flag.")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"TensorBoard logs: tensorboard --logdir {args.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
