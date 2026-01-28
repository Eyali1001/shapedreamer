"""
Synthetic Geometric Shapes Dataset for Diffusion Model Training

This module generates binary images (32x32) containing simple geometric shapes:
circles, rectangles, triangles, and lines. Each image contains 1-3 shapes
randomly placed and sized.

The binary nature (black/white only) makes it easy to:
1. Generate synthetic data on-the-fly
2. Visually interpret the quality of generated samples
3. Evaluate whether the model learned the shape distributions
"""

import random
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class GeometricShapesDataset(Dataset):
    """
    A PyTorch Dataset that generates binary images of geometric shapes.

    The dataset generates images on-the-fly using a seeded random generator,
    ensuring reproducibility while avoiding the need to store images on disk.

    Each image contains 1-3 shapes randomly selected from:
    - Circles: varying radii (3-12 pixels)
    - Rectangles: varying sizes (4-16 pixels per side)
    - Triangles: varying sizes (6-14 pixels base)
    - Lines: varying lengths (5-20 pixels)

    Attributes:
        num_samples: Number of images in the dataset
        image_size: Size of square images (default 32x32)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 32,
        seed: int = 42,
        transform: Optional[callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            num_samples: Total number of images to generate
            image_size: Height and width of generated images
            seed: Random seed for reproducible generation
            transform: Optional transform to apply to images
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.transform = transform

        # Pre-generate seeds for each image to ensure reproducibility
        # even when accessing images in random order
        rng = random.Random(seed)
        self.image_seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Generate and return a single image.

        The image is generated deterministically based on the index,
        allowing for reproducible dataset access.

        Args:
            idx: Index of the image to generate

        Returns:
            Tensor of shape (1, image_size, image_size) with values in [-1, 1]
            where -1 is black (background) and 1 is white (shapes)
        """
        # Use the pre-generated seed for this index
        rng = random.Random(self.image_seeds[idx])

        # Create a black image (background)
        img = Image.new('L', (self.image_size, self.image_size), color=0)
        draw = ImageDraw.Draw(img)

        # Randomly choose how many shapes to draw (1-3)
        num_shapes = rng.randint(1, 3)

        for _ in range(num_shapes):
            # Randomly select a shape type
            shape_type = rng.choice(['circle', 'rectangle', 'triangle', 'line'])
            self._draw_shape(draw, shape_type, rng)

        # Convert to numpy array and normalize to [-1, 1]
        # This normalization is standard for diffusion models:
        # - Input images are scaled to [-1, 1]
        # - The model predicts noise in this same range
        # - Final samples are rescaled back to [0, 255]
        img_array = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        img_array = img_array * 2 - 1  # [-1, 1]

        # Convert to tensor with channel dimension: (1, H, W)
        tensor = torch.from_numpy(img_array).unsqueeze(0)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor

    def _draw_shape(self, draw: ImageDraw.Draw, shape_type: str, rng: random.Random) -> None:
        """
        Draw a single shape on the image.

        Args:
            draw: PIL ImageDraw object
            shape_type: Type of shape to draw
            rng: Random number generator for this image
        """
        size = self.image_size

        if shape_type == 'circle':
            # Circle with random center and radius
            radius = rng.randint(3, 12)
            # Ensure circle stays mostly within bounds
            cx = rng.randint(radius, size - radius - 1)
            cy = rng.randint(radius, size - radius - 1)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=255
            )

        elif shape_type == 'rectangle':
            # Rectangle with random position and size
            w = rng.randint(4, 16)
            h = rng.randint(4, 16)
            x = rng.randint(0, size - w - 1)
            y = rng.randint(0, size - h - 1)
            draw.rectangle([x, y, x + w, y + h], fill=255)

        elif shape_type == 'triangle':
            # Triangle with random position and size
            base = rng.randint(6, 14)
            height = rng.randint(6, 14)
            # Random position for bottom-left corner
            x = rng.randint(0, size - base - 1)
            y = rng.randint(height, size - 1)
            # Three vertices: bottom-left, bottom-right, top-center
            points = [
                (x, y),  # bottom-left
                (x + base, y),  # bottom-right
                (x + base // 2, y - height),  # top-center
            ]
            draw.polygon(points, fill=255)

        elif shape_type == 'line':
            # Line with random start and end points
            length = rng.randint(5, 20)
            x1 = rng.randint(0, size - 1)
            y1 = rng.randint(0, size - 1)
            # Random angle
            angle = rng.uniform(0, 2 * np.pi)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            # Clamp to image bounds
            x2 = max(0, min(size - 1, x2))
            y2 = max(0, min(size - 1, y2))
            # Use width > 1 for visibility
            draw.line([x1, y1, x2, y2], fill=255, width=2)

    def visualize_samples(self, num_samples: int = 16, save_path: Optional[str] = None) -> None:
        """
        Visualize a grid of sample images from the dataset.

        Args:
            num_samples: Number of samples to show (will be arranged in a square grid)
            save_path: If provided, save the visualization to this path
        """
        import matplotlib.pyplot as plt

        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_samples)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()

        for i in range(num_samples):
            img = self[i]
            # Convert from [-1, 1] to [0, 1] for display
            img_display = (img.squeeze().numpy() + 1) / 2
            axes[i].imshow(img_display, cmap='gray', vmin=0, vmax=1)
            axes[i].axis('off')

        # Hide unused axes
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Quick test of the dataset
    dataset = GeometricShapesDataset(num_samples=100)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")
    print(f"Sample value range: [{dataset[0].min():.2f}, {dataset[0].max():.2f}]")

    # Visualize some samples
    dataset.visualize_samples(16)
