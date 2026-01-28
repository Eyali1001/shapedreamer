"""
Time Embeddings for Diffusion Models

In diffusion models, the network needs to know which timestep t it's denoising.
We use sinusoidal position embeddings (similar to Transformers) to encode the
timestep into a high-dimensional vector that the model can easily learn from.

Why sinusoidal embeddings?
1. They provide a smooth, continuous representation of time
2. The model can easily learn to extrapolate to unseen timesteps
3. Different frequency components capture different time scales
4. They're deterministic - no learned parameters needed

The embedding formula (for position p and dimension i):
    PE(p, 2i) = sin(p / 10000^(2i/d))
    PE(p, 2i+1) = cos(p / 10000^(2i/d))

Where d is the embedding dimension.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal time embeddings for diffusion models.

    This module takes integer timesteps and converts them into continuous
    vector representations using sinusoidal functions at different frequencies.

    The output can be further processed by an MLP to create time embeddings
    that are added to the network at various points.

    Example:
        >>> embed = SinusoidalPositionEmbeddings(dim=128)
        >>> t = torch.tensor([0, 100, 500, 999])  # batch of timesteps
        >>> embeddings = embed(t)  # shape: (4, 128)
    """

    def __init__(self, dim: int):
        """
        Initialize the embedding layer.

        Args:
            dim: Output dimension of the embeddings. Should be even for
                 the sin/cos pairing to work correctly.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Convert timesteps to sinusoidal embeddings.

        Args:
            time: Tensor of shape (batch_size,) containing integer timesteps

        Returns:
            Tensor of shape (batch_size, dim) containing the embeddings
        """
        device = time.device
        half_dim = self.dim // 2

        # Calculate the frequency scaling factors
        # These create a geometric sequence of frequencies from 1 to 1/10000
        # Lower frequencies (higher indices) capture coarse time information
        # Higher frequencies (lower indices) capture fine time information
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Multiply timesteps by frequencies to get the arguments for sin/cos
        # time[:, None] has shape (batch_size, 1)
        # embeddings[None, :] has shape (1, half_dim)
        # Result has shape (batch_size, half_dim)
        embeddings = time[:, None] * embeddings[None, :]

        # Concatenate sin and cos embeddings
        # Final shape: (batch_size, dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class TimeEmbedding(nn.Module):
    """
    Complete time embedding module with MLP projection.

    This combines sinusoidal embeddings with a learned MLP to create
    time embeddings that can modulate the network's behavior at each timestep.

    The MLP allows the network to learn task-specific transformations of
    the time information, rather than using raw sinusoidal features.

    Architecture:
        Sinusoidal(dim) -> Linear(dim, dim*4) -> SiLU -> Linear(dim*4, dim)
    """

    def __init__(self, dim: int):
        """
        Initialize the time embedding module.

        Args:
            dim: Base dimension for embeddings. The hidden layer uses dim*4.
        """
        super().__init__()

        self.sinusoidal = SinusoidalPositionEmbeddings(dim)

        # MLP to project sinusoidal embeddings to a learned representation
        # The expansion to 4x dimension is a common practice that gives
        # the network more capacity to learn complex time representations
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),  # SiLU (Swish) is commonly used in diffusion models
            nn.Linear(dim * 4, dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Compute time embeddings.

        Args:
            time: Tensor of shape (batch_size,) containing integer timesteps

        Returns:
            Tensor of shape (batch_size, dim) containing learned time embeddings
        """
        # Get sinusoidal embeddings
        x = self.sinusoidal(time)
        # Project through MLP
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    # Test the embeddings
    embed = SinusoidalPositionEmbeddings(dim=128)
    time_embed = TimeEmbedding(dim=128)

    # Test with various timesteps
    t = torch.tensor([0, 100, 500, 999])

    sin_emb = embed(t)
    full_emb = time_embed(t)

    print(f"Input timesteps: {t}")
    print(f"Sinusoidal embedding shape: {sin_emb.shape}")
    print(f"Full time embedding shape: {full_emb.shape}")

    # Visualize the sinusoidal patterns
    import matplotlib.pyplot as plt

    all_t = torch.arange(0, 1000)
    all_emb = embed(all_t).numpy()

    plt.figure(figsize=(12, 4))
    plt.imshow(all_emb.T, aspect='auto', cmap='RdBu')
    plt.xlabel('Timestep')
    plt.ylabel('Embedding Dimension')
    plt.title('Sinusoidal Time Embeddings')
    plt.colorbar(label='Value')
    plt.tight_layout()
    plt.show()
