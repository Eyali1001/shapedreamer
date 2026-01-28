"""
Gaussian Diffusion Process

This module implements the core diffusion process: both the forward (noising)
process and the reverse (denoising) process.

=== DIFFUSION THEORY (Simplified) ===

The key insight of diffusion models is that:
1. We can gradually add noise to data until it becomes pure noise (forward process)
2. We can learn to reverse this process to generate new data (reverse process)

FORWARD PROCESS (q):
    Given a clean image x_0, we add noise over T timesteps to get x_T ~ N(0, I)

    The noising at each step is:
        q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) * x_{t-1}, β_t * I)

    But we can skip directly to any timestep t using:
        q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) * x_0, (1-ᾱ_t) * I)

    Where:
        α_t = 1 - β_t           (how much signal remains at step t)
        ᾱ_t = ∏_{s=1}^t α_s     (cumulative product - total signal remaining)

    In practice, we sample x_t as:
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε,  where ε ~ N(0, I)

REVERSE PROCESS (p):
    To generate, we start from x_T ~ N(0, I) and iteratively denoise:
        p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² * I)

    The neural network predicts the noise ε_θ(x_t, t), and we use it to
    compute the mean μ:
        μ = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-ᾱ_t)) * ε_θ(x_t, t))

TRAINING:
    Loss = E[||ε - ε_θ(x_t, t)||²]

    We train by:
    1. Sample x_0 from data
    2. Sample t uniformly from [1, T]
    3. Sample ε ~ N(0, I)
    4. Compute x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
    5. Predict ε_θ(x_t, t)
    6. Loss = MSE(ε, ε_θ)

NOISE SCHEDULE:
    β_t defines how much noise is added at each step.
    Linear schedule: β increases linearly from β_1 to β_T
    This gives a smooth, gradual noising process.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn


class GaussianDiffusion:
    """
    Implements the Gaussian diffusion process for training and sampling.

    This class manages:
    1. The noise schedule (β values and derived quantities)
    2. Forward process (adding noise to images)
    3. Reverse process (denoising to generate images)
    4. Training loss computation
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        """
        Initialize the diffusion process.

        Args:
            num_timesteps: Total number of diffusion steps (T)
            beta_start: Starting value of β (noise added at t=1)
            beta_end: Ending value of β (noise added at t=T)
            device: Device to store tensors on
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # === NOISE SCHEDULE ===
        # Linear schedule: β increases linearly from beta_start to beta_end
        # This is the original DDPM schedule, simple and effective
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

        # Compute derived quantities
        # α_t = 1 - β_t (how much of the original signal remains after one step)
        alphas = 1.0 - betas

        # ᾱ_t = ∏_{s=1}^t α_s (cumulative product - total signal remaining)
        # At t=T, this is very small (almost no original signal left)
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # ᾱ_{t-1} (shifted by one, used in sampling)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Store all these as buffers (not parameters, but moved with model)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)

        # Precompute values used in forward process
        # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        # Precompute values used in reverse process (sampling)
        # These are used in the formula for the posterior mean
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)

        # Coefficient for predicted noise in the posterior mean formula
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).to(device)

        # Posterior variance (used when adding noise during sampling)
        # This is σ²_t in the reverse process
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance.to(device)

        # Clamp log variance to avoid log(0)
        self.posterior_log_variance = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        ).to(device)

    def to(self, device: str) -> "GaussianDiffusion":
        """Move all tensors to a new device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance = self.posterior_log_variance.to(device)
        return self

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extract values from a 1D tensor at indices t, then reshape for broadcasting.

        Args:
            a: 1D tensor of values (one per timestep)
            t: Tensor of timestep indices
            x_shape: Shape of the tensor to broadcast to

        Returns:
            Tensor of shape (batch_size, 1, 1, 1) for broadcasting with images
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)  # Select values at indices t
        # Reshape to (batch, 1, 1, 1) for broadcasting with (batch, C, H, W)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # ==================== FORWARD PROCESS ====================

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: add noise to x_0 to get x_t.

        This is the "noising" step used during training. Given a clean image
        and a timestep, we add the appropriate amount of noise.

        Formula: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε

        Args:
            x_start: Clean images x_0, shape (batch, C, H, W)
            t: Timesteps, shape (batch,)
            noise: Optional pre-sampled noise (if None, sample fresh)

        Returns:
            Tuple of (noisy image x_t, noise ε that was added)
        """
        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get the coefficients for this timestep
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # Apply the forward process formula
        # x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_noisy, noise

    # ==================== REVERSE PROCESS ====================

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given x_t and predicted noise, reconstruct x_0.

        This is derived from rearranging the forward process formula:
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        =>
        x_0 = (x_t - sqrt(1-ᾱ_t) * ε) / sqrt(ᾱ_t)

        Args:
            x_t: Noisy image at timestep t
            t: Current timestep
            noise: Predicted noise

        Returns:
            Predicted clean image x_0
        """
        sqrt_recip_alphas_cumprod_t = self._extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of p(x_{t-1} | x_t).

        This is the reverse process step: given a noisy image at timestep t,
        compute the distribution of the slightly less noisy image at t-1.

        Args:
            model: Noise prediction network
            x_t: Noisy image at timestep t
            t: Current timestep

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance)
        """
        # Predict the noise that was added
        predicted_noise = model(x_t, t)

        # Predict x_0 from x_t and predicted noise
        x_recon = self.predict_start_from_noise(x_t, t, predicted_noise)

        # Optionally clip x_0 to valid range (helps with stability)
        x_recon = torch.clamp(x_recon, -1.0, 1.0)

        # Compute posterior mean using the formula from the DDPM paper
        # This is a weighted combination of x_t and x_0
        posterior_mean = (
            self._extract(self.betas, t, x_t.shape)
            * self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
            * x_recon
            + self._extract(self.alphas, t, x_t.shape)
            * self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            / (1.0 - self._extract(self.alphas_cumprod, t, x_t.shape))
            * x_t
        )

        # Actually, let's use the simpler formulation that's more numerically stable:
        # μ = (1/sqrt(α_t)) * (x_t - (β_t/sqrt(1-ᾱ_t)) * ε_θ)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        posterior_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).

        This is a single reverse step: take a noisy image and make it
        slightly less noisy.

        Args:
            model: Noise prediction network
            x_t: Noisy image at timestep t
            t: Current timestep (tensor of shape (batch,))

        Returns:
            Slightly denoised image x_{t-1}
        """
        # Get the posterior distribution parameters
        mean, variance, log_variance = self.p_mean_variance(model, x_t, t)

        # Sample from the posterior
        # x_{t-1} = μ + σ * z, where z ~ N(0, I)
        noise = torch.randn_like(x_t)

        # No noise when t = 0 (final step)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples by running the full reverse process.

        Starting from pure noise x_T ~ N(0, I), iteratively denoise
        for T steps to get a clean image x_0.

        Args:
            model: Noise prediction network
            shape: Shape of samples to generate (batch, C, H, W)
            return_intermediates: If True, return all intermediate x_t

        Returns:
            Generated samples x_0 (and optionally intermediates)
        """
        model.eval()
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=self.device)

        # Store intermediates if requested
        intermediates = [x.clone()] if return_intermediates else None

        # Reverse process: t = T-1, T-2, ..., 0
        for t in reversed(range(self.num_timesteps)):
            # Create batch of timesteps
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Single reverse step
            x = self.p_sample(model, x, t_batch)

            if return_intermediates:
                intermediates.append(x.clone())

        if return_intermediates:
            return x, intermediates
        return x

    # ==================== TRAINING ====================

    def training_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the training loss for a batch of images.

        The loss is simply MSE between predicted and actual noise:
        L = E[||ε - ε_θ(x_t, t)||²]

        Args:
            model: Noise prediction network
            x_start: Clean images x_0
            t: Timesteps (if None, sample uniformly)

        Returns:
            Scalar loss value
        """
        batch_size = x_start.shape[0]

        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

        # Forward process: add noise to get x_t
        x_noisy, noise = self.q_sample(x_start, t)

        # Predict the noise
        predicted_noise = model(x_noisy, t)

        # Simple MSE loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        return loss


if __name__ == "__main__":
    # Test the diffusion process
    import matplotlib.pyplot as plt
    from ..data.dataset import GeometricShapesDataset

    # Create dataset and get a sample
    dataset = GeometricShapesDataset(num_samples=1)
    x_0 = dataset[0].unsqueeze(0)  # Add batch dimension

    # Initialize diffusion
    diffusion = GaussianDiffusion(num_timesteps=1000)

    # Visualize forward process at different timesteps
    timesteps = [0, 100, 250, 500, 750, 999]

    fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 3))

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t])
        x_t, _ = diffusion.q_sample(x_0, t_tensor)

        # Convert to displayable format
        img = (x_t.squeeze().numpy() + 1) / 2

        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f't = {t}')
        axes[i].axis('off')

    plt.suptitle('Forward Diffusion Process')
    plt.tight_layout()
    plt.show()

    # Verify that at t=999, the image is nearly pure noise
    t_final = torch.tensor([999])
    x_T, _ = diffusion.q_sample(x_0, t_final)
    print(f"At t=999:")
    print(f"  Mean: {x_T.mean():.4f} (should be ~0)")
    print(f"  Std:  {x_T.std():.4f} (should be ~1)")
