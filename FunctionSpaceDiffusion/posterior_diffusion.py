from __future__ import annotations

import math
import torch

from .prior_diffusion import PriorDiffusionModel
from .likelihood import LogLikelihood
from .posterior_denoiser import PosteriorDenoiser
from .time_discretization import reverse_time_grid, heun_reverse_predictor, heun_reverse_corrector


class PosteriorDiffusionModel:
    def __init__(
        self,
        prior_model: PriorDiffusionModel,
        log_likelihood: LogLikelihood,
        t_min: float,
        t_max: float,
        num_mc_particles: int,
        batch_size_inference: int,
        num_denoiser_steps: int,
    ) -> None:
        self.prior_model = prior_model
        self.device = prior_model.device
        self.dtype = prior_model.dtype
        self.num_modes = prior_model.num_modes
        self.cm_radius = prior_model.cm_radius 
        self.alpha = prior_model.alpha
        self.drift_rates = prior_model.drift_rates
        self.reference_rate = prior_model.reference_rate
        self.t_min, self.t_max = t_min, t_max
        assert prior_model.t_min < self.t_min < self.t_max <= prior_model.t_max
        kmax = math.ceil(math.sqrt(2 * self.num_modes))
        pairs = [(k, l) for k in range(kmax + 1) for l in range(kmax + 1)]
        pairs.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]))
        self.modes = torch.tensor(pairs[:self.num_modes], device=self.device)
        self.eigenvalues = math.pi**2 * self.modes.to(self.dtype).square().sum(-1)
        self.c = (1 + self.eigenvalues).pow(-self.alpha)
        self.sqrt_c = self.c.sqrt()
        self.prior_denoiser = prior_model.denoiser.to(device=self.device, dtype=self.dtype)
        self.log_likelihood = log_likelihood
        self.posterior_denoiser = PosteriorDenoiser(
            prior_model=prior_model,
            log_likelihood=log_likelihood,
            num_mc_particles=num_mc_particles,
            num_diffusion_steps=num_denoiser_steps,
            batch_size_inference=batch_size_inference)

    def denoise(self, samples: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        assert samples.ndim == 2 and time.ndim == 0
        denoised = self.posterior_denoiser(time=time.item(), samples=samples)
        return self.prior_model.project_cm_ball(denoised)

    def to_physical(self, samples: torch.Tensor):
        # Whitened coordinates -> coefficients in the L2-orthonormal basis.
        return self.sqrt_c * torch.as_tensor(samples, dtype=self.dtype, device=self.device)

    def reverse_simulate(
        self, 
        num_samples: int, 
        num_steps: int,
    ) -> torch.Tensor:
        samples = torch.randn((num_samples, self.num_modes), dtype=self.dtype, device=self.device)
        samples = reverse_simulate(
            posterior_model=self,
            start_time=self.t_max,
            samples=samples,
            num_steps=num_steps)
        return samples 

    def field(
        self, 
        samples: torch.Tensor, 
        resolution: int
    ) -> torch.Tensor:
        return self.prior_model.field(samples=samples, resolution=resolution)



@torch.no_grad()
def reverse_simulate(
    posterior_model: PosteriorDiffusionModel, 
    start_time: float, 
    samples: torch.Tensor, 
    num_steps: int, 
    grid_spacing: str = "log_snr",
) -> torch.Tensor:
    samples = samples.detach()
    num_samples, dim = samples.shape
    particles = samples.clone()
    assert start_time > posterior_model.t_min
    time_grid = reverse_time_grid(
        start_time=start_time, 
        stop_time=posterior_model.t_min, 
        num_steps=num_steps,
        dtype=posterior_model.dtype, 
        device=posterior_model.device,
        spacing=grid_spacing,
        reference_rate=posterior_model.reference_rate)
    for current_time, next_time in zip(time_grid[:-1], time_grid[1:]):
        print(f"current time: {current_time.item():>.3e}")
        denoised = posterior_model.denoise(particles, current_time)
        particles = heun_reverse_predictor(
            samples=particles, 
            denoised=denoised, 
            current_time=current_time, 
            next_time=next_time,
            drift_rates=posterior_model.drift_rates)
        next_denoised = posterior_model.denoise(particles, next_time)
        particles = heun_reverse_corrector(
            samples=particles, 
            denoised=denoised, 
            next_denoised=next_denoised,
            current_time=current_time, 
            next_time=next_time,
            drift_rates=posterior_model.drift_rates)
    particles = posterior_model.denoise(
        samples=particles, 
        time=torch.tensor(posterior_model.t_min))
    return particles.reshape(num_samples, dim)


