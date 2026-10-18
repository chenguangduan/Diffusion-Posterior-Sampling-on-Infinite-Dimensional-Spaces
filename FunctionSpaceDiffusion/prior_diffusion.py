from __future__ import annotations

import math
from typing import Tuple
import torch

from .nn import TimeConditionedMLP
from .time_discretization import reverse_time_grid, heun_reverse_predictor, heun_reverse_corrector


class PriorDiffusionModel:
    def __init__(
        self,
        denoiser: TimeConditionedMLP,
        num_modes: int,
        cm_radius: float,
        alpha: float,
        t_min: float,
        t_max: float,
        device: torch.device,
        dtype: torch.dtype,
        gamma: float | None = None,
    ) -> None:
        # gamma=None preserves A=I; gamma>0 selects A=gamma I-Delta.
        if gamma is not None:
            gamma = float(gamma)
            if not math.isfinite(gamma) or gamma <= 0:
                raise ValueError("gamma must be finite and positive, or None for A=I")
        self.device = device
        self.dtype = dtype
        self.num_modes = num_modes
        self.cm_radius = cm_radius 
        self.alpha = alpha
        self.t_min, self.t_max = t_min, t_max
        kmax = math.ceil(math.sqrt(2 * num_modes))
        pairs = [(k, l) for k in range(kmax + 1) for l in range(kmax + 1)]
        pairs.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]))
        self.modes = torch.tensor(pairs[:num_modes], device=self.device)
        self.eigenvalues = math.pi * math.pi * self.modes.to(dtype).square().sum(-1)
        # The existing cosine basis diagonalizes the Neumann Laplacian.
        self.gamma = gamma
        self.drift_rates = (torch.ones_like(self.eigenvalues) if gamma is None
                            else gamma + self.eigenvalues)
        if not bool(torch.isfinite(self.drift_rates).all() & (self.drift_rates > 0).all()):
            raise ValueError("drift rates must be finite and positive in the chosen dtype")
        self.reference_rate = 1.0 if gamma is None else gamma
        # Keep the original covariance C and physical prior unchanged.
        # The physical noise covariance is then B B* = 2 A C.
        self.c = (1.0 + self.eigenvalues).pow(-alpha)
        self.sqrt_c = self.c.sqrt()
        self.denoiser = denoiser.to(device=self.device, dtype=self.dtype)

    def _mean_variance(
        self, 
        time_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert time_points.ndim == 1
        scaled_time = time_points[:, None] * self.drift_rates[None, :]
        return torch.exp(-scaled_time), -torch.expm1(-2 * scaled_time)

    def sample_marginal(
        self, 
        samples: torch.Tensor, 
        time_points: torch.Tensor 
    ) -> torch.Tensor:
        assert samples.ndim == 2 and time_points.ndim == 1
        assert samples.size(0) == time_points.size(0)
        mean, variance = self._mean_variance(time_points=time_points)
        noise = torch.randn_like(samples)
        return mean * samples + variance.sqrt() * noise

    def project_cm_ball(self, samples: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(samples, dim=-1, keepdim=True)
        scale = (self.cm_radius / norm.clamp_min(torch.finfo(samples.dtype).tiny)).clamp(max=1)
        return samples * scale

    def denoise(self, samples: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        assert samples.ndim == 2 and time.ndim == 0
        time_points = time * torch.ones((samples.size(0),), device=self.device, dtype=self.dtype)
        denoised = self.denoiser(samples, time_points)
        return self.project_cm_ball(denoised)

    def to_physical(self, samples: torch.Tensor):
        # Whitened coordinates -> coefficients in the L2-orthonormal basis.
        return self.sqrt_c * torch.as_tensor(samples, dtype=self.dtype, device=self.device)

    def train(
        self, 
        prior_samples: torch.Tensor, 
        num_epoches: int, 
        batch_size: int,
        learning_rate: float, 
        validation_fraction: float = 0.1, 
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ) -> TimeConditionedMLP:
        return _train_score(
            self, 
            samples=prior_samples, 
            denoiser=self.denoiser, 
            num_epoches=num_epoches, 
            batch_size=batch_size,
            learning_rate=learning_rate, 
            t_min=self.t_min, 
            t_max=self.t_max, 
            validation_fraction=validation_fraction, 
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm)

    def reverse_simulate(
        self, 
        start_time: float, 
        samples: torch.Tensor, 
        num_chains: int, 
        num_steps: int,
    ) -> torch.Tensor:
        return reverse_simulate(
            prior_model=self,
            start_time=start_time,
            samples=samples,
            num_chains=num_chains,
            num_steps=num_steps)

    def field(self, samples: torch.Tensor, resolution: int) -> torch.Tensor:
        return field(
            num_modes=self.num_modes,
            alpha=self.alpha,
            samples=samples,
            resolution=resolution,
            device=self.device,
            dtype=self.dtype)


@torch.no_grad()
def field(
    num_modes: int,
    alpha: float,
    samples: torch.Tensor, 
    resolution: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # eigenvalues
    kmax = math.ceil(math.sqrt(2 * num_modes))
    pairs = [(k, l) for k in range(kmax + 1) for l in range(kmax + 1)]
    pairs.sort(key=lambda p: (p[0] * p[0] + p[1] * p[1], p[0], p[1]))
    modes = torch.tensor(pairs[:num_modes], device=device)
    eigenvalues = math.pi * math.pi * modes.to(dtype).square().sum(-1)
    sqrt_c = (1.0 + eigenvalues).pow(-alpha).sqrt()
    # from samples to field
    grid = torch.linspace(0.0, 1.0, resolution, dtype=dtype, device=device)
    k, l = modes[:, 0], modes[:, 1]
    scale_k = torch.ones_like(k, dtype=dtype)
    scale_l = torch.ones_like(l, dtype=dtype)
    scale_k[k > 0] = math.sqrt(2.)
    scale_l[l > 0] = math.sqrt(2.)
    hx = torch.cos(math.pi * k[:, None] * grid) * scale_k[:, None]
    hy = torch.cos(math.pi * l[:, None] * grid) * scale_l[:, None]
    return torch.einsum("...j,jx,jy->...xy", sqrt_c * samples, hx, hy)


def _train_score(
    prior_model: PriorDiffusionModel, 
    samples: torch.Tensor, 
    denoiser: TimeConditionedMLP, 
    num_epoches: int, 
    batch_size: int,
    learning_rate: float, 
    t_min: float, 
    t_max: float, 
    validation_fraction: float, 
    weight_decay: float,
    max_grad_norm: float
) -> TimeConditionedMLP:
    prior_model = prior_model
    permutation = torch.randperm(len(samples), device=prior_model.device)
    num_valid = min(int(len(samples) * validation_fraction), 4096)
    validation_samples = samples[permutation[:num_valid]]
    train_samples = samples[permutation[num_valid:]]

    def corrupt(batch_samples):
        unif_rand = torch.rand((len(batch_samples),), dtype=prior_model.dtype, device=prior_model.device)
        time_points = torch.exp(math.log(t_min) + math.log(t_max / t_min) * unif_rand)
        noisy_samples = prior_model.sample_marginal(batch_samples, time_points)
        return batch_samples, noisy_samples, time_points

    def objective(batch_samples, noisy_samples, time_points):
        prediction = denoiser(noisy_samples, time_points)
        return (prediction - batch_samples).square().mean()

    validation_data = corrupt(validation_samples)
    def validate():
        denoiser.eval()
        with torch.no_grad():
            value = objective(*validation_data).item()
        return value

    optimizer = torch.optim.AdamW(params=denoiser.parameters(), lr=learning_rate, weight_decay=weight_decay)
    initial_loss = validate()
    best_loss, best_epoch = initial_loss, 0
    best_state = {k: v.detach().cpu().clone() for k, v in denoiser.state_dict().items()}
    running, count = 0.0, 0
    for epoch in range(1, num_epoches + 1):
        denoiser.train()
        indices = torch.randint(len(train_samples), (batch_size,), device=prior_model.device)
        optimizer.zero_grad()
        loss = objective(*corrupt(train_samples[indices]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=denoiser.parameters(), 
            max_norm=max_grad_norm, 
            error_if_nonfinite=True)
        optimizer.step()
        running += loss.item()
        count += 1
        if epoch % 500 == 0 or epoch == num_epoches:
            validation_loss = validate()
            print(f"epoch = {epoch:>5d}/{num_epoches:>5d}, train loss = {running / count:>.3e}, validation loss = {validation_loss:>.3e}")
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in denoiser.state_dict().items()}
            running, count = 0.0, 0
    denoiser.load_state_dict(best_state)
    denoiser.eval()
    return denoiser


@torch.no_grad()
def reverse_simulate(
    prior_model: PriorDiffusionModel, 
    start_time: float, 
    samples: torch.Tensor, 
    num_chains: int,
    num_steps: int, 
    grid_spacing: str = "log_snr",
) -> torch.Tensor:
    prior_model.denoiser.eval()
    samples = samples.detach()
    num_samples, dim = samples.shape
    particles = samples.repeat_interleave(num_chains, dim=0)
    assert start_time > prior_model.t_min
    time_grid = reverse_time_grid(start_time, prior_model.t_min, num_steps,
                                    dtype=prior_model.dtype, device=prior_model.device, 
                                    spacing=grid_spacing,
                                    reference_rate=prior_model.reference_rate)
    for current_time, next_time in zip(time_grid[:-1], time_grid[1:]):
        denoised = prior_model.denoise(particles, current_time)
        particles = heun_reverse_predictor(
            samples=particles, 
            denoised=denoised, 
            current_time=current_time, 
            next_time=next_time,
            drift_rates=prior_model.drift_rates)
        next_denoised = prior_model.denoise(particles, next_time)
        particles = heun_reverse_corrector(
            samples=particles, 
            denoised=denoised, 
            next_denoised=next_denoised,
            current_time=current_time, 
            next_time=next_time,
            drift_rates=prior_model.drift_rates)
    particles = prior_model.denoise(
        samples=particles, 
        time=torch.tensor(prior_model.t_min))
    return particles.reshape(num_samples, num_chains, dim)

