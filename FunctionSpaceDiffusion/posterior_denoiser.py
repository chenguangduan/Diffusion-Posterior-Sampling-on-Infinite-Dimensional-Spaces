from __future__ import annotations

import math
import torch

from .prior_diffusion import PriorDiffusionModel
from .likelihood import LogLikelihood
from .time_discretization import noise_log_ratio


class PosteriorDenoiser:
    def __init__(
        self,
        prior_model: PriorDiffusionModel,
        log_likelihood: LogLikelihood,
        num_mc_particles: int,
        batch_size_inference: int,
        num_diffusion_steps: int,
    ) -> None:
        self.prior_model = prior_model
        self.log_likelihood = log_likelihood
        self.num_mc_particles = num_mc_particles
        self.batch_size_inference = batch_size_inference
        self.num_diffusion_steps = num_diffusion_steps

    def __call__(
            self, 
            samples: torch.Tensor,
            time: float
        ) -> torch.Tensor:
        num_samples = len(samples)
        batch_size_inference = self.batch_size_inference or max(1, num_samples)
        denoised = torch.empty_like(samples)
        for start in range(0, num_samples, batch_size_inference):
            stop = min(start + batch_size_inference, num_samples)
            print(f"Denoiser: {start + 1} to {stop} of {num_samples} samples")
            denoised[start:stop] = _evaluate_posterior_denoiser(
                time=time,
                samples=samples[start:stop],
                prior_model=self.prior_model,
                log_likelihood_fn=self.log_likelihood,
                num_mc_particles=self.num_mc_particles,
                num_diffusion_steps=self.num_diffusion_steps)
        return denoised


@torch.no_grad()
def _evaluate_posterior_denoiser(
    time: float,
    samples: torch.Tensor,
    prior_model: PriorDiffusionModel,
    log_likelihood_fn: LogLikelihood,
    num_mc_particles: int,
    num_diffusion_steps: int,
) -> torch.Tensor:
    times = torch.tensor([time, prior_model.t_min, prior_model.t_max], dtype=torch.float64)
    transformed = noise_log_ratio(t=times, rate=prior_model.reference_rate)
    ratio = float((transformed[0] - transformed[1]) / (transformed[2] - transformed[1]))
    actual_num_steps = max(1, math.ceil(num_diffusion_steps * ratio))
    # endpoints: (num_samples, num_chains, num_modes)
    endpoints = prior_model.reverse_simulate(
        start_time=time, 
        samples=samples,
        num_chains=num_mc_particles,
        num_steps=actual_num_steps) 
    # theta: (num_samples, num_chains, resolution, resolution)
    theta = prior_model.field(
        endpoints, resolution=log_likelihood_fn.resolution)
    # theta: (num_samples, num_chains)
    log_weights = log_likelihood_fn(theta)
    # normalized: (num_samples, num_chains)
    normalized = torch.softmax(log_weights, dim=-1)
    denoiser = torch.sum(normalized[..., None] * endpoints, dim=1)
    return denoiser


@torch.no_grad()
def _evaluate_posterior_denoiser_smc(
    time: float,
    samples: torch.Tensor,
    prior_model: PriorDiffusionModel,
    log_likelihood_fn: LogLikelihood,
    num_mc_particles: int,
    num_diffusion_steps: int,
    num_smc_steps: int = 4,
    cess_target: float = 0.8,
    resample_threshold: float = 0.5,
    num_mh_steps: int = 2,
) -> torch.Tensor:
    times = torch.tensor([time, prior_model.t_min, prior_model.t_max], dtype=torch.float64)
    transformed = noise_log_ratio(times, rate=prior_model.reference_rate)
    ratio = float((transformed[0] - transformed[1]) / (transformed[2] - transformed[1]))
    actual_num_steps = max(1, math.ceil(num_diffusion_steps * ratio))

    # endpoints: (num_samples, num_chains, num_modes)
    endpoints = prior_model.reverse_simulate(
        start_time=time, 
        samples=samples,
        num_chains=num_mc_particles,
        num_steps=actual_num_steps) 
    # theta: (num_samples, num_chains, resolution, resolution)
    theta = prior_model.field(
        endpoints, resolution=log_likelihood_fn.resolution)
    # log_likelihood: (num_samples, num_chains)
    log_likelihood = log_likelihood_fn(theta)
    log_weights = torch.full_like(log_likelihood, -math.log(num_mc_particles))
    
    num_samples = samples.size(0)
    device = samples.device
    dtype = samples.dtype
    accepted_total = torch.zeros(num_samples, device=device, dtype=dtype)
    resampling_count = torch.zeros(num_samples, device=device, dtype=torch.long)

    beta = 0.0
    stages = 0
    # K+1 endpoints define exactly K temperature updates, ending at beta=1.
    betas = [(step / num_smc_steps) ** 2 for step in range(num_smc_steps + 1)]
    for previous_beta, beta in zip(betas[:-1], betas[1:]):
        centered_log_likelihood = log_likelihood - log_likelihood.max(dim=1, keepdim=True).values
        log_weights = _normalize(log_weights + (beta - previous_beta) * centered_log_likelihood)
        ess_old = _ess(log_weights=log_weights)
        # Resample only the input rows whose particle weights have low ESS.
        did_resample = ess_old < resample_threshold * num_mc_particles
        if bool(did_resample.any()):
            rows = did_resample.nonzero(as_tuple=True)[0]
            indices = _systematic_indices(log_weights[rows])
            endpoints[rows] = endpoints[rows].gather(1, indices[..., None].expand(-1, -1, endpoints.shape[-1]))
            log_likelihood[rows] = log_likelihood[rows].gather(1, indices)
            log_weights[rows] = -math.log(num_mc_particles)
            resampling_count[rows] += 1
        accepted_stage = torch.zeros_like(accepted_total)

        for _ in range(num_mh_steps):
            # endpoints: (num_samples, num_chains, num_modes)
            proposal_endpoints = prior_model.reverse_simulate(
                start_time=time, 
                samples=samples,
                num_chains=num_mc_particles,
                num_steps=actual_num_steps) 
            # theta: (num_samples, num_chains, resolution, resolution)
            proposal_theta = prior_model.field(
                proposal_endpoints, resolution=log_likelihood_fn.resolution)
            # log_likelihood: (num_samples, num_chains)
            proposal_log_likelihood = log_likelihood_fn(proposal_theta)
            log_accept = (beta * (proposal_log_likelihood - log_likelihood)).clamp_max(0.0)
            uniforms = torch.rand_like(log_likelihood)
            accept = uniforms.clamp_min(torch.finfo(dtype).tiny).log() < log_accept
            endpoints = torch.where(accept[..., None], proposal_endpoints, endpoints)
            log_likelihood = torch.where(accept, proposal_log_likelihood, log_likelihood)
            accepted_stage += accept.sum(dim=1)
            # Retain weights: this MH kernel leaves the current target invariant.

        accepted_total += accepted_stage
        stages += 1

    weights = log_weights.exp()
    denoiser = (weights[..., None] * endpoints).sum(dim=1)
    return denoiser


@torch.no_grad()
def _evaluate_posterior_denoiser_smc_adaptive(
    time: float,
    samples: torch.Tensor,
    prior_model: PriorDiffusionModel,
    log_likelihood_fn: LogLikelihood,
    num_mc_particles: int,
    num_diffusion_steps: int,
    max_smc_steps: int = 16,
    cess_target: float = 0.8,
    resample_threshold: float = 0.5,
    num_mh_steps: int = 2,
) -> torch.Tensor:
    times = torch.tensor([time, prior_model.t_min, prior_model.t_max], dtype=torch.float64)
    transformed = noise_log_ratio(times, rate=prior_model.reference_rate)
    ratio = float((transformed[0] - transformed[1]) / (transformed[2] - transformed[1]))
    actual_num_steps = max(1, math.ceil(num_diffusion_steps * ratio))

    # endpoints: (num_samples, num_chains, num_modes)
    endpoints = prior_model.reverse_simulate(
        start_time=time, 
        samples=samples,
        num_chains=num_mc_particles,
        num_steps=actual_num_steps) 
    # theta: (num_samples, num_chains, resolution, resolution)
    theta = prior_model.field(
        endpoints, resolution=log_likelihood_fn.resolution)
    # log_likelihood: (num_samples, num_chains)
    log_likelihood = log_likelihood_fn(theta)
    log_weights = torch.full_like(log_likelihood, -math.log(num_mc_particles))
    
    num_samples = samples.size(0)
    device = samples.device
    dtype = samples.dtype
    accepted_total = torch.zeros(num_samples, device=device, dtype=dtype)
    resampling_count = torch.zeros(num_samples, device=device, dtype=torch.long)

    beta = 0.0
    stages = 0
    while beta < 1.0 and stages < max_smc_steps:
        new_beta = _next_beta(
            beta=beta, 
            log_weights=log_weights, 
            log_likelihood=log_likelihood,
            cess_fraction=cess_target)
        centered_log_likelihood = log_likelihood - log_likelihood.max(dim=1, keepdim=True).values
        log_weights = _normalize(log_weights + (new_beta - beta) * centered_log_likelihood)
        beta = new_beta
        ess_old = _ess(log_weights=log_weights)
        # Resample only the input rows whose particle weights have low ESS.
        did_resample = ess_old < resample_threshold * num_mc_particles
        if bool(did_resample.any()):
            rows = did_resample.nonzero(as_tuple=True)[0]
            indices = _systematic_indices(log_weights[rows])
            endpoints[rows] = endpoints[rows].gather(1, indices[..., None].expand(-1, -1, endpoints.shape[-1]))
            log_likelihood[rows] = log_likelihood[rows].gather(1, indices)
            log_weights[rows] = -math.log(num_mc_particles)
            resampling_count[rows] += 1
        accepted_stage = torch.zeros_like(accepted_total)

        for _ in range(num_mh_steps):
            # endpoints: (num_samples, num_chains, num_modes)
            proposal_endpoints = prior_model.reverse_simulate(
                start_time=time, 
                samples=samples,
                num_chains=num_mc_particles,
                num_steps=actual_num_steps) 
            # theta: (num_samples, num_chains, resolution, resolution)
            proposal_theta = prior_model.field(
                proposal_endpoints, resolution=log_likelihood_fn.resolution)
            # log_likelihood: (num_samples, num_chains)
            proposal_log_likelihood = log_likelihood_fn(proposal_theta)
            log_accept = (beta * (proposal_log_likelihood - log_likelihood)).clamp_max(0.0)
            uniforms = torch.rand_like(log_likelihood)
            accept = uniforms.clamp_min(torch.finfo(dtype).tiny).log() < log_accept
            endpoints = torch.where(accept[..., None], proposal_endpoints, endpoints)
            log_likelihood = torch.where(accept, proposal_log_likelihood, log_likelihood)
            accepted_stage += accept.sum(dim=1)
            # Retain weights: this MH kernel leaves the current target invariant.

        accepted_total += accepted_stage
        stages += 1

    if beta < 1.0:
        raise RuntimeError(f"SMC stopped at beta={beta:.6g} after {stages} stages.")

    weights = log_weights.exp()
    denoiser = (weights[..., None] * endpoints).sum(dim=1)
    return denoiser


def _normalize(log_weights: torch.Tensor) -> torch.Tensor:
    return log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)


def _ess(log_weights: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.logsumexp(2.0 * log_weights, dim=1))


def _systematic_indices(log_weights: torch.Tensor) -> torch.Tensor:
    batch, particles = log_weights.shape
    cdf = log_weights.exp().cumsum(dim=1)
    cdf[:, -1] = 1.0
    offsets = torch.rand((batch, 1), device=log_weights.device, dtype=log_weights.dtype)
    positions = (torch.arange(particles, device=log_weights.device, dtype=log_weights.dtype)[None, :] + offsets) / particles
    return torch.searchsorted(cdf.contiguous(), positions.contiguous(), right=True).clamp_max(particles - 1)


def _next_beta(
    beta: float, 
    log_weights: torch.Tensor, 
    log_likelihood: torch.Tensor,
    cess_fraction: float,
) -> float:
    centered = log_likelihood - log_likelihood.max(dim=1, keepdim=True).values
    log_threshold = math.log(cess_fraction)

    def acceptable(delta: float) -> bool:
        u = delta * centered
        log_cess = (
            2.0 * torch.logsumexp(log_weights + u, dim=1)
            - torch.logsumexp(log_weights + 2.0 * u, dim=1))
        return bool((log_cess >= log_threshold).all())

    remaining = 1.0 - beta
    if acceptable(remaining):
        return 1.0
    lo, hi = 0.0, remaining
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if acceptable(mid):
            lo = mid
        else:
            hi = mid
    return beta + lo

