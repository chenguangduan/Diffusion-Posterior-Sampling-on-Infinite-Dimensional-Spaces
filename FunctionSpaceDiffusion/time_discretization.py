from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def noise_log_ratio(t, rate: float = 1.0):
    scaled_time = rate * t
    return 2 * scaled_time + torch.log(-torch.expm1(-2 * scaled_time))


def reverse_time_grid(
    start_time: float, 
    stop_time: float, 
    num_steps: int, 
    dtype: torch.dtype,
    device: torch.device, 
    spacing: str = "log_snr",
    reference_rate: float = 1.0,
) -> torch.Tensor:
    if not math.isfinite(reference_rate) or reference_rate <= 0:
        raise ValueError("reference_rate must be finite and positive")
    if spacing == "log_snr":
        endpoints = torch.tensor([start_time, stop_time], dtype=dtype, device=device)
        # One shared time grid, parameterized by the slowest mode's log-SNR.
        lambdas = noise_log_ratio(endpoints, rate=reference_rate)
        grid = 0.5 * F.softplus(torch.linspace(lambdas[0], lambdas[1], num_steps + 1, dtype=dtype, device=device)) / reference_rate
    elif spacing == "geometric":
        grid = torch.linspace(math.log(start_time), math.log(stop_time), num_steps + 1, dtype=dtype, device=device).exp()
    elif spacing == "linear":
        grid = torch.linspace(start_time, stop_time, num_steps + 1, dtype=dtype, device=device)
    else:
        raise ValueError("spacing must be 'log_snr', 'geometric', or 'linear'")
    grid[0], grid[-1] = start_time, stop_time
    return grid


def heun_reverse_predictor(
    samples: torch.Tensor, 
    denoised: torch.Tensor, 
    current_time: torch.Tensor, 
    next_time: torch.Tensor,
    drift_rates: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    assert current_time.ndim == 0 and next_time.ndim == 0 
    time_step = current_time - next_time
    noise = torch.randn(samples.shape, dtype=samples.dtype, device=samples.device)
    # Whitened OU modes: dz_j = -a_j z_j dt + sqrt(2 a_j) dW_j.
    # These are the Gaussian bridge coefficients given the clean endpoint.
    current_variance = -torch.expm1(-2.0 * drift_rates * current_time)
    next_variance = -torch.expm1(-2.0 * drift_rates * next_time)
    variance_gap = -torch.expm1(-2.0 * drift_rates * time_step)
    a = torch.exp(-drift_rates * time_step) * next_variance / current_variance
    b = torch.exp(-drift_rates * next_time) * variance_gap / current_variance
    q = next_variance * variance_gap / current_variance
    return a * samples + b * denoised + q.clamp_min(0).sqrt() * noise


def heun_reverse_corrector(
    samples: torch.Tensor, 
    denoised: torch.Tensor, 
    next_denoised: torch.Tensor,
    current_time: torch.Tensor, 
    next_time: torch.Tensor,
    drift_rates: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    assert current_time.ndim == 0 and next_time.ndim == 0 
    time_step = current_time - next_time
    current_variance = -torch.expm1(-2.0 * drift_rates * current_time)
    variance_gap = -torch.expm1(-2.0 * drift_rates * time_step)
    b = torch.exp(-drift_rates * next_time) * variance_gap / current_variance
    return samples + 0.5 * b * (next_denoised - denoised)
