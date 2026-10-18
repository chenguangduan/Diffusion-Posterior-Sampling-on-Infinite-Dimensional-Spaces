import os
from typing import Tuple
import argparse
import math
import numpy as np
import torch

from FunctionSpaceDiffusion.nn import TimeConditionedMLP
from FunctionSpaceDiffusion.prior_diffusion import PriorDiffusionModel, field
from FunctionSpaceDiffusion.likelihood import LogLikelihood, generate_measurement
from FunctionSpaceDiffusion.posterior_diffusion import PosteriorDiffusionModel

from plot import plot_sample


@torch.no_grad()
def generate_prior_samples(
    num_modes: int, 
    cm_radius: float, 
    num_samples: int, 
    dtype: torch.dtype, 
    device: torch.device
) -> torch.Tensor:
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    j = torch.arange(1, num_modes + 1, dtype=dtype, device=device)
    b = math.sqrt(6.0) / (math.pi * j)
    tau = 2.0 * torch.rand((num_samples, num_modes), dtype=dtype, device=device) - 1.0
    return cm_radius * b * tau


@torch.no_grad()
def generate_reference(
    num_modes: int,
    resolution: int, 
    cm_radius: float, 
    alpha: float, 
    sensors: torch.Tensor,
    forcing: float,
    noise_std: float,
    dtype: torch.dtype, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    reference_sample = generate_prior_samples(
        num_modes=num_modes, 
        cm_radius=cm_radius, 
        num_samples=1, 
        dtype=dtype, 
        device=device)
    reference_theta = field(
        num_modes=num_modes,
        alpha=alpha,
        samples=reference_sample,
        resolution=resolution,
        dtype=dtype,
        device=device)
    measurement = generate_measurement(
        theta=reference_theta,
        sensors=sensors,
        forcing=forcing, 
        noise_std=noise_std)
    return reference_sample[0], measurement[0]


@torch.no_grad()
def posterior_importance_sampling(
    prior_samples: torch.Tensor,
    log_likelihood: LogLikelihood,
    resolution: int,
    batch_size: int,
) -> torch.Tensor:
    num_samples = len(prior_samples)
    # Compute all prior fields once.
    prior_theta = field(
        num_modes=prior_samples.shape[-1],
        alpha=alpha,
        samples=prior_samples,
        resolution=resolution,
        dtype=prior_samples.dtype,
        device=prior_samples.device)
    # Compute likelihoods in minibatches.
    log_weights = prior_samples.new_empty(num_samples)
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        print(f"Importance sampling: {start + 1} to {stop} of {num_samples} samples")
        log_weights[start:stop] = log_likelihood(theta=prior_theta[start:stop].unsqueeze(0))[0]
    # Normalize globally across all samples.
    weights = log_weights.softmax(dim=0)
    # Weighted mean without creating another full weighted-field tensor.
    return torch.einsum("n,ni->i", weights, prior_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diffusion posterior sampling.")
    parser.add_argument("--mode", choices=["train", "sample", "plot"], required=True)
    parser.add_argument("--gamma", type=float, default=-1.0)
    args = parser.parse_args()
    mode = args.mode
    gamma = None if args.gamma < 0.0 else args.gamma

    seed = 1234
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float64

    # reference setting
    reference_num_modes = 32
    reference_resolution = 17
    cm_radius = 5.0
    alpha = 2.0
    sensors_per_axis = 64
    sensor_axis = torch.linspace(0.1, 0.9, sensors_per_axis, dtype=dtype, device=device)
    sensors = torch.cartesian_prod(sensor_axis, sensor_axis)
    forcing = 1.0
    noise_std = 1.0e-5

    # hyperparameters
    num_modes = 16
    t_max = 2.0
    t_min_prior = 1.0e-6
    t_min_posterior = 1.0e-4
    resolution = 17
    num_steps = 64 # 32
    num_mc_particles = 10000 # 50000
    num_denoiser_steps = 64 # 32
    num_samples_inference = 64
    batch_size_inference = 8

    hidden_dim = 128
    num_samples_train = 10000
    num_epoches = 50000
    batch_size = 2048
    learning_rate = 1.0e-3
    batch_size_likelihood = 128

    # reference 
    reference_sample, measurement = generate_reference(
        num_modes=reference_num_modes,
        resolution=reference_resolution,
        cm_radius=cm_radius,
        alpha=alpha,
        sensors=sensors,
        forcing=forcing,
        noise_std=noise_std,
        dtype=dtype,
        device=device)

    # likelihood
    log_likelihood = LogLikelihood(
        measurement=measurement,
        sensors=sensors,
        forcing=forcing,
        noise_std=noise_std,
        resolution=resolution,
        batch_size=batch_size_likelihood)

    denoiser = TimeConditionedMLP(
        num_modes=num_modes, 
        hidden_dim=hidden_dim
    ).to(device=device)
    prior_model = PriorDiffusionModel(
        denoiser=denoiser,
        num_modes=num_modes,
        cm_radius=cm_radius,
        alpha=alpha, 
        t_min=t_min_prior,
        t_max=t_max,
        dtype=dtype,
        device=device,
        gamma=gamma)

    folder = "./checkpoints"
    os.makedirs(folder, exist_ok=True)
    checkpoint_name = "denoiser.pt" if gamma is None else f"denoiser_gamma_{gamma:.17g}.pt"
    path = os.path.join(folder, checkpoint_name)
    diffusion_sample_name = "diffusion_sample.txt" if gamma is None else f"diffusion_sample_{gamma:.17g}.txt"

    # prior diffusion model 
    if mode == "train":
        denoiser.train()
        prior_samples = generate_prior_samples(
            num_modes=num_modes, 
            cm_radius=cm_radius, 
            num_samples=num_samples_train, 
            dtype=dtype, 
            device=device)
        prior_model.train(
            prior_samples=prior_samples,
            num_epoches=num_epoches,
            batch_size=batch_size,
            learning_rate=learning_rate)
        torch.save(denoiser.state_dict(), path)
    elif mode == "sample":
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run the example with --mode train first.")
        denoiser.load_state_dict(torch.load(path, map_location=device))
        denoiser.eval()
        with torch.no_grad():
            # diffusion posterior sampling
            posterior = PosteriorDiffusionModel(
                prior_model=prior_model,
                log_likelihood=log_likelihood,
                t_min=t_min_posterior,
                t_max=t_max,
                num_mc_particles=num_mc_particles,
                batch_size_inference=batch_size_inference,
                num_denoiser_steps=num_denoiser_steps)
            estimated_samples = posterior.reverse_simulate(
                num_samples=num_samples_inference,
                num_steps=num_steps)
        np.savetxt(diffusion_sample_name, estimated_samples.detach().cpu().numpy(), fmt="%.18e")
    elif mode == "plot":
        plot_resolution = 129
        reference_theta = field(
                num_modes=reference_num_modes,
                alpha=alpha,
                samples=reference_sample[None, :].detach(),
                resolution=plot_resolution,
                dtype=dtype,
                device=device)[0].cpu()
        # diffusion with gamma=None 
        diffusion_sample = torch.from_numpy(np.loadtxt("diffusion_sample.txt")).to(device)
        diffusion_sample_mean = diffusion_sample.mean(dim=0, keepdim=True)
        diffusion_theta_mean = field(
                num_modes=num_modes,
                alpha=alpha,
                samples=diffusion_sample_mean.detach(),
                resolution=plot_resolution,
                dtype=dtype,
                device=device)[0].cpu()
        # diffusion with gamma is not None 
        diffusion_sample_gamma = torch.from_numpy(np.loadtxt(diffusion_sample_name)).to(device)
        diffusion_sample_gamma_mean = diffusion_sample_gamma.mean(dim=0, keepdim=True)
        diffusion_theta_gamma_mean = field(
                num_modes=num_modes,
                alpha=alpha,
                samples=diffusion_sample_gamma_mean.detach(),
                resolution=plot_resolution,
                dtype=dtype,
                device=device)[0].cpu()
        # importance sampling 
        prior_samples = generate_prior_samples(
            num_modes=num_modes, 
            cm_radius=cm_radius, 
            num_samples=num_samples_train, 
            dtype=dtype, 
            device=device)
        batch_size_is = 10000
        is_sample_mean = posterior_importance_sampling(
            prior_samples=prior_samples, 
            log_likelihood=log_likelihood,
            resolution=resolution,
            batch_size=batch_size_is)
        is_theta_mean = field(
                num_modes=num_modes,
                alpha=alpha,
                samples=is_sample_mean[None, :].detach(),
                resolution=plot_resolution,
                dtype=dtype,
                device=device)[0].cpu()
        # plot 
        plot_sample(
            reference_theta=reference_theta,
            diffusion_theta_mean=diffusion_theta_mean,
            diffusion_theta_gamma_mean=diffusion_theta_gamma_mean,
            is_theta_mean=is_theta_mean,
            filename=f"result_{gamma:.17g}.pdf")
    else:
        ValueError(f"Invalid mode: {mode}.")
    
