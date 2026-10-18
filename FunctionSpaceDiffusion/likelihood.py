import math
import torch 


def _theta2diffusivity(theta: torch.Tensor):
    return theta.exp() + 1.0


class LogLikelihood:
    def __init__(
        self,
        measurement: torch.Tensor,
        sensors: torch.Tensor,
        forcing: float,
        noise_std: float,
        resolution: int,
        batch_size: int
    ) -> None:
        self.measurement = measurement 
        self.sensors = sensors
        self.forcing = forcing 
        self.noise_std = noise_std
        self.resolution = resolution
        self.batch_size = batch_size

    def __call__(
        self, 
        theta: torch.Tensor
    ) -> torch.Tensor:
        # theta: (batch_size, num_chains, resolution, resolution)
        # diffusivity: (batch_size, num_chains, resolution, resolution)
        assert theta.ndim == 4
        diffusivity = _theta2diffusivity(theta)
        solution = _solve_elliptic_minibatch(
            diffusivity=diffusivity,
            forcing=self.forcing,
            batch_size=self.batch_size)
        weights = _local_average_weights(
            resolution=diffusivity.size(-1),
            sensors=self.sensors)
        prediction = _observe(
            solution=solution,
            weights=weights)
        log_likelihood = _gaussian_log_likelihood(
            measurement=self.measurement,
            prediction=prediction,
            noise_std=self.noise_std)
        return log_likelihood


@torch.no_grad()
def generate_measurement(
    theta: torch.Tensor,
    sensors: torch.Tensor,
    forcing: float,
    noise_std: float,
) -> torch.Tensor:
    # theta: (batch_size, resolution, resolution)
    # diffusivity: (batch_size, resolution, resolution)
    assert theta.ndim == 3
    diffusivity = _theta2diffusivity(theta)
    solution = _solve_elliptic(
        diffusivity=diffusivity,
        forcing=forcing)
    weights = _local_average_weights(
        resolution=diffusivity.size(-1),
        sensors=sensors)
    prediction = _observe(
        solution=solution,
        weights=weights)
    noise = noise_std * torch.randn_like(prediction)
    return prediction + noise
    

@torch.no_grad()
def _solve_elliptic_minibatch(
    diffusivity: torch.Tensor, 
    forcing: float, 
    batch_size: int,
) -> torch.Tensor:
    # diffusivity: (batch_size, num_chains, resolution, resolution)
    diffusivity_flat = diffusivity.flatten(0, 1)
    solution_list = []
    for batch in diffusivity_flat.split(batch_size, dim=0):
        solution_list.append(_solve_elliptic(
            diffusivity=batch,
            forcing=forcing))
    solution = torch.cat(solution_list, dim=0)
    return solution.reshape(diffusivity.shape)


@torch.no_grad()
def _local_average_weights(
    resolution: int, 
    sensors: torch.Tensor, 
    width: float = 0.05, 
) -> torch.Tensor:
    # sensors: (num_sensors, 2)
    device = sensors.device
    dtype = sensors.dtype
    grid = torch.linspace(0.0, 1.0, resolution, dtype=dtype, device=device)
    X1, X2 = torch.meshgrid(grid, grid, indexing="ij")
    distance_sq = ((X1[None] - sensors[:, 0, None, None]).square()
                 + (X2[None] - sensors[:, 1, None, None]).square())
    log_density = -0.5 * distance_sq / (width * width)
    log_density -= log_density.amax(dim=(1, 2), keepdim=True)
    one_dimensional_weights = torch.ones(resolution, dtype=dtype, device=device)
    one_dimensional_weights[[0, -1]] = .5
    quadrature = (one_dimensional_weights[:, None] * one_dimensional_weights[None, :]
                  / ((resolution - 1) * (resolution - 1)))
    weights = log_density.exp() * quadrature
    # weights: (num_sensors, resolution, resolution)
    return weights / weights.sum(dim=(1, 2), keepdim=True)


@torch.no_grad()
def _observe(
    solution: torch.Tensor, 
    weights: torch.Tensor
) -> torch.Tensor:
    # solution: (..., resolution, resolution)
    # weights: (num_centers, resolution, resolution)
    assert solution.ndim == 3 or solution.ndim == 4
    return torch.einsum("mij,...ij->...m", weights, solution)


@torch.no_grad()
def _gaussian_log_likelihood(
    measurement: torch.Tensor, 
    prediction: torch.Tensor, 
    noise_std: float
) -> torch.Tensor:
    # measurement: (num_centers, )
    # prediction: (..., num_centers)
    residual = measurement - prediction
    standardized = residual / noise_std
    result = -0.5 * standardized.square().sum(dim=-1)
    return result


# --------------- elliptic solver ---------------
@torch.no_grad()
def _solve_elliptic(
    diffusivity: torch.Tensor, 
    forcing: float, 
) -> torch.Tensor:
    # diffusivity: (batch_size, resolution, resolution)
    device = diffusivity.device
    dtype = diffusivity.dtype
    resolution = diffusivity.shape[-1]
    batch = diffusivity.shape[:-2]

    # discretization parameters 
    h = 1.0 / (resolution - 1)
    num_nodes = resolution * resolution
    num_elements = (resolution - 1) * (resolution - 1)

    # Nodes of every element
    i = torch.arange(resolution - 1, device=device)[:, None]
    j = torch.arange(resolution - 1, device=device)[None, :]
    n0 = i * resolution + j
    # nodes: (num_elements, 4)
    nodes = torch.stack([
        n0,               # bottom-left
        n0 + 1,           # bottom-right
        n0 + resolution + 1,       # top-right
        n0 + resolution,           # top-left
    ], dim=-1).reshape(num_elements, 4)             

    # Element coefficient: average of four nodal values
    diffusivity_element_values = 0.25 * (
        diffusivity[:, :-1, :-1]
        + diffusivity[:, :-1, 1:]
        + diffusivity[:, 1:, 1:]
        + diffusivity[:, 1:, :-1]).reshape(batch + (num_elements,))    

    # Local stiffness matrix
    K0 = diffusivity.new_tensor([
        [ 4, -1, -2, -1],
        [-1,  4, -1, -2],
        [-2, -1,  4, -1],
        [-1, -2, -1,  4],
    ]) / 6.0
    # Ke: (batch_size, num_elements, 4, 4)
    Ke = diffusivity_element_values[:, :, None, None] * K0  

    # Vectorized global stiffness assembly
    row = nodes[:, :, None].expand(num_elements, 4, 4)
    col = nodes[:, None, :].expand(num_elements, 4, 4)
    # Flatten an entry of the num_nodes × num_nodes matrix.
    K_indices = (row * num_nodes + col).reshape(-1)
    stiffness = diffusivity.new_zeros(batch + (num_nodes * num_nodes,))
    stiffness.index_add_(-1, K_indices, Ke.reshape(batch + (16 * num_elements,)))
    stiffness = stiffness.reshape(batch + (num_nodes, num_nodes))    

    # Vectorized load-vector assembly
    # f: (:, num_elements)
    forcing_element_values = forcing * torch.ones_like(diffusivity_element_values)
    # local_load: (:, num_elements, 4)
    local_load = 0.25 * (forcing_element_values * h * h)[:, :, None].expand(batch + (num_elements, 4))
    load = diffusivity.new_zeros(batch + (num_nodes,))
    load.index_add_(-1, nodes.reshape(-1), local_load.reshape(batch + (-1,)))

    # Homogeneous Dirichlet boundary condition
    i = torch.arange(resolution, device=device)[:, None]
    j = torch.arange(resolution, device=device)[None, :]
    boundary = (
        (i == 0) | (i == resolution - 1) |
        (j == 0) | (j == resolution - 1)
    ).reshape(-1)
    interior = ~boundary
    # Keep only interior-interior entries
    mask = (interior[:, None] & interior[None, :]).to(dtype)
    stiffness = stiffness * mask
    # Put identity on boundary rows
    stiffness = stiffness + torch.diag(boundary.to(dtype))
    # Boundary RHS = 0
    load = load * interior.to(dtype)

    # Batched solve
    u = torch.linalg.solve(stiffness, load.unsqueeze(-1)).squeeze(-1) 
    return u.reshape(diffusivity.shape)

