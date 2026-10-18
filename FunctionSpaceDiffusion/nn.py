from typing import cast
import torch


class TimeConditionedMLP(torch.nn.Module):
    def __init__(self, num_modes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.register_buffer("time_frequencies", torch.tensor([1.0, 2.0, 4.0, 8.0]))
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_modes + 9, hidden_dim), torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, num_modes))
        torch.nn.init.zeros_(cast(torch.Tensor, self.layers[-1].weight))
        torch.nn.init.zeros_(cast(torch.Tensor, self.layers[-1].bias))

    def forward(self, samples, time):
        assert samples.ndim == 2 and samples.size(-1) == self.num_modes
        assert time.ndim == 1 and time.size(0) == samples.size(0)
        log_time = time.log()[:, None]
        phase = log_time * self.time_frequencies
        features = torch.cat((samples, log_time, phase.sin(), phase.cos()), dim=-1)
        return samples + self.layers(features)

