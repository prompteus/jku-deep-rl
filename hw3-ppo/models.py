from __future__ import annotations
import abc

import numpy as np
import torch


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        squeeze_dim: int,
        use_skips: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        
        linear_1 = torch.nn.Linear(dim, squeeze_dim)
        linear_2 = torch.nn.Linear(squeeze_dim, dim)
        with torch.no_grad():
            linear_2.weight.zero_()
            linear_2.bias.zero_()

        self.nn = torch.nn.Sequential(
            linear_1,
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),
            linear_2,
            torch.nn.ELU(),
        )
        self.use_skip = use_skips
        self.layer_norm = torch.nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.nn(x)
        if self.use_skip:
            x += skip
        x = self.layer_norm(x)
        return x


class FeedForward(torch.nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int,
        use_skips: bool,
        squeeze_dim: int = None,
        dropout: float = 0.1,
    ) -> None:
        if squeeze_dim is None:
            squeeze_dim = hidden_dim
        flatten = torch.nn.Flatten()
        linear_in = torch.nn.Linear(input_dim, hidden_dim)
        blocks = [
            ResBlock(hidden_dim, squeeze_dim, use_skips, dropout)
            for _ in range(num_blocks)
        ]
        linear_out = torch.nn.Linear(hidden_dim, output_dim)
        super().__init__(flatten, linear_in, *blocks, linear_out)


class DiscreteActorHead(torch.nn.Module):
    def forward(self, logits: torch.Tensor) -> torch.distributions.Distribution:
        return torch.distributions.Categorical(logits=logits)


class SimpleContinuousActorHead(torch.nn.Module):
    """
    Simplest possible actor head for continuous action space.
    It outputs a gaussian distribution with diagonal covariance matrix.
    Diagonal entries (variances) are learned, but not dependent on the observation.
    """
    def __init__(self, action_dim: int | tuple[int]) -> None:
        super().__init__()
        if isinstance(action_dim, int):
            action_dim = (action_dim,)
        self.layer_sigma = torch.nn.Parameter(torch.ones(1, *action_dim))

    def forward(self, logits_mu: torch.Tensor) -> torch.distributions.Distribution:
        sigma = self.layer_sigma.expand_as(logits_mu)
        sigma = torch.nn.functional.softplus(sigma) + 1e-5
        distribution = torch.distributions.Normal(loc=logits_mu, scale=sigma)
        distribution = torch.distributions.Independent(distribution, 1)
        return distribution


class Agent(abc.ABC):
    def select_action(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        "Returns the action and the log probability of the action"
        raise NotImplementedError
    
    def predict_value(self, state: np.ndarray) -> np.ndarray:
        "Returns the value of the state"
        raise NotImplementedError

