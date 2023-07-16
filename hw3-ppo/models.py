from __future__ import annotations
import abc
import math

import numpy as np
import torch


class FeedForward(torch.nn.Module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        layers = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(dim_in, dim_out))
            layers.append(torch.nn.Tanh())
        layers.pop()
        self.nn = torch.nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self) -> None:
        for layer in self.nn.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, math.sqrt(2))
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class DiscreteActor(torch.nn.Module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.nn = FeedForward(dims)

    def init_weights(self) -> None:
        torch.nn.init.orthogonal_(self.nn.nn[-1].weight, 0.01)

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        logits = self.nn(x)
        return torch.distributions.Categorical(logits=logits)


class ContinuousActor(torch.nn.Module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.nn_means = FeedForward(dims)
        self.layer_sigma = torch.nn.Parameter(torch.ones(1, dims[-1]))
        self.init_weights()

    def init_weights(self) -> None:
        torch.nn.init.orthogonal_(self.nn_means.nn[-1].weight, 0.01)
        self.layer_sigma.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = self.nn_means(x)
        mu = torch.tanh(x)
        sigma = self.layer_sigma.expand_as(mu)
        sigma = torch.nn.functional.softplus(sigma) + 1e-5
        distribution = torch.distributions.Normal(loc=mu, scale=sigma)
        distribution = torch.distributions.Independent(distribution, 1)
        # This is a gaussian distribution with diagonal covariance matrix
        # log_prob will give a single value even for whole multidimensional action
        return distribution


class Critic(torch.nn.Module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.nn = FeedForward(dims)
        self.init_weights()

    def init_weights(self) -> None:
        torch.nn.init.orthogonal_(self.nn.nn[-1].weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


class Agent(abc.ABC):
    def select_action(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        "Returns the action and the log probability of the action"
        raise NotImplementedError
    
    def predict_value(self, state: np.ndarray) -> np.ndarray:
        "Returns the value of the state"
        raise NotImplementedError
        