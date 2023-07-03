from __future__ import annotations
import abc

import torch


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        use_skips: bool,
        input_dim=None,
        output_dim=None,
        squeeze_dim=None,
    ) -> None:
        super().__init__()
        if squeeze_dim is None:
            squeeze_dim = hidden_dim

        if input_dim is None or input_dim == hidden_dim:
            self.layer_in = torch.nn.Identity()
        else:
            self.layer_in = torch.nn.Linear(input_dim, hidden_dim)

        if output_dim is None or output_dim == hidden_dim:
            self.layer_out = torch.nn.Identity()
        else:            
            self.layer_out = torch.nn.Linear(hidden_dim, output_dim)

        self.use_skips = use_skips
        self.blocks = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        for _ in range(num_blocks):
            block = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, squeeze_dim),
                torch.nn.ELU(),
                torch.nn.Linear(squeeze_dim, hidden_dim),
                torch.nn.ELU(),
            )
            self.blocks.append(block)
            self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_in(x)
        for block, layer_norm in zip(self.blocks, self.layer_norms):
            if self.use_skips:
                x = block(x) + x
            else:
                x = block(x)
            x = layer_norm(x)
        x = self.layer_out(x)
        return x


class ContinuousActor(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, action_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_mu = torch.nn.LazyLinear(action_dim)
        self.layer_sigma = torch.nn.LazyLinear(action_dim)

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = self.backbone(x)
        mu_logits = self.layer_mu(x)
        sigma_logits = self.layer_sigma(x)
        mu = torch.tanh(mu_logits)
        sigma = torch.nn.functional.softplus(sigma_logits) + 1e-5
        distribution = torch.distributions.Normal(loc=mu, scale=sigma)
        distribution = torch.distributions.Independent(distribution, 1)
        # This is a gaussian distribution with diagonal covariance matrix
        # log_prob will give a single value even for whole multidimensional action
        return distribution


class Critic(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.layer_value = torch.nn.LazyLinear(1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = self.backbone(state)
        value = self.layer_value(state)
        return value


class Agent(abc.ABC):
    def select_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        "Returns the action and the log probability of the action"
        raise NotImplementedError
        