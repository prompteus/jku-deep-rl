from __future__ import annotations

import math
import dataclasses
from typing import Any, NamedTuple, Iterator, Iterable
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
import torch.utils.data
import numpy as np
import gymnasium as gym
import lightning
import lightning.pytorch.loggers
import lightning.pytorch.callbacks
import typer
import torchdata.datapipes as dp
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.distributions import Distribution
from lovely_tensors import lovely

from envs import create_env
from models import FeedForward, ContinuousActor, Critic, Agent


class Transition(NamedTuple):
    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor
    terminated: Tensor
    truncated: Tensor
    log_prob: Tensor
    future_return: Tensor
    episodic_return: Tensor
    episodic_length: Tensor


@dataclasses.dataclass
class Scaler:
    negative_scale: float = 1.0
    positive_scale: float = 1.0
    min_clip: float = None
    max_clip: float = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.negative_scale != 1.0:
            x = torch.where(x < 0, x * self.negative_scale, x)
        if self.positive_scale != 1.0:
            x = torch.where(x > 0, x * self.positive_scale, x)
        if self.min_clip is not None:
            x = torch.where(x < self.min_clip, self.min_clip, x)
        if self.max_clip is not None:
            x = torch.where(x > self.max_clip, self.max_clip, x)
        return x

@torch.no_grad()
def global_grad_norm(tensors: Iterable[Tensor]) -> Tensor:
    return torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in tensors if p.grad is not None))


class PPO(lightning.LightningModule, Agent):
    def __init__(
        self,
        backbone_config: dict[str, Any],
        action_dim: int,
        optimizer_config_actor: dict[str, Any] = dict(lr=1e-4),
        optimizer_config_critic: dict[str, Any] = dict(lr=1e-4),
        backbone_is_shared: bool = False,
        optimizer_class: str = "Adam",
        critic_loss_fn: str = "smooth_l1_loss",
        epsilon: float = 0.1,
        discount_factor: float = 0.99,
        normalize_advantage: bool = False,
        entropy_coeff: float = 0.0,
        critic_coeff: float = 0.5,
        entropy_limit_steps: int = None,
        reward_scaler_config: dict[str, Any] = None,
        clip_grad_norm_actor: float = None,
        clip_grad_norm_critic: float = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if backbone_is_shared:
            backbone = FeedForward(**backbone_config)
            self.actor = ContinuousActor(backbone, action_dim)
            self.critic = Critic(backbone)
        else:
            self.actor = ContinuousActor(FeedForward(**backbone_config), action_dim)
            self.critic = Critic(FeedForward(**backbone_config))
        
        self.critic_loss_fn = getattr(torch.nn.functional, critic_loss_fn)
        self.log_ratio_bounds: torch.Tensor
        self.register_buffer("log_ratio_bounds", torch.log1p(torch.tensor([-epsilon, epsilon])))
        assert torch.allclose(
            self.log_ratio_bounds.exp(),
            torch.tensor([1-epsilon, 1+epsilon]),
        )

        if reward_scaler_config is None:
            self.reward_scaler = torch.nn.Identity()
        else:
            self.reward_scaler = Scaler(**reward_scaler_config)

    def forward(self, x: Tensor) -> Tensor:
        return self.actor(x)
    
    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_class)
        optimizer = optimizer_class([
            {"params": self.actor.parameters(), **self.hparams.optimizer_config_actor},
            {"params": self.critic.parameters(), **self.hparams.optimizer_config_critic},
        ])
        return optimizer
    
    def _entropy_coeff(self) -> float:
        if self.hparams.entropy_limit_steps is None:
            return self.hparams.entropy_coeff
        if self.global_step >= self.hparams.entropy_limit_steps:
            return 0.0
        return self.hparams.entropy_coeff

    def critic_loss(self, batch: Transition) -> tuple[Tensor, Tensor]:
        curr_value = self.critic(batch.state).flatten()
        with torch.no_grad():
            reward = self.reward_scaler(batch.reward)
            next_value = self.critic(batch.next_state).flatten().detach()
            td_target = (reward + (~batch.terminated) * self.hparams.discount_factor * next_value).detach()
            td_advantage = (td_target - curr_value).detach()
            if self.hparams.normalize_advantage:
                td_advantage = self._normalize_advantage(td_advantage)
        critic_td_loss = self.critic_loss_fn(curr_value, td_target.detach())
        critic_td_loss = self.hparams.critic_coeff * critic_td_loss
        return critic_td_loss, td_advantage

    def actor_loss(self, batch: Transition, td_advantage: Tensor) -> tuple[Tensor, Tensor]:
        distribution: Distribution = self.actor(batch.state)
        log_prob = distribution.log_prob(batch.action)
        log_ratio = log_prob - batch.log_prob
        ratio = log_ratio.exp()
        ratio_clipped = log_ratio.clamp(*self.log_ratio_bounds).exp()
        surrogate = torch.min(ratio * td_advantage, ratio_clipped * td_advantage)
        actor_ppo_loss = -torch.mean(surrogate)
        entropy = distribution.entropy().mean(dim=0)
        self.log_dict({
            "importance_ratio_min": ratio.min(),
            "importance_ratio_max": ratio.max(),
            "was_clipped": torch.isclose(ratio, ratio_clipped).float().mean(),
        })
        return actor_ppo_loss, entropy

    def entropy_loss(self, entropy: Tensor) -> Tensor:
        entropy_coeff = self._entropy_coeff()
        entropy_loss = - entropy_coeff * entropy
        return entropy_loss, entropy_coeff

    def training_step(self, batch: Transition, batch_idx: int) -> Tensor:
        critic_td_loss, td_advantage = self.critic_loss(batch)
        actor_ppo_loss, entropy = self.actor_loss(batch, td_advantage)
        entropy_loss, entropy_coeff = self.entropy_loss(entropy)
        loss = actor_ppo_loss + critic_td_loss + entropy_loss
        self.log_dict({
            "actor_ppo_loss": actor_ppo_loss,
            "critic_td_loss": critic_td_loss,
            "actor_entropy_loss": entropy_loss,
            "actor_entropy_coeff": entropy_coeff,
            "actor_entropy": entropy,
            "critic_loss_coeff": self.hparams.critic_coeff,
            "td_advantage": td_advantage.mean(),
            "total_loss": loss,
        })
        return loss

    def _normalize_advantage(self, td_advantage: Tensor) -> Tensor:
        if self.hparams.normalize_advantage and td_advantage.numel() > 1:
            with torch.no_grad():
                loc = td_advantage.mean().item()
                scale = td_advantage.std().clamp_min(1e-6).item()
                return (td_advantage - loc) / scale
        return td_advantage

    def select_action(self, state: Tensor) -> tuple[Tensor, Tensor]:
        distribution: Distribution = self.actor(state.to(self.device))
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob.detach()
    
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # torch clip_grad_norm_ returns norm before clipping
        if self.hparams.clip_grad_norm_actor is not None:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hparams.clip_grad_norm_actor)
            actor_grad_norm_clipped = min(actor_grad_norm, self.hparams.clip_grad_norm_actor)
        else:
            actor_grad_norm = global_grad_norm(self.actor.parameters())
            actor_grad_norm_clipped = actor_grad_norm

        if self.hparams.clip_grad_norm_critic is not None:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.clip_grad_norm_critic)
            critic_grad_norm_clipped = min(critic_grad_norm, self.hparams.clip_grad_norm_critic)
        else:
            critic_grad_norm = global_grad_norm(self.critic.parameters())
            critic_grad_norm_clipped = critic_grad_norm

        self.log_dict({
            "actor_grads/norm_unclipped": actor_grad_norm,
            "actor_grads/norm_clipped": actor_grad_norm_clipped,
            "critic_grads/norm_unclipped": critic_grad_norm,
            "critic_grads/norm_clipped": critic_grad_norm_clipped,
        })


class MonteCarloReinforce(lightning.LightningModule, Agent):
    def __init__(self, backbone_config: dict[str, Any], action_dim: int, optim_config: dict[str, Any], return_scaler_config: dict[str, Any], normalize_returns: bool,) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.actor = ContinuousActor(FeedForward(**backbone_config), action_dim)
        # if use_baseline:
        #     self.baseline = Critic(FeedForward(**backbone_config))
        # else:
        #     self.baseline = None
        if return_scaler_config is None:
            self.return_scaler = torch.nn.Identity()
        else:
            self.return_scaler = Scaler(**return_scaler_config)

    def forward(self, x: Tensor) -> Tensor:
        return self.actor(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.actor.parameters(), **self.hparams.optim_config)
        return optimizer
    
    def training_step(self, batch: Transition, batch_idx: int) -> Tensor:
        distribution: Distribution = self.actor(batch.state)
        returns = self.return_scaler(batch.future_return)
        log_prob = distribution.log_prob(batch.action)
        if self.hparams.normalize_returns:
            returns = (returns - returns.mean()) / returns.std()
        actor_loss = -(log_prob * returns).mean()
        self.log("actor_loss", actor_loss)
        return actor_loss 
    
    def select_action(self, state: Tensor) -> tuple[Tensor, Tensor]:
        distribution: Distribution = self.actor(state.to(self.device))
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob.detach()


class DataCollection:
    def __init__(self, agent: Agent, env_name: str, compute_returns: bool, gamma=None, env_options=None) -> None:
        self.agent = agent
        self.env_name = env_name
        self.env_options = env_options
        self.track_future_returns = compute_returns
        self.gamma = gamma

    def __iter__(self) -> Iterator[Transition]:
        env, _ = create_env(self.env_name, self.env_options)
        state, _ = env.reset()
        transitions: list[Transition] = []
        state = torch.from_numpy(state).float().unsqueeze(0)

        while True:
            with torch.no_grad():
                action, log_prob = self.agent.select_action(state)
            action = action.cpu().squeeze(0).numpy()
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)

            if "episode" in info:
                episodic_return = float(info["episode"]["r"])
                episodic_length = int(info["episode"]["l"])
            else:
                episodic_return = math.nan
                episodic_length = math.nan

            transition = Transition(
                state=state,
                action=torch.from_numpy(action),
                reward=float(reward),
                next_state=next_state,
                log_prob=log_prob.cpu(),
                terminated=bool(terminated),
                truncated=bool(truncated),
                future_return=math.nan,
                episodic_length=episodic_length,
                episodic_return=episodic_return,
            )

            if self.track_future_returns:
                transitions.append(transition)
            else:
                yield transition

            state = next_state

            if terminated or truncated:
                if self.track_future_returns:
                    self._compute_future_returns(transitions)
                    yield from transitions
                    transitions.clear()

                state, _ = env.reset()
                state = torch.from_numpy(state).float().unsqueeze(0)

    def _compute_future_returns(self, transitions: list[Transition]) -> None:
        returns = 0.0
        for transition in reversed(transitions):
            returns = transition.reward + self.gamma * returns
            transition.future_return.fill_(returns)


class DataTracker(lightning.Callback):
    def __init__(self) -> None:
        self.finished_episodes = 0
        self.episodic_returns = []
        self.episodic_lengths = []

    def __call__(self, transition: Transition) -> Any:
        if not math.isnan(transition.episodic_return):
            self.finished_episodes += 1
            self.episodic_returns.append(transition.episodic_return)
            self.episodic_lengths.append(transition.episodic_length)
        return transition
        
    def on_train_batch_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        if batch_idx % trainer.log_every_n_steps == 0 and len(self.episodic_returns) > 0:
            pl_module.log_dict({
                "collect_experience/episodic_return": np.mean(self.episodic_returns),
                "collect_experience/episodic_length": np.mean(self.episodic_lengths),
                "collect_experience/num_finished_episodes_so_far": float(self.finished_episodes),
            })
            self.episodic_returns.clear()
            self.episodic_lengths.clear()


def get_dataloader(
    agent: Agent,
    env_name: str,
    batch_size: int,
    buffer_size: int,
    buffer_repeat: int | None,
    compute_returns: bool,
    gamma: float = None,
    env_options = None,
) -> tuple[torch.utils.data.DataLoader, DataTracker]:
    data_collection = DataCollection(agent, env_name, compute_returns, gamma, env_options)
    data_tracker = DataTracker()
    pipe: dp.iter.IterDataPipe
    pipe = dp.iter.IterableWrapper(data_collection)
    pipe = pipe.map(data_tracker)
    if buffer_repeat is not None and buffer_repeat > 1:
        pipe = pipe.repeat(buffer_repeat)
    pipe = pipe.shuffle(buffer_size=buffer_size)
    loader = torch.utils.data.DataLoader(
        pipe,
        batch_size,
        pin_memory=True,
        shuffle=False
    )
    return loader, data_tracker


def main(
    env_name = 'pendulum',
    batch_size: int = 256,
    buffer_size: int = 1024,
    buffer_repeat: int = 10,
    device: str = "cuda",
    backbone_config = None,
) -> None:
    if backbone_config is None:
        backbone_config = dict(
            hidden_dim=64,
            num_blocks=1,
            use_skips=False,
        )

    env, name = create_env(env_name)
    input_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    backbone_config["input_dim"] = input_dim

    algo = PPO(
        backbone_config,
        action_dim,
        normalize_advantage=False,
        epsilon=0.2,
        clip_grad_norm_actor=0.5,
        clip_grad_norm_critic=0.5,
        backbone_is_shared=False,
    )

    wandb_logger = lightning.pytorch.loggers.WandbLogger(
        project="jku-deep-rl_ppo",
        group=name,
        save_dir="./wandb",
        tags=[env_name, algo.__class__.__name__]
    )
    wandb_logger.experiment.config.update({
        "algo": algo.__class__.__name__,
        "replay_buffer_size": buffer_size,
        "batch_size": batch_size,
    })

    loader, tracker = get_dataloader(
        algo,
        env_name,
        batch_size,
        buffer_size,
        buffer_repeat,
        compute_returns=False,
        gamma=0.99,
    )

    trainer = lightning.Trainer(
        accelerator=device,
        max_epochs=-1,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[tracker],
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.5,
    )
    trainer.fit(algo, loader)
    


if __name__ == "__main__":
    typer.run(main)    
