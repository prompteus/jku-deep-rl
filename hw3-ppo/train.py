from __future__ import annotations

import math
import functools
from typing import Any, NamedTuple, Iterator, Iterable, Callable, Optional

import torch
import torch.utils.data
import numpy as np
import gymnasium as gym
import lightning
import lightning.pytorch.loggers
import lightning.pytorch.callbacks
import typer
import torchdata.datapipes as dp
import wandb
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.distributions import Distribution
from lovely_tensors import lovely

from envs import create_env
from models import Critic, DiscreteActor, Agent


class Transition(NamedTuple):
    state: Tensor
    value: Tensor
    action: Tensor
    reward: Tensor
    log_prob: Tensor
    gae_advantage: Tensor
    gae_return: Tensor
    ep_length: Tensor
    ep_return: Tensor


class Buffer:
    def __init__(self, gamma: float, gae_lambda: float) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.truncateds = []
        self.terminateds = []
        self.values = []
        self.ep_lengths = []
        self.ep_returns = []
        self.gae_advantages = None
        self.gae_returns = None

    def append(self, s, a, r, lp, v, ter, trun, ep_l, ep_r) -> None:
        self.states.append(torch.tensor(s))
        self.actions.append(torch.tensor(a))
        self.rewards.append(torch.tensor(r))
        self.log_probs.append(torch.tensor(lp))
        self.values.append(torch.tensor(v))
        self.terminateds.append(torch.tensor(ter))
        self.truncateds.append(torch.tensor(trun))
        self.ep_lengths.append(torch.tensor(ep_l))
        self.ep_returns.append(torch.tensor(ep_r))

    def flush(self) -> Iterator[Transition]:
        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)
        self.rewards = torch.stack(self.rewards)
        self.log_probs = torch.stack(self.log_probs)
        self.values = torch.stack(self.values)
        self.terminateds = torch.stack(self.terminateds)
        self.truncateds = torch.stack(self.truncateds)
        self.ep_lengths = torch.stack(self.ep_lengths)
        self.ep_returns = torch.stack(self.ep_returns)

        self._compute_gae_advantages()

        for t in range(len(self.actions)):
            batch = zip(self.states[t], self.values[t], self.actions[t], self.rewards[t], self.log_probs[t], self.gae_advantages[t], self.gae_returns[t], self.ep_lengths[t], self.ep_returns[t])
            for s, v, a, r, lp, gae_adv, gae_ret, ep_l, ep_r in batch:
                yield Transition(
                    state=s,
                    value=v,
                    action=a,
                    reward=r,
                    log_prob=lp,
                    gae_advantage=gae_adv,
                    gae_return=gae_ret,
                    ep_length=ep_l,
                    ep_return=ep_r,
                )
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.terminateds = []
        self.truncateds = []
        self.ep_lengths = []
        self.ep_returns = []
        self.gae_advantages = None
        self.gae_returns = None

    def finish_batch(self, next_state, next_value, ) -> Iterator[Transition]:
        self.states.append(torch.tensor(next_state))
        self.values.append(torch.tensor(next_value))

    def _compute_gae_advantages(self) -> None:
        dones = self.terminateds | self.truncateds
        td_targets = self.rewards + self.gamma * (~dones) * self.values[1:]
        td_advantage = td_targets - self.values[:-1]
        gae_advantages = torch.zeros_like(self.values)
        for t in reversed(range(len(gae_advantages) - 1)):
            gae_advantages[t] = td_advantage[t] + self.gamma * self.gae_lambda * (~dones[t]) * gae_advantages[t+1]
        self.gae_advantages = gae_advantages[:-1]
        self.gae_returns = self.gae_advantages + self.values[:-1]
        ...


@torch.no_grad()
def global_grad_norm(tensors: Iterable[Tensor]) -> Tensor:
    return torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in tensors if p.grad is not None))


class PPO(lightning.LightningModule, Agent):
    def __init__(
        self,
        architecture_dims: list[int],
        clip_coef: float,
        entropy_loss_coef: float,
        critic_td_loss_coef: float,
        gamma: float,
        critic_loss_fn: str,
        lr_actor: float,
        lr_critic: float,
        normalize_advantage: bool,
        optimizer_config_actor: dict[str, Any] = dict(eps=1e-5),
        optimizer_config_critic: dict[str, Any] = dict(eps=1e-5),
        optimizer_class: str = "Adam",
        entropy_limit_steps: int = -1,
        clip_grad_norm_actor: float = math.inf,
        clip_grad_norm_critic: float = math.inf,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.actor = DiscreteActor(architecture_dims)
        self.critic = Critic(architecture_dims[:-1] + [1])
        self.critic_loss_fn = getattr(torch.nn.functional, critic_loss_fn)

    def forward(self, x: Tensor) -> Tensor:
        return self.actor(x)
    
    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_class)
        optimizer = optimizer_class([
            {"params": self.actor.parameters(), "lr": self.hparams.lr_actor, **self.hparams.optimizer_config_actor},
            {"params": self.critic.parameters(), "lr": self.hparams.lr_critic, **self.hparams.optimizer_config_critic},
        ])
        return optimizer
    
    def _entropy_loss_coef(self) -> float:
        if self.hparams.entropy_limit_steps == -1:
            return self.hparams.entropy_loss_coef
        if self.global_step >= self.hparams.entropy_limit_steps:
            return 0.0
        return self.hparams.entropy_loss_coef

    def critic_loss(self, batch: Transition) -> Tensor:
        curr_value = self.critic(batch.state).flatten()
        critic_td_loss = self.critic_loss_fn(curr_value, batch.gae_return)
        critic_td_loss *= self.hparams.critic_td_loss_coef
        self.log_dict({
            "loss/critic_td_loss": critic_td_loss,
            "loss/critic_ld_loss_coef": self.hparams.critic_td_loss_coef,
        })
        return critic_td_loss

    def actor_loss(self, batch: Transition) -> tuple[Tensor, Tensor]:
        distr: Distribution = self.actor(batch.state)
        log_prob = distr.log_prob(batch.action)
        ratio = (log_prob - batch.log_prob).exp()
        ratio_clipped = ratio.clamp(1-self.hparams.clip_coef, 1+self.hparams.clip_coef)
        advantage = self._normalize_advantage(batch.gae_advantage)
        surr_1 = advantage * ratio
        surr_2 = advantage * ratio_clipped
        surr = torch.min(surr_1, surr_2)
        actor_ppo_loss = -surr.mean()
        entropy = distr.entropy().mean(dim=0)
        self.log_dict({
            "loss/actor_ppo_loss": actor_ppo_loss,
            "importance_ratio_min": ratio.min(),
            "importance_ratio_max": ratio.max(),
            "ratio_outside_bounds": 1 - torch.isclose(ratio, ratio_clipped).float().mean(),
            "clipped_fraction": torch.isclose(surr, surr_2).float().mean(),
            "advantage": advantage.mean(),
        })
        return actor_ppo_loss, entropy

    def entropy_loss(self, entropy: Tensor) -> Tensor:
        entropy_loss_coef = self._entropy_loss_coef()
        entropy_loss = -entropy
        entropy_loss *= entropy_loss_coef
        self.log_dict({
            "actor_entropy": entropy,
            "loss/actor_entropy_loss": entropy_loss,
            "loss/actor_entropy_loss_coef": entropy_loss_coef,
        })
        return entropy_loss

    def training_step(self, batch: Transition, batch_idx: int) -> Tensor:
        critic_td_loss = self.critic_loss(batch)
        actor_ppo_loss, entropy = self.actor_loss(batch)
        entropy_loss = self.entropy_loss(entropy)
        loss = actor_ppo_loss + critic_td_loss + entropy_loss
        self.log("loss/total_loss", loss)
        return loss

    def _normalize_advantage(self, advantage: Tensor) -> Tensor:
        if self.hparams.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            return (advantage - loc) / scale
        return advantage

    def select_action(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = torch.atleast_2d(torch.tensor(state).to(self.device).float())
        distr: Distribution = self.actor(state)
        action = distr.sample()
        log_prob = distr.log_prob(action)
        if isinstance(distr, torch.distributions.Categorical):
            action = action.long()
        else:
            action = action.float()
        return (
            action.detach().cpu().numpy(),
            log_prob.detach().cpu().float().numpy()
        )
    
    def predict_value(self, state: np.ndarray) -> np.ndarray:
        state = torch.atleast_2d(torch.tensor(state).to(self.device).float())
        value = self.critic(state).flatten()
        return value.detach().cpu().float().numpy()
    
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hparams.clip_grad_norm_actor)
        actor_grad_norm_clipped = min(actor_grad_norm, self.hparams.clip_grad_norm_actor)

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.clip_grad_norm_critic)
        critic_grad_norm_clipped = min(critic_grad_norm, self.hparams.clip_grad_norm_critic)

        self.log_dict({
            "grad_norm/actor_unclipped": actor_grad_norm,
            "grad_norm/actor_clipped": actor_grad_norm_clipped,
            "grad_norm/critic_unclipped": critic_grad_norm,
            "grad_norm/critic_clipped": critic_grad_norm_clipped,
        })


class DataCollector:
    def __init__(
        self,
        agent: Agent,
        gamma: float,
        gae_lambda: float,
        buffer_size: int,
        make_env: Callable[[int | None, bool], gym.Env],
        make_env_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        num_parallel_envs: int | None = None,
        seed: int | None = None,
    ) -> None:
        if num_parallel_envs is None:
            if make_env_kwargs is None:
                raise ValueError("Either make_env_kwargs or num_parallel_envs must be set")
            if isinstance(make_env_kwargs, dict):
                raise ValueError("num_parallel_envs must be set if make_env_kwargs is a dict")
        
        if make_env_kwargs is None:
            make_env_kwargs = {}
        if isinstance(make_env_kwargs, dict):
            make_env_kwargs = [make_env_kwargs for _ in range(num_parallel_envs)]
        if num_parallel_envs is None:
            num_parallel_envs = len(make_env_kwargs)
        if len(make_env_kwargs) != num_parallel_envs:
            raise ValueError("make_env_kwargs must have the same length as num_parallel_envs")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.agent = agent
        self.buffer_size = buffer_size
        self.make_env = make_env
        self.make_env_kwargs = make_env_kwargs
        self.num_parallel_envs = num_parallel_envs
        self.seed = seed

    def __iter__(self) -> Iterator[Transition]:
        envs = gym.vector.AsyncVectorEnv([
            functools.partial(self.make_env, **kwargs)
            for kwargs in self.make_env_kwargs
        ])

        buffer = Buffer(self.gamma, self.gae_lambda)
        state, _ = envs.reset()
        num_finished_episodes = 0
        total_env_steps = 0

        while True:
            for _ in range(self.buffer_size // self.num_parallel_envs):
                action, log_prob = self.agent.select_action(state)
                next_state, reward, terminated, truncated, infos = envs.step(action)
                value = self.agent.predict_value(next_state)
                ep_returns = np.full(self.num_parallel_envs, np.nan)
                ep_lengths = np.full(self.num_parallel_envs, np.nan)
                if "final_info" in infos:
                    for ep, info in enumerate(infos["final_info"]):
                        if info is not None:
                            num_finished_episodes += 1
                            wandb.log({
                                "collect_experience/episodic_return": float(info["episode"]["r"]),
                                "collect_experience/episodic_length": int(info["episode"]["l"]),
                                "collect_experience/finished_episodes": num_finished_episodes,
                                "collect_experience/total_env_steps": total_env_steps,
                            })
                            ep_returns[ep] = info["episode"]["r"]
                            ep_lengths[ep] = info["episode"]["l"]

                total_env_steps += self.num_parallel_envs
                buffer.append(s=state, a=action, r=reward, lp=log_prob, ter=terminated, trun=truncated, v=value, ep_l=ep_lengths, ep_r=ep_returns)
                state = next_state

            buffer.finish_batch(next_state, self.agent.predict_value(next_state))
            yield from buffer.flush()


def get_dataloader(
    agent: Agent,
    env_name: str,
    buffer_size: int,
    batch_size: int,
    repeat: int,
    gamma: float,
    gae_lambda: float,
    num_parallel_envs: int,
    env_options = None,
    seed: int | None = None,
) -> torch.utils.data.DataLoader:
    data_collector = DataCollector(agent, gamma, gae_lambda, buffer_size, env_name, env_options, num_parallel_envs, seed)
    pipe: dp.iter.IterDataPipe

    # CRITICAL to have deepcopy=False,
    # otherwise data_collector will be copied and it's agent will be copied too,
    # causing that the agent for collecting data is a different instance than the
    # one that is updated during training
    pipe = dp.iter.IterableWrapper(data_collector, deepcopy=False)
    pipe = pipe.batch(buffer_size)
    if repeat > 1:
        pipe = pipe.repeat(repeat)
    pipe = pipe.in_batch_shuffle()
    pipe = pipe.unbatch()

    loader = torch.utils.data.DataLoader(pipe, batch_size, pin_memory=True, shuffle=False)
    return loader


def main(
    env_name = 'CartPole-v1',
    buffer_size: int = 1024,
    buffer_repeat: int = 10,
    batch_size: int = 64,
    num_parallel_envs: int = 8,
    device: str = "cuda",
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    clip_grad_norm_actor: float = math.inf,
    clip_grad_norm_critic: float = math.inf,
    entropy_limit_steps: int = -1,
    entropy_loss_coef: float = 0.01,
    critic_td_loss_coef: float = 1.0,
    clip_coef: float = 0.2,
    limit_env_steps: int = -1,
    critic_loss_fn: str = "smooth_l1_loss",
    lr_actor: float = 1e-4,
    lr_critic: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantage: bool = False,
    wandb_project_name: str = "jku-deep-rl_ppo",
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_save_code: bool = True,
    seed: int = 0,
) -> None:
    if wandb_group is None:
        wandb_group = env_name
    
    args = locals().copy()

    if device == "cuda":
        import torch.backends.cudnn
        torch.backends.cudnn.deterministic = True
    
    lightning.seed_everything(seed)

    env = create_env(env_name)
    input_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        action_dim = int(np.prod(env.action_space.shape))
    architecture_dims = [input_dim] + [hidden_dim] * hidden_layers + [action_dim]

    algo = PPO(
        architecture_dims,
        clip_coef=clip_coef,
        critic_td_loss_coef=critic_td_loss_coef,
        entropy_loss_coef=entropy_loss_coef,
        gamma=gamma,
        normalize_advantage=normalize_advantage,
        clip_grad_norm_actor=clip_grad_norm_actor,
        clip_grad_norm_critic=clip_grad_norm_critic,
        entropy_limit_steps=entropy_limit_steps,
        critic_loss_fn=critic_loss_fn,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
    )

    wandb_logger = lightning.pytorch.loggers.WandbLogger(
        project=wandb_project_name,
        save_code=wandb_save_code,
        entity=wandb_entity,
        group=wandb_group,
        save_dir="./wandb",
        tags=[env_name, algo.__class__.__name__],
    )
    wandb_logger.experiment.config.update({"args": args})

    def make_env(**kwargs):
        return create_env(env_name, limit_env_steps, options=kwargs)

    loader = get_dataloader(
        algo,
        make_env,
        buffer_size,
        batch_size,
        buffer_repeat,
        gamma,
        gae_lambda,
        num_parallel_envs,
        seed=seed,
    )

    trainer = lightning.Trainer(
        accelerator=device,
        max_epochs=-1,
        precision="16-mixed",
        logger=wandb_logger,
    )

    trainer.fit(algo, loader)
    

if __name__ == "__main__":
    typer.run(main)
