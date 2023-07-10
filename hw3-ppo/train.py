from __future__ import annotations

import math
import functools
from typing import Any, NamedTuple, Iterator, Iterable, Callable

import torch
import torch.utils.data
import numpy as np
import gymnasium as gym
import lightning
import lightning.pytorch.loggers
import lightning.pytorch.callbacks
import typer
import torchdata.datapipes as dp
import torchviz
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.distributions import Distribution
from lovely_tensors import lovely

from envs import create_env
from models import Critic, ContinuousActor, DiscreteActor, Agent


class Transition(NamedTuple):
    state: Tensor
    value: Tensor
    action: Tensor
    reward: Tensor
    log_prob: Tensor
    gae_advantage: Tensor
    gae_return: Tensor


class Episode:
    def __init__(self, gamma: float, gae_lambda: float) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.states = []
        self.values = None
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.ep_return = None
        self.gae_advantages = []
        self.truncated = None

    def append(self, s, a, r, lp) -> None:
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.log_probs.append(lp)

    def finish(self, agent: Agent, final_state, ep_return: float, truncated: bool) -> None:
        self.states.append(final_state)
        self.values = agent.predict_value(np.array(self.states))
        if not truncated:
            self.values[-1] = 0.0
        self.ep_return = ep_return
        self.truncated = truncated
        self._compute_gae_advantages()

    def _compute_gae_advantages(self) -> None:
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        td_targets = rewards + self.gamma * values[1:]
        td_advantages = td_targets - values[:-1]
        gae_advantages = [0.0] * len(values)
        for t in reversed(range(len(gae_advantages) - 1)):
            gae_advantages[t] = td_advantages[t] + self.gamma * self.gae_lambda * gae_advantages[t + 1]
        self.gae_advantages = gae_advantages[:-1]

    def __len__(self) -> int:
        return len(self.actions)
    
    def to_transitions(self) -> Iterator[Transition]:
        if self.ep_return is None:
            raise ValueError("Episode not finished")
        
        for i in range(len(self)):
            yield Transition(
                state=self.states[i],
                value=self.values[i],
                action=self.actions[i],
                reward=self.rewards[i],
                log_prob=self.log_probs[i],
                gae_advantage=self.gae_advantages[i],
                gae_return=self.gae_advantages[i] + self.values[i],
            )


@torch.no_grad()
def global_grad_norm(tensors: Iterable[Tensor]) -> Tensor:
    return torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in tensors if p.grad is not None))


class PPO(lightning.LightningModule, Agent):
    def __init__(
        self,
        architecture_dims: list[int],
        epsilon,
        entropy_coeff: float,
        critic_coeff: float,
        gamma: float,
        critic_loss_fn: str,
        lr_actor: float,
        lr_critic: float,
        #use_new_value_estimate: bool,
        optimizer_config_actor: dict[str, Any] = dict(eps=1e-5),
        optimizer_config_critic: dict[str, Any] = dict(eps=1e-5),
        optimizer_class: str = "Adam",
        normalize_advantage: bool = False,
        entropy_limit_steps: int = -1,
        clip_grad_norm_actor: float = None,
        clip_grad_norm_critic: float = None,
        clip_grad_norm_total: float = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.actor = DiscreteActor(architecture_dims)
        self.critic = Critic(architecture_dims[:-1] + [1])
        self.critic_loss_fn = getattr(torch.nn.functional, critic_loss_fn)
        self.log_ratio_bounds: torch.Tensor
        self.register_buffer("log_ratio_bounds", torch.log1p(torch.tensor([-epsilon, epsilon])))
        if clip_grad_norm_total is not None and (clip_grad_norm_critic is not None or clip_grad_norm_actor is not None):
            raise ValueError("Cannot specify both clip_grad_norm_total and clip_grad_norm_critic/actor")

    def forward(self, x: Tensor) -> Tensor:
        return self.actor(x)
    
    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_class)
        optimizer = optimizer_class([
            {"params": self.actor.parameters(), "lr": self.hparams.lr_actor, **self.hparams.optimizer_config_actor},
            {"params": self.critic.parameters(), "lr": self.hparams.lr_critic, **self.hparams.optimizer_config_critic},
        ])
        return optimizer
    
    def _entropy_coeff(self) -> float:
        if self.hparams.entropy_limit_steps == -1:
            return self.hparams.entropy_coeff
        if self.global_step >= self.hparams.entropy_limit_steps:
            return 0.0
        return self.hparams.entropy_coeff

    def critic_loss(self, batch: Transition) -> tuple[Tensor, Tensor]:
        curr_value = self.critic(batch.state).flatten()
        critic_td_loss = self.critic_loss_fn(curr_value, batch.gae_return)
        critic_td_loss *= self.hparams.critic_coeff
        return critic_td_loss

    def actor_loss(self, batch: Transition) -> tuple[Tensor, Tensor]:
        distr: Distribution = self.actor(batch.state)
        log_prob = distr.log_prob(batch.action)
        ratio = (log_prob - batch.log_prob).exp()
        ratio_clipped = ratio.clamp(1-self.hparams.epsilon, 1+self.hparams.epsilon)
        #assert (ratio_clipped >= 1 - self.hparams.epsilon - 0.0001).all()
        #assert (ratio_clipped <= 1 + self.hparams.epsilon + 0.0001).all()
        advantage = self._normalize_advantage(batch.gae_advantage)
        surr_1 = advantage * ratio
        surr_2 = advantage * ratio_clipped
        surr = torch.min(surr_1, surr_2)
        actor_ppo_loss = -surr.mean()
        entropy = distr.entropy().mean(dim=0)
        self.log_dict({
            "importance_ratio_min": ratio.min(),
            "importance_ratio_max": ratio.max(),
            "ratio_outside_bounds": 1 - torch.isclose(ratio, ratio_clipped).float().mean(),
            "clipped_fraction": torch.isclose(surr, surr_2).float().mean()
        })
        return actor_ppo_loss, entropy

    def entropy_loss(self, entropy: Tensor) -> Tensor:
        entropy_coeff = self._entropy_coeff()
        entropy_loss = -entropy
        entropy_loss *= entropy_coeff
        return entropy_loss, entropy_coeff

    def training_step(self, batch: Transition, batch_idx: int) -> Tensor:
        critic_td_loss = self.critic_loss(batch)
        actor_ppo_loss, entropy = self.actor_loss(batch)
        entropy_loss, entropy_coeff = self.entropy_loss(entropy)
        loss = actor_ppo_loss + critic_td_loss + entropy_loss
        self.log_dict({
            "actor_ppo_loss": actor_ppo_loss,
            "critic_td_loss": critic_td_loss,
            "actor_entropy_loss": entropy_loss,
            "actor_entropy_coeff": entropy_coeff,
            "actor_entropy": entropy,
            "critic_loss_coeff": self.hparams.critic_coeff,
            "gae_advantage": batch.gae_advantage.mean(),
            "total_loss": loss,
        })
        
        # Check if losses don't send gradients to the wrong network:
        # torchviz.make_dot(loss, params=dict(self.named_parameters()))
        # torch.autograd.grad(actor_ppo_loss, self.critic.parameters(), allow_unused=True)
        # torch.autograd.grad(entropy_loss, self.critic.parameters(), allow_unused=True)
        # torch.autograd.grad(critic_td_loss, self.actor.parameters(), allow_unused=True)
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
        # torch clip_grad_norm_ returns norm before clipping

        grad_norm_total = global_grad_norm(self.parameters())
        grad_norm_actor = global_grad_norm(self.actor.parameters())
        grad_norm_critic = global_grad_norm(self.critic.parameters())

        if self.hparams.clip_grad_norm_total is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.hparams.clip_grad_norm_total)    
        if self.hparams.clip_grad_norm_actor is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.hparams.clip_grad_norm_actor)
        if self.hparams.clip_grad_norm_critic is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.hparams.clip_grad_norm_critic)
        
        grad_norm_clipped_actor = global_grad_norm(self.actor.parameters())
        grad_norm_clipped_critic = global_grad_norm(self.critic.parameters())
        grad_norm_clipped_total = global_grad_norm(self.parameters())

        self.log_dict({
            "actor_grads/norm_unclipped": grad_norm_actor,
            "actor_grads/norm_clipped": grad_norm_clipped_actor,
            "critic_grads/norm_unclipped": grad_norm_critic,
            "critic_grads/norm_clipped": grad_norm_clipped_critic,
            "total_grads/norm_unclipped": grad_norm_total,
            "total_grads/norm_clipped": grad_norm_clipped_total,
        })


class EpisodeCollector:
    def __init__(
        self,
        agent: Agent,
        gamma: float,
        gae_lambda: float,
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
        self.make_env = make_env
        self.make_env_kwargs = make_env_kwargs
        self.num_parallel_envs = num_parallel_envs
        self.seed = seed

    def __iter__(self) -> Iterator[Episode]:
        envs = gym.vector.AsyncVectorEnv([
            functools.partial(self.make_env, **kwargs)
            for kwargs in self.make_env_kwargs
        ])
        episodes = [Episode(self.gamma, self.gae_lambda) for _ in range(self.num_parallel_envs)]
        state, _ = envs.reset(seed=self.seed)

        while True:
            action, log_prob = self.agent.select_action(state)
            next_state, reward, terminated, truncated, infos = envs.step(action)
            done = terminated | truncated
            for i, episode in enumerate(episodes):
                episode.append(s=state[i], a=action[i], r=reward[i], lp=log_prob[i])

            # AsyncVectorEnv automatically resets environments when they are done
            # so next_state might be a resetted state instead of a final state
            # which is desired for running the agent, but we still want the true 
            # final state for training
            for ep in done.nonzero()[0]:
                ep_returns = float(infos["final_info"][ep]["episode"]["r"])
                final_state = infos["final_observation"][ep]
                episodes[ep].finish(self.agent, final_state, ep_returns, truncated[ep])
                yield episodes[ep]
                episodes[ep] = Episode(self.gamma, self.gae_lambda)

            state = next_state


class EpisodeTracker(lightning.Callback):
    def __init__(self) -> None:
        self.finished_episodes = 0
        self.episodic_returns = []
        self.episodic_lengths = []

    def __call__(self, episode: Episode) -> Episode:
        self.finished_episodes += 1
        self.episodic_returns.append(episode.ep_return)
        self.episodic_lengths.append(len(episode))
        return episode
        
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
    buffer_size: int,
    batch_size: int,
    repeat: int,
    gamma: float,
    gae_lambda: float,
    num_parallel_envs: int,
    env_options = None,
    seed: int | None = None,
) -> tuple[torch.utils.data.DataLoader, EpisodeTracker]:
    episode_collector = EpisodeCollector(agent, gamma, gae_lambda, env_name, env_options, num_parallel_envs, seed)
    episode_tracker = EpisodeTracker()
    pipe: dp.iter.IterDataPipe

    pipe = dp.iter.IterableWrapper(episode_collector)
    pipe = pipe.map(episode_tracker)
    pipe = pipe.flatmap(Episode.to_transitions)
    pipe = pipe.batch(buffer_size)
    if repeat > 1:
        pipe = pipe.repeat(repeat)
    pipe = pipe.in_batch_shuffle()
    pipe = pipe.unbatch()

    loader = torch.utils.data.DataLoader(pipe, batch_size, pin_memory=True, shuffle=False)
    return loader, episode_tracker


def main(
    env_name = 'cartpole',
    buffer_size: int = 2048*8, # TODO
    buffer_repeat: int = 10,
    batch_size: int = 64,
    num_parallel_envs: int = 8,
    device: str = "cuda",
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    clip_grad_norm_actor: float = None,
    clip_grad_norm_critic: float = None,
    clip_grad_norm_total: float = 0.5,
    entropy_limit_steps: int = -1,
    entropy_coeff: float = 0.00,
    critic_coeff: float = 1.0,
    epsilon: float = 0.2,
    limit_steps: int = -1,
    critic_loss_fn: str = "smooth_l1_loss",
    lr_actor: float = 2.5e-4,
    lr_critic: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.5,
    normalize_advantage: bool = True,
    seed: int = 0,
) -> None:
    
    args = locals().copy()

    if device == "cuda":
        import torch.backends.cudnn
        torch.backends.cudnn.deterministic = True
    
    lightning.seed_everything(seed)

    env, name = create_env(env_name)
    input_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        action_dim = int(np.prod(env.action_space.shape))
    architecture_dims = [input_dim] + [hidden_dim] * hidden_layers + [action_dim]

    algo = PPO(
        architecture_dims,
        epsilon=epsilon,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff,
        gamma=gamma,
        normalize_advantage=normalize_advantage,
        clip_grad_norm_actor=clip_grad_norm_actor,
        clip_grad_norm_critic=clip_grad_norm_critic,
        clip_grad_norm_total=clip_grad_norm_total,
        entropy_limit_steps=entropy_limit_steps,
        critic_loss_fn=critic_loss_fn,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
    )

    wandb_logger = lightning.pytorch.loggers.WandbLogger(
        project="jku-deep-rl_ppo",
        group=name,
        save_dir="./wandb",
        tags=[env_name, algo.__class__.__name__]
    )
    wandb_logger.experiment.config.update({"args": args})

    def make_env(**kwargs):
        return create_env(env_name, limit_steps, options=kwargs)[0]

    loader, tracker = get_dataloader(
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
        callbacks=[tracker],
    )
    trainer.fit(algo, loader)
    


if __name__ == "__main__":
    typer.run(main)    
