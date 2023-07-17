from __future__ import annotations

import math
import time
import random
import pathlib
import functools
import collections
from typing import Any, NamedTuple, Iterator, Iterable, Optional, Tuple

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
from tqdm import tqdm
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.distributions import Distribution
from lovely_tensors import lovely

from envs import make_env
from models import Critic, DiscreteActor, ContinuousActor, Agent


class Transition(NamedTuple):
    state: Tensor
    action: Tensor
    reward: Tensor
    log_prob: Tensor
    value: Tensor
    next_value: Tensor
    terminated: Tensor
    truncated: Tensor


class Batch(NamedTuple):
    state: Tensor
    action: Tensor
    log_prob: Tensor
    gae_advantage: Tensor
    gae_return: Tensor


class Buffer:
    def __init__(self, discount_factor: float, gae_lambda: float) -> None:
        self.storage = collections.defaultdict(list)
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda

    def append(self, transition: Transition) -> None:
        for key, val in transition._asdict().items():
            self.storage[key].append(torch.tensor(val))

    def flush(self) -> Iterator[Batch]:
        store = Transition(**{key: torch.stack(val) for key, val in self.storage.items()})
        self.storage.clear()

        gae_advantage, gae_return = self._compute_gae(store)
        
        for step, env_idx in np.ndindex(store.reward.shape):
            yield Batch(
                store.state[step, env_idx],
                store.action[step, env_idx],
                store.log_prob[step, env_idx],
                gae_advantage[step, env_idx],
                gae_return[step, env_idx],
            )

    def _compute_gae(self, store: Transition) -> tuple[Tensor, Tensor]:
        not_dones = ~(store.terminated | store.truncated)
        td_targets = store.reward + self.discount_factor * (~store.terminated) * store.next_value
        td_advantage = td_targets - store.value
        num_steps, num_envs = store.reward.shape
        gae_advantage = torch.zeros((num_steps + 1, num_envs), dtype=torch.float32)
        for t in reversed(range(num_steps)):
            gae_advantage[t] = td_advantage[t] + self.discount_factor * self.gae_lambda * not_dones[t] * gae_advantage[t+1]
        gae_advantage = gae_advantage[:-1]
        gae_return = gae_advantage + store.value
        return gae_advantage, gae_return


def measure_scores_parallel(
    agent: Agent,
    make_env_kwargs: list[dict[str, Any]],
    video_folder: pathlib.Path | str | None = None,
) -> Iterable[float]:
    """
    Measure scores of multiple environments in parallel.
    """
    if video_folder is not None:
        video_folder = pathlib.Path(video_folder)
        video_folder.mkdir(parents=True, exist_ok=True)
    num_episodes = len(make_env_kwargs)

    def make_env_with_recoding(i, kwargs):
        env = make_env(**kwargs)
        if video_folder is not None:
            env = gym.wrappers.RecordVideo(env, video_folder/str(i), disable_logger=True)
        return env

    # lambda: make_env(i, kwargs) would not work because of late binding
    env = gym.vector.AsyncVectorEnv([
        functools.partial(make_env_with_recoding, i, kwargs) 
        for i, kwargs in enumerate(make_env_kwargs)
    ])
    
    limit_env_steps = env.get_attr("spec")[0].max_episode_steps
    
    state, _ = env.reset()
    ep_returns = np.zeros(num_episodes, dtype=np.float32)
    done_in_past = np.zeros(num_episodes, dtype=bool)
    done_at_the_moment = np.zeros(num_episodes, dtype=bool)

    for step in tqdm(range(limit_env_steps), "Measuring score - step"):
        action, log_prob = agent.select_action(state)
        state, _, terminated, truncated, infos = env.step(action)
        done_at_the_moment = terminated | truncated
    
        if "final_info" in infos:
            env_idxs = infos["_final_info"].nonzero()[0]
            for env_idx, info in zip(env_idxs, infos["final_info"][env_idxs]):
                ep_returns[env_idx] = info["episode"]["r"]
        done = done_in_past | done_at_the_moment
        done_for_first_time = done_in_past != done # this is not the same as done_at_the_moment, because of autoreset
        done_in_past = done
        
        for idx in done_for_first_time.nonzero()[0]:
            yield ep_returns[idx].item()

        if done_in_past.all():
            break
      
    env.close()


class PPO(lightning.LightningModule, Agent):
    def __init__(
        self,
        architecture_dims: list[int],
        actions_are_discrete: bool,
        clip_coef: float,
        lr_actor: float,
        lr_critic: float,
        critic_td_loss_coef: float,
        entropy_loss_coef: float,
        critic_loss_fn: str,
        normalize_advantage: bool,
        optimizer_class: str,
        optimizer_config_actor: dict[str, Any] = dict(eps=1e-5),
        optimizer_config_critic: dict[str, Any] = dict(eps=1e-5),
        entropy_limit_steps: int = -1,
        clip_grad_norm_actor: float = math.inf,
        clip_grad_norm_critic: float = math.inf,
        video_root_folder: pathlib.Path | str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if actions_are_discrete:
            self.actor = DiscreteActor(architecture_dims)
            self.action_dtype = torch.long
        else:
            self.actor = ContinuousActor(architecture_dims)
            self.action_dtype = torch.float32
        self.critic = Critic(architecture_dims[:-1] + [1])
        self.critic_loss_fn = getattr(torch.nn.functional, critic_loss_fn)
        
        if video_root_folder is not None:
            video_root_folder = pathlib.Path(video_root_folder)
            video_root_folder.mkdir(parents=True, exist_ok=True)
        self.video_root_folder = video_root_folder

        self.validation_scores = []
        self.validation_start_time: float = math.nan

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

    def critic_loss(self, batch: Batch) -> Tensor:
        curr_value = self.critic(batch.state).flatten()
        critic_td_loss = self.critic_loss_fn(curr_value, batch.gae_return)
        critic_td_loss *= self.hparams.critic_td_loss_coef
        self.log_dict({
            "loss/critic_td_loss": critic_td_loss,
            "loss/critic_ld_loss_coef": self.hparams.critic_td_loss_coef,
        })
        return critic_td_loss

    def actor_loss(self, batch: Batch) -> tuple[Tensor, Tensor]:
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
            "clipped_fraction": (surr_2 < surr_1).float().mean(),
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

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
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
        action = distr.sample().to(self.action_dtype)
        log_prob = distr.log_prob(action)
        return (
            action.detach().cpu().numpy(),
            log_prob.detach().cpu().float().numpy()
        )
    
    def predict_value(self, state: np.ndarray) -> np.ndarray:
        state = torch.atleast_2d(torch.tensor(state).to(self.device).float())
        value: Tensor = self.critic(state)
        return value.flatten().detach().cpu().float().numpy()
    
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
    
    def on_validation_epoch_start(self):
        self.validation_scores.clear()
        self.validation_start_time = time.time()

    def on_validation_epoch_end(self):
        scores = torch.tensor(self.validation_scores)
        validation_num_seconds = time.time() - self.validation_start_time
        self.log_simulation_metrics(scores, validation_num_seconds)

    def validation_step(self, make_env_kwargs: list[dict[str, Any]], batch_idx) -> list[float]:
        if self.video_root_folder is None:
            video_folder = None
        else:
            video_folder = self.video_root_folder / f"step-{self.trainer.global_step}-batch-{batch_idx}"
        
        scores = list(measure_scores_parallel(self, make_env_kwargs, video_folder))
        self.validation_scores.extend(scores)

        if video_folder is not None:
            for logger in self.loggers:
                if isinstance(logger, lightning.pytorch.loggers.WandbLogger):
                    for worker_dir in video_folder.glob("*"):
                        logger.log_metrics({
                            f"video/batch_{batch_idx}_worker_{worker_dir.name}": wandb.Video(str(video))
                            for video in worker_dir.glob("*0.mp4")
                        }, step=self.trainer.global_step)
                    break

        return scores
    
    def simulation_metrics(self, scores: Tensor, num_seconds: float | None = None) -> dict[str, float]:
        metrics = {
            "score/avg": scores.mean(),
            "score/min": scores.min(),
            "score/max": scores.max(),
            "score/std": scores.std(),
            "score/num_episodes": float(len(scores)),
            "score/percentile_25": float(np.percentile(scores, 25)),
            "score/percentile_50": float(np.percentile(scores, 50)),
            "score/percentile_75": float(np.percentile(scores, 75))
        }
        if num_seconds is not None:
            metrics["score/validation_total_num_seconds"] = num_seconds
        return metrics

    def log_simulation_metrics(self, scores: Tensor, num_seconds: float) -> None:
        metrics = self.simulation_metrics(scores, num_seconds)
        self.log_dict(metrics)
        for logger in self.loggers:
            if isinstance(logger, lightning.pytorch.loggers.WandbLogger):
                logger.log_metrics({f"score/score": wandb.Histogram(scores)})



class DataCollector:
    def __init__(
        self,
        agent: Agent,
        envs: gym.vector.VectorEnv,
        buffer_size: int,
        discount_factor: float,
        gae_lambda: float,
    ) -> None:
        self.agent = agent
        self.envs = envs
        self.buffer_size = buffer_size
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda

    def __iter__(self) -> Iterator[Batch]:
        buffer = Buffer(self.discount_factor, self.gae_lambda)
        state, _ = self.envs.reset()
        value = self.agent.predict_value(state)
        num_finished_episodes = 0
        total_env_steps = 0

        while True:
            for _ in range(self.buffer_size // self.envs.num_envs):
                action, log_prob = self.agent.select_action(state)
                next_state, reward, terminated, truncated, infos = self.envs.step(action)
                total_env_steps += self.envs.num_envs
                next_value = self.agent.predict_value(next_state)
                ep_returns = np.full(self.envs.num_envs, np.nan)
                ep_lengths = np.full(self.envs.num_envs, np.nan)
                next_value_final = next_value.copy()

                if "final_info" in infos:
                    env_idxs = infos["_final_info"].nonzero()[0]
                    final_state = np.stack(infos["final_observation"][env_idxs])
                    next_value_final[env_idxs] = self.agent.predict_value(final_state)
                    for env_idx, info in zip(env_idxs, infos["final_info"][env_idxs]):
                        num_finished_episodes += 1
                        wandb.log({
                            "collect_experience/episodic_return": float(info["episode"]["r"]),
                            "collect_experience/episodic_length": int(info["episode"]["l"]),
                            "collect_experience/finished_episodes": num_finished_episodes,
                            "collect_experience/total_env_steps": total_env_steps,
                        })
                        ep_returns[env_idx] = info["episode"]["r"]
                        ep_lengths[env_idx] = info["episode"]["l"]

                transition = Transition(state, action, reward, log_prob, value, next_value_final, terminated, truncated)
                buffer.append(transition)
                state = next_state
                value = next_value

            yield from buffer.flush()


def get_train_dataloader(
    agent: Agent,
    envs: gym.vector.VectorEnv,
    buffer_size: int,
    batch_size: int,
    repeat: int,
    discount_factor: float,
    gae_lambda: float,
) -> torch.utils.data.DataLoader:
    data_collector = DataCollector(agent, envs, buffer_size, discount_factor, gae_lambda)
    buffer: dp.iter.IterDataPipe

    # CRITICAL to have deepcopy=False,
    # otherwise data_collector will be copied and it's agent will be copied too,
    # causing that the agent for collecting data is a different instance than the
    # one that is updated during training
    buffer = dp.iter.IterableWrapper(data_collector, deepcopy=False)
    buffer = buffer.batch(buffer_size)
    if repeat > 1:
        buffer = buffer.repeat(repeat)
    buffer = buffer.in_batch_shuffle()
    buffer = buffer.unbatch()

    loader = torch.utils.data.DataLoader(buffer, batch_size, pin_memory=True, shuffle=False)
    return loader


def get_valid_dataloader(
    env_kwargs: dict[str, Any],
    valid_num_episodes: int,
    valid_num_parallel_envs: int,
    seed: int,
) -> torch.utils.data.DataLoader:
    
    # validation loader just gives parameters for creating the gym env
    validation_seed_gen = random.Random(seed)

    validation_make_env_kwargs = []
    for _ in range(valid_num_episodes):
        env_kwargs_copy = env_kwargs.copy()
        env_kwargs_copy["seed"] = validation_seed_gen.randint(0, 1000000)
        validation_make_env_kwargs.append(env_kwargs_copy)
        
    return torch.utils.data.DataLoader(
        validation_make_env_kwargs,
        batch_size=valid_num_parallel_envs,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda x: x,
    )
    

def main(
    env_name = 'BipedalWalker-v3',
    device: str = "cuda",
    buffer_size: int = 1024,
    buffer_repeat: int = 5,
    batch_size: int = 64,
    train_num_parallel_envs: int = 8,
    valid_num_parallel_envs: int = 8,
    valid_num_episodes: int = 8,
    validate_every_n_updates: int = 3000,
    max_train_updates: int = -1,
    hidden_dim: int = 128,
    hidden_layers: int = 3,
    clip_grad_norm_actor: float = 0.5,
    clip_grad_norm_critic: float = 0.5,
    entropy_limit_steps: int = -1,
    entropy_loss_coef: float = 0.01,
    critic_td_loss_coef: float = 1.0,
    clip_coef: float = 0.2,
    limit_env_steps: int = -1,
    critic_loss_fn: str = "smooth_l1_loss",
    optimizer: str = "Adam",
    lr_actor: float = 1e-4,
    lr_critic: float = 1e-4,
    discount_factor: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantage: bool = True,
    normalize_reward: bool = True,
    normalize_observation: bool = False,
    clip_action: bool = True,
    clip_reward: Tuple[float, float] = (-math.inf, math.inf),
    clip_observation: Tuple[float, float] = (-math.inf, math.inf),
    wandb_project_name: str = "jku-deep-rl_ppo",
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_save_code: bool = True,
    render_mode: str = "rgb_array",
    use_early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
    validate_at_start: bool = True,
    seed: int = 0,
) -> None:
    if wandb_group is None:
        wandb_group = env_name

    if valid_num_parallel_envs is None:
        valid_num_parallel_envs = train_num_parallel_envs
    
    args = locals().copy()

    if device == "cuda":
        import torch.backends.cudnn
        torch.backends.cudnn.deterministic = True
    
    lightning.seed_everything(seed)

    env_kwargs = dict(
        name=env_name,
        normalize_reward=normalize_reward,
        normalize_observation=normalize_observation,
        clip_action=clip_action,
        clip_reward=clip_reward,
        clip_observation=clip_observation,
        limit_env_steps=limit_env_steps,
        render_mode=render_mode,
    )

    env: gym.Env = make_env(**env_kwargs)
    input_dim = int(np.prod(env.observation_space.shape))
    actions_are_dicrete = isinstance(env.action_space, gym.spaces.Discrete)
    if actions_are_dicrete:
        action_dim = env.action_space.n
    else:
        action_dim = int(np.prod(env.action_space.shape))
    architecture_dims = [input_dim] + [hidden_dim] * hidden_layers + [action_dim]

    wandb_logger = lightning.pytorch.loggers.WandbLogger(
        project=wandb_project_name,
        save_code=wandb_save_code,
        entity=wandb_entity,
        group=wandb_group,
        save_dir="./wandb",
        tags=[env_name],
        config={"args": args},
    )

    agent = PPO(
        architecture_dims,
        actions_are_dicrete,
        clip_coef,
        lr_actor,
        lr_critic,
        critic_td_loss_coef,
        entropy_loss_coef,
        normalize_advantage=normalize_advantage,
        entropy_limit_steps=entropy_limit_steps,
        critic_loss_fn=critic_loss_fn,
        optimizer_class=optimizer,
        clip_grad_norm_actor=clip_grad_norm_actor,
        clip_grad_norm_critic=clip_grad_norm_critic,
        video_root_folder=f"./videos/{wandb_logger.experiment.name}"
    )

    envs = gym.vector.AsyncVectorEnv([
        functools.partial(make_env, **env_kwargs)
        for _ in range(train_num_parallel_envs)
    ])

    train_loader = get_train_dataloader(agent, envs, buffer_size, batch_size, buffer_repeat, discount_factor, gae_lambda)
    valid_loader = get_valid_dataloader(env_kwargs, valid_num_episodes, valid_num_parallel_envs, seed)
    wandb_logger.experiment.config.update({"validation_make_env_kwargs": valid_loader.dataset})

    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor="score/avg",
        mode="max",
        save_last=True,
        save_top_k=10,
        dirpath=f"checkpoints/{wandb_logger.experiment.name}/",
        filename="step={step}_score_avg={score/avg:.2f}",
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint_callback]

    if use_early_stopping:
        early_stopping_callback = lightning.pytorch.callbacks.EarlyStopping(
            monitor="score/avg",
            mode="max",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
        )
        callbacks.append(early_stopping_callback)


    trainer = lightning.Trainer(
        accelerator=device,
        max_epochs=-1,
        max_steps=max_train_updates,
        precision="16-mixed",
        logger=wandb_logger,
        val_check_interval=validate_every_n_updates,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    if validate_at_start:
        trainer.validate(agent, valid_loader)
    trainer.fit(agent, train_loader, valid_loader)
    

if __name__ == "__main__":
    typer.run(main)
