from __future__ import annotations
from typing import NamedTuple, Iterator, Iterable, Optional
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torchdata.datapipes as dp
import typer
import wandb
import lightning
import lightning.pytorch.loggers

from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer


from models import DiscreteActor, Critic, Agent


class Transition(NamedTuple):
    state: Tensor
    action: Tensor
    log_prob: Tensor
    gae_advantage: Tensor
    gae_return: Tensor



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer


# class Agent(nn.Module):
#     def __init__(self, envs):
#         super().__init__()
#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 1), std=1.0),
#         )
#         self.actor = nn.Sequential(
#             layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
#         )

#     def get_value(self, x):
#         return self.critic(x)

#     def get_action_and_value(self, x, action=None):
#         logits = self.actor(x)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(x)


@torch.no_grad()
def global_grad_norm(tensors: Iterable[Tensor]) -> Tensor:
    return torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in tensors if p.grad is not None))


class PPO(lightning.LightningModule, Agent):
    def __init__(
        self,
        norm_adv: bool,
        clip_coef: float,
        ent_coef: float,
        lr_actor: float,
        lr_critic: float,
        critic_td_loss_coef: float,
        clip_grad_norm_actor: float = None,
        clip_grad_norm_critic: float = None,
        clip_grad_norm_total: float = None,
        architecture_dims: list[int] = None,
    ):
        super().__init__()
        self.actor = DiscreteActor(architecture_dims)
        self.critic = Critic(architecture_dims[:-1] + [1])
        self.critic_loss_fn = torch.nn.functional.smooth_l1_loss
        self.save_hyperparameters(ignore="agent")

    def critic_loss(self, batch: Transition) -> Tensor:
        curr_value = self.critic(batch.state).flatten()
        critic_td_loss = self.critic_loss_fn(curr_value, batch.gae_return)
        critic_td_loss *= self.hparams.critic_td_loss_coef
        self.log_dict({
            "loss/critic_td_loss": critic_td_loss,
            "loss/critic_td_loss_coef": self.hparams.critic_td_loss_coef,
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

    def entropy_loss(self, entropy) -> Tensor:
        entropy_loss_coef = self.hparams.ent_coef
        entropy_loss = -entropy
        entropy_loss *= entropy_loss_coef
        self.log_dict({
            "actor_entropy": entropy,
            "loss/actor_entropy_loss": entropy_loss,
            "loss/actor_entropy_loss_coef": entropy_loss_coef,
        })
        return entropy_loss

    def _normalize_advantage(self, advantage: Tensor) -> Tensor:
        if self.hparams.norm_adv and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            return (advantage - loc) / scale
        return advantage
    
    def training_step(self, batch: Transition, batch_idx) -> Tensor:
        critic_td_loss = self.critic_loss(batch)
        actor_ppo_loss, entropy = self.actor_loss(batch)
        entropy_loss = self.entropy_loss(entropy)
        loss = actor_ppo_loss + critic_td_loss + entropy_loss
        self.log("loss/total_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam([
            {"params": self.actor.parameters(), "lr": self.hparams.lr_actor, "eps": 1e-5},
            {"params": self.critic.parameters(), "lr": self.hparams.lr_critic, "eps": 1e-5},
        ])

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
            "grad_norm/actor_unclipped": grad_norm_actor,
            "grad_norm/actor_clipped": grad_norm_clipped_actor,
            "grad_norm/critic_unclipped": grad_norm_critic,
            "grad_norm/critic_clipped": grad_norm_clipped_critic,
            "grad_norm/total_unclipped": grad_norm_total,
            "grad_norm/total_clipped": grad_norm_clipped_total,
        })

    def predict_value(self, state: np.ndarray) -> np.ndarray:
        return self.critic(state)
    
    def select_action(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distr = self.actor(state)
        action = distr.sample()
        log_prob = distr.log_prob(action)
        return action, log_prob


class DataCollector:
    def __init__(self, envs: gym.vector.VectorEnv, agent: Agent, num_steps: int, gamma: float, gae_lambda: float, batch_size: int, buffer_size: int, buffer_repeats: int, device) -> None:
        self.agent = agent
        self.envs = envs
        self.num_envs = envs.num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer_repeats = buffer_repeats
        self.device = device

    def __iter__(self) -> Iterator[tuple[int, Transition]]:
        envs = self.envs
        # ALGO Logic: Storage setup
        states = torch.zeros((self.num_steps, self.num_envs) + envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        total_env_steps = 0
        finished_episodes = 0
        next_obs, _ = envs.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        while True:
            for step in range(0, self.num_steps):
                total_env_steps += self.num_envs
                states[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob = self.agent.select_action(next_obs)
                    value = self.agent.predict_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = terminated | truncated
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

                if "final_info" not in infos:
                    continue

                for info in infos["final_info"]:
                    if info is None:
                        continue
                    finished_episodes += 1
                    print(info['episode']['r'].item())
                    wandb.log({
                        "collect_experience/episodic_return": info["episode"]["r"],
                        "collect_experience/episodic_length": info["episode"]["l"],
                        "collect_experience/finished_episodes": finished_episodes,
                        "collect_experience/total_env_steps": total_env_steps,
                    })

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.predict_value(next_obs).reshape(1, -1)
                next_values = torch.cat([values[1:], next_value], dim=0)
                next_dones = torch.cat([dones[1:], next_done.reshape(1, -1)], dim=0)
                td_targets = rewards + self.gamma * (1 - next_dones) * next_values
                td_advantage = td_targets - values
                gae_advantages = torch.zeros_like(rewards).to(self.device)
                gae_advantages = torch.cat([gae_advantages, torch.zeros_like(gae_advantages[:1])], dim=0)
                for t in reversed(range(self.num_steps)):
                    gae_advantages[t] = td_advantage[t] + self.gamma * self.gae_lambda * (1-next_dones[t]) * gae_advantages[t+1]
                gae_advantages = gae_advantages[:-1]
                gae_returns = gae_advantages + values


            # flatten the batch
            b_states = states.reshape((-1,) + envs.single_observation_space.shape)
            b_values = values.flatten()
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_rewards = rewards.flatten()
            b_log_probs = logprobs.flatten()
            b_gae_advantages = gae_advantages.flatten()
            b_gae_returns = gae_returns.flatten()

            for step in range(self.num_steps):
                for env in range(self.num_envs):
                    yield Transition(
                        state=states[step, env],
                        action=actions[step, env],
                        log_prob=logprobs[step, env],
                        gae_return=gae_returns[step, env],
                        gae_advantage=gae_advantages[step, env],
                    )


def get_dataloader(
    data_collector: DataCollector,
    batch_size: int,
    buffer_size: int,
    repeat: int,
) -> torch.utils.data.DataLoader:
    pipe: dp.iter.IterDataPipe

    pipe = dp.iter.IterableWrapper(data_collector, deepcopy=False)
    pipe = pipe.batch(buffer_size)
    if repeat > 1:
        pipe = pipe.repeat(repeat)
    pipe = pipe.in_batch_shuffle()
    pipe = pipe.unbatch()

    loader = torch.utils.data.DataLoader(pipe, batch_size, shuffle=False)
    return loader


def main(
    env_name = 'CartPole-v1',
    buffer_size: int = 2048, # TODO
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
    entropy_loss_coef: float = 0.00,
    critic_td_loss_coef: float = 1.0,
    clip_coef: float = 0.2,
    critic_loss_fn: str = "smooth_l1_loss",
    lr_actor: float = 1e-4,
    lr_critic: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.9,
    normalize_advantage: bool = False,
    seed: int = 0,
    wandb_project_name: str = "jku-deep-rl_ppo",
    wandb_entity: Optional[str] = None,
) -> None:
    args = locals().copy()

    wandb_logger = lightning.pytorch.loggers.WandbLogger(
        project=wandb_project_name,
        entity=wandb_entity,
        group=env_name,
        config=args,
        monitor_gym=True,
        save_code=True,
    )
    wandb_logger.experiment.name = wandb_logger.experiment.name + "-clean-rl"

    lightning.seed_everything(seed)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv(
        [make_env(env_name, seed + i, i, False, wandb_logger.experiment.name) for i in range(num_parallel_envs)]
    )

    input_dim = int(np.prod(envs.single_observation_space.shape))
    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        action_dim = envs.single_action_space.n
    else:
        action_dim = int(np.prod(envs.single_action_space.shape))
    architecture_dims = [input_dim] + [hidden_dim] * hidden_layers + [action_dim]

    ppo = PPO(
        norm_adv=normalize_advantage,
        clip_coef=clip_coef,
        ent_coef=entropy_loss_coef,
        critic_td_loss_coef=critic_td_loss_coef,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        clip_grad_norm_actor=clip_grad_norm_actor,
        clip_grad_norm_critic=clip_grad_norm_critic,
        clip_grad_norm_total=clip_grad_norm_total,
        architecture_dims = architecture_dims
    )

    data_generator = DataCollector(
        envs,
        ppo,
        num_steps=buffer_size // num_parallel_envs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        batch_size=batch_size,
        buffer_repeats=buffer_repeat,
        buffer_size=buffer_size,
        device=device,
    )

    loader = get_dataloader(data_generator, batch_size, buffer_size, buffer_repeat)

    trainer = lightning.Trainer(
        accelerator=device,
        max_epochs=-1,
        max_steps=-1,
        logger=wandb_logger,
        precision="16-mixed",
    )

    trainer.fit(ppo, loader)
    envs.close()


if __name__ == "__main__":
    typer.run(main)