# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
from __future__ import annotations
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import lightning
import lightning.pytorch.loggers

from typing import NamedTuple, Iterator, Iterable
from torch.distributions import Distribution
from torch.optim.optimizer import Optimizer
from torch import Tensor
import wandb


class Transition(NamedTuple):
    state: Tensor
    action: Tensor
    log_prob: Tensor
    gae_advantage: Tensor
    gae_return: Tensor


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=2048,
        help="total buffer size before updating")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.0, # TODO
        help="the lambda for the general advantage estimation")
    parser.add_argument("--batch-size", type=int, default=64,
        help="the size of minibatches")
    parser.add_argument("--buffer-repeat", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--critic-td-loss-coef", type=float, default=1.0,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    args = parser.parse_args()
    args.num_steps = args.buffer_size // args.num_envs
    args.num_batches = int(args.buffer_size // args.batch_size)
    # fmt: on
    return args


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


@torch.no_grad()
def global_grad_norm(tensors: Iterable[Tensor]) -> Tensor:
    return torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in tensors if p.grad is not None))


class PPO(lightning.LightningModule):
    def __init__(
        self,
        agent: Agent,
        args: argparse.Namespace,
        clip_grad_norm_actor: float = None,
        clip_grad_norm_critic: float = None,
        clip_grad_norm_total: float = None,
    ):
        super().__init__()
        self.critic = agent.critic
        self.actor = agent.actor
        self.critic_loss_fn = torch.nn.functional.smooth_l1_loss
        self.args = args
        self.save_hyperparameters(ignore="agent")

    def critic_loss(self, batch: Transition) -> Tensor:
        curr_value = self.critic(batch.state).flatten()
        critic_td_loss = self.critic_loss_fn(curr_value, batch.gae_return)
        critic_td_loss *= self.args.critic_td_loss_coef
        self.log_dict({
            "loss/critic_td_loss": critic_td_loss,
            "loss/critic_td_loss_coef": self.args.critic_td_loss_coef,
        })
        return critic_td_loss

    def actor_loss(self, batch: Transition) -> tuple[Tensor, Tensor]:
        logits = self.actor(batch.state)
        distr = Categorical(logits=logits)
        log_prob = distr.log_prob(batch.action)
        ratio = (log_prob - batch.log_prob).exp()
        ratio_clipped = ratio.clamp(1-self.args.clip_coef, 1+self.args.clip_coef)
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
        entropy_loss_coef = self.args.ent_coef
        entropy_loss = -entropy
        entropy_loss *= entropy_loss_coef
        self.log_dict({
            "actor_entropy": entropy,
            "loss/actor_entropy_loss": entropy_loss,
            "loss/actor_entropy_loss_coef": entropy_loss_coef,
        })
        return entropy_loss

    def _normalize_advantage(self, advantage: Tensor) -> Tensor:
        if self.args.norm_adv and advantage.numel() > 1:
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
            {"params": self.actor.parameters(), "lr": self.args.learning_rate, "eps": 1e-5},
            {"params": self.critic.parameters(), "lr": self.args.learning_rate, "eps": 1e-5},
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


class DataGenerator:
    def __init__(self, envs: gym.Env, agent: Agent, args) -> None:
        self.agent = agent
        self.args = args
        self.envs = envs

    def __iter__(self) -> Iterator[tuple[int, Transition]]:
        envs = self.envs
        # ALGO Logic: Storage setup
        states = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        total_env_steps = 0
        finished_episodes = 0
        next_obs, _ = envs.reset()
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        while True:
            for step in range(0, args.num_steps):
                total_env_steps += args.num_envs
                states[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = terminated | truncated
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

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
                next_value = agent.get_value(next_obs).reshape(1, -1)
                next_values = torch.cat([values[1:], next_value], dim=0)
                next_dones = torch.cat([dones[1:], next_done.reshape(1, -1)], dim=0)
                td_targets = rewards + args.gamma * (1 - next_dones) * next_values
                td_advantage = td_targets - values
                gae_advantages = torch.zeros_like(rewards).to(device)
                gae_advantages = torch.cat([gae_advantages, torch.zeros_like(gae_advantages[:1])], dim=0)
                for t in reversed(range(args.num_steps)):
                    gae_advantages[t] = td_advantage[t] + args.gamma * args.gae_lambda * (1-next_dones[t]) * gae_advantages[t+1]
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

            # Optimizing the policy and value network
            b_inds = np.arange(args.buffer_size)

            for _ in range(args.buffer_repeat):
                np.random.shuffle(b_inds)
                for start in range(0, args.buffer_size, args.batch_size):
                    end = start + args.batch_size
                    mb_inds = b_inds[start:end]

                    transition = Transition(
                        state=b_states[mb_inds],
                        action=b_actions[mb_inds],
                        log_prob=b_log_probs[mb_inds],
                        gae_return=b_gae_returns[mb_inds],              
                        gae_advantage=b_gae_advantages[mb_inds],
                    )

                    yield transition



if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb_logger = lightning.pytorch.loggers.WandbLogger(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        monitor_gym=True,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    agent = Agent(envs)
    ppo = PPO(
        agent,
        args,
        clip_grad_norm_total=args.max_grad_norm,
    )

    data_generator = DataGenerator(envs, agent, args)

    trainer = lightning.Trainer(
        accelerator=device,
        max_epochs=-1,
        max_steps=-1,
        logger=wandb_logger,
        precision="16-mixed",
    )
    trainer.fit(ppo, data_generator)
    envs.close()
