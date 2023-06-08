# PyTorch imports
import numpy as np
import torch
import torch.nn as nn
import onnx
from torch.distributions import Normal
from onnx2pytorch import ConvertModel
import torch.nn.functional as F

# Environment import and set logger level to display error only
import gymnasium as gym
from gymnasium import logger as gymlogger

from pyvirtualdisplay import Display

import argparse
import os

gymlogger.set_level(40)  # error only
pydisplay = Display(visible=0, size=(640, 480))
pydisplay.start()

# Seed random number generators
if os.path.exists("seed.rnd"):
    with open("seed.rnd", "r") as f:
        seed = int(f.readline().strip())
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = None


class Env():
    """
    Environment wrapper for BipedalWalker
    """
    def __init__(self):
        self.gym_env = gym.make('BipedalWalker-v3', hardcore=False)
        self.env = self.gym_env
        self.action_space = self.env.action_space

    def reset(self, seed):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class Agent():
    """
    Agent for training
    """
    def __init__(self, net):
        self.actor = net

    def distribution(self, mu_logits, sigma_logits):
        # tanh to ensure value range stays centered between -1 and 1
        mu = F.tanh(mu_logits)
        # softplus logits to ensure only positive values
        sigma = F.softplus(sigma_logits) + 1e-5
        # create a distribution based on mu and sigma
        policy_dist = Normal(loc=mu, scale=sigma)
        return policy_dist

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            output = self.actor(state)
            policy_dist = self.distribution(*output)
            action = policy_dist.sample()
            action = action.cpu().squeeze()
            action = torch.clamp(action, -1, 1)
            action = action.numpy()
        return action


def run_episode(agent, img_stack, seed=None, max_steps=2000):
    env = Env()
    state, _ = env.reset(seed)
    score = 0
    done_or_die = False
    step = 0
    while not done_or_die:
        action = agent.select_action(state)
        state, reward, done, _, _ = env.step(action)
        score += reward
        step += 1

        if done or step >= max_steps:
            done_or_die = True
    env.close()
    return score


if __name__ == "__main__":
    N_EPISODES = 50
    IMG_STACK = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    args = parser.parse_args()
    model_file = args.submission

    device = torch.device("cpu")

    # Network
    net = ConvertModel(onnx.load(model_file))
    net = net.to(device)
    net.eval()
    agent = Agent(net)

    scores = []
    for i in range(N_EPISODES):
        if seed is not None:
            seed = np.random.randint(1e7)
        scores.append(run_episode(agent, IMG_STACK, seed=seed))

    print(np.mean(scores))

