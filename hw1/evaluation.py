# imports
import argparse
import os
import numpy as np
import random

import gymnasium as gym
from gymnasium.spaces import Box

from tqdm.auto import tqdm

import torch
from torch.distributions.categorical import Categorical

import onnx
from onnx2pytorch import ConvertModel


# Seed random number generators
torch.backends.cudnn.deterministic = True
if os.path.exists("seed.rnd"):
    with open("seed.rnd", "r") as f:
        seed = int(f.readline().strip())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = None
print(f"Seed: {seed}")


# Custom wrapper to crop HUD
class CropObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        gym.ObservationWrapper.__init__(self, env)
        self.shape = shape
        obs_shape = self.shape + env.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        return observation[:self.shape[0], :self.shape[1]]


def make_env(seed):
    env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    env = CropObservation(env, (84, 96))
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


class Agent():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def select_action(self, state):        
        with torch.no_grad():
            state = torch.Tensor(state).unsqueeze(0).unsqueeze(0).to(device) / 255.0 # rescale
            #print(state.shape)
            logits = self.model(state)
            if type(logits) is tuple:
                logits = logits[0]
            probs = Categorical(logits=logits)
            return probs.sample().cpu().numpy()[0]


def run_episode(agent, seed=None):
    env = make_env(seed=seed)
    state, _ = env.reset()
    score = 0
    done = False
    with tqdm(total=1000, desc="Steps", leave=False) as pbar:
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
            pbar.update(1)
    env.close()
    return score


if __name__ == "__main__":
    N_EPISODES = 50

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    args = parser.parse_args()
    model_file = args.submission

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ConvertModel(onnx.load(model_file))
    model.eval()
    model = model.to(device)
    agent = Agent(model=model, device=device)

    # Evaluate model
    scores = []
    for i in tqdm(range(N_EPISODES), "Episodes"):
        if seed is not None:
            seed = np.random.randint(1e7)
        scores.append(run_episode(agent, seed=seed))
        print("avg:", np.mean(scores))

    # Print result
    print(np.mean(scores))