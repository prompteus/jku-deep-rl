from __future__ import annotations
from typing import Any

import numpy as np
import gymnasium as gym


def create_bipedal_walker_env(hardcore=False) -> tuple[gym.Env, str]:
    env = gym.make("BipedalWalker-v3", hardcore=hardcore, render_mode="rgb_array")
    return env, "bipedal_hardcore=" + ('on' if hardcore else 'off')

def create_pendulum_env(gamma=0.99) -> tuple[gym.Env, str]:
    env = gym.make('Pendulum-v1')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env, "pendulum"

def create_env(name: str, options: dict[str, Any] = None) -> tuple[gym.Env, str]:
    if options is None:
        options = {}
    if name == "bipedal_walker":
        env = create_bipedal_walker_env
    elif name == "pendulum":
        env = create_pendulum_env
    else:
        raise ValueError(f"Unknown environment: {name}")
    return env(**options)
