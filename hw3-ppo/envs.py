from __future__ import annotations
from typing import Any

import numpy as np
import gymnasium as gym


def create_bipedal_walker_env(hardcore=False) -> tuple[gym.Env, str]:
    env = gym.make("BipedalWalker-v3", hardcore=hardcore, render_mode="rgb_array")
    return env, "bipedal_hardcore=" + ('on' if hardcore else 'off')

def create_pendulum_env() -> tuple[gym.Env, str]:
    env = gym.make('Pendulum-v1')
    return env, "pendulum"

def create_cheetach_env() -> tuple[gym.Env, str]:
    env = gym.make('HalfCheetah-v4')
    return env, "half_cheetach"

def create_cartpole_env() -> tuple[gym.Env, str]:
    env = gym.make('CartPole-v1')
    return env, "cartpole"

def create_env(name: str, limit_steps: int = -1, options: dict[str, Any] = None, gamma=0.99) -> tuple[gym.Env, str]:
    if options is None:
        options = {}
    if name == "bipedal_walker":
        make_env = create_bipedal_walker_env
    elif name == "pendulum":
        make_env = create_pendulum_env
    elif name == "half_cheetach":
        make_env = create_cheetach_env
    elif name == "cartpole":
        make_env = create_cartpole_env
    else:
        raise ValueError(f"Unknown environment: {name}")
    env, env_name = make_env(**options)
    if limit_steps is not None and limit_steps > 0:
        env = gym.wrappers.TimeLimit(env, limit_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env, env_name
