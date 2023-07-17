from __future__ import annotations
from typing import Any
import math

import numpy as np
import gymnasium as gym


def create_env(
    name: str,
    normalize_reward: bool,
    normalize_observation: bool,
    clip_action: bool = True,
    clip_reward: tuple[float, float] = (-math.inf, math.inf),
    clip_observation: tuple[float, float] = (-math.inf, math.inf),
    limit_env_steps: int = -1,
    options: dict[str, Any] = None,
) -> gym.Env:
    if options is None:
        options = {}
    
    env = gym.make(name, **options)
    if limit_env_steps is not None and limit_env_steps > 0:
        env = gym.wrappers.TimeLimit(env, limit_env_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if clip_action and isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    if normalize_observation:
        env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, *clip_observation))
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, *clip_reward))
    return env
