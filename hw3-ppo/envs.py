from __future__ import annotations
from typing import Any

import numpy as np
import gymnasium as gym


def create_env(name: str, limit_steps: int = -1, options: dict[str, Any] = None, gamma=0.99) -> gym.Env:
    if options is None:
        options = {}
    
    env = gym.make(name, **options)
    if limit_steps is not None and limit_steps > 0:
        env = gym.wrappers.TimeLimit(env, limit_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    #if isinstance(env.action_space, gym.spaces.Box):
    #    env = gym.wrappers.ClipAction(env)
    #env = gym.wrappers.NormalizeObservation(env)
    #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    #env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env
