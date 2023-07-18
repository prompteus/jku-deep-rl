from __future__ import annotations
from typing import Any
import math
import functools

import numpy as np
import gymnasium as gym


class ResetArgsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reset_kwargs: dict[str, Any]):
        super().__init__(env)
        self.reset_kwargs = reset_kwargs

    def reset(self, *args, **kwargs):
        kwargs = {**self.reset_kwargs, **kwargs}
        return self.env.reset(*args, **kwargs)
    
    def __repr__(self):
        return f"ResetArgsWrapper({self.env}, {self.reset_kwargs})"


def make_env(
    name: str,
    normalize_reward: bool,
    normalize_observation: bool,
    clip_action: bool = True,
    clip_reward: tuple[float, float] = (-math.inf, math.inf),
    clip_observation: tuple[float, float] = (-math.inf, math.inf),
    limit_env_steps: int = -1,
    seed: int | None = None,
    render_mode: str | None = None,
    options: dict[str, Any] = None,
) -> gym.Env:
    if options is None:
        options = {}
    
    env = gym.make(name, render_mode=render_mode, **options)
    if limit_env_steps is not None and limit_env_steps > 0:
        env = gym.wrappers.TimeLimit(env, limit_env_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if clip_action and isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    if normalize_observation:
        env = gym.wrappers.NormalizeObservation(env)
    min_obs, max_obs = clip_observation
    env = gym.wrappers.TransformObservation(env, functools.partial(np.clip, a_min=min_obs, a_max=max_obs))
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env)
    min_reward, max_reward = clip_reward
    env = gym.wrappers.TransformReward(env, functools.partial(np.clip, a_min=min_reward, a_max=max_reward))
    
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env = ResetArgsWrapper(env, {"seed": seed})

    return env
