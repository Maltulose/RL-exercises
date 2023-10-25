from __future__ import annotations

from typing import Callable, List

import os

import compiler_gym
import gymnasium as gym
import numpy as np
from compiler_gym.envs.llvm import make_benchmark
from gymnasium.wrappers import TimeLimit
from policy import create_policy
from tqdm import tqdm


def evaluate(env: gym.Env, policy: Callable[[np.ndarray], int], episodes: int = 100) -> float:
    """
    Evaluate a given Policy on an Environment

    Parameters
    ----------
    env: gym.Env
        Environment to evaluate on
    policy: Callable[[np.ndarray], int]
        Policy to evaluate
    episodes: int
        Evaluation episodes

    Returns
    -------
    mean_rewards
        Mean evaluation rewards
    """
    episode_rewards: List[float] = []
    pbar = tqdm(total=episodes)
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_rewards.append(0)
        done = False
        episode_steps = 0
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_rewards[-1] += reward
            episode_steps += 1
            if terminated or truncated:
                pbar.set_postfix({"episode reward": episode_rewards[-1], "episode step": episode_steps})
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)


if __name__ == "__main__":
    print(compiler_gym.COMPILER_GYM_ENVS)
    custom_benchmark = make_benchmark(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "custom_benchmarks", "rot13.cpp")
    )
    benchmark = "cbench-v1/dijkstra"
    env = gym.make(
        "llvm-autophase-ic-v0", benchmark=benchmark, reward_space="IrInstructionCountNorm", apply_api_compatibility=True
    )
    env = TimeLimit(env, max_episode_steps=100)
    policy = create_policy(env)
    return_mean = evaluate(env, policy)
    print(return_mean)
