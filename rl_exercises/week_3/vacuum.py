from __future__ import annotations

from typing import Any
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from rl_exercises.agent.buffer import Transition


class VacuumEnv(gym.Env):
    """So far, a non-functional env"""

    def __init__(self) -> None:
        """Use this function to initialize the environment"""
        self.action_space: gym.spaces.Space = None  # type: ignore[assignment]
        self.observation_space: gym.spaces.Space = None  # type: ignore[assignment]
        self.reward_range: gym.spaces.Space = None  # type: ignore[assignment]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """Reset the environment"""
        state: list[Any] = []
        info: dict = {}
        return state, info  # type: ignore[return-value]

    def step(self, action: ActType) -> Transition:
        """This should move the vacuum"""
        state = action
        reward = 0
        terminated = True
        truncated = True
        info: dict = {}
        return state, reward, terminated, truncated, info

    def close(self) -> bool:
        """Make sure environment is closed"""
        return True
