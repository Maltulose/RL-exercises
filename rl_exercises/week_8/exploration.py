from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch  # noqa: F401
import torch.nn as nn
from rl_exercises.week_5 import EpsilonGreedyPolicy


class EpsilonDecayPolicy(EpsilonGreedyPolicy):
    """Policy implementing Epsilon Greedy Exploration with linearly decaying epsilon."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        total_timesteps: int = 100000,
        final_epsilon: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        """Init

        Parameters
        ----------
        env : gym.Env
            Environment
        epsilon: float
            Exploration rate
        seed : int, optional
            Seed, by default None
        """
        self.env = env
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed=seed)
        self.total_timesteps = total_timesteps
        self.starting_epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.timestep = 0

    def update_epsilon(self) -> None:
        """Decay the epsilon value."""
        # TODO: implement decay
        self.epsilon = self.epsilon
        self.timestep += 1

    def __call__(self, Q: nn.Module, state: np.array, evaluate: bool = False) -> int:
        """Select action

        Parameters
        ----------
        Q : nn.Module
            State-Value function
        state : np.array
            State
        evaluate: bool
            evaluation mode - if true, exploration should be turned off.

        Returns
        -------
        int
            action
        """
        action = 0  # just assigned sth
        # TODO implement algorithm

        return action


class EZGreedyPolicy(EpsilonGreedyPolicy):
    """Policy for Exploration with ε(z)-greedy"""

    def __init__(
        self,
        env: gym.Env,
        duration_max: int = 100,
        epsilon: float = 0.1,
        mu: float = 3,
        seed: int | None = None,
    ) -> None:
        """Init

        Parameters
        ----------
        duration_max : int, optional
            Maximum number of action repetition, by default 100
        mu : float, optional
            Zeta/Zipf distribution parameter, by default 2
        seed : int, optional
            Seed, by default None
        """
        self.env = env
        self.duration_max = duration_max
        self.mu = mu
        self.epsilon = epsilon

        self.n: int = 1  # number of times left to perform action
        self.w: int = -1  # random action in memory
        self.rng = np.random.default_rng(seed=seed)
        self.step = -1

    def sample_duration(self) -> int:
        """Sample duration from a zeta/zipf distribution

        The duration is capped at `self.duration_max`.

        Returns
        -------
        int
            duration (how often the action is repeated)
        """
        duration = 1  # just assigned sth
        # TODO implement sampling
        return duration

    def __call__(self, Q: nn.Module, state: np.array, evaluate: bool = False) -> int:
        """Select action

        εz-greedy algorithm B.1 [Dabney et al., 2020].
        The while loop is happening outside, in the training loop.
        This is what is inside the while loop.

        Parameters
        ----------
        Q : nn.Module
            State-Value function
        state : np.array
            State
        evaluate: bool
            evaluation mode - if true, exploration should be turned off.

        Returns
        -------
        int
            action
        """
        action = 0  # just assigned sth
        # TODO implement algorithm

        return action
