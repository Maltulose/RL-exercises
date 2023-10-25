"""GridCore Env taken from https://github.com/automl/TabularTempoRL/"""
from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
import gymnasium


class MarsRover(gymnasium.Env):
    """Simple Environment for a Mars Rover that can move in a 1D Space

    Actions
    -------
    Discrete, 2:
    - 0: go left
    - 1: go right

    Observations
    ------------
    The current position of the rover (int).

    Reward
    ------
    Certain amount per field.

    Start/Reset State
    -----------------
    Position 2.
    """

    def __init__(
        self,
        transition_probabilities: np.ndarray = np.ones((5, 2)),
        rewards: list[float] = [1, 0, 0, 0, 10],
        horizon: int = 10,
        seed: int | None = None,
    ):
        """Init the environment

        Parameters
        ----------
        transition_probabilities : np.ndarray, optional
            [Nx2] Array for N positions and 2 actions each, by default np.ones((5, 2)).
        rewards : list[float], optional
            [Nx1] Array for rewards. rewards[pos] determines the reward for a given
            position `pos`, by default [1, 0, 0, 0, 10].
        horizon : int, optional
            Number of total steps for this environment until it is done (e.g. battery drained), by default 10.
        """
        self.rng = np.random.default_rng(seed=seed)

        self.rewards: list[float] = rewards
        self.transition_probabilities: np.ndarray = transition_probabilities
        self.current_steps: int = 0
        self.horizon: int = horizon
        self.position: int = 2

        n = len(self.transition_probabilities)
        self.observation_space = gymnasium.spaces.Discrete(n=n)
        self.action_space = gymnasium.spaces.Discrete(n=2)

        self.states = np.arange(0, n)
        self.actions = np.arange(0, 2)
        self.transition_matrix = self.T = self.get_transition_matrix(
            S=self.states, A=self.actions, P=self.transition_probabilities
        )

    def get_reward_per_action(self) -> np.ndarray:
        """Determine the reward per action.

        Returns
        -------
        np.ndarray
            Reward per action as a |S|x|A| matrix.
        """
        R_sa = np.zeros((len(self.states), len(self.actions)))  # same shape as P
        for s in range(R_sa.shape[0]):
            for a in range(R_sa.shape[1]):
                delta_s = -1 if a == 0 else 1
                s_index = max(0, min(len(self.states) - 1, s + delta_s))
                R_sa[s, a] = self.rewards[s_index]

        return R_sa

    def get_next_state(self, s: int, a: int, S: np.ndarray, p: float = 1) -> int:
        """Get next state for deterministic action.

        - The action will always be followed = deterministic
        - Translate action into delta s
        - Respect limits of the environment (min and max state).

        Parameters
        ----------
        s : int
            Current state.
        a : int
            Action.
        S : np.ndarray, |S|
            All states.
        p : float
            Probability that action a is followed, by default 1.

        Returns
        -------
        int
            Next state.
        """
        # TODO Implement logic of getting next state
        s_next = s
        return s_next

    def get_transition_matrix(self, S: np.ndarray, A: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Get transition matrix T

        Parameters
        ----------
        S : np.ndarray, |S|
            States
        A : np.ndarray, |A|
            Actions
        P : np.ndarray, |S|x|A|
            Transition probabilities. One entry P[s,a] means the probability of applying the
            desired action a instead of the opposite action.

        Returns
        -------
        np.ndarray, |S|x|A|x|S|
            Transition matrix. T[s,a,s_next] means the probability of being in state s, applying
            action a and landing in state s_next.
        """
        T = np.zeros((len(S), len(A), len(S)))
        # TODO Build transition matrix
        return T

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the environment.

        The rover will always be set to position 2.

        Parameters
        ----------
        seed : int | None, optional
            Seed, not used, by default None
        options : dict[str, Any] | None, optional
            Options, not used, by default None

        Returns
        -------
        tuple[Any, dict[str, Any]]
            Observation, info
        """
        self.current_steps = 0
        self.position = 2

        observation = self.position
        info: dict = {}

        return observation, info

    def step(self, action: int) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment

        Executes an action and return next_state, reward and whether the environment is done (horizon reached).

        Parameters
        ----------
        action : int
            Action. Has to be either 0 (go left) or 1 (go right).

        Returns
        -------
        tuple[int, SupportsFloat, bool, bool, dict[str, Any]]
            Next state, reward, terminated, truncated, info.
        """
        # Determine move given an action and transition probabilities for environment
        action = int(action)
        if action not in [0, 1]:
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1

        self.position = self.get_next_state(
            s=self.position, a=action, S=self.states, p=self.transition_probabilities[self.position][action]
        )

        # Get reward
        reward = self.rewards[self.position]

        terminated = False
        truncated = self.current_steps >= self.horizon

        info: dict = {}

        return self.position, reward, terminated, truncated, info
