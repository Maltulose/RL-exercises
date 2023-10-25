from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Tuple
import numpy as np

from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4 import EpsilonGreedyPolicy as TabularEpsilonGreedyPolicy

import gymnasium as gym


def to_bins(value: np.ndarray, bins: np.ndarray) -> float:  # type: ignore
    """Put a single float value into closest bin."""
    return np.digitize(x=[value], bins=bins)[0]


def to_discrete_state(obs: Tuple[float, float, float, float], num_bins: int) -> Tuple[float, float, float, float]:
    """Transform an observation from continuous to discrete space."""
    x, v, theta, omega = obs
    CART_POSITION = np.linspace(-4.8, 4.8, num_bins)
    CART_VELOCITY = np.linspace(-1, 1, num_bins)
    POLE_ANGLE = np.linspace(-0.418, 0.418, num_bins)
    POLE_ANGULAR_VELOCITY = np.linspace(-3, 3, num_bins)
    state = (
        to_bins(x, CART_POSITION),  # type: ignore
        to_bins(v, CART_VELOCITY),  # type: ignore
        to_bins(theta, POLE_ANGLE),  # type: ignore
        to_bins(omega, POLE_ANGULAR_VELOCITY),  # type: ignore
    )
    return state


def default_q_value() -> float:
    """Retrn a randomly sampled Q value

    Returns
    -------
    float
        Uniformly sampled float between 1 and -1
    """
    return np.random.uniform(1, -1)


class TabularQAgent(AbstractAgent):
    """Q-Learning Agent Class."""

    def __init__(
        self,
        env: gym.Env,
        policy: Callable[[gym.Env], TabularEpsilonGreedyPolicy],
        learning_rate: float,
        gamma: float,
        num_bins: int = 20,
        **kwargs: dict,
    ) -> None:
        """
        Make Tabular Q-Learning agent.

        Parameters
        ----------
        env : gym.Env
            Environment to train on
        policy : Callable[[gym.Env], EpsilonGreedyPolicy]
            Make function for policy
        learning_rate : float
            Learning rate
        gamma : float
            Discount factor
        num_bins : int
            Number of bins to discretize state space into
        """
        self.env = env
        self.Q: DefaultDict[np.array, float] = defaultdict(lambda: np.random.uniform(1, -1))  # type: ignore
        self.policy = policy(self.env)
        self.learning_rate = learning_rate
        self.gamma = gamma

        # This adds the option to pass a function via the kwargs
        # You'll only need this if you want to use a different environment
        if "discretize_state" in kwargs.keys():
            self.discretize_state = kwargs["discretize_state"]
        else:
            self.discretize_state = to_discrete_state  # type: ignore
        self.num_bins = num_bins

    def predict_action(self, state: tuple, info: Dict) -> tuple[int, dict]:  # type: ignore
        """
        Predict action from state.

        Parameters
        ----------
        state : np.array
            Env state
        info : Dict
            Info dict
        evaluate : bool, optional
            Whether to predict in evaluation mode (i.e. without exploration)

        Returns
        -------
        action, info
            action to take and info dict
        """
        discrete_state = self.discretize_state(state, self.num_bins)  # type: ignore # noqa: F841

        # TODO: predict an action
        action = ...
        info = {}
        return action, info  # type: ignore

    def save(self, path: str) -> Any:  # type: ignore
        """
        Save Q function.

        Parameters
        ----------
        path : str
            Path to save to.
        """
        np.save(path, dict(self.Q))  # type: ignore

    def load(self, path: str) -> Any:  # type: ignore
        """
        Load Q function.

        Parameters
        ----------
        path : str
            Path to checkpoint
        """
        self.Q = defaultdict(lambda: np.random.uniform(1, -1))
        checkpoint = np.load(path)
        self.Q.update(checkpoint.item())

    def update_agent(self, transition: list[np.array]) -> float:  # type: ignore
        """Value Function Update.

        Parameters
        ----------
        transition : list[np.array]
            Transition to train on

        Returns
        -------
        float
            TD-Error
        """
        # TODO: Implement Q-Learning Update
        td_error = 0
        return td_error
