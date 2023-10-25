from __future__ import annotations
from collections import defaultdict
from typing import DefaultDict, Any

import gymnasium as gym
import numpy as np

from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4 import EpsilonGreedyPolicy


# FIXME: I don't follow the AbstractAgent class
class SARSAAgent(AbstractAgent):
    """SARSA algorithm"""

    def __init__(self, env: gym.Env, policy: EpsilonGreedyPolicy, alpha: float = 0.5, gamma: float = 1.0) -> None:
        """Initialize the SARSA agent

        Parameters
        ----------
        env : gym.Env
            Environment for the agent
        num_episodes : int
            Number of episodes
        gamma : float, optional
            Discount Factor , by default 1.0
        alpha : float, optional
            Learning Rate, by default 0.5
        epsilon : float, optional
            Exploration Parameter, by default 0.1
        """
        # Check hyperparameter boundaries
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"

        self.env = env
        self.gamma = gamma
        self.alpha = alpha

        self.n_actions = self.env.action_space.n  # type: ignore

        # create Q structure
        self.Q: DefaultDict[int, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))

        self.policy = policy(self.Q, self.env)  # type: ignore

    def predict_action(self, state: np.array, info: dict = {}, evaluate: bool = False) -> Any:  # type: ignore # noqa
        """Predict the action for a given state"""
        action = self.policy(self.Q, state, evaluate=evaluate)  # type: ignore
        info = {}
        return action, info

    def save(self, path: str) -> Any:  # type: ignore
        """Save the Q table

        Parameters
        ----------
        path :
            Path to save the Q table

        """
        np.save(path, self.Q)  # type: ignore

    def load(self, path) -> Any:  # type: ignore
        """Load the Q table

        Parameters
        ----------
        path :
            Path to saved the Q table

        """
        self.Q = np.load(path)

    def update(  # type: ignore
        self,
        transition: list[np.array],  # type: ignore
        next_action: int,
        done: bool,
    ) -> float:
        """Perform a TD update

        Parameters
        ----------
        transition : list[np.array]
            Transition to train on -- (state, action, reward, next_state)
        next_action : int
            Next action for lookahead
        done : bool
            done flag

        Returns
        -------
        float
            New Q value for the state action pair
        """
        # TODO: Impelment the TD update
        new_Q = self.Q[transition[0]][transition[1]] + ...
        self.Q[transition[0]][transition[1]] = new_Q
        return new_Q
