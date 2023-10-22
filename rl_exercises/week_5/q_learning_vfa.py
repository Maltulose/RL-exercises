from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from rl_exercises.agent import AbstractAgent


class EpsilonGreedyPolicy(object):
    """A Policy doing Epsilon Greedy Exploration."""

    def __init__(
        self,
        env: gym.Env,
        epsilon: float,
        seed: Optional[int] = None,
    ) -> None:
        """
        Make Epsilon Greedy Policy.

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

    def __call__(self, Q: nn.Module, state: np.array, evaluate: bool = False) -> int:  # type: ignore
        """
        Select action.

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
        if not evaluate and np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        q_values = Q(torch.from_numpy(state).float()).detach().numpy()
        action = np.argmax(q_values)
        return action  # type: ignore


class VFAQAgent(AbstractAgent):
    """VFA Agent Class."""

    def __init__(
        self,
        env: gym.Env,
        policy: Callable[[gym.Env], EpsilonGreedyPolicy],
        learning_rate: float,
        gamma: float,
        **kwargs: Dict,
    ) -> None:
        """
        Make Q-Learning agent using linear function approximation.

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
        """
        self.env = env
        self.Q = self.make_Q()
        self.policy = policy(self.env)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = ...

    def make_Q(self) -> nn.Module:
        """
        Make Q-function approximator.

        The Q-function is using linear function approximation for Q-value prediction.
        You can use tensors with 'requires_grad=True' to represent the weights of the linear function.
        Q should then be a function combining the weights and state into a Q-value prediction.
        Use `env.observation_space.shape` to get the shape of the input data.
        Use `env.action_space.n` to get number of possible actions for this environment.

        Returns
        -------
        Q
            An prediction function
        """
        # TODO: Create Q-Function from env.
        self.W = ...
        self.b = ...
        Q = ...
        return Q  # type: ignore

    def predict_action(self, state: np.array, info: Dict, evaluate: bool = False) -> Any:  # type: ignore
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
        # TODO: predict an action
        action = ...
        info = {}
        return action, info

    def save(self, path: str) -> Any:  # type: ignore
        """
        Save Q function and optimizer.

        Parameters
        ----------
        path : str
            Path to save to.
        """
        train_state = {"W": self.W, "b": self.b, "optimizer_state": self.optimizer.state_dict()}  # type: ignore
        torch.save(train_state, path)

    def load(self, path: str) -> Any:  # type: ignore
        """
        Load Q function and optimizer.

        Parameters
        ----------
        path : str
            Path to checkpoint
        """
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint["parameters"])  # type: ignore
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])  # type: ignore

    def update(self, training_batch: list[np.array]) -> float:  # type: ignore
        """
        Value Function Update for a Batch of Transitions.

        Use MSE loss.

        Parameters
        ----------
        training_batch : list[np.array]
            Batch to train on

        Returns
        -------
        float
            Loss
        """
        # TODO: Implement Value Function Update Step
        # Convert data into torch tensors

        # Compute MSE loss

        # Optimize the model

        loss = 0
        return float(loss)
