"""Implements a q-learning agent with value function approximation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    """DQN Agent Class."""

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
        self.optimizer = optim.Adam([self.W, self.b], lr=self.learning_rate)

    def make_Q(self) -> Callable:
        """
        Make Q-function approximator.

        The Q-function is using linear function approximation for Q-value prediction.
        You can use tensors with 'requires_grad=True' to represent the weights of the linear function.
        Use `env.observation_space.shape` to get the shape of the input data.
        Use `env.action_space.n` to get number of possible actions for this environment.

        Returns
        -------
        Q
            An intialized policy
        """
        self.W = torch.randn(
            self.env.action_space.n,  # type: ignore
            self.env.observation_space.low.shape[0],  # type: ignore
            requires_grad=True,
        )
        self.b = torch.randn(self.env.action_space.n, requires_grad=True)  # type: ignore

        def Q(state: np.array):  # type: ignore
            return state @ self.W.t() + self.b

        return Q

    def predict(self, state: np.array, info: Dict, evaluate: bool = False) -> Any:  # type: ignore
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
        action = self.policy(self.Q, state, evaluate=evaluate)  # type: ignore
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
        train_state = {"W": self.W, "b": self.b, "optimizer_state": self.optimizer.state_dict()}
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
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def update(  # type: ignore
        self,
        training_batch: list[np.array],  # type: ignore
    ) -> float:
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
        # Convert data into torch tensors
        states, actions, rewards, next_states, dones, infos = training_batch
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states)

        # Compute MSE loss
        pred = self.Q(states)[actions]
        target = rewards + self.gamma * self.Q(next_states).max()
        loss = nn.MSELoss()(pred, target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
