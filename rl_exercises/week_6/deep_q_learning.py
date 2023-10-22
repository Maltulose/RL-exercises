"""Implements a simple DQN agent with replay buffer."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl_exercises.agent import AbstractAgent, AbstractBuffer
from rl_exercises.week_5 import EpsilonGreedyPolicy


class ReplayBuffer(AbstractBuffer):
    """Buffer for storing and sampling transitions."""

    def __init__(self, capacity: int) -> None:
        """
        Make batch replay buffer.

        Parameters
        ----------
        capacity : int
            Max buffer size
        """
        self.capacity = int(capacity)
        self.states: List[np.array] = []  # type: ignore
        self.actions: List[Any[float, int, Tuple]] = []
        self.rewards: List[float] = []
        self.next_states: List[np.array] = []  # type: ignore
        self.dones: List[bool] = []
        self.infos: List[Dict] = []

    def add(
        self,
        state: np.array,  # type: ignore
        action: Any[float, int, Tuple],
        reward: float,
        next_state: np.array,  # type: ignore
        done: bool,
        info: Dict,
    ) -> None:
        """
        Add transition to buffer.

        Parameters
        ----------
        state : np.array
            Env state
        action: Any[float, int, Tuple]
            Action taken
        reward : float
            Reward received
        next_state : np.array
            Next state
        done : bool
            Whether episode is done
        info : Dict
            Info dict
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.infos.append(info)
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)

    def sample(self, batch_size: int = 32) -> Tuple[List, List, List, List, List, List]:  # type: ignore
        """
        Sample batch.

        Parameters
        ----------
        batch_size : int
            Batch size to sample

        Returns
        -------
        states, actions, rewards, next_states, dones, infos
            Transition batch
        """
        # TODO: sample transitions
        transition_ids = ...
        batch_states = [self.states[i] for i in transition_ids]  # type: ignore
        batch_actions = [self.actions[i] for i in transition_ids]  # type: ignore
        batch_rewards = [self.rewards[i] for i in transition_ids]  # type: ignore
        batch_next_states = [self.next_states[i] for i in transition_ids]  # type: ignore
        batch_dones = [self.dones[i] for i in transition_ids]  # type: ignore
        batch_infos = [self.infos[i] for i in transition_ids]  # type: ignore
        return (batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_infos)  # type: ignore

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.states)


class DQN(AbstractAgent):
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
        Make DQN agent.

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
        self.target_Q = self.make_Q()
        self.target_Q.load_state_dict(self.Q.state_dict())

        self.policy = policy(self.env)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        self.n_updates = 0

    def make_Q(self) -> nn.Module:
        """Create Q-Function from env.

        Use 1 hidden layer with 64 units.
        Use ReLU as an activation function after all layers except the last.
        Use `env.observation_space.shape` to get the shape of the input data.
        Use `env.action_space.n` to get number of possible actions for this environment.

        Returns
        -------
        nn.Module
            State-Value Function
        """
        # TODO: Make Q-network
        Q = ...

        return Q  # type: ignore

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
        return self.policy(self.Q, state, evaluate=evaluate), {}

    def save(self, path: str) -> Any:  # type: ignore
        """
        Save Q function and optimizer.

        Parameters
        ----------
        path : str
            Path to save to.
        """
        train_state = {
            "parameters": self.Q.state_dict(),  # type: ignore
            "optimizer_state": self.optimizer.state_dict(),  # type: ignore
        }
        torch.save(train_state, path)  # type: ignore

    def load(self, path: str) -> Any:  # type: ignore
        """
        Load Q function and optimizer.

        Parameters
        ----------
        path : str
            Path to checkpoint
        """
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])  # type: ignore

    def update(  # type: ignore
        self,
        training_batch: list[np.array],  # type: ignore
    ) -> float:
        """Value Function Update for a Batch of Transitions.

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
        states = torch.tensor(np.array(states))
        actions = torch.tensor(np.array(actions))[:, None]
        rewards = torch.tensor(np.array(rewards))
        next_states = torch.tensor(np.array(next_states))

        # TODO: Compute MSE loss
        pred = ...  # noqa: F841
        target = ...  # noqa: F841
        loss = ...

        # TODO: Optimize the model
        self.optimizer.zero_grad()  # type: ignore
        ...

        # TODO: Update target network
        if self.n_updates % 100 == 0:
            pass
        self.n_updates += 1

        return float(loss.item())  # type: ignore
