from __future__ import annotations
from typing import List, Tuple, Any

import gymnasium as gym
import numpy as np  # noqa: F401
import torch
import torch.nn as nn
import torch.optim as optim  # noqa: F401

# NOTE This could be useful to you:
# import torch.nn.functional as F

# NOTE This is potentially a helpful class:
# from torch.distributions import Categorical

from rl_exercises.agent import AbstractAgent, AbstractBuffer  # noqa: F401


class Policy(nn.Module):
    """Define policy network"""

    def __init__(
        self,
        state_space: gym.spaces.box.Box,
        action_space: gym.spaces.discrete.Discrete,
        hidden_size: int = 128,
    ):
        """Initialize the policy network

        Parameters
        ----------
        state_space : gym.spaces.box.Box
            Space for inputs to the network
        action_space : gym.spaces.discrete.Discrete
            Space for outputs of the network
        hidden_size : int, optional
            size of hidden layer, by default 128

        for more information about gym.spaces, please refer to https://www.gymlibrary.dev/api/spaces/

        """
        # TODO Initialize 2 linear layers to map the state to an output equal to the number of actions

        super().__init__()

    def forward(self, x: List[float]) -> torch.Tensor:
        """Forward pass of the policy network

        Parameters
        ----------
        x : List[float]
            State of the environment

        Returns
        -------
        torch.Tensor
            Probabilites over actions
        """
        # TODO pass the input through each layer

        # TODO compute the softmax to normalize the probabilities
        probs = ...

        return probs  # type: ignore


# policy = Policy(env.observation_space, env.action_space)
# optimizer = optim.Adam(policy.parameters(), lr=1e-2)


class REINFORCE(AbstractAgent):
    def __init__(self, env: gym.Env, learning_rate: float, gamma: float) -> None:
        self.env = env
        self.policy = Policy(self.env.observation_space, self.env.action_space)  # type: ignore
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = ...

    def predict(self, state: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use policy to sample an action and return probability for gradient update

        Parameters
        ----------
        state : List[float]
            State of the environment -- 4D array

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            computed action and log probability of the action
        """
        # TODO pass the state through the policy network
        probs = ...  # noqa: F841

        # TODO create the outupt into a categorical distribution and sample and action from it
        action = ...  # noqa: F841

        # TODO compute the log probabilitiy of the action
        log_prob = ...  # noqa: F841

        return action, log_prob  # type: ignore

    def save(self, path: str) -> Any:  # type: ignore
        """Save the policy

        Parameters
        ----------
        path :
            Path to save

        """
        train_state = {
            "parameters": self.policy.state_dict(),  # type: ignore
            "optimizer_state": self.optimizer.state_dict(),  # type: ignore
        }  # type: ignore
        torch.save(train_state, path)

    def load(self, path: str) -> Any:  # type: ignore
        """Load the policy

        Parameters
        ----------
        path :
            Path to load

        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])  # type: ignore

    def compute_returns(self, rewards: List[int]) -> List[float]:
        """Compute discounted returns

        Parameters
        ----------
        rewards : List[int]
            rewards accumulated during trajectory sampling
        Returns
        -------
        List[float]
            List of discounted returns
        """
        returns: list = []

        # Compute the returns_to_go by discounting rewardsand add them sequentially to the list
        ...

        return returns

    def update(self, log_probs: torch.Tensor, rewards: list(float)) -> float:  # type: ignore
        """Perform Policy Improvement using a batch op transitions from a rollout

        Parameters
        ----------
        training_batch : list[np.array]
            Transition to train on -- (state, action, reward, next_state)
        log_probs : torch.Tensor
            Log proabilities of the actions taken during the rollout

        Returns
        -------
        loss    : float
            Loss of the policy network
        """
        # TODO compute the returns
        returns = torch.tensor(self.compute_returns(rewards))  # noqa: F841

        # TODO compute advantages using returns and normalized them
        advantages = ...  # noqa: F841

        log_probs = torch.stack(log_probs)  # type: ignore

        self.optimizer.zero_grad()  # type: ignore

        # TODO Compute loss as the sum of log probs weighted by advantages
        loss = ...

        loss.backward()  # type: ignore
        self.optimizer.step()  # type: ignore

        return loss.item()  # type: ignore
