from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Callable


def create_policy(env: gym.Env) -> Callable:
    """
    Create Policy

    Parameters
    ----------
    env: gym.Env
        Training Environment

    Returns
    -------
    policy
        Initialiyed Policy
    """
    print(
        f"""Creating policy for environment {env}
        with observation space {env.observation_space}
        and action space {env.action_space}"""
    )

    def policy(obs: np.ndarray) -> int:
        """
        Receives an observation `obs` and returns the action for the given environment.

        Parameters
        ----------
        obs : np.ndarray
            Observation of the current state of the environment.

        Returns
        -------
        int
            Action to take in the given state.
        """
        assert env.action_space is not None
        # TODO: replace this random policy
        return env.action_space.sample()

    return policy
