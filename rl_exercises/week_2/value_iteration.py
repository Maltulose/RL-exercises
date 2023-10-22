from __future__ import annotations
from typing import Any, Callable

import numpy as np
import gymnasium
from rl_exercises.environments import MarsRover
from rl_exercises.agent import AbstractAgent

from rich import print as printr


class ValueIteration(AbstractAgent):
    def __init__(self, env: MarsRover | gymnasium.Env, gamma: float = 0.9, seed: int = 333, **kwargs: dict) -> None:
        """Value Iteration.

        Parameters
        ----------
        env : MarsRover
            Environment. Only applicable to the MarsRover env.
        gamma : float, optional
            Discount factor, by default 0.9
        seed : int, optional
            Seed, by default 333
        """
        if hasattr(env, "unwrapped"):
            env: MarsRover = env.unwrapped  # type: ignore[assignment,no-redef]
        self.env: MarsRover = env  # type: ignore[assignment]
        self.seed = seed

        super().__init__(**kwargs)

        self.n_obs = self.env.observation_space.n  # type: ignore[attr-defined]
        self.n_actions = self.env.action_space.n  # type: ignore[attr-defined]

        # Get the MDP from the env
        self.S = self.env.states
        self.A = self.env.actions
        self.T = self.env.transition_matrix
        self.R_sa = self.env.get_reward_per_action()
        self.gamma = gamma

        # Value Function
        self.V = np.zeros_like(self.S)

        self.policy_fitted: bool = False

    def predict_action(  # type: ignore[override]
        self, observation: int, info: dict | None = None, evaluate: bool = False
    ) -> tuple[int, dict]:
        """Predict action based on observation.

        Parameters
        ----------
        observation : int
            Observation.
        info : dict | None, optional
            Info dict, by default None
        evaluate : bool, optional
            Whether to predict in evaluation mode (i.e. without exploration)

        Returns
        -------
        tuple[int, dict]
            Action, info dict.
        """
        if not self.policy_fitted:
            self.update_agent()
        action = self.pi(observation)
        info = {}
        return action, info

    def update_agent(self, *args: tuple[Any], **kwargs: dict) -> None:
        """Update policy

        In this case, determine the policy once by value iteration.
        """
        if not self.policy_fitted:
            self.V, self.pi = do_value_iteration(
                V=self.V,
                MDP=(self.S, self.A, self.T, self.R_sa, self.gamma),
                seed=self.seed,
            )
            printr("V: ", self.V)
            printr("Final policy: ", self.pi)
            self.policy_fitted = True


def do_value_iteration(
    V: np.ndarray,
    MDP: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    seed: int | None = None,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, Callable]:
    """Value Iteration

    First, update the value function until it is converged and then determine
    the policy.

    Parameters
    ----------
    V : np.ndarray
    MDP : tuple
        Markov Decision Process, defined as (S, A, T, R_sa, gamma).
        S: np.ndarray: States, dim: |S|
        A: np.ndarray: Actions, dim: |A|
        T: np.ndarray: Transition matrix, dim: |S| x |A| x |S|
        R_sa: np.ndarray: Reward being in state s and using action a, dim: |S| x |A|
        gamma: float: Discount factor.
    seed : int | None, optional
        Optional seed, by default None
    epsilon : float, optional
        Convergence criterion, the difference in the value must be lower than this to be
        classified as converged, by default 1e-8

    Returns
    -------
    np.ndarray, Callable
        V, pi (The policy)
    """
    converged = False
    while not converged:
        V, converged = update_value_function(
            V=V,
            MDP=MDP,
            epsilon=epsilon,
        )

    pi = determine_pi(V=V, seed=seed)

    return V, pi


def determine_pi(V: np.ndarray, seed: int | None = None) -> Callable:
    """Generate the policy.

    Parameters
    ----------
    V : np.ndarray
        Value function as value per state.
    seed : int | None, optional
        Optional seed, by default None

    Returns
    -------
    Callable
        Policy receiving the state and outputting action.
    """
    # Seed the random generator
    rng = np.random.default_rng(seed=seed)  # noqa: F841

    # TODO Determine the policy based on the current value function.
    def pi(s: int) -> int:
        """Policy

        Parameters
        ----------
        s : int
            State

        Returns
        -------
        int
            Action
        """
        action: int = 0
        return action  # type: ignore[return-value]

    return pi


# TODO Complete `update_value_function`. This is what happens for each state s inside the while/convergence loop.
def update_value_function(
    V: np.ndarray, MDP: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float], epsilon: float = 1e-8
) -> tuple[np.ndarray, bool]:
    """Update Value Function.

    Parameters
    ----------
    V : np.ndarray
        Value function of len |S|.
    MDP : tuple
        Markov Decision Process, defined as (S, A, T, R_sa, gamma).
        S: np.ndarray: States, dim: |S|
        A: np.ndarray: Actions, dim: |A|
        T: np.ndarray: Transition matrix, dim: |S| x |A| x |S|
        R_sa: np.ndarray: Reward being in state s and using action a, dim: |S| x |A|
        gamma: float: Discount factor.
    epsilon : float, optional
        Convergence criterion, the difference in the value must be lower than this to be
        classified as converged, by default 1e-8

    Returns
    -------
    tuple[np.ndarray, bool]
        V, converged
    """
    S, A, T, R_sa, gamma = MDP
    converged: bool = False

    return V, converged
