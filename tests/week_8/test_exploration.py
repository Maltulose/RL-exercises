import unittest
from unittest.mock import Mock

import numpy as np
import torch
from rl_exercises.week_5 import EpsilonGreedyPolicy
from rl_exercises.week_8.exploration import EpsilonDecayPolicy, EZGreedyPolicy

policy_classes = {
    "ez-greedy": EZGreedyPolicy,
    "epsilon_decay": EpsilonDecayPolicy,
    "epsilon_greedy": EpsilonGreedyPolicy,
}


class MockActionSpace(Mock):
    def sample(self):
        return -2


class MockEnv(Mock):
    action_space = MockActionSpace()


class MockQ(Mock):
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        qvalues = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int)
        return qvalues


class MockQ2(Mock):
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        qvalues = torch.tensor([4, 3, 2, 1, 0], dtype=torch.int)
        return qvalues


class TestExploration(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_duration(self):
        policy = EZGreedyPolicy(env=MockEnv(), seed=333, epsilon=0)
        Q = MockQ()
        action = policy(Q, state=np.random.rand(3))
        policy.n = 2
        self.assertEqual(action, policy.w)
        Q = MockQ2()
        new_action = policy(Q, state=np.random.rand(3))
        self.assertEqual(action, new_action)
        new_action = policy(Q, state=np.random.rand(3))
        self.assertNotEqual(action, new_action)
        self.assertEqual(policy.w, new_action)

    def test_decay(self):
        policy = EpsilonDecayPolicy(env=MockEnv(), seed=333, epsilon=1.0, total_timesteps=10, final_epsilon=0.01)
        Q = MockQ()
        last_epsilon = 1.0
        for _ in range(10):
            policy(Q, state=np.random.rand(3))
            self.assertNotEqual(policy.epsilon, last_epsilon)
            last_epsilon = policy.epsilon
        self.assertAlmostEqual(policy.epsilon, 0.01)
        policy(Q, state=np.random.rand(3))
        self.assertAlmostEqual(policy.epsilon, 0.01)

    def test_action_selection(self):
        for pclass in policy_classes.values():
            policy = pclass(env=MockEnv(), seed=333, epsilon=0)
            Q = MockQ()
            action = policy(Q, state=np.random.rand(3))
            self.assertEqual(action, 4)
            policy.epsilon = 1
            action = policy(Q, state=np.random.rand(3))
            self.assertEqual(action, -2)


if __name__ == "__main__":
    unittest.main()
