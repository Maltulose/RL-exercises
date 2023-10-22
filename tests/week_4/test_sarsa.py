import unittest
import numpy as np

from functools import partial

from rl_exercises.environments import MarsRover
from rl_exercises.week_4 import SARSAAgent, EpsilonGreedyPolicy


class TestSARSA(unittest.TestCase):
    def test_no_exploration(self):
        env = MarsRover()
        np.random.seed(0)
        _ = env.reset(seed=0)  # set the seed via reset once
        env.action_space.seed(0)
        policy = partial(EpsilonGreedyPolicy, epsilon=0.0)

        agent = SARSAAgent(env, policy, 0.1, 0.99)
        rewards = []
        state, _ = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.predict_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            next_action = agent.predict_action(state)

            agent.update_agent((state, action, reward, next_state), next_action, (truncated or terminated))
            rewards.append(reward)

        self.assertAlmostEqual(sum(rewards), 9)

    def test_low_exploration(self):
        env = MarsRover()
        np.random.seed(42)
        _ = env.reset(seed=42)  # set the seed via reset once
        env.action_space.seed(42)
        policy = partial(EpsilonGreedyPolicy, epsilon=0.2)

        agent = SARSAAgent(env, policy, 0.1, 0.99)
        rewards = []
        state, _ = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.predict(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            next_action = agent.predict_action(state)

            agent.update_agent((state, action, reward, next_state), next_action, (truncated or terminated))
            rewards.append(reward)

        self.assertAlmostEqual(sum(rewards), 8)

    def test_high_exploration(self):
        env = MarsRover()
        np.random.seed(10)
        _ = env.reset(seed=10)  # set the seed via reset once
        env.action_space.seed(10)
        policy = partial(EpsilonGreedyPolicy, epsilon=0.8)

        agent = SARSAAgent(env, policy, 0.1, 0.99)
        rewards = []
        state, _ = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.predict(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            next_action = agent.predict_action(state)

            agent.update_agent((state, action, reward, next_state), next_action, (truncated or terminated))
            rewards.append(reward)

        self.assertAlmostEqual(sum(rewards), 22)


if __name__ == "__main__":
    unittest.main()
