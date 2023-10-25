import torch
import numpy as np
import unittest
import gymnasium as gym

from rl_exercises.week_7 import REINFORCE


class TestPolicyGradient(unittest.TestCase):
    def test_compute_returns(self):
        env = gym.make("LunarLander-v2")

        agent = REINFORCE(env, 0.1, 0.99)

        self.assertAlmostEqual(
            agent.compute_returns([1, 1, 1, 1, 1]),
            [4.90099501, 3.9403989999999998, 2.9701, 1.99, 1.0],
        )

    def test_policy_improvement(self):
        env = gym.make("CartPole-v1")
        global_seed = 10
        _ = env.reset(seed=global_seed)  # set the seed via reset once
        env.action_space.seed(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)

        num_episodes = 10
        max_episode_length = 1000

        agent = REINFORCE(env, 1e-2, 0.99)

        rewards = []

        for episode in range(num_episodes):
            rewards.append(0)
            trajectory = []
            state, info = env.reset()

            for t in range(max_episode_length):
                # Generate an action and its log_probability given a state
                action, log_prob = agent.predict_action(state)

                # Take a step in the environment using this action
                next_state, reward, terminated, truncated, _ = env.step(action.item())

                # Append the log probability and reward to the trajectory
                trajectory.append((log_prob, reward))

                state = next_state

                # accumulate the reward for the given episode
                rewards[-1] += reward

                if terminated or truncated:
                    break

            # Policy improvement step
            agent.update_agent(*zip(*trajectory))

        self.assertAlmostEqual(sum(rewards), 262)


if __name__ == "__main__":
    unittest.main()
