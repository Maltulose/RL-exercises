import unittest
import numpy as np

from rl_exercises.environments import MarsRover
from rl_exercises.week_2.policy_iteration import PolicyIteration, do_policy_improvement
from rl_exercises.train_agent import evaluate


def get_agent() -> PolicyIteration:
    agent = PolicyIteration(env=MarsRover())
    return agent


class TestPolicyIteration(unittest.TestCase):
    def test_policy_quality(self):
        seeds = range(1, 11)
        r = []
        steps = []
        for seed in seeds:
            env = MarsRover()
            agent = PolicyIteration(env=env, seed=seed)
            agent.update_agent()
            # Get mean reward per episode
            mean_r = evaluate(env=env, agent=agent, episodes=1)  # deterministic policy
            r.append(mean_r)
            # Get the number of policy improvement steps
            steps.append(agent.steps)

        self.assertTrue(np.mean(steps) > 1)
        self.assertTrue(sum(r) > 0)

    def test_policy_improvement(self):
        Q = np.array([[1, 0], [0, 1], [1, 0]])
        pi_before = [0, 1, 0]
        pi, converged = do_policy_improvement(Q, pi_before)
        self.assertTrue(np.all(pi_before == pi))
        self.assertTrue(converged)


if __name__ == "__main__":
    unittest.main()
