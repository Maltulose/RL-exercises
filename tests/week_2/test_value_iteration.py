import unittest
from rl_exercises.week_2.value_iteration import ValueIteration
from rl_exercises.environments import MarsRover
from rl_exercises.train_agent import evaluate


class TestValueIteration(unittest.TestCase):
    def test_value_quality(self):
        seeds = range(1, 11)
        r = []
        for seed in seeds:
            env = MarsRover()
            agent = ValueIteration(env=env, seed=seed)
            agent.update_agent()
            # Get mean reward per episode
            mean_r = evaluate(env=env, agent=agent, episodes=1)  # deterministic policy
            r.append(mean_r)

        self.assertTrue(sum(r) > 0)


if __name__ == "__main__":
    unittest.main()
