import unittest
from rl_exercises.week_3.vacuum import VacuumEnv


class TestVacuumEnv(unittest.TestCase):
    def test_env_reset(self):
        env = VacuumEnv()
        state, info = env.reset()
        assert len(state) > 0

    def test_env_random_actions(self):
        env = VacuumEnv()
        state, info = env.reset()
        truncated, terminated = False, False

        counter = 0
        max_steps = 10000

        while not (truncated or terminated) and counter < max_steps:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)

            counter += 1

        assert terminated or truncated, "Env did not terminate episode."
