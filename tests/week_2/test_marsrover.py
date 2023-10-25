import unittest
from rl_exercises.environments import MarsRover


class TestMarsRover(unittest.TestCase):
    def test_get_reward_per_action(self):
        env = MarsRover()
        print(env.rng)
        R_sa = env.get_reward_per_action()

        self.assertEqual(R_sa[2, 1], 0)
        self.assertEqual(R_sa[3, 1], 10)

    def test_get_next_state(self):
        env = MarsRover()
        S = env.states
        s_next = env.get_next_state(0, 1, S, p=1)
        self.assertEqual(s_next, 1)
        s_next = env.get_next_state(0, 1, S, p=0)
        self.assertEqual(s_next, 0)
        s_next = env.get_next_state(4, 1, S, p=1)
        self.assertEqual(s_next, 4)
        s_next = env.get_next_state(4, 1, S, p=0)
        self.assertEqual(s_next, 3)

    def test_get_transition_matrix(self):
        env = MarsRover()
        T = env.get_transition_matrix(env.states, env.actions, env.transition_probabilities)
        self.assertEqual(T[0, 0, 0], 1)
        self.assertEqual(T[4, 1, 4], 1)
        self.assertEqual(T[2, 1, 4], 0)


if __name__ == "__main__":
    unittest.main()
