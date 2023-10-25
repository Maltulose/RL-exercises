import unittest
from functools import partial

import gymnasium as gym
from rl_exercises.week_5 import EpsilonGreedyPolicy
from rl_exercises.week_6.deep_q_learning import DQN, ReplayBuffer


def check_nets(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False

    return True


class TestDeepQLearning(unittest.TestCase):
    def test_init(self):
        env = gym.make("LunarLander-v2")
        policy = partial(EpsilonGreedyPolicy, epsilon=0.1)
        dqn = DQN(env, policy, 0.1, 0.99)
        assert dqn.Q is not None
        assert dqn.target_Q is not None
        assert dqn.optimizer is not None
        assert dqn.policy is not None
        assert dqn.learning_rate == 0.1
        assert dqn.gamma == 0.99
        assert dqn.n_updates == 0

    def test_buffer(self):
        buffer = ReplayBuffer(capacity=5)
        self.assertEqual(len(buffer), 0)
        buffer.add([9, 0], 1, 2, [3], False, {})
        self.assertEqual(len(buffer), 1)
        buffer.add([0, 9], 1, 2, [3], False, {})
        buffer.add([0, 9], 1, 2, [3], False, {})
        transition = buffer.sample(2)
        self.assertEqual(len(transition[0]), 2)
        buffer.add([0, 9], 1, 2, [3], False, {})
        buffer.add([0, 9], 1, 2, [3], False, {})
        buffer.add([0, 9], 1, 2, [3], False, {})
        self.assertEqual(len(buffer), 5)
        self.assertNotEqual(buffer.states[0], [9, 0])

    def test_deep_q_learning(self):
        env = gym.make("LunarLander-v2")
        policy = partial(EpsilonGreedyPolicy, epsilon=0.1)
        dqn = DQN(env, policy, 0.1, 0.99)
        buffer = ReplayBuffer(capacity=100)

        # Copy Q
        state_before_training = dqn.Q.state_dict()
        Q_before_training = DQN(env, policy, 0.1, 0.99)
        Q_before_training.Q.load_state_dict(state_before_training)

        # Train
        state, info = env.reset()
        for _ in range(10):
            action, info = dqn.predict(state, info)
            next_state, reward, terminated, truncated, info = env.step(action)

            buffer.add(state, action, reward, next_state, (truncated or terminated), info)
            batch = buffer.sample(1)
            dqn.update(batch)

        # Check whether something was learned
        assert not check_nets(dqn.Q, Q_before_training.Q)


if __name__ == "__main__":
    unittest.main()
