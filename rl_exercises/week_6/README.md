# Week 6: Deep Q-Learning

This week you will extend Q-Learning with function approximation to even more complex environments by implementing a Deep Q-Network (DQN).
Since this is our transition to Deep Learning, it will also mean transitioning to the experimental standards in Deep RL. If you haven't been running at least 5 seeds for all your experiments, you should start doing so. Reporting some measure of deviation between your runs, e.g. a confidence interval or a standard deviation, in a addition to the mean performance will provide a better estimate of how reliable your method is in the given setting.

## Level 1
### Deep Q Learning
This week’s exercise aims to develop an intuition about how adding deep learning to value function approximation impacts the learning process. We will use the LunarLander-v2 environment for this week's experiments. Your tasks are the following:
- Complete the DQN implementation in `deep_q_learning.py` by adding a deep network as a function approximator and a replay buffer to store and sample transitions from.
- Create a configuration file for your DQN experiments settings. Start with a buffer large enough to store 1e6 samples and a batch size of 32.
- Vary the network architecture (wider, deeper) and the size of the replay buffer and batch. Record your observations for each choice of architecture and buffer size. Please plot the training curve for your experiments with the number of steps on the x-axis and the mean reward on the y-axis. The plots should have your choice of architecture as the title and should be stored in a new folder `plots`.

*Note*: The tests provided for this exercise are only an indicator of whether the plots and answers were generated or not, and whether the Q network learned something or not. We will look into the plots as well as the code to determine the quality of the submitted solutions.

You can run the exercise with
```bash
python rl_exercises/train_agent.py +exercise=w6_dqn
```

## Level 2
The seed can drastically impact your experiment outcome, so it is a common practice in Reinforcement Learning to repeat experiments across multiple seeds and record the training curves as mean values across these seeds with a standard deviation around this value. Mean and standard deviation are not always the best metrics, however, and there are new emerging best practices to report training outcomes. In preparation for your project and further RL experiments, use [RLiable](https://github.com/google-research/rliable), a library for more robust reporting, to improve your plots. What changes? Do you feel more confident in the results? Why? We will discuss these questions during the exercise session. 

## Level 3
A base DQN can have some drawbacks which can be fixed by improving certain algorithm components - the [Rainbow DQN paper](https://arxiv.org/pdf/1710.02298.pdf) illustrates an improved DQN version with several improvements. One of the more important ones is the [Prioritized Experience Replay Buffer](https://arxiv.org/pdf/1511.05952.pdf). Take a look at the paper and implement a priorization to your replay buffer. Does it outperform the base DQN on LunarLander?