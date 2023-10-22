# Week 5: Q-Learning

This week you will implement Q-Learning, another model-free RL algorithm. By using linear function approximation, it is able to scale to infinitely large state spaces.

## Level 1
### 1. Tabular Q-Learning
Implement the Q-Learning update step in `q_learning_tabular.py` and try different state discretizations (bins) and learning rates. How does the number of states and the learning rate affect the training of the RL algorithm?

You can run the exercise with
```bash
python rl_exercises/train_agent.py +exercise=w5_q_learning_tabular
```

### 2. Q-Learning with Linear Value Function Approximation
Implement Q-Learning with Linear Value Function Approximation. First, complete the `make_Q` function of the agent class to create a PyTorch Model. 
Then implement the value function update step in using the Q module and the optimizer. How does the training differ from the tabular case? How sensitive is the algorithm to the weight initialization?
Update the hyperparameters and the model to achieve a mean reward of more than 50 for the CartPole environment.

You can run the exercise with
```bash
python rl_exercises/train_agent.py +exercise=w5_q_learning_vfa
```

## Level 2
Implement double Q-learning for the value function approximation agent. Do you see any improvements? Is overestimation going down? Visualize your Q-values to verify your results.

## Level 3
Instead of predicting only a single Q-value as the expected return, we can also predict the distribution of expected returns. This way we explicitly model the agent's belief about the return structure of the environment. This is called [Distributional Reinforcement Learning](http://proceedings.mlr.press/v89/bellemare19a/bellemare19a.pdf). Try and implement the approach from this paper into your VFA agent. Do you see improvements? What does the learnt return distribution look like?