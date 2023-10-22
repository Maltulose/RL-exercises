# Week 8: Exploration
This exercise showcases the impact of different exploration strategies. In this assignment you will implement a policy using no exploration, a policy using ε-greedy and one using εz-greedy [Dabney et al., 2020](https://arxiv.org/pdf/2006.01782.pdf). 
The εz-greedy policy samples not only a random action but also a duration for which the action will be played. You can find the algorithm in Appendix B of the linked paper. 
We will use gridworld environments. 

## Level 1: Implement ε(z)-greedy & Rpsilon Decay Exploration Strategies
Your task is to extend the Policy class from Week 6 implement an ε(z)-greedy policy as well as an epsilon greedy policy using linear decay of the epsilon value:

1. For the decay schedule, you'll have to think about how to track the decay steps, the rest should remain fairly similar to the `EpsilonGreedyPolicy` class.

2. To implement ε(z)-greedy, you will have to implement a sampling mechansim for the exploration duration in the 'sample_duration' method - checking the paper for the hyperparameter μ might be helpful for this. The '__call__' method should then implement the repetition behaviour.

3. Make sure both of your policy classes stay compatible with the DQN agent, import them to train and make config files to run DQN with a decaying epsilon schedule as well as ε(z)-greedy.

## Level 2: Experiment & Document
Run all three algorithms - which one performs best in your experiments? We recommend starting with the MiniGrid-Empty-5x5-v0 environment for this exercise.
Is the current algorithm well suited for the problem? What could be a way to improve it (think of the previous lectures)? You can also play with the hyperparameters (e.g., γand ε) and try different environments (e.g., bigger grid).

## Level 3: Count-Based Exploration
Implement a slightly more directed exploration concept: a count-based exploration policy. The idea is to provide an incentive for agents to explore areas of the state space that have not been visited a lot yet. There are many ways to implement count-based exploration, e.g. using [density models of pixel-based environment](https://arxiv.org/pdf/1606.01868.pdf) or [hashing observations](https://arxiv.org/pdf/1611.04717.pdf), but for our gridworlds, we can use a basic version of this method:
1. Count the number of times a state-action pair (s, a) has been visited as N(s, a)
2. Add an exploration bonus to the reward, e.g. like $r_t^i += N(s_t, a_t)^{-1/2}$ (Strehl and Littmann. 2008)
Refer to the slides for more details. Compare this method to the previous ones. Is there a difference? Do we see better performance or behaviour?

## Additional Material:
- Blog Post on Exploration Paradigms in Deep RL: https://lilianweng.github.io/posts/2020-06-07-exploration-drl/
- A list of interesting exploration papers: https://spinningup.openai.com/en/latest/spinningup/keypapers.html#exploration
