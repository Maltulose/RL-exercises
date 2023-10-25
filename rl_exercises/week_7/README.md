# Week 7: Policy Gradient

This week you will implement the REINFORCE policy gradient algorithm in order to learn a stochastic policy for the CartPole environment.

## Level 1
### Policy Gradient Implementation
- Complete the Policy class in the code with 2 Linear units to map the states to probabilities over actions.
- Implement compute returns method to compute the discounted returns Gt for each state in a trajectory.
- Implement the policy improvement step to update the policy given the rewards and probabilities from the last trajectory.
- Use the policy in the act method to sample action and return its log probability.
TODO: describe & discuss the tricks we need to make this work

## Level 2
### Questions
- How does the length of the trajectories affect the training?
- How could a baseline be implemented to stabilize the training?
- Does the same network architecture and learning rate work for LunarLander-v2?
- How is the sample complexity (how many steps it takes to solve the environment) of this algorithm related to the DQN from the last exercise?
Please write your answers in answers.txt

## Level 3
### Implement A2C
A2C is an actor-critic method i.e. a hybrid architecture combining value-based and Policy-Based methods that helps to stabilize the training by reducing the variance using. The actor is a policy-based method thatcontrols how the agetn behaves, while the critic is a value-based method that measures how good the taken action is. By baselining the policy-gradient using the citic's value, A2C stabilizes the learning.
Please follow the API provided to you to implement an A2C agent and evaluate it on the given environment.

Resource: https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f