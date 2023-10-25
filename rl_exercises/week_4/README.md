# Week 4: Model-free Control
This week you will implement you first real model-free learning algorithm, SARSA, as well as conduct some experiments concerning its hyperparameters.

## Level 1
### Model-free Control with SARSA
You will complete the code stubs in `sarsa.py` to implement the SARSA algorithm from the lecture. 
You should include epsilon greedy exploration, as exploration is an important part of model-free learning algorithms. 
As always, use the methods provided as guidance as to what is queried in the tests, but feel free to extend our suggestions in any way you like.

## Level 2
### Hyperparameter Optimization for SARSA
Many concepts of SARSA also apply in more powerful RL algorithms, for example the effect of its hyperparameters. 
Therefore you now have an opportunity to experiment with different hyperparameter values and how they influence how successful the algorithm runs. 
Use the [Hydra SMAC sweeper](https://github.com/automl/hydra-smac-sweeper.git) to tune your algorithm. Try answering the following questions:
- What is the overall performance improvement with tuned hyperparaemters?
- What is the impact of learning rate on the number of training steps? 
- What is the value of $\epsilon$ for which you get the best performance?


## Level 3
### Implementing TD($\lambda$)
In the same format as the SARSA code, implement the TD($\lambda(n)$) algorithm on the Gridcore environment. Make $n$ a configurable parameter signifying the number of lookahead steps. Try to ablate the peformance for multiple values of $n$ and verify the theoretical claims in the lecture. 
