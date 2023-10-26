# Week 2: Policy and Value Iteration
This week you will implement the fundamental algorithms of policy and value iteration. You'll see how your agent's behaviour changes over time and hopefully have your first successful training runs.

⚠ Before you start, make sure to have read the general `README.md`.
You should add your solutions to the central train and eval script.

## Level 1
### 1. The MarsRover Environment
In the `mars_rover_env.py` file you’ll find the first environment we’ll work with: the MarsRover. 
You have seen it as an example in the lecture: the agent can move left or right with each step and should ideally move to the rightmost state. 
Your task here is to implement the environment dynamics `get_next_state` and determine
the transition matrix `get_transition_matrix`. This is needed for the algorithms policy and value iteration.
The script `rl_exercises/week_2/mars_rover.py` is for you to play with and to check whether your implementation
makes sense. Feel free to vary it however you need, e.g. what you log and how you initialize the environment.
Also, this is a little practice for week 3, where you need to develop an environment.
```bash
python rl_exercises/week_2/mars_rover.py
```

### 2. Policy Iteration for the MarsRover
In this first exercise, the environment will be deterministic, that means the rover
will always execute the given action. Your task is to implement the algorithm policy iteration.
The code stub to be completed is in `policy_iteration.py`.

You can run the exercise with:
```bash
# Policy Iteration
 python rl_exercises/train_agent.py +exercise=w2_policy_iteration
```

Please note that in this exercise we work with the state-value / Q function. In principle, the same formula applies.

### 3. Value Iteration for the probibalistic MarsRover
For this second exercise, we modify the MarsRover environment, now the rover may or may not execute the requested action, the probability is 50%. 
You will complete the code in `value_iteration.py` in order
to evaluate a policy on this variation of our environment.
What happens if you different initial policies? Will you always converge to the same policy? What if you vary gamma?

You can run the exercise with:
```bash
# Value Iteration
python rl_exercises/train_agent.py +exercise=w2_value_iteration
```

## Level 2
What happens if you only have access to `step()` instead of the dynamics and reward? Do both methods still work? This setting will be what we'll work with for the rest of the semester.

## Level 3
Implement Generalized Policy Iteration from the Sutton & Barto book. It is different from your Level 2 solution? Can you match the performance of policy and value iteration?