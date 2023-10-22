# Mini-Project: The Vacuum Environment
This week you will have no tests and no set guidelines. 
Your task will be to complete the code stub you will find in the `vacuum.py` file and build an environment to control an automatic vacuum cleaner. 
Your environment should adhere to the gymnasium format, though you don’t need to render anything. 
Start by implementing the basic requirements that should be learned (e.g. moving) before adding options for greater difficulty. 
This could include extra dirty spots in the room, different apartment layouts, breakable vacuum cleaners or even additional functions like dusting. 
To test if your design works as expected, you should try to run an agent (you can use stable baselines3 for this like in week 1's example) on your environment from time to time. Don't forget to wrap your environment in `gymnasium.wrappers.TimeLimit` if you want your env to be resetted after n steps.
Ideally, it will learn slower or not at all whenever you add a new difficulty. 
If your design decisions are flawed this may make the task easier or much too hard, so make sure to check. Good luck and have fun!


## Level 2
Add a rendering mechanism for your environment. Rendering is an important concept for environments, it makes the agent's behavior easily interpretable by humans. There are different rendering frameworks. You can take inspiration from the rendering example from [CartPole](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py). 
Now, when you added your rendering, wrap your environment in a monitor wrapper rendering the env from time to time and train the agent of your choice. You can use stable-baselines3 for this, week 1's example also shows how to render.
Visualize the abilities of your agent at different points in training. Does your agent improve?

## Level 3 
Of course, in reality our vacuum cleaner should be able to handle any room and maybe have different strategies for different rooms and situations.
We can model this by Contextual Markov Decision Processes ([Hallak et al., 2015](https://arxiv.org/abs/1502.02259)).
In cMDPs, depending on the *context*, we get a slightly different MDP. For example, in the context (😉) of vaccum cleaners we might have a dusty context or a task with loads of obstacles.
For more information you can check out our benchmark library CARL for examples and more info ([github](https://github.com/automl/CARL), [paper](https://arxiv.org/abs/2202.04500)).
Maybe you already have some parameters in your environment changing the behavior, these might actually be context features!
If you have defined a cMDP, try training an agent on them. How does it perform?
Btw, we are actively working on CARL/cMDPs and are always happy for support and looking for theses!