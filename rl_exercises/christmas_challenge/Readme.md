# Reinforcement Learning - Christmas Challenge

In this challenge, your task is to act better than random on the `cbench-v1` benchmark dataset in the `llvm-autophase-ic-v0` environment. 
The environment only works on Linux or MacOS.
You can use [Google Colab](colab.research.google.com/) or Linux Subsystem for Windows (we did not test this but should work) if you do not have such a system available.

You are free to use whatever RL library and algorithms you know!

Please optimize your algorithm on the `dijkstra` benchmark as this will determine your [leaderboard](https://github.com/facebookresearch/CompilerGym#leaderboards) (a team from last year is on 1st place!) position.
However, there will also be a *generalization winner* that has the best performance over all benchmarks in the dataset.
The benchmarks are listed in the cbench-v1.txt file.

You can find other benchmarks [here](https://compilergym.com/llvm/index.html#datasets). The reward space is the [normalized instruction count](https://compilergym.com/llvm/index.html#ir-instruction-count).

**The episodes have no step limit by default, so remember to use the `TimeLimit` wrapper!**

## Installation
For this challenge you need to install `compiler_gym` which you can do with
```bash
pip install -U compiler_gym
```
If this does not work for you, try these [instructions](https://github.com/facebookresearch/CompilerGym/blob/development/INSTALL.md).


## Getting Started

```bash
# test your setup
python evaluate.py
```

## Extra

You can also try your RL algorithms on your own custom benchmarks such as the example in the `custom_benchmarks` directory.
