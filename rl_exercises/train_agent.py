# Ignore "imported but unused"
# flake8: noqa: F401
import os
import hydra
from omegaconf import DictConfig
import numpy as np
from rich import print as printr
import gymnasium as gym
import rl_exercises
from gymnasium.wrappers import TimeLimit
from minigrid.wrappers import FlatObsWrapper
import warnings

try:
    import compiler_gym
except:
    warnings.warn("Could not import compiler_gym. Probably it is not installed.")
from stable_baselines3.common.monitor import Monitor
from rl_exercises.environments import MarsRover
from stable_baselines3 import SAC, PPO
from typing import List
from tqdm import tqdm
from functools import partial

from rl_exercises.agent.abstract_agent import AbstractAgent
from rl_exercises.week_2 import PolicyIteration, ValueIteration
from rl_exercises.week_4 import EpsilonGreedyPolicy as TabularEpsilonGreedyPolicy
from rl_exercises.week_4 import SARSAAgent
from rl_exercises.week_5 import TabularQAgent, VFAQAgent, EpsilonGreedyPolicy
from rl_exercises.week_6 import DQN
from rl_exercises.week_7 import REINFORCE
from rl_exercises.week_8 import EpsilonDecayPolicy, EZGreedyPolicy


@hydra.main("configs", "base", version_base="1.1")  # type: ignore[misc]
def train(cfg: DictConfig) -> float:
    """Train the agent.

    Parameters
    ----------
    cfg : DictConfig
        Agent/experiment configuration

    Returns
    -------
    float
        Mean return of n eval episodes

    Raises
    ------
    NotImplementedError
        _description_
    """
    env = make_env(cfg.env_name)
    printr(cfg)
    if cfg.agent == "sb3":
        return train_sb3(env, cfg)
    elif cfg.agent in ["policy_iteration", "value_iteration"]:
        agent = eval(cfg.agent_class)(env=env, **cfg.agent_kwargs)
    elif cfg.agent in ["tabular_q_learning", "vfa_q_learning", "dqn"]:
        policy_class = eval(cfg.policy_class)
        policy = partial(policy_class, **cfg.policy_kwargs)
        agent_class = eval(cfg.agent_class)
        agent = agent_class(env, policy, **cfg.agent_kwargs)
    else:
        # TODO: add your agent options here
        raise NotImplementedError

    # TODO: TE: this is not a smart way of doing this, why not import all buffers and handle like above?
    # I know it's not ideal either because the imports aren't called explicitly, but at least we don't have to do this...
    buffer_cls = getattr(rl_exercises.agent.buffer, cfg.buffer_cls, None)
    if buffer_cls is None:
        buffer_cls = getattr(rl_exercises.week_6, cfg.buffer_cls, None)
    if buffer_cls is not None:
        buffer = buffer_cls(**cfg.buffer_kwargs)
    state, info = env.reset()

    num_episodes = 0
    for step in range(cfg.training_steps):
        action, info = agent.predict_action(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        buffer.add(state, action, reward, next_state, (truncated or terminated), info)

        if len(buffer) > cfg.batch_size or (cfg.update_after_episode_end and (terminated or truncated)):
            batch = buffer.sample(cfg.batch_size)
            agent.update_agent(batch)

        state = next_state

        if terminated or truncated:
            state = env.reset()

        if step % cfg.eval_every_n_steps == 0:
            eval_performance = evaluate(make_env(cfg.env_name, cfg.env_kwargs), agent, cfg.n_eval_episodes)
            print(f"Eval reward after {step} steps was {eval_performance}.")

    agent.save(str(os.path.abspath("model.csv")))
    final_eval = evaluate(env, agent, cfg.n_eval_episodes)
    print(f"Final eval reward was: {final_eval}")
    return final_eval


def train_sb3(env: gym.Env, cfg: DictConfig) -> float:
    """Train stablebaselines agent on env.

    Parameters
    ----------
    env : gym.Env
        Environment
    cfg : DictConfig
        Agent/experiment configuration

    Returns
    -------
    float
        Mean rewards
    """
    # Create agent
    model = eval(cfg.agent_class)("MlpPolicy", env, verbose=cfg.verbose, tensorboard_log=cfg.log_dir, seed=cfg.seed)

    # Train agent
    model.learn(total_timesteps=cfg.total_timesteps)

    # Save agent
    model.save(cfg.model_fn)

    # Evaluate
    env = Monitor(gym.make(cfg.env_id))
    means = evaluate(env, model, episodes=cfg.n_eval_episodes)
    performance = np.mean(means)
    return performance


def evaluate(env: gym.Env, agent: AbstractAgent, episodes: int = 100) -> float:
    """Evaluate a given Policy on an Environment.

    Parameters
    ----------
    env: gym.Env
        Environment to evaluate on
    policy: Callable[[np.ndarray], int]
        Policy to evaluate
    episodes: int
        Evaluation episodes

    Returns
    -------
    float
        Mean evaluation rewards
    """
    episode_rewards: List[float] = []
    pbar = tqdm(total=episodes)
    for _ in range(episodes):
        obs, info = env.reset()
        episode_rewards.append(0)
        done = False
        episode_steps = 0
        while not done:
            action, _ = agent.predict_action(obs, info, evaluate=True)  # type: ignore[arg-type]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_rewards[-1] += reward
            episode_steps += 1
            if terminated or truncated:
                done = True
                pbar.set_postfix({"episode reward": episode_rewards[-1], "episode step": episode_steps})
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)


def make_env(env_name: str, env_kwargs: dict = {}) -> gym.Env:
    """Make environment based on name and kwargs.

    Parameters
    ----------
    env_name : str
        Environment name
    env_kwargs : dict, optional
        Optional env config, by default {}

    Returns
    -------
    gym.Env
        Instantiated env
    """
    if "compiler" in env_name:
        benchmark = "cbench-v1/dijkstra"
        env = gym.make(
            "llvm-autophase-ic-v0",
            benchmark=benchmark,
            reward_space="IrInstructionCountNorm",
            apply_api_compatibility=True,
        )
        env = TimeLimit(env, max_episode_steps=100)
    elif env_name == "MarsRover":
        env = MarsRover(**env_kwargs)
        # env = TimeLimit(env, max_episode_steps=env.horizon)
    elif "MiniGrid" in env_name:
        env = gym.make(env_name, **env_kwargs)
        # env = RGBImgObsWrapper(env)
        env = FlatObsWrapper(env)
    else:
        env = gym.make(env_name, **env_kwargs)
    env = Monitor(env, filename="logs.csv")
    return env


if __name__ == "__main__":
    train()
