"""
train.py — DQN training loop for the Sovereign Risk environment.

This function orchestrates the interaction between the DQN agent and the
SovereignRiskEnv. It follows the standard RL training loop:

    for each step:
        1. Agent observes state s
        2. Agent selects action a (epsilon-greedy)
        3. Environment returns (s', r, done, info)
        4. Transition (s, a, r, s', done) stored in replay buffer
        5. If buffer is large enough, one gradient update is performed

Evaluation is interleaved during training: every eval_freq steps, the agent
is run greedily (epsilon=0) on a separate evaluation environment for
eval_episodes episodes. This gives an unbiased estimate of the policy's
true performance at each checkpoint.

The learning_starts parameter implements a "random warm-up" phase: for the
first learning_starts steps, actions are random regardless of epsilon. This
fills the replay buffer with diverse transitions before training begins,
preventing the early-learning instability that occurs when the buffer
contains only similar transitions from the initial state.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .agent import DQNAgent

logger = logging.getLogger(__name__)


def train_dqn(
    env,
    total_timesteps: int = 100_000,
    agent_kwargs: Optional[dict] = None,
    eval_env=None,
    eval_freq: int = 5_000,
    eval_episodes: int = 20,
    learning_starts: int = 1_000,
    train_freq: int = 4,
    verbose: bool = True,
) -> Tuple[DQNAgent, Dict]:
    """Train a DQN agent on the Sovereign Risk environment.

    Parameters
    ----------
    env:
        Training environment (SovereignRiskEnv instance).
    total_timesteps:
        Total number of environment steps to train for.
    agent_kwargs:
        Keyword arguments passed to DQNAgent constructor.
    eval_env:
        Separate evaluation environment. If None, no evaluation is performed.
    eval_freq:
        Evaluate the policy every this many environment steps.
    eval_episodes:
        Number of greedy episodes per evaluation checkpoint.
    learning_starts:
        Steps of random exploration before gradient updates begin.
    train_freq:
        Perform one gradient update every this many steps (after learning starts).
    verbose:
        Print progress every eval_freq steps.

    Returns
    -------
    (agent, training_log)
        agent: fully trained DQNAgent.
        training_log: dict with keys:
            "episode_rewards": cumulative reward per training episode.
            "episode_lengths": steps per training episode.
            "eval_mean_rewards": mean reward at each evaluation checkpoint.
            "eval_steps": environment step at each evaluation checkpoint.
    """
    agent_kwargs = agent_kwargs or {}
    agent = DQNAgent(**agent_kwargs)

    training_log: Dict[str, List] = {
        "episode_rewards":   [],
        "episode_lengths":   [],
        "eval_mean_rewards": [],
        "eval_steps":        [],
    }

    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episodes_done  = 0

    for step in range(total_timesteps):
        # ----------------------------------------------------------
        # Action selection: random warm-up, then epsilon-greedy
        # ----------------------------------------------------------
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        # ----------------------------------------------------------
        # Environment step
        # ----------------------------------------------------------
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition (total_steps counter incremented inside)
        agent.store_transition(
            obs, action, reward, next_obs, float(terminated)
            # Note: use terminated (not truncated) as the "done" signal for
            # Q-learning. A truncated episode (max steps reached) does not mean
            # the state is truly terminal — the future is non-zero. Only a debt
            # crisis (terminated=True) is genuinely terminal.
        )

        # ----------------------------------------------------------
        # Training update
        # ----------------------------------------------------------
        if step >= learning_starts and step % train_freq == 0:
            agent.train_step()

        episode_reward += reward
        episode_length += 1

        if done:
            training_log["episode_rewards"].append(episode_reward)
            training_log["episode_lengths"].append(episode_length)
            episodes_done += 1
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
        else:
            obs = next_obs

        # ----------------------------------------------------------
        # Periodic evaluation
        # ----------------------------------------------------------
        if eval_env is not None and (step + 1) % eval_freq == 0:
            eval_rewards = []
            for _ in range(eval_episodes):
                eval_obs, _   = eval_env.reset()
                ep_r          = 0.0
                eval_done     = False
                while not eval_done:
                    eval_action = agent.select_action(eval_obs, evaluate=True)
                    eval_obs, eval_r, eval_t, eval_tr, _ = eval_env.step(eval_action)
                    ep_r     += eval_r
                    eval_done = eval_t or eval_tr
                eval_rewards.append(ep_r)

            mean_eval = float(np.mean(eval_rewards))
            training_log["eval_mean_rewards"].append(mean_eval)
            training_log["eval_steps"].append(step + 1)

            if verbose:
                recent = training_log["episode_rewards"][-50:] or [0.0]
                eps    = agent.get_epsilon()
                print(
                    f"  Step {step+1:>7,d}/{total_timesteps:,d} | "
                    f"Episodes: {episodes_done:>4d} | "
                    f"Train(50ep): {np.mean(recent):>8.1f} | "
                    f"Eval({eval_episodes}ep): {mean_eval:>8.1f} | "
                    f"eps={eps:.3f}"
                )

    # Log any partial episode
    if episode_length > 0:
        training_log["episode_rewards"].append(episode_reward)
        training_log["episode_lengths"].append(episode_length)

    return agent, training_log
