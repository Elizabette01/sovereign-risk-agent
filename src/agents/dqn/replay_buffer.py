"""
replay_buffer.py — Experience replay buffer for DQN.

Why experience replay?
In standard Q-learning, the agent trains on consecutive (s, a, r, s') transitions.
These are highly correlated — the state at step t+1 is almost identical to step t.
Training on correlated data violates the i.i.d. assumption of stochastic gradient
descent and causes Q-values to oscillate or diverge.

Experience replay (Lin, 1992; Mnih et al., 2015) solves this by:
1. Storing all observed transitions in a fixed-size circular buffer.
2. Sampling random mini-batches for each gradient update.

This decorrelates the training data, stabilises learning, and improves sample
efficiency by reusing each transition multiple times.

Capacity choice: 50,000 transitions. At 30 steps per episode, this holds
~1,667 full episodes of fiscal policy experience. This is large enough to
maintain diverse, representative samples from different economic conditions
(recessions, booms, climate events) without using excessive memory.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular buffer storing (s, a, r, s', done) transitions.

    When the buffer is full, the oldest transition is automatically overwritten
    (deque with maxlen handles this). This ensures the buffer always contains
    the most recent experiences.

    Parameters
    ----------
    capacity:
        Maximum number of transitions. Default 50,000.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """Store one transition.

        Parameters
        ----------
        state:
            Observation before the action (shape: state_dim).
        action:
            Integer action index.
        reward:
            Scalar reward received.
        next_state:
            Observation after the action.
        done:
            1.0 if the episode ended, 0.0 otherwise.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Sample a random mini-batch of transitions.

        Returns numpy arrays for efficient batch conversion to PyTorch tensors.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.

        Returns
        -------
        (states, actions, rewards, next_states, dones) — all numpy arrays.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self.buffer)} transitions but batch_size={batch_size}. "
                "Wait for more experience before sampling."
            )

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
