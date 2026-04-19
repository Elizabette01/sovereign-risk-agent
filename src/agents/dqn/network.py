"""
network.py — Q-Network for the Deep Q-Network (DQN) agent.

The Q-network is the core learnable component of DQN. It approximates the
action-value function Q(s, a): the expected cumulative discounted reward
of taking action a in state s and then following the optimal policy.

Architecture choice rationale:
- Two hidden layers with 128 units each is the standard baseline for
  low-dimensional state spaces. With only 7 input dimensions and 6 actions,
  a larger network would have too many parameters relative to the signal,
  increasing variance and training instability.
- ReLU activations avoid the vanishing gradient problem that affects tanh
  in deep networks, and are computationally efficient.
- No activation on the output layer: Q-values are unbounded real numbers,
  so any output activation would artificially constrain the estimate.
- Huber loss (not MSE) in the agent: this architecture produces raw Q-values
  that are then used with Huber loss, which is less sensitive to large
  Bellman errors during early training.

Reference:
    Mnih et al. (2015). "Human-level control through deep reinforcement
    learning." Nature, 518, 529–533.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Feedforward neural network mapping states to per-action Q-values.

    Parameters
    ----------
    state_dim:
        Dimensionality of the state space. Default 7 (matching the
        SovereignRiskEnv observation space).
    action_dim:
        Number of discrete actions. Default 6.
    hidden_dim:
        Number of units in each hidden layer. Default 128.
    """

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 6,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of states to Q-values for all actions.

        Parameters
        ----------
        x:
            Tensor of shape (batch_size, state_dim).

        Returns
        -------
        Tensor of shape (batch_size, action_dim) containing Q(s, a) for
        every action a.
        """
        return self.network(x)
