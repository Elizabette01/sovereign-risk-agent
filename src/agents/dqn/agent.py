"""
agent.py — Deep Q-Network (DQN) agent with experience replay and target network.

DQN Algorithm Overview (Mnih et al., 2015)
==========================================

Classic Q-learning maintains a table Q(s, a) updated by the Bellman equation:
    Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

DQN replaces this table with a neural network and adds two stabilising tricks:

1. Experience Replay (Section above in replay_buffer.py):
   Store transitions, train on random mini-batches.

2. Target Network:
   Maintain a second, slowly-updated copy of the Q-network (θ⁻) for computing
   target values. Without this, both the prediction Q(s,a;θ) and the target
   r + γ max Q(s',a';θ) change each update step — chasing a moving target
   destabilises training. The target network θ⁻ is only updated every K steps
   (a hard copy of θ), giving stable regression targets.

3. Epsilon-Greedy Exploration:
   With probability ε, take a random action (exploration).
   With probability 1-ε, take the action with the highest Q-value (exploitation).
   ε decays from 1.0 (pure exploration) to 0.05 (mostly exploitation) over
   training, transitioning from learning about the environment to acting on
   what was learned.

4. Gradient Clipping:
   Clip gradients to max norm 10 to prevent explosive updates during
   early training when Q-value estimates are poor.

5. Huber Loss:
   Behaves like L2 (MSE) for small errors and L1 for large errors. This
   dampens the effect of large Bellman errors (outlier transitions) that
   would otherwise dominate gradient updates early in training.

Reference:
    Mnih, V. et al. (2015). "Human-level control through deep reinforcement
    learning." Nature, 518(7540), 529–533.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import QNetwork
from .replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DQNAgent:
    """DQN agent: Q-network + target network + replay buffer + epsilon-greedy.

    Parameters
    ----------
    state_dim:
        Input dimension (7 for SovereignRiskEnv).
    action_dim:
        Number of discrete actions (6).
    learning_rate:
        Adam optimizer learning rate. 1e-4 is the standard DQN value from
        Mnih et al. (2015); lower than SGD defaults to account for the
        non-stationary target signal.
    gamma:
        Discount factor. 0.99 gives a planning horizon of ~100 steps
        (1/(1-0.99)), appropriate for our 30-year episodes.
    epsilon_start, epsilon_end, epsilon_decay_steps:
        Linear epsilon schedule. 30,000 steps = ~33% of 100k total steps,
        ensuring meaningful exploration in the early phase.
    buffer_size:
        Replay buffer capacity (see replay_buffer.py for rationale).
    batch_size:
        Mini-batch size. 64 is standard; larger batches give lower-variance
        gradients but are slower per update.
    target_update_freq:
        Hard-copy target network every N gradient steps. 500 gives stable
        targets while allowing the Q-network to improve.
    seed:
        Random seed for PyTorch and numpy RNG.
    """

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 6,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 30_000,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        seed: int = 42,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon decay schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Networks
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.q_network      = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target network is never directly trained

        # Optimizer and loss
        self.optimiser = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn   = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Step counter (incremented in train_step(), not in the env loop)
        self.total_steps: int = 0

        # Gradient-update counter (for target network sync)
        self._update_count: int = 0

        # Training metrics
        self.training_losses: List[float] = []

        # RNG for epsilon-greedy decisions
        self.rng = np.random.default_rng(seed)

        logger.info(
            "DQNAgent created: device=%s, state_dim=%d, action_dim=%d, "
            "lr=%.0e, γ=%.2f, ε: %.2f→%.2f over %d steps",
            self.device, state_dim, action_dim,
            learning_rate, gamma,
            epsilon_start, epsilon_end, epsilon_decay_steps,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_epsilon(self) -> float:
        """Return the current exploration rate using a linear decay schedule."""
        progress = min(self.total_steps / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Choose an action using the epsilon-greedy policy.

        Parameters
        ----------
        state:
            Current normalised observation of shape (state_dim,).
        evaluate:
            If True, always pick the greedy action (no exploration).
            Used during evaluation runs to measure policy performance.

        Returns
        -------
        int: action index in {0, ..., action_dim-1}.
        """
        if not evaluate and self.rng.random() < self.get_epsilon():
            # Explore: uniform random action
            return int(self.rng.integers(0, self.action_dim))

        # Exploit: select action with highest Q-value
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """Store one (s, a, r, s', done) tuple in the replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self) -> float:
        """Perform one mini-batch gradient update.

        The DQN loss is the mean Huber error between predicted Q-values and
        Bellman targets:
            target = r + γ × max_{a'} Q_target(s', a') × (1 - done)
            loss   = HuberLoss(Q(s, a), target)

        The (1 - done) term zeroes the future-reward component for terminal
        transitions — if the episode ended, there is no future Q-value.

        Returns
        -------
        float: loss value for this update step (0.0 if buffer too small).
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q(s, a) — only the Q-value for the action actually taken
        current_q = (
            self.q_network(states_t)
            .gather(1, actions_t.unsqueeze(1))
            .squeeze(1)
        )

        # Bellman target: r + γ × max_a' Q_target(s', a') × (1 - done)
        with torch.no_grad():
            next_q_max = self.target_network(next_states_t).max(dim=1)[0]
            target_q   = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        # NaN guard
        if torch.isnan(current_q).any() or torch.isnan(target_q).any():
            logger.warning("NaN detected in Q-values — skipping update step %d", self._update_count)
            return 0.0

        loss = self.loss_fn(current_q, target_q)

        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping prevents explosive updates when Bellman errors are large
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimiser.step()

        self._update_count += 1

        # Periodically sync target network with Q-network (hard copy)
        if self._update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters in the Q-network."""
        return sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent state to a .pt checkpoint file."""
        torch.save(
            {
                "q_network":      self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimiser":      self.optimiser.state_dict(),
                "total_steps":    self.total_steps,
                "_update_count":  self._update_count,
            },
            path,
        )
        logger.info("DQNAgent saved to %s", path)

    def load(self, path: str) -> None:
        """Load agent state from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        self.total_steps   = checkpoint.get("total_steps", 0)
        self._update_count = checkpoint.get("_update_count", 0)
        logger.info("DQNAgent loaded from %s (total_steps=%d)", path, self.total_steps)
