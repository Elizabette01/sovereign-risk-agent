from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class QTable:
    """
    A simple tabular Q function:
    Q[state_bins, action] -> value
    """

    n_bins: int
    n_actions: int
    q: np.ndarray

    @classmethod
    def create(cls, n_bins: int, n_actions: int) -> "QTable":
        # Shape: (bins, bins, bins, bins, actions)
        q = np.zeros((n_bins, n_bins, n_bins, n_bins, n_actions), dtype=np.float32)
        return cls(n_bins=n_bins, n_actions=n_actions, q=q)

    def get(self, s: tuple[int, int, int, int], a: int) -> float:
        d, g, i, sh = s
        return float(self.q[d, g, i, sh, a])

    def set(self, s: tuple[int, int, int, int], a: int, value: float) -> None:
        d, g, i, sh = s
        self.q[d, g, i, sh, a] = value

    def best_action(self, s: tuple[int, int, int, int]) -> int:
        d, g, i, sh = s
        return int(np.argmax(self.q[d, g, i, sh, :]))

    def best_value(self, s: tuple[int, int, int, int]) -> float:
        d, g, i, sh = s
        return float(np.max(self.q[d, g, i, sh, :]))
