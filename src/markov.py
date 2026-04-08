"""Markov Chain model: build transition matrices, generate sequences, temperature sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Hashable, Optional, Tuple

import numpy as np


@dataclass
class MarkovModel:
    """A first-order Markov Chain with temperature-controlled sampling."""
    states: List[Hashable] = field(default_factory=list)
    transition_counts: Dict[Hashable, Dict[Hashable, int]] = field(default_factory=dict)
    _index: Dict[Hashable, int] = field(default_factory=dict)
    _matrix: Optional[np.ndarray] = None

    def add_transition(self, from_state: Hashable, to_state: Hashable) -> None:
        for s in (from_state, to_state):
            if s not in self._index:
                self._index[s] = len(self.states)
                self.states.append(s)
        self.transition_counts.setdefault(from_state, {})
        self.transition_counts[from_state][to_state] = (
            self.transition_counts[from_state].get(to_state, 0) + 1
        )
        self._matrix = None  # invalidate cache

    def build_matrix(self) -> np.ndarray:
        """Build and cache the row-stochastic transition matrix."""
        if self._matrix is not None:
            return self._matrix
        n = len(self.states)
        mat = np.zeros((n, n), dtype=np.float64)
        for from_s, targets in self.transition_counts.items():
            i = self._index[from_s]
            for to_s, count in targets.items():
                j = self._index[to_s]
                mat[i, j] = count
        # Normalise rows
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat /= row_sums
        self._matrix = mat
        return mat

    def sample_next(
        self, current: Hashable, rng: np.random.Generator, temperature: float = 1.0
    ) -> Hashable:
        """Sample the next state given current state and temperature."""
        mat = self.build_matrix()
        if current not in self._index:
            # Fallback: uniform random over all states
            return self.states[rng.integers(len(self.states))]

        row = mat[self._index[current]].copy()

        if temperature != 1.0 and temperature > 0:
            # Apply temperature: raise probabilities to 1/T, then re-normalise
            row = np.power(row + 1e-12, 1.0 / temperature)
            row /= row.sum()

        return self.states[rng.choice(len(self.states), p=row)]

    def generate_sequence(
        self,
        length: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
        start: Optional[Hashable] = None,
    ) -> List[Hashable]:
        """Generate a sequence of given length."""
        if not self.states:
            return []
        if start is None or start not in self._index:
            current = self.states[rng.integers(len(self.states))]
        else:
            current = start
        seq = [current]
        for _ in range(length - 1):
            current = self.sample_next(current, rng, temperature)
            seq.append(current)
        return seq


def build_model_from_sequences(sequences: List[List[Hashable]]) -> MarkovModel:
    """Train a MarkovModel from a list of sequences."""
    model = MarkovModel()
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            model.add_transition(a, b)
    model.build_matrix()
    return model
