"""Markov Chain model: build transition matrices, generate sequences, temperature sampling.

Supports first-order and higher-order (n-gram) models with Laplace smoothing
and automatic backoff from higher to lower order when a context is unseen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Hashable, Optional, Tuple

import numpy as np


@dataclass
class MarkovModel:
    """A Markov Chain with temperature-controlled sampling and Laplace smoothing.

    Supports arbitrary order via n-gram context keys.  When used as a
    higher-order model the *context* (from_state) is a tuple of the last
    ``order`` output states; the *emission* (to_state) is a single state.
    """
    states: List[Hashable] = field(default_factory=list)
    transition_counts: Dict[Hashable, Dict[Hashable, int]] = field(default_factory=dict)
    _index: Dict[Hashable, int] = field(default_factory=dict)
    _matrix: Optional[np.ndarray] = None
    smoothing_alpha: float = 0.01  # Laplace smoothing constant

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
        """Build and cache the row-stochastic transition matrix with Laplace smoothing."""
        if self._matrix is not None:
            return self._matrix
        n = len(self.states)
        mat = np.full((n, n), self.smoothing_alpha, dtype=np.float64)
        for from_s, targets in self.transition_counts.items():
            i = self._index[from_s]
            for to_s, count in targets.items():
                j = self._index[to_s]
                mat[i, j] += count
        # Normalise rows
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat /= row_sums
        self._matrix = mat
        return mat

    def get_distribution(
        self, context: Hashable, temperature: float = 1.0
    ) -> Optional[np.ndarray]:
        """Return probability distribution over states for a given context, or None if unknown."""
        mat = self.build_matrix()
        if context not in self._index:
            return None
        row = mat[self._index[context]].copy()
        if temperature != 1.0 and temperature > 0:
            row = np.power(row + 1e-12, 1.0 / temperature)
            row /= row.sum()
        return row

    def sample_next(
        self, current: Hashable, rng: np.random.Generator, temperature: float = 1.0
    ) -> Hashable:
        """Sample the next state given current state and temperature."""
        dist = self.get_distribution(current, temperature)
        if dist is None:
            return self.states[rng.integers(len(self.states))]
        return self.states[rng.choice(len(self.states), p=dist)]

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


class HighOrderMarkovModel:
    """Second-order (or higher) Markov model with automatic backoff.

    Uses n-gram tuples as context keys.  When sampling, if the full
    n-gram context has not been seen, it backs off to (n-1)-gram, then
    (n-2)-gram, …, down to uniform random.

    Uses a clean dict-based approach that separates context keys from
    emission states, avoiding the MarkovModel's state-space mixing issue.
    """

    def __init__(self, order: int = 2, smoothing_alpha: float = 0.01) -> None:
        self.order = order
        self.smoothing_alpha = smoothing_alpha
        # Per n-gram level: context_tuple → {emission_state: count}
        self._counts: Dict[int, Dict[Tuple, Dict[Hashable, int]]] = {
            k: {} for k in range(1, order + 1)
        }
        # All unique emission states (the actual musical tokens)
        self._emission_states: List[Hashable] = []
        self._emission_index: Dict[Hashable, int] = {}

    def _register_emission(self, state: Hashable) -> None:
        if state not in self._emission_index:
            self._emission_index[state] = len(self._emission_states)
            self._emission_states.append(state)

    def add_sequence(self, seq: List[Hashable]) -> None:
        """Train from a single sequence, building all n-gram levels."""
        for s in seq:
            self._register_emission(s)
        for k in range(1, self.order + 1):
            for i in range(len(seq) - k):
                context = tuple(seq[i:i + k])
                target = seq[i + k]
                ctx_dict = self._counts[k].setdefault(context, {})
                ctx_dict[target] = ctx_dict.get(target, 0) + 1

    def build(self) -> None:
        """No-op — kept for API consistency; sampling reads counts directly."""
        pass

    def _get_emission_dist(
        self, context: Tuple, level: int, temperature: float
    ) -> Optional[np.ndarray]:
        """Build a probability distribution over emission states for a context."""
        if context not in self._counts[level]:
            return None
        count_map = self._counts[level][context]
        n = len(self._emission_states)
        dist = np.full(n, self.smoothing_alpha, dtype=np.float64)
        for state, count in count_map.items():
            dist[self._emission_index[state]] += count
        # Temperature scaling
        if temperature != 1.0 and temperature > 0:
            dist = np.power(dist, 1.0 / temperature)
        dist /= dist.sum()
        return dist

    def sample_next(
        self,
        history: List[Hashable],
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> Hashable:
        """Sample the next emission state with automatic backoff through n-gram levels."""
        for k in range(min(self.order, len(history)), 0, -1):
            context = tuple(history[-k:])
            dist = self._get_emission_dist(context, k, temperature)
            if dist is not None:
                return self._emission_states[rng.choice(len(self._emission_states), p=dist)]

        # Ultimate fallback: uniform over emission states
        if self._emission_states:
            return self._emission_states[rng.integers(len(self._emission_states))]
        raise ValueError("Model has no states — was it trained?")

    def generate_sequence(
        self,
        length: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
        start: Optional[List[Hashable]] = None,
    ) -> List[Hashable]:
        """Generate a sequence, seeding with an optional start list."""
        if not self._emission_states:
            return []
        if start is not None:
            history = list(start)
        else:
            history = [self._emission_states[rng.integers(len(self._emission_states))]]

        while len(history) < length:
            nxt = self.sample_next(history, rng, temperature)
            history.append(nxt)

        return history[:length]


def build_model_from_sequences(
    sequences: List[List[Hashable]], order: int = 1
) -> MarkovModel | HighOrderMarkovModel:
    """Train a Markov model from a list of sequences.

    order=1  → returns a first-order MarkovModel (backward-compatible).
    order>=2 → returns a HighOrderMarkovModel with automatic backoff.
    """
    if order <= 1:
        model = MarkovModel()
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                model.add_transition(a, b)
        model.build_matrix()
        return model

    ho_model = HighOrderMarkovModel(order=order)
    for seq in sequences:
        ho_model.add_sequence(seq)
    ho_model.build()
    return ho_model
