"""Trip-hop drum pattern generation with positional priors and density modulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.markov import MarkovModel, build_model_from_sequences
from src.parser import Analysis
from src.training_data import TRIP_HOP_DRUM_PATTERNS

STEPS_PER_BAR = 16  # 16th-note resolution


@dataclass
class DrumTrack:
    """Generated drum track — per-step hits for each instrument."""
    kick: List[int] = field(default_factory=list)    # 0 or velocity
    snare: List[int] = field(default_factory=list)
    hihat: List[int] = field(default_factory=list)
    steps_per_bar: int = STEPS_PER_BAR
    num_bars: int = 0


def generate_drums(
    analysis: Analysis,
    num_bars: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    density_scale: float = 1.0,
) -> DrumTrack:
    """Generate a drum track using a Markov Chain trained on trip-hop patterns.

    The rhythmic density from the analysis modulates pattern density:
    sparse melody sections get sparser drums, dense sections get busier drums.
    """
    # Build Markov model from training patterns
    # Encode each step as a tuple (kick, snare, hihat) — the state space
    sequences = [[(k, s, h) for k, s, h in pat] for pat in TRIP_HOP_DRUM_PATTERNS]
    model = build_model_from_sequences(sequences)

    track = DrumTrack(num_bars=num_bars)

    # Normalised density per measure (0.0 – 1.0)
    densities = _get_density_per_bar(analysis, num_bars)

    for bar_idx in range(num_bars):
        bar_density = densities[bar_idx] * density_scale

        # Generate 16 steps for this bar via Markov chain
        bar_steps: List[Tuple[int, int, int]] = model.generate_sequence(
            length=STEPS_PER_BAR,
            rng=rng,
            temperature=temperature,
            start=(1, 0, 1),  # start with kick + hihat
        )

        for step_idx, (k, s, h) in enumerate(bar_steps):
            # Density gating: with lower density, randomly mute some hits
            gate = rng.random()
            if gate > (bar_density * 0.6 + 0.4):
                # Mute non-essential hits (keep structural kick on 1, snare on 9)
                if step_idx not in (0, 8):
                    k, s, h = 0, 0, min(h, 1)

            # Velocity humanisation
            vel_base = 90
            vel_var = rng.integers(-10, 11)
            kick_vel = (vel_base + 15 + vel_var) if k else 0
            snare_vel = (vel_base + 5 + vel_var) if s else 0
            hat_vel = (vel_base - 10 + vel_var) if h else 0

            track.kick.append(max(0, min(127, kick_vel)))
            track.snare.append(max(0, min(127, snare_vel)))
            track.hihat.append(max(0, min(127, hat_vel)))

    return track


def _get_density_per_bar(analysis: Analysis, num_bars: int) -> List[float]:
    """Map analysis rhythmic density to 0-1 range for each output bar."""
    if not analysis.rhythmic_density:
        return [0.5] * num_bars

    max_d = max(analysis.rhythmic_density) or 1.0
    normed = [d / max_d for d in analysis.rhythmic_density]

    # Stretch or tile to fit requested num_bars
    result: List[float] = []
    for i in range(num_bars):
        src_idx = int(i * len(normed) / num_bars) % len(normed)
        result.append(normed[src_idx])
    return result
