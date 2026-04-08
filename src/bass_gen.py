"""Scale-constrained bass line generation using interval Markov chains."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.markov import build_model_from_sequences
from src.parser import Analysis
from src.training_data import TRIP_HOP_BASS_PATTERNS

STEPS_PER_BAR = 16
BASS_OCTAVE = 2  # MIDI octave for bass (C2 = 36)


@dataclass
class BassTrack:
    """Generated bass track — MIDI pitches and velocities per step."""
    pitches: List[int] = field(default_factory=list)      # MIDI pitch, 0 = rest
    velocities: List[int] = field(default_factory=list)
    steps_per_bar: int = STEPS_PER_BAR
    num_bars: int = 0


def generate_bass(
    analysis: Analysis,
    num_bars: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> BassTrack:
    """Generate a bass line from interval Markov chain, constrained to the scale."""
    # Build interval Markov model from training patterns
    model = build_model_from_sequences(TRIP_HOP_BASS_PATTERNS)

    track = BassTrack(num_bars=num_bars)

    # D Hungarian Minor scale pitch classes
    scale_pcs = set(analysis.scale_pitches) if analysis.scale_pitches else {2, 4, 5, 8, 9, 10, 1}

    for bar_idx in range(num_bars):
        # Get chord root for this bar
        chord_root_pc, _ = _get_chord_for_bar(analysis, bar_idx)
        bass_root_midi = BASS_OCTAVE * 12 + chord_root_pc  # e.g. D2 = 38

        # Generate interval sequence for this bar
        intervals = model.generate_sequence(
            length=STEPS_PER_BAR,
            rng=rng,
            temperature=temperature,
            start=0,  # start on root
        )

        for interval in intervals:
            if interval == -1:
                # Rest
                track.pitches.append(0)
                track.velocities.append(0)
            else:
                raw_pitch = bass_root_midi + interval
                # Constrain to scale
                pitch = _snap_to_scale(raw_pitch, scale_pcs)
                # Keep in bass range (28–60 = E1–C4)
                pitch = max(28, min(60, pitch))
                vel = 80 + rng.integers(-8, 9)
                track.pitches.append(pitch)
                track.velocities.append(max(1, min(127, vel)))

    return track


def _get_chord_for_bar(analysis: Analysis, bar_idx: int) -> tuple[int, str]:
    """Get the chord root pitch class for a given output bar."""
    if not analysis.chord_progression:
        return (2, "min")  # default D minor
    src_idx = bar_idx % len(analysis.chord_progression)
    return analysis.chord_progression[src_idx]


def _snap_to_scale(pitch: int, scale_pcs: set[int]) -> int:
    """Snap a MIDI pitch to the nearest pitch in the scale."""
    pc = pitch % 12
    if pc in scale_pcs:
        return pitch
    # Search up and down for nearest scale tone
    for offset in range(1, 7):
        if (pc + offset) % 12 in scale_pcs:
            return pitch + offset
        if (pc - offset) % 12 in scale_pcs:
            return pitch - offset
    return pitch
