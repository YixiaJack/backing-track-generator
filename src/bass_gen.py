"""Scale-constrained bass line generation using interval Markov chains."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from src.markov import build_model_from_sequences
from src.parser import Analysis
from src.training_data import TRIP_HOP_BASS_PATTERNS

if TYPE_CHECKING:
    from src.drum_gen import DrumTrack

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
    drum_track: Optional["DrumTrack"] = None,
) -> BassTrack:
    """Generate a bass line from interval Markov chain, constrained to the scale.

    When drum_track is provided, bass notes are biased toward kick positions:
    steps where the kick hits are more likely to play a note, and steps where
    the kick is silent are more likely to rest.  This bass-kick lock is a
    defining characteristic of trip-hop production.
    """
    # Build second-order interval Markov model for more coherent bass lines.
    # Using order=2 means intervals depend on the previous two intervals,
    # producing smoother melodic contours with automatic backoff.
    model = build_model_from_sequences(TRIP_HOP_BASS_PATTERNS, order=2)

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
            start=[0, -1],  # start on root then rest — typical trip-hop bass opening
        )

        for step_in_bar, interval in enumerate(intervals):
            global_step = bar_idx * STEPS_PER_BAR + step_in_bar

            # Bass-kick synchronisation: bias rests/notes toward kick pattern
            if drum_track is not None and global_step < len(drum_track.kick):
                kick_here = drum_track.kick[global_step] > 0
                if interval == -1 and kick_here:
                    # Kick hit but Markov chose rest → play root with 65% probability
                    if rng.random() < 0.65:
                        interval = 0
                elif interval != -1 and not kick_here:
                    # No kick but Markov chose note → convert to rest with 40% probability
                    if rng.random() < 0.40:
                        interval = -1

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
