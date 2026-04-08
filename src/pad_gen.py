"""Chord-based ambient pad generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.parser import Analysis

PAD_OCTAVE = 4  # MIDI octave for pad (C4 = 60)


@dataclass
class PadTrack:
    """Generated pad track — chords sustained per measure."""
    chords: List[List[int]] = field(default_factory=list)  # list of MIDI pitch lists per bar
    velocities: List[int] = field(default_factory=list)     # one velocity per bar
    durations: List[float] = field(default_factory=list)    # quarter-note lengths per bar
    num_bars: int = 0


def generate_pad(
    analysis: Analysis,
    num_bars: int,
    rng: np.random.Generator,
) -> PadTrack:
    """Generate sustained pad chords derived from the melody's harmony analysis."""
    track = PadTrack(num_bars=num_bars)
    scale_pcs = set(analysis.scale_pitches) if analysis.scale_pitches else {2, 4, 5, 8, 9, 10, 1}
    beats_per_bar = analysis.time_sig_num

    for bar_idx in range(num_bars):
        root_pc, quality = _get_chord_for_bar(analysis, bar_idx)

        # Build chord voicing
        chord_pitches = _build_voicing(root_pc, quality, scale_pcs)

        # Velocity — softer pads, slight variation
        vel = 55 + rng.integers(-5, 6)

        # Duration — whole bar with slight variance for humanisation
        dur = float(beats_per_bar) - 0.1 + rng.random() * 0.1

        track.chords.append(chord_pitches)
        track.velocities.append(max(1, min(127, vel)))
        track.durations.append(dur)

    return track


def _get_chord_for_bar(analysis: Analysis, bar_idx: int) -> Tuple[int, str]:
    """Get chord root and quality for a bar."""
    if not analysis.chord_progression:
        return (2, "min")
    src_idx = bar_idx % len(analysis.chord_progression)
    return analysis.chord_progression[src_idx]


def _build_voicing(root_pc: int, quality: str, scale_pcs: set[int]) -> List[int]:
    """Build a 3-4 note pad voicing in the PAD_OCTAVE range."""
    base = PAD_OCTAVE * 12 + root_pc

    if quality == "min":
        intervals = [0, 3, 7, 12]  # root, minor 3rd, 5th, octave
    else:
        intervals = [0, 4, 7, 12]  # root, major 3rd, 5th, octave

    pitches = []
    for iv in intervals:
        p = base + iv
        # Snap 3rd to scale if needed
        pc = p % 12
        if pc not in scale_pcs:
            for off in (1, -1, 2, -2):
                if (pc + off) % 12 in scale_pcs:
                    p += off
                    break
        pitches.append(p)

    return pitches
