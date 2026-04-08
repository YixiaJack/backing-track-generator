"""Chord-based ambient pad generation with rich voicings and slow evolution.

Trip-hop pads sustain chords over multiple bars with slow CC envelopes,
using extended voicings (sus2, sus4, add9) for atmospheric colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.parser import Analysis

PAD_OCTAVE = 4  # MIDI octave for pad (C4 = 60)

# Chord hold duration in bars — trip-hop uses "big block" harmonies
_MIN_HOLD_BARS = 2
_MAX_HOLD_BARS = 4


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
    """Generate sustained pad chords with extended voicings held over multiple bars."""
    track = PadTrack(num_bars=num_bars)
    scale_pcs = set(analysis.scale_pitches) if analysis.scale_pitches else {2, 4, 5, 8, 9, 10, 1}
    beats_per_bar = analysis.time_sig_num

    bar_idx = 0
    while bar_idx < num_bars:
        root_pc, quality = _get_chord_for_bar(analysis, bar_idx)

        # Determine how many bars to hold this chord (2-4 bars, trip-hop style)
        hold_bars = rng.integers(_MIN_HOLD_BARS, _MAX_HOLD_BARS + 1)
        hold_bars = min(hold_bars, num_bars - bar_idx)

        # Pick a voicing type — occasionally use extended voicings for colour
        chord_pitches = _build_voicing(root_pc, quality, scale_pcs, rng)

        for hold_i in range(hold_bars):
            # Velocity — softer pads, slight swell across the hold
            base_vel = 50 + hold_i * 3  # gentle crescendo across held bars
            vel = base_vel + rng.integers(-4, 5)

            # Duration — whole bar with slight variance for humanisation
            dur = float(beats_per_bar) - 0.05 + rng.random() * 0.05

            track.chords.append(chord_pitches)
            track.velocities.append(max(1, min(127, vel)))
            track.durations.append(dur)

        bar_idx += hold_bars

    return track


def _get_chord_for_bar(analysis: Analysis, bar_idx: int) -> Tuple[int, str]:
    """Get chord root and quality for a bar."""
    if not analysis.chord_progression:
        return (2, "min")
    src_idx = bar_idx % len(analysis.chord_progression)
    return analysis.chord_progression[src_idx]


# Voicing types with their interval sets (semitones from root)
_VOICING_TYPES = {
    "triad_min":  [0, 3, 7, 12],       # root, minor 3rd, 5th, octave
    "triad_maj":  [0, 4, 7, 12],       # root, major 3rd, 5th, octave
    "add9_min":   [0, 3, 7, 14],       # root, minor 3rd, 5th, 9th
    "add9_maj":   [0, 4, 7, 14],       # root, major 3rd, 5th, 9th
    "sus2":       [0, 2, 7, 12],       # root, 2nd, 5th, octave
    "sus4":       [0, 5, 7, 12],       # root, 4th, 5th, octave
    "min7":       [0, 3, 7, 10],       # root, minor 3rd, 5th, minor 7th
    "open_5th":   [-12, 0, 7, 12],     # low root, root, 5th, octave (power voicing)
}


def _build_voicing(
    root_pc: int, quality: str, scale_pcs: set[int], rng: np.random.Generator
) -> List[int]:
    """Build a pad voicing with occasional extended chord colours.

    70% standard triad, 30% extended voicing (sus2/sus4/add9/min7/open_5th).
    """
    base = PAD_OCTAVE * 12 + root_pc

    if rng.random() < 0.30:
        # Extended voicing for atmospheric colour
        extended_options = ["add9_min", "add9_maj", "sus2", "sus4", "min7", "open_5th"]
        vtype = extended_options[rng.integers(len(extended_options))]
        intervals = _VOICING_TYPES[vtype]
    else:
        # Standard triad
        key = "triad_min" if quality == "min" else "triad_maj"
        intervals = _VOICING_TYPES[key]

    pitches = []
    for iv in intervals:
        p = base + iv
        # Snap to scale if needed
        pc = p % 12
        if pc not in scale_pcs:
            for off in (1, -1, 2, -2):
                if (pc + off) % 12 in scale_pcs:
                    p += off
                    break
        pitches.append(p)

    return pitches
