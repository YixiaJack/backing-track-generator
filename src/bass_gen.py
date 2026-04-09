"""Scale-constrained bass line generation using interval Markov chains,
with intensity-aware climax response.

Climax behaviour (driven by intensity_curve from analyzer):
- High intensity (>0.7): higher octave option, louder velocity, fewer rests,
  wider intervals allowed
- Low intensity (<0.3): softer, sparser, stick to root and fifth
- Medium: standard trip-hop bass groove
"""
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

    Responds to intensity curve: climax sections get louder, busier bass with
    wider register; calm sections become sparser and softer.
    """
    model = build_model_from_sequences(TRIP_HOP_BASS_PATTERNS, order=2)

    track = BassTrack(num_bars=num_bars)

    scale_pcs = set(analysis.scale_pitches) if analysis.scale_pitches else {2, 4, 5, 8, 9, 10, 1}
    intensities = _get_intensity_per_bar(analysis, num_bars)

    for bar_idx in range(num_bars):
        chord_root_pc, _ = _get_chord_for_bar(analysis, bar_idx)
        bar_intensity = intensities[bar_idx]

        # Intensity-adjusted octave: occasionally use higher octave at climax
        if bar_intensity > 0.75 and rng.random() < (bar_intensity - 0.75) * 2.0:
            octave = BASS_OCTAVE + 1  # one octave higher for punch
        else:
            octave = BASS_OCTAVE
        bass_root_midi = octave * 12 + chord_root_pc

        # Intensity-adjusted temperature: slightly hotter at climax
        bar_temp = temperature + (bar_intensity - 0.5) * 0.2

        intervals = model.generate_sequence(
            length=STEPS_PER_BAR,
            rng=rng,
            temperature=bar_temp,
            start=[0, -1],
        )

        for step_in_bar, interval in enumerate(intervals):
            global_step = bar_idx * STEPS_PER_BAR + step_in_bar

            # ── Intensity-driven rest/note bias ───────────────────
            # High intensity: convert more rests to notes (busier bass)
            if bar_intensity > 0.6 and interval == -1:
                fill_prob = (bar_intensity - 0.6) * 0.6  # up to 24%
                if rng.random() < fill_prob:
                    interval = rng.choice([0, 7, 5])  # root, fifth, or fourth

            # Low intensity: convert more notes to rests (sparser bass)
            if bar_intensity < 0.3 and interval != -1:
                thin_prob = (0.3 - bar_intensity) * 1.0  # up to 30%
                if rng.random() < thin_prob:
                    interval = -1

            # ── Bass-kick synchronisation ─────────────────────────
            if drum_track is not None and global_step < len(drum_track.kick):
                kick_here = drum_track.kick[global_step] > 0
                if interval == -1 and kick_here:
                    if rng.random() < 0.65:
                        interval = 0
                elif interval != -1 and not kick_here:
                    if rng.random() < 0.40:
                        interval = -1

            if interval == -1:
                track.pitches.append(0)
                track.velocities.append(0)
            else:
                raw_pitch = bass_root_midi + interval
                pitch = _snap_to_scale(raw_pitch, scale_pcs)

                # Range depends on intensity: wider at climax
                if bar_intensity > 0.7:
                    pitch = max(28, min(67, pitch))  # up to G4
                else:
                    pitch = max(28, min(60, pitch))  # standard E1–C4

                # Velocity: full dynamic range — ppp(35) at calm → fff(120) at climax
                vel_base = int(35 + 85 * bar_intensity)  # 35–120
                vel = vel_base + int(rng.integers(-6, 7))
                track.pitches.append(pitch)
                track.velocities.append(max(1, min(127, vel)))

    return track


def _get_chord_for_bar(analysis: Analysis, bar_idx: int) -> tuple[int, str]:
    if not analysis.chord_progression:
        return (2, "min")
    src_idx = bar_idx % len(analysis.chord_progression)
    return analysis.chord_progression[src_idx]


def _snap_to_scale(pitch: int, scale_pcs: set[int]) -> int:
    pc = pitch % 12
    if pc in scale_pcs:
        return pitch
    for offset in range(1, 7):
        if (pc + offset) % 12 in scale_pcs:
            return pitch + offset
        if (pc - offset) % 12 in scale_pcs:
            return pitch - offset
    return pitch


def _get_intensity_per_bar(analysis: Analysis, num_bars: int) -> list[float]:
    """Map intensity curve to output bars."""
    if not analysis.intensity_curve:
        return [0.5] * num_bars
    curve = analysis.intensity_curve
    result: list[float] = []
    for i in range(num_bars):
        src_idx = int(i * len(curve) / num_bars) % len(curve)
        result.append(curve[src_idx])
    return result
