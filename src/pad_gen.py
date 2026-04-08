"""Chord-based ambient pad generation with voice leading, staggered entry,
and internal movement — inspired by Massive Attack / Portishead pad textures.

Key techniques (from algorithmic composition research & trip-hop production):
- Voice leading via minimal pitch movement between chords
- Staggered note entry/exit (lower voices first) to avoid block-chord stiffness
- Ghost re-attacks: periodic re-triggering of individual voices at low velocity
- Tone subtraction/addition: randomly muting or adding a tension tone
- Per-voice velocity curves with phase-offset sine LFOs
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.parser import Analysis

PAD_OCTAVE = 4  # MIDI octave for pad (C4 = 60)

# Chord hold duration in bars
_MIN_HOLD_BARS = 2
_MAX_HOLD_BARS = 4


@dataclass
class PadNoteEvent:
    """A single note event within the pad track."""
    pitch: int
    time: float       # in quarter-note units from start
    duration: float   # in quarter-note units
    velocity: int
    is_ghost: bool = False  # ghost re-attack (low-velocity re-trigger)


@dataclass
class PadTrack:
    """Generated pad track — list of individual note events."""
    events: List[PadNoteEvent] = field(default_factory=list)
    num_bars: int = 0
    beats_per_bar: int = 4

    # Legacy fields kept for midi_writer CC envelope generation
    chords: List[List[int]] = field(default_factory=list)
    velocities: List[int] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)


def generate_pad(
    analysis: Analysis,
    num_bars: int,
    rng: np.random.Generator,
) -> PadTrack:
    """Generate evolving pad chords with voice leading, staggered entry, and internal motion."""
    scale_pcs = set(analysis.scale_pitches) if analysis.scale_pitches else {2, 4, 5, 8, 9, 10, 1}
    beats_per_bar = analysis.time_sig_num
    track = PadTrack(num_bars=num_bars, beats_per_bar=beats_per_bar)

    prev_voicing: List[int] = []
    bar_idx = 0

    while bar_idx < num_bars:
        root_pc, quality = _get_chord_for_bar(analysis, bar_idx)

        # How many bars to hold this chord
        hold_bars = int(rng.integers(_MIN_HOLD_BARS, _MAX_HOLD_BARS + 1))
        hold_bars = min(hold_bars, num_bars - bar_idx)

        # Build target voicing
        raw_voicing = _build_voicing(root_pc, quality, scale_pcs, rng)

        # Apply voice leading: move each voice to nearest pitch in target
        if prev_voicing:
            voicing = _voice_lead(prev_voicing, raw_voicing)
        else:
            voicing = raw_voicing

        # Generate note events for this chord hold
        _emit_chord_hold(
            track, voicing, bar_idx, hold_bars, beats_per_bar, rng, scale_pcs
        )

        prev_voicing = voicing
        bar_idx += hold_bars

    return track


# ── Voice leading ────────────────────────────────────────────────────

def _voice_lead(prev: List[int], target: List[int]) -> List[int]:
    """Move each voice in `prev` to the nearest pitch in `target`'s pitch-class set.

    This produces smooth voice leading where each voice moves by the
    smallest possible interval, a technique formalised by Tymoczko (2006).
    """
    target_pcs = [p % 12 for p in target]
    result: List[int] = []

    for prev_pitch in prev:
        best_pitch = prev_pitch
        best_dist = 999
        # Search within ±6 semitones for the nearest target pitch class
        for offset in range(-6, 7):
            candidate = prev_pitch + offset
            if candidate % 12 in target_pcs:
                if abs(offset) < best_dist:
                    best_dist = abs(offset)
                    best_pitch = candidate
        result.append(best_pitch)

    # Remove duplicates (two voices landing on same pitch) — shift one up an octave
    seen = set()
    for i, p in enumerate(result):
        while p in seen:
            p += 12
        seen.add(p)
        result[i] = p

    return sorted(result)


# ── Chord hold emission ─────────────────────────────────────────────

def _emit_chord_hold(
    track: PadTrack,
    voicing: List[int],
    start_bar: int,
    hold_bars: int,
    beats_per_bar: int,
    rng: np.random.Generator,
    scale_pcs: set[int],
) -> None:
    """Emit all note events for a chord held over several bars.

    Techniques applied:
    - Staggered entry on first bar (voices enter one by one)
    - Per-voice velocity envelopes (sine LFO with phase offset)
    - Ghost re-attacks within sustained bars
    - Occasional tone subtraction/addition
    """
    total_beats = hold_bars * beats_per_bar
    bar_time = start_bar * beats_per_bar

    num_voices = len(voicing)

    # ── Per-voice parameters ──────────────────────────────────────
    # Phase offsets for velocity LFO (different per voice → movement)
    vel_phases = [rng.uniform(0, 2 * math.pi) for _ in range(num_voices)]
    vel_period = rng.uniform(3.0, 6.0)  # LFO period in bars

    for hold_i in range(hold_bars):
        current_bar = start_bar + hold_i
        current_bar_time = current_bar * beats_per_bar
        is_first_bar = (hold_i == 0)
        is_last_bar = (hold_i == hold_bars - 1)

        # Decide which voices are active (tone subtraction)
        active_voices = list(range(num_voices))
        if not is_first_bar and num_voices >= 3 and rng.random() < 0.15:
            # Drop one non-root voice for a bar (tone subtraction)
            drop_idx = int(rng.integers(1, num_voices))
            active_voices = [v for v in active_voices if v != drop_idx]

        # Occasional tension tone addition (add a 9th or 11th)
        extra_pitch = None
        if not is_first_bar and rng.random() < 0.10:
            root = voicing[0]
            tension_intervals = [14, 17, 10]  # 9th, 11th, minor 7th
            iv = tension_intervals[int(rng.integers(len(tension_intervals)))]
            candidate = root + iv
            candidate = _snap_to_scale(candidate, scale_pcs)
            if candidate not in voicing:
                extra_pitch = candidate

        for voice_idx in active_voices:
            pitch = voicing[voice_idx]

            # ── Staggered entry on first bar ──────────────────────
            if is_first_bar:
                # Lower voices enter first, each offset by ~0.15-0.3 beats
                stagger = voice_idx * rng.uniform(0.12, 0.30)
                note_start = current_bar_time + stagger
                note_dur = float(beats_per_bar) - stagger - 0.05
            else:
                note_start = current_bar_time
                note_dur = float(beats_per_bar) - 0.05

            # ── Staggered exit on last bar ────────────────────────
            if is_last_bar:
                # Upper voices release first
                release_offset = (num_voices - 1 - voice_idx) * rng.uniform(0.1, 0.25)
                note_dur -= release_offset

            note_dur = max(0.5, note_dur)

            # ── Per-voice velocity envelope (sine LFO) ────────────
            phase = vel_phases[voice_idx]
            lfo_pos = (hold_i / max(hold_bars, 1)) * vel_period * 2 * math.pi + phase
            # Velocity range: 42-65, with gentle swell
            base_vel = 50 + int(12 * math.sin(lfo_pos))
            # Crescendo across hold
            base_vel += hold_i * 2
            vel = max(35, min(80, base_vel + int(rng.integers(-3, 4))))

            track.events.append(PadNoteEvent(
                pitch=pitch,
                time=note_start,
                duration=note_dur,
                velocity=vel,
            ))

        # ── Extra tension tone ────────────────────────────────────
        if extra_pitch is not None:
            # Tension tone enters mid-bar, shorter duration
            t_start = current_bar_time + rng.uniform(1.0, 2.0)
            t_dur = rng.uniform(1.5, float(beats_per_bar) - 1.0)
            t_vel = max(30, min(55, 40 + int(rng.integers(-5, 6))))
            track.events.append(PadNoteEvent(
                pitch=extra_pitch,
                time=t_start,
                duration=t_dur,
                velocity=t_vel,
            ))

        # ── Ghost re-attacks (internal motion) ────────────────────
        # On non-first bars, periodically re-trigger one voice at low velocity
        if not is_first_bar and rng.random() < 0.40:
            ghost_voice = int(rng.integers(num_voices))
            ghost_pitch = voicing[ghost_voice]
            # Ghost appears on beat 2 or 3
            ghost_beat = rng.choice([1.0, 2.0, 2.5])
            ghost_time = current_bar_time + ghost_beat + rng.uniform(-0.05, 0.05)
            ghost_vel = int(rng.integers(25, 42))
            ghost_dur = rng.uniform(0.5, 1.5)

            track.events.append(PadNoteEvent(
                pitch=ghost_pitch,
                time=ghost_time,
                duration=ghost_dur,
                velocity=ghost_vel,
                is_ghost=True,
            ))

        # Populate legacy fields for CC envelope generation in midi_writer
        track.chords.append(voicing[:])
        track.velocities.append(vel if active_voices else 50)
        track.durations.append(float(beats_per_bar))


# ── Chord voicing ────────────────────────────────────────────────────

def _get_chord_for_bar(analysis: Analysis, bar_idx: int) -> Tuple[int, str]:
    if not analysis.chord_progression:
        return (2, "min")
    src_idx = bar_idx % len(analysis.chord_progression)
    return analysis.chord_progression[src_idx]


_VOICING_TYPES = {
    "triad_min":  [0, 3, 7, 12],
    "triad_maj":  [0, 4, 7, 12],
    "add9_min":   [0, 3, 7, 14],
    "add9_maj":   [0, 4, 7, 14],
    "sus2":       [0, 2, 7, 12],
    "sus4":       [0, 5, 7, 12],
    "min7":       [0, 3, 7, 10],
    "open_5th":   [-12, 0, 7, 12],
}


def _build_voicing(
    root_pc: int, quality: str, scale_pcs: set[int], rng: np.random.Generator
) -> List[int]:
    """Build a pad voicing — 70% standard triad, 30% extended voicing."""
    base = PAD_OCTAVE * 12 + root_pc

    if rng.random() < 0.30:
        extended_options = ["add9_min", "add9_maj", "sus2", "sus4", "min7", "open_5th"]
        vtype = extended_options[int(rng.integers(len(extended_options)))]
        intervals = _VOICING_TYPES[vtype]
    else:
        key = "triad_min" if quality == "min" else "triad_maj"
        intervals = _VOICING_TYPES[key]

    pitches = []
    for iv in intervals:
        p = base + iv
        p = _snap_to_scale(p, scale_pcs)
        pitches.append(p)

    return sorted(pitches)


def _snap_to_scale(pitch: int, scale_pcs: set[int]) -> int:
    pc = pitch % 12
    if pc in scale_pcs:
        return pitch
    for off in (1, -1, 2, -2):
        if (pc + off) % 12 in scale_pcs:
            return pitch + off
    return pitch
