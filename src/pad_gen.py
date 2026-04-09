"""Chord-based ambient pad generation — trip-hop / downtempo style.

Redesigned for warmth and atmosphere:
- Register: C3–C4 (MIDI 48–64), below melody, warm and dark
- Voicings: open voicings with 7ths/9ths, drop-2 style
- Voice leading: minimal movement between chords
- Dynamics: gentle, sustained, less randomness
- Trip-hop aesthetic: sparser at high intensity, longer holds
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Set

import numpy as np

from src.parser import Analysis

# ── MIDI dynamic marks ──────────────────────────────────────────
VEL_PPP = 20
VEL_PP = 35
VEL_P = 50
VEL_MP = 64
VEL_MF = 80
VEL_F = 96
VEL_FF = 112
VEL_FFF = 120


@dataclass
class PadNoteEvent:
    """A single note event within the pad track."""
    pitch: int
    time: float       # in quarter-note units from start
    duration: float   # in quarter-note units
    velocity: int
    is_ghost: bool = False


@dataclass
class PadTrack:
    """Generated pad track — list of individual note events."""
    events: List[PadNoteEvent] = field(default_factory=list)
    num_bars: int = 0
    beats_per_bar: int = 4

    # Legacy fields for midi_writer CC envelope generation
    chords: List[List[int]] = field(default_factory=list)
    velocities: List[int] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)


def generate_pad(
    analysis: Analysis,
    num_bars: int,
    rng: np.random.Generator,
) -> PadTrack:
    """Generate evolving pad chords below the melody, following the intensity curve."""
    scale_pcs = set(analysis.scale_pitches) if analysis.scale_pitches else {0, 2, 4, 5, 7, 9, 10}
    beats_per_bar = analysis.time_sig_num
    track = PadTrack(num_bars=num_bars, beats_per_bar=beats_per_bar)
    intensities = _get_intensity_per_bar(analysis, num_bars)

    prev_voicing: List[int] = []
    bar_idx = 0

    while bar_idx < num_bars:
        root_pc, quality = _get_chord_for_bar(analysis, bar_idx)
        avg_intensity = _avg_intensity(intensities, bar_idx, num_bars)

        # Trip-hop: hold LONGER at high intensity (anchor), change more at low
        if avg_intensity > 0.7:
            hold_bars = int(rng.integers(3, 5))   # 3-4 bars, stable
        elif avg_intensity < 0.3:
            hold_bars = int(rng.integers(1, 3))   # 1-2 bars, evolving
        else:
            hold_bars = int(rng.integers(2, 4))   # 2-3 bars
        hold_bars = min(hold_bars, num_bars - bar_idx)

        # Build voicing in warm register (C3–C4)
        voicing = _build_voicing(root_pc, quality, scale_pcs, rng, avg_intensity)

        # Gentle consonance: only remove harsh semitone clashes
        melody_pcs = set()
        for b in range(bar_idx, min(bar_idx + hold_bars, num_bars)):
            melody_pcs |= _get_melody_pcs_for_bar(analysis, b)
        voicing = _filter_gentle(voicing, melody_pcs, scale_pcs)

        # Voice leading: minimize movement from previous chord
        if prev_voicing:
            voicing = _voice_lead(prev_voicing, voicing)

        if not voicing:
            bar_idx += hold_bars
            continue

        # Emit sustained notes
        _emit_sustained(track, voicing, bar_idx, hold_bars, beats_per_bar, rng, intensities)

        prev_voicing = voicing
        bar_idx += hold_bars

    return track


# ── Voicing construction (warm low register) ───────────────────

def _build_voicing(
    root_pc: int, quality: str, scale_pcs: Set[int],
    rng: np.random.Generator, intensity: float,
) -> List[int]:
    """Build a warm pad voicing in C3–C4 range (MIDI 48–64).

    Uses open voicings with extensions (7ths, 9ths) for trip-hop atmosphere.
    Sparser at high intensity (trip-hop aesthetic).
    """
    # Root in octave 3 (C3=48)
    base = 48 + root_pc
    third = 3 if quality == "min" else 4
    seventh = 10 if quality == "min" else 11  # minor 7th / major 7th

    if intensity > 0.7:
        # High intensity: sparse — root + fifth only, dark and open
        roll = rng.random()
        if roll < 0.5:
            intervals = [0, 7]              # root + P5
        else:
            intervals = [0, 7, 12]          # root + P5 + octave
    elif intensity < 0.3:
        # Low intensity: richer — add 7th or 9th for color
        roll = rng.random()
        if roll < 0.35:
            intervals = [0, third, 7, seventh]          # full 7th chord
        elif roll < 0.65:
            intervals = [0, 7, seventh, 12 + 2]         # root, 5th, 7th, 9th (drop-2)
        else:
            intervals = [0, third, 7, 12 + 2]           # root, 3rd, 5th, 9th
    else:
        # Medium: triad + occasional 7th
        roll = rng.random()
        if roll < 0.4:
            intervals = [0, third, 7]                    # simple triad
        elif roll < 0.7:
            intervals = [0, seventh, 12 + third]         # root, 7th, 10th (open)
        else:
            intervals = [0, third, 7, seventh]           # 7th chord

    pitches = []
    for iv in intervals:
        p = base + iv
        p = _snap_to_scale(p, scale_pcs)
        # Enforce warm range: C3(48) to E4(64)
        if 45 <= p <= 64:
            pitches.append(p)

    if not pitches:
        pitches = [_snap_to_scale(base, scale_pcs)]

    return sorted(set(pitches))


def _snap_to_scale(pitch: int, scale_pcs: Set[int]) -> int:
    pc = pitch % 12
    if pc in scale_pcs:
        return pitch
    for off in (1, -1, 2, -2):
        if (pc + off) % 12 in scale_pcs:
            return pitch + off
    return pitch


# ── Gentle consonance filter ───────────────────────────────────

def _filter_gentle(
    voicing: List[int], melody_pcs: Set[int], scale_pcs: Set[int]
) -> List[int]:
    """Only remove notes that create harsh semitone clashes with melody.

    Keeps diatonic dissonance (2nds, tritones) for atmospheric color.
    Much less aggressive than before — preserves chord richness.
    """
    if not melody_pcs:
        return voicing

    result: List[int] = []
    for p in voicing:
        pc = p % 12
        # Only reject if semitone (1 or 11) away from a melody note AND not in scale
        semitone_clash = any(abs(pc - m_pc) % 12 in (1, 11) for m_pc in melody_pcs)
        if semitone_clash and pc not in scale_pcs:
            continue
        result.append(p)

    # Always keep at least root + one other note
    if len(result) < 2 and len(voicing) >= 2:
        result = voicing[:2]
    elif not result:
        result = [voicing[0]]

    return result


# ── Voice leading ──────────────────────────────────────────────

def _voice_lead(prev: List[int], target: List[int]) -> List[int]:
    """Move each voice to nearest pitch in target's pitch-class set.

    Prioritizes common tones (no movement) and stepwise motion.
    """
    target_pcs = [p % 12 for p in target]

    # If different number of voices, just use target directly
    if len(prev) != len(target):
        return target

    result: List[int] = []
    for prev_pitch in prev:
        best_pitch = prev_pitch
        best_dist = 999
        # Check common tone first
        if prev_pitch % 12 in target_pcs:
            result.append(prev_pitch)
            continue
        # Find nearest target pitch class
        for offset in range(-7, 8):
            candidate = prev_pitch + offset
            if candidate % 12 in target_pcs and 45 <= candidate <= 64:
                if abs(offset) < best_dist:
                    best_dist = abs(offset)
                    best_pitch = candidate
        result.append(best_pitch)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for p in result:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return sorted(unique)


# ── Sustained note emission ────────────────────────────────────

def _emit_sustained(
    track: PadTrack,
    voicing: List[int],
    start_bar: int,
    hold_bars: int,
    beats_per_bar: int,
    rng: np.random.Generator,
    intensities: List[float],
) -> None:
    """Emit long sustained notes — minimal articulation, smooth and continuous.

    Trip-hop pads: long notes, gentle velocity, subtle swell, no busy re-attacks.
    """
    total_duration = hold_bars * beats_per_bar
    start_time = start_bar * beats_per_bar
    bar_intensity = intensities[start_bar] if start_bar < len(intensities) else 0.5

    # Velocity: warm range, not too loud, not too soft
    # intensity 0→mp(64), 0.5→mf(80), 1.0→f(96)
    base_vel = int(VEL_MP + (VEL_F - VEL_MP) * bar_intensity)

    for i, pitch in enumerate(voicing):
        # Gentle stagger on chord changes: 0.0 to 0.3 beats max
        stagger = i * rng.uniform(0.05, 0.15)
        note_start = start_time + stagger
        note_dur = total_duration - stagger - 0.1  # slight gap before next chord

        # Per-voice velocity: subtle variation only
        vel = base_vel + int(rng.integers(-4, 5))
        vel = max(VEL_P, min(VEL_F, vel))

        track.events.append(PadNoteEvent(
            pitch=pitch, time=note_start, duration=max(0.5, note_dur), velocity=vel,
        ))

    # One subtle ghost re-attack per hold (low probability, gentle)
    if hold_bars >= 3 and rng.random() < 0.25:
        ghost_bar = start_bar + int(rng.integers(1, hold_bars))
        ghost_time = ghost_bar * beats_per_bar + rng.uniform(0.5, 1.5)
        ghost_pitch = voicing[int(rng.integers(len(voicing)))]
        ghost_vel = max(VEL_PPP, base_vel - 25 + int(rng.integers(-3, 4)))
        track.events.append(PadNoteEvent(
            pitch=ghost_pitch, time=ghost_time,
            duration=rng.uniform(1.0, 2.0), velocity=ghost_vel, is_ghost=True,
        ))

    # Legacy fields for CC envelopes in midi_writer
    for h in range(hold_bars):
        track.chords.append(voicing[:])
        track.velocities.append(base_vel)
        track.durations.append(float(beats_per_bar))


# ── Helpers ────────────────────────────────────────────────────

def _get_chord_for_bar(analysis: Analysis, bar_idx: int) -> Tuple[int, str]:
    if not analysis.chord_progression:
        return (2, "min")
    src_idx = bar_idx % len(analysis.chord_progression)
    return analysis.chord_progression[src_idx]


def _get_melody_pcs_for_bar(analysis: Analysis, bar_idx: int) -> Set[int]:
    """Get melody pitch classes for a given bar."""
    num_bars = analysis.num_measures
    src_m = bar_idx % num_bars + 1
    return {e.pitch % 12 for e in analysis.note_events
            if e.measure == src_m and e.pitch is not None}


def _get_intensity_per_bar(analysis: Analysis, num_bars: int) -> List[float]:
    if not analysis.intensity_curve:
        return [0.5] * num_bars
    curve = analysis.intensity_curve
    result: List[float] = []
    for i in range(num_bars):
        src_idx = int(i * len(curve) / num_bars) % len(curve)
        result.append(curve[src_idx])
    return result


def _avg_intensity(intensities: List[float], start: int, total: int) -> float:
    end = min(start + 4, total)
    if start >= len(intensities):
        return 0.5
    segment = intensities[start:end]
    return sum(segment) / len(segment) if segment else 0.5
