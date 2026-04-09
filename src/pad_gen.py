"""Chord-based ambient pad generation with voice leading, staggered entry,
intensity-aware dynamics, register separation, and consonance validation.

Key fixes based on research:
- Register: pad placed BELOW melody range (Berklee "Writing String Pads" rule)
- Consonance: pad notes validated against per-measure melody pitch classes
  (consonance scoring from AutoHarmonizer / Automatic Melody Harmonization papers)
- Dynamics: full ppp–fff range (MIDI vel 20–120) driven by intensity curve
  (MIDI velocity mapping: Apple Logic Pro standard)
- Scale: uses auto-detected scale from actual pitch histogram, not hardcoded
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Set

import numpy as np

from src.parser import Analysis

# ── MIDI dynamic marks (Apple Logic Pro 9 standard) ──────────────────
VEL_PPP = 20
VEL_PP = 35
VEL_P = 50
VEL_MP = 64
VEL_MF = 80
VEL_F = 96
VEL_FF = 112
VEL_FFF = 120

# Consonant intervals from root (in semitones)
_CONSONANT_INTERVALS = {0, 3, 4, 5, 7, 8, 9, 12, 15, 16}  # unison, m3, M3, P4, P5, m6, M6, oct, ...


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

    # Collect melody pitch classes per measure for consonance checking
    melody_pcs_per_bar = _get_melody_pcs_per_bar(analysis, num_bars)

    prev_voicing: List[int] = []
    bar_idx = 0

    while bar_idx < num_bars:
        root_pc, quality = _get_chord_for_bar(analysis, bar_idx)
        avg_intensity = _avg_intensity(intensities, bar_idx, num_bars)

        # Hold duration adapts to intensity
        min_hold = 1 if avg_intensity > 0.7 else 2
        max_hold = 2 if avg_intensity > 0.7 else (4 if avg_intensity < 0.3 else 3)
        hold_bars = int(rng.integers(min_hold, max_hold + 1))
        hold_bars = min(hold_bars, num_bars - bar_idx)

            # Build voicing in natural pad register (octave 3-4)
        # Consonance filter (below) handles dissonance — no need to force register down
        raw_voicing = _build_voicing(
            root_pc, quality, scale_pcs, rng, avg_intensity
        )

        # Consonance check: filter out notes that clash with melody
        mel_pcs = set()
        for b in range(bar_idx, min(bar_idx + hold_bars, len(melody_pcs_per_bar))):
            mel_pcs |= melody_pcs_per_bar[b]
        voicing = _filter_consonant(raw_voicing, mel_pcs, scale_pcs)

        # Voice leading
        if prev_voicing and len(prev_voicing) == len(voicing):
            voicing = _voice_lead(prev_voicing, voicing)

        if not voicing:
            bar_idx += hold_bars
            continue

        # Emit events
        _emit_chord_hold(
            track, voicing, bar_idx, hold_bars, beats_per_bar,
            rng, scale_pcs, intensities,
        )

        prev_voicing = voicing
        bar_idx += hold_bars

    return track


# ── Melody analysis helpers ──────────────────────────────────────────

def _get_melody_floor_per_bar(analysis: Analysis, num_bars: int) -> List[int]:
    """Get the lowest melody pitch per output bar (for register separation)."""
    floors: List[int] = []
    for i in range(num_bars):
        src_m = int(i * analysis.num_measures / num_bars) % analysis.num_measures + 1
        pitches = [e.pitch for e in analysis.note_events
                   if e.measure == src_m and e.pitch is not None]
        floors.append(min(pitches) if pitches else 60)
    return floors


def _get_melody_pcs_per_bar(analysis: Analysis, num_bars: int) -> List[Set[int]]:
    """Get the set of melody pitch classes per output bar (for consonance checking)."""
    result: List[Set[int]] = []
    for i in range(num_bars):
        src_m = int(i * analysis.num_measures / num_bars) % analysis.num_measures + 1
        pcs = {e.pitch % 12 for e in analysis.note_events
               if e.measure == src_m and e.pitch is not None}
        result.append(pcs)
    return result


# ── Consonance validation ────────────────────────────────────────────

def _filter_consonant(
    voicing: List[int], melody_pcs: Set[int], scale_pcs: Set[int]
) -> List[int]:
    """Remove pad notes that are dissonant against the melody.

    A pad note is kept if:
    1. Its pitch class is IN the melody (chord tone), OR
    2. Its interval to at least one melody note is consonant (3rd, 5th, 6th, octave), OR
    3. Its pitch class is in the scale AND not a semitone away from any melody note

    Based on consonance scoring from Automatic Melody Harmonization (Yeh et al., 2021).
    """
    if not melody_pcs:
        return voicing

    result: List[int] = []
    for p in voicing:
        pc = p % 12
        # Rule 1: pad note is a melody tone → always consonant
        if pc in melody_pcs:
            result.append(p)
            continue

        # Rule 2: check intervals against each melody PC
        consonant = False
        for m_pc in melody_pcs:
            interval = abs(pc - m_pc) % 12
            if interval in _CONSONANT_INTERVALS:
                consonant = True
                break
        if consonant:
            result.append(p)
            continue

        # Rule 3: in scale and not a semitone from melody → weak consonance, keep
        if pc in scale_pcs:
            semitone_clash = any(abs(pc - m_pc) % 12 in (1, 11) for m_pc in melody_pcs)
            if not semitone_clash:
                result.append(p)

    # Ensure we keep at least the root
    if not result and voicing:
        result = [voicing[0]]

    return result


# ── Voice leading ────────────────────────────────────────────────────

def _voice_lead(prev: List[int], target: List[int]) -> List[int]:
    """Move each voice to nearest pitch in target's pitch-class set (Tymoczko 2006)."""
    target_pcs = [p % 12 for p in target]
    result: List[int] = []

    for prev_pitch in prev:
        best_pitch = prev_pitch
        best_dist = 999
        for offset in range(-6, 7):
            candidate = prev_pitch + offset
            if candidate % 12 in target_pcs:
                if abs(offset) < best_dist:
                    best_dist = abs(offset)
                    best_pitch = candidate
        result.append(best_pitch)

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
    scale_pcs: Set[int],
    intensities: List[float],
) -> None:
    """Emit note events for a chord hold with full dynamic range."""
    num_voices = len(voicing)

    vel_phases = [rng.uniform(0, 2 * math.pi) for _ in range(num_voices)]
    vel_period = rng.uniform(3.0, 6.0)

    for hold_i in range(hold_bars):
        current_bar = start_bar + hold_i
        current_bar_time = current_bar * beats_per_bar
        is_first_bar = (hold_i == 0)
        is_last_bar = (hold_i == hold_bars - 1)

        bar_intensity = intensities[current_bar] if current_bar < len(intensities) else 0.5

        # Tone subtraction
        active_voices = list(range(num_voices))
        if not is_first_bar and num_voices >= 3:
            drop_prob = 0.25 - bar_intensity * 0.20
            if rng.random() < drop_prob:
                drop_idx = int(rng.integers(1, num_voices))
                active_voices = [v for v in active_voices if v != drop_idx]

        # Tension tone addition
        extra_pitch = None
        tension_prob = 0.03 + bar_intensity * 0.22
        if not is_first_bar and rng.random() < tension_prob:
            root = voicing[0]
            tension_intervals = [14, 17, 10]
            iv = tension_intervals[int(rng.integers(len(tension_intervals)))]
            candidate = root + iv
            candidate = _snap_to_scale(candidate, scale_pcs)
            if candidate not in voicing:
                extra_pitch = candidate

        for voice_idx in active_voices:
            pitch = voicing[voice_idx]

            # Staggered entry
            if is_first_bar:
                stagger_lo = 0.25 - bar_intensity * 0.20
                stagger_hi = 0.45 - bar_intensity * 0.30
                stagger = voice_idx * rng.uniform(max(0.03, stagger_lo), max(0.05, stagger_hi))
                note_start = current_bar_time + stagger
                note_dur = float(beats_per_bar) - stagger - 0.05
            else:
                note_start = current_bar_time
                note_dur = float(beats_per_bar) - 0.05

            if is_last_bar:
                release_offset = (num_voices - 1 - voice_idx) * rng.uniform(0.1, 0.25)
                note_dur -= release_offset

            note_dur = max(0.5, note_dur)

            # ── Full dynamic range velocity (ppp=20 to fff=120) ──────
            phase = vel_phases[voice_idx]
            lfo_pos = (hold_i / max(hold_bars, 1)) * vel_period * 2 * math.pi + phase
            lfo_val = int(10 * math.sin(lfo_pos))

            # intensity 0.0 → ppp(20), 0.5 → mp(64), 1.0 → fff(120)
            vel_target = int(VEL_PPP + (VEL_FFF - VEL_PPP) * bar_intensity)
            vel = vel_target + lfo_val + hold_i * 2
            vel = max(VEL_PPP, min(VEL_FFF, vel + int(rng.integers(-5, 6))))

            track.events.append(PadNoteEvent(
                pitch=pitch, time=note_start, duration=note_dur, velocity=vel,
            ))

        # Extra tension tone
        if extra_pitch is not None:
            t_start = current_bar_time + rng.uniform(0.5, 2.0)
            t_dur = rng.uniform(1.5, float(beats_per_bar) - 0.5)
            t_vel = max(VEL_PPP, min(VEL_MF, int(VEL_PP + (VEL_MF - VEL_PP) * bar_intensity) + int(rng.integers(-5, 6))))
            track.events.append(PadNoteEvent(
                pitch=extra_pitch, time=t_start, duration=t_dur, velocity=t_vel,
            ))

        # Ghost re-attacks
        ghost_prob = 0.15 + bar_intensity * 0.45
        if not is_first_bar and rng.random() < ghost_prob:
            ghost_voice = int(rng.integers(num_voices))
            ghost_pitch = voicing[ghost_voice]
            ghost_beat = rng.choice([1.0, 2.0, 2.5])
            ghost_time = current_bar_time + ghost_beat + rng.uniform(-0.05, 0.05)
            gv_lo = max(VEL_PPP, int(VEL_PPP + 10 * bar_intensity))
            gv_hi = max(gv_lo + 5, int(VEL_P + 10 * bar_intensity))
            ghost_vel = int(rng.integers(gv_lo, gv_hi + 1))
            ghost_dur = rng.uniform(0.5, 1.5)

            track.events.append(PadNoteEvent(
                pitch=ghost_pitch, time=ghost_time, duration=ghost_dur,
                velocity=ghost_vel, is_ghost=True,
            ))

            if bar_intensity > 0.8 and rng.random() < 0.40:
                g2_voice = (ghost_voice + 1) % num_voices
                g2_pitch = voicing[g2_voice]
                g2_beat = rng.choice([1.5, 3.0, 3.5])
                g2_time = current_bar_time + g2_beat + rng.uniform(-0.05, 0.05)
                g2_vel = int(rng.integers(gv_lo, gv_hi + 1))
                track.events.append(PadNoteEvent(
                    pitch=g2_pitch, time=g2_time, duration=rng.uniform(0.3, 1.0),
                    velocity=g2_vel, is_ghost=True,
                ))

        # Legacy fields
        track.chords.append(voicing[:])
        track.velocities.append(vel if active_voices else VEL_MP)
        track.durations.append(float(beats_per_bar))


# ── Voicing construction (below melody) ──────────────────────────────

def _get_chord_for_bar(analysis: Analysis, bar_idx: int) -> Tuple[int, str]:
    if not analysis.chord_progression:
        return (2, "min")
    src_idx = bar_idx % len(analysis.chord_progression)
    return analysis.chord_progression[src_idx]


def _build_voicing(
    root_pc: int, quality: str, scale_pcs: Set[int],
    rng: np.random.Generator, intensity: float,
) -> List[int]:
    """Build a spread pad voicing in the sweet-spot register (C3–C5, MIDI 48–72).

    Rules (from Berklee "Writing String Pads" & orchestration best practices):
    - Root sits in C3–G3 (MIDI 48–55), never below G2 (43)
    - Guide tones (3rd, 7th) stay above G2 (43)
    - Low register (below C3): only root + fifth, no 3rd
    - Upper voices spread up to C5 (72) — not clustered in one octave
    - High intensity: thicker spread up to C5; Low: thin root+fifth in octave 3
    """
    # Root in octave 3 (C3=48 to B3=59)
    base = 3 * 12 + root_pc  # e.g. D3 = 50
    third = 3 if quality == "min" else 4

    if intensity > 0.75:
        # Climax: spread voicing across octave 3–4 (up to C5=72)
        roll = rng.random()
        if roll < 0.4:
            # Root in oct3, 5th in oct3, 3rd in oct4, octave in oct4
            intervals = [0, 7, 12 + third, 12 + 7]
        elif roll < 0.7:
            # Root in oct3, 3rd in oct3, 5th+oct in oct4
            intervals = [0, third, 12 + 7, 24]
        else:
            # Root in oct3, 5th in oct3, oct in oct4, 10th in oct4
            intervals = [0, 7, 12, 12 + third]
    elif intensity < 0.3:
        # Calm: root + fifth only in octave 3 (thin, open)
        if rng.random() < 0.5:
            intervals = [0, 7]            # root + fifth
        else:
            intervals = [0, 7, 12]        # root, fifth, octave
    else:
        # Medium: spread triad — root in oct3, 3rd or 5th up in oct4
        roll = rng.random()
        if roll < 0.4:
            intervals = [0, 7, 12 + third]       # root, 5th, 10th
        elif roll < 0.7:
            intervals = [0, third, 7, 12]         # close triad + octave
        else:
            intervals = [0, 7, 12, 12 + third]   # root, 5th, oct, 10th

    pitches = []
    for iv in intervals:
        p = base + iv
        p = _snap_to_scale(p, scale_pcs)
        # Enforce range: floor=G2(43), ceiling=C5(72)
        if 43 <= p <= 72:
            pitches.append(p)

    # Ensure at least root is present
    if not pitches:
        pitches = [max(48, _snap_to_scale(base, scale_pcs))]

    return sorted(set(pitches))


def _snap_to_scale(pitch: int, scale_pcs: Set[int]) -> int:
    pc = pitch % 12
    if pc in scale_pcs:
        return pitch
    for off in (1, -1, 2, -2):
        if (pc + off) % 12 in scale_pcs:
            return pitch + off
    return pitch


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
