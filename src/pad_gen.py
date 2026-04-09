"""Section-aware pad generation for trip-hop / downtempo.

Based on:
- LBDM phrase boundaries (Cambouropoulos 2001) for structural awareness
- Self-similarity section detection for texture variation
- PopMAG density profiles (Ren et al. 2020) mapped to sections
- GTTM harmonic rhythm principles (Lerdahl & Jackendoff)
- Tymoczko (2006) voice leading for smooth transitions

Each section type gets a distinct pad texture:
  intro  — sparse, single notes, mysterious
  verse  — warm triads, medium density
  chorus — open voicings with 7ths/9ths, fuller
  bridge — suspended chords, tension, sparse
  outro  — thinning out, fading
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict

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

# ── Section texture profiles ────────────────────────────────────
# Each section type defines: num_voices, extensions, velocity_range, hold_bars_range
SECTION_PROFILES: Dict[str, dict] = {
    "intro": {
        "voices": (1, 2),         # sparse
        "use_extensions": False,
        "vel_range": (VEL_PP, VEL_MP),
        "hold_range": (3, 6),     # long holds, slow evolution
        "register": (48, 60),     # low, dark
        "ghost_prob": 0.05,
    },
    "verse": {
        "voices": (2, 3),         # warm triads
        "use_extensions": False,
        "vel_range": (VEL_MP, VEL_MF),
        "hold_range": (2, 4),
        "register": (48, 64),
        "ghost_prob": 0.15,
    },
    "chorus": {
        "voices": (3, 4),         # fuller, richer
        "use_extensions": True,    # add 7ths, 9ths
        "vel_range": (VEL_MF, VEL_F),
        "hold_range": (2, 3),     # more harmonic movement
        "register": (48, 67),     # slightly wider
        "ghost_prob": 0.25,
    },
    "bridge": {
        "voices": (2, 3),         # suspended, tense
        "use_extensions": True,
        "vel_range": (VEL_P, VEL_MF),
        "hold_range": (1, 3),     # more restless
        "register": (50, 62),     # narrower, mid
        "ghost_prob": 0.10,
    },
    "outro": {
        "voices": (1, 2),         # thinning
        "use_extensions": False,
        "vel_range": (VEL_PPP, VEL_P),
        "hold_range": (4, 8),     # very long, fading
        "register": (48, 57),     # low, intimate
        "ghost_prob": 0.02,
    },
}


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
    """Generate section-aware pad chords with distinct textures per section."""
    scale_pcs = set(analysis.scale_pitches) if analysis.scale_pitches else {0, 2, 4, 5, 7, 9, 10}
    beats_per_bar = analysis.time_sig_num
    track = PadTrack(num_bars=num_bars, beats_per_bar=beats_per_bar)
    intensities = _get_intensity_per_bar(analysis, num_bars)
    section_labels = _get_section_labels(analysis, num_bars)

    prev_voicing: List[int] = []
    prev_section: str = ""
    bar_idx = 0

    while bar_idx < num_bars:
        root_pc, quality = _get_chord_for_bar(analysis, bar_idx)
        section = section_labels[bar_idx]
        profile = SECTION_PROFILES.get(section, SECTION_PROFILES["verse"])

        # Section change → force new voicing (no carry-over from different texture)
        section_changed = (section != prev_section)

        # Hold duration from section profile
        hold_lo, hold_hi = profile["hold_range"]
        hold_bars = int(rng.integers(hold_lo, hold_hi + 1))
        # Don't hold across section boundaries
        for h in range(1, hold_bars):
            if bar_idx + h < num_bars and section_labels[bar_idx + h] != section:
                hold_bars = h
                break
        hold_bars = min(hold_bars, num_bars - bar_idx)

        # Build voicing based on section profile
        voicing = _build_section_voicing(
            root_pc, quality, scale_pcs, rng, profile
        )

        # Gentle consonance: only remove harsh semitone clashes
        melody_pcs = set()
        for b in range(bar_idx, min(bar_idx + hold_bars, num_bars)):
            melody_pcs |= _get_melody_pcs_for_bar(analysis, b)
        voicing = _filter_gentle(voicing, melody_pcs, scale_pcs)

        # Voice leading (only within same section, not across transitions)
        if prev_voicing and not section_changed:
            voicing = _voice_lead(prev_voicing, voicing, profile["register"])

        if not voicing:
            bar_idx += hold_bars
            continue

        # Emit notes with section-appropriate dynamics
        _emit_section_notes(
            track, voicing, bar_idx, hold_bars, beats_per_bar,
            rng, intensities, profile, section_changed,
        )

        prev_voicing = voicing
        prev_section = section
        bar_idx += hold_bars

    return track


# ── Section-aware voicing ──────────────────────────────────────

def _build_section_voicing(
    root_pc: int, quality: str, scale_pcs: Set[int],
    rng: np.random.Generator, profile: dict,
) -> List[int]:
    """Build voicing tailored to the section's texture profile."""
    reg_lo, reg_hi = profile["register"]
    min_voices, max_voices = profile["voices"]
    use_ext = profile["use_extensions"]

    # Root in the low end of register
    base = reg_lo + root_pc % 12
    if base < reg_lo:
        base += 12
    if base > reg_hi:
        base -= 12

    third = 3 if quality == "min" else 4
    fifth = 7
    seventh = 10 if quality == "min" else 11
    ninth = 14  # 9th = octave + 2nd

    # Build interval pool based on section
    if use_ext:
        # Chorus/bridge: include extensions
        interval_pool = [
            [0, fifth, seventh],                # root + 5th + 7th
            [0, third, fifth, seventh],          # full 7th
            [0, fifth, seventh, ninth],           # root + 5th + 7th + 9th
            [0, third, seventh, 12 + 2],          # root + 3rd + 7th + 9th
            [0, fifth, 12, 12 + third],           # spread: root + 5th + oct + 10th
        ]
    else:
        # Verse/intro/outro: simpler
        interval_pool = [
            [0, fifth],                           # root + 5th (open fifth)
            [0, third, fifth],                    # simple triad
            [0, fifth, 12],                       # root + 5th + octave
            [0, third, 12],                       # root + 3rd + octave
        ]

    # Pick voicing pattern, constrain to desired voice count
    pattern = interval_pool[int(rng.integers(len(interval_pool)))]

    # Trim or extend to match voice count
    target_voices = int(rng.integers(min_voices, max_voices + 1))
    if len(pattern) > target_voices:
        pattern = pattern[:target_voices]

    pitches = []
    for iv in pattern:
        p = base + iv
        p = _snap_to_scale(p, scale_pcs)
        if reg_lo - 3 <= p <= reg_hi + 3:  # small tolerance
            pitches.append(p)

    if not pitches:
        pitches = [_snap_to_scale(base, scale_pcs)]

    return sorted(set(pitches))


# ── Note emission with section dynamics ────────────────────────

def _emit_section_notes(
    track: PadTrack,
    voicing: List[int],
    start_bar: int,
    hold_bars: int,
    beats_per_bar: int,
    rng: np.random.Generator,
    intensities: List[float],
    profile: dict,
    section_changed: bool,
) -> None:
    """Emit sustained notes with section-appropriate dynamics."""
    total_duration = hold_bars * beats_per_bar
    start_time = start_bar * beats_per_bar
    bar_intensity = intensities[start_bar] if start_bar < len(intensities) else 0.5

    vel_lo, vel_hi = profile["vel_range"]
    # Scale velocity within section's range based on intensity
    base_vel = int(vel_lo + (vel_hi - vel_lo) * bar_intensity)

    for i, pitch in enumerate(voicing):
        # Stagger: more on section changes (new texture entrance), less mid-section
        if section_changed:
            stagger = i * rng.uniform(0.1, 0.25)
        else:
            stagger = i * rng.uniform(0.02, 0.08)

        note_start = start_time + stagger
        note_dur = total_duration - stagger - 0.1

        vel = base_vel + int(rng.integers(-3, 4))
        vel = max(vel_lo, min(vel_hi, vel))

        # Fade-in on section change: first voice softer, grows
        if section_changed and i == 0:
            vel = max(vel_lo, vel - 10)

        track.events.append(PadNoteEvent(
            pitch=pitch, time=note_start, duration=max(0.5, note_dur), velocity=vel,
        ))

    # Ghost re-attack (section-dependent probability)
    if hold_bars >= 2 and rng.random() < profile["ghost_prob"]:
        ghost_bar = start_bar + int(rng.integers(1, hold_bars))
        ghost_time = ghost_bar * beats_per_bar + rng.uniform(0.5, 2.0)
        ghost_pitch = voicing[int(rng.integers(len(voicing)))]
        ghost_vel = max(VEL_PPP, base_vel - 20 + int(rng.integers(-3, 4)))
        track.events.append(PadNoteEvent(
            pitch=ghost_pitch, time=ghost_time,
            duration=rng.uniform(1.0, 2.5), velocity=ghost_vel, is_ghost=True,
        ))

    # Legacy fields
    for h in range(hold_bars):
        track.chords.append(voicing[:])
        track.velocities.append(base_vel)
        track.durations.append(float(beats_per_bar))


# ── Consonance / voice leading (unchanged from previous) ──────

def _filter_gentle(
    voicing: List[int], melody_pcs: Set[int], scale_pcs: Set[int]
) -> List[int]:
    """Only remove notes that create harsh semitone clashes with melody."""
    if not melody_pcs:
        return voicing

    result: List[int] = []
    for p in voicing:
        pc = p % 12
        semitone_clash = any(abs(pc - m_pc) % 12 in (1, 11) for m_pc in melody_pcs)
        if semitone_clash and pc not in scale_pcs:
            continue
        result.append(p)

    if len(result) < 2 and len(voicing) >= 2:
        result = voicing[:2]
    elif not result:
        result = [voicing[0]]

    return result


def _voice_lead(prev: List[int], target: List[int], register: Tuple[int, int]) -> List[int]:
    """Move each voice to nearest pitch in target's pitch-class set."""
    target_pcs = [p % 12 for p in target]
    reg_lo, reg_hi = register

    if len(prev) != len(target):
        return target

    result: List[int] = []
    for prev_pitch in prev:
        if prev_pitch % 12 in target_pcs:
            result.append(prev_pitch)
            continue
        best_pitch = prev_pitch
        best_dist = 999
        for offset in range(-7, 8):
            candidate = prev_pitch + offset
            if candidate % 12 in target_pcs and reg_lo - 3 <= candidate <= reg_hi + 3:
                if abs(offset) < best_dist:
                    best_dist = abs(offset)
                    best_pitch = candidate
        result.append(best_pitch)

    seen = set()
    unique = []
    for p in result:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return sorted(unique)


def _snap_to_scale(pitch: int, scale_pcs: Set[int]) -> int:
    pc = pitch % 12
    if pc in scale_pcs:
        return pitch
    for off in (1, -1, 2, -2):
        if (pc + off) % 12 in scale_pcs:
            return pitch + off
    return pitch


# ── Helpers ────────────────────────────────────────────────────

def _get_chord_for_bar(analysis: Analysis, bar_idx: int) -> Tuple[int, str]:
    if not analysis.chord_progression:
        return (2, "min")
    src_idx = bar_idx % len(analysis.chord_progression)
    return analysis.chord_progression[src_idx]


def _get_melody_pcs_for_bar(analysis: Analysis, bar_idx: int) -> Set[int]:
    num_bars = analysis.num_measures
    src_m = bar_idx % num_bars + 1
    return {e.pitch % 12 for e in analysis.note_events
            if e.measure == src_m and e.pitch is not None}


def _get_intensity_per_bar(analysis: Analysis, num_bars: int) -> List[float]:
    if not analysis.intensity_curve:
        return [0.5] * num_bars
    curve = analysis.intensity_curve
    return [curve[int(i * len(curve) / num_bars) % len(curve)] for i in range(num_bars)]


def _get_section_labels(analysis: Analysis, num_bars: int) -> List[str]:
    """Map section labels to output bars (which may differ from source measures)."""
    if not analysis.section_labels:
        return ["verse"] * num_bars
    src = analysis.section_labels
    return [src[int(i * len(src) / num_bars) % len(src)] for i in range(num_bars)]
