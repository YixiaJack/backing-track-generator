"""Trip-hop drum pattern generation with positional priors, density modulation,
and intensity-aware climax response.

Climax behaviour (driven by intensity_curve from analyzer):
- High intensity (>0.7): higher base velocity, more hits survive density gating,
  occasional snare fills, open hi-hat substitutions
- Low intensity (<0.3): sparser patterns, softer velocity, more space
- Medium: standard trip-hop groove
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.markov import build_model_from_sequences
from src.parser import Analysis
from src.training_data import TRIP_HOP_DRUM_PATTERNS

STEPS_PER_BAR = 16  # 16th-note resolution


@dataclass
class DrumTrack:
    """Generated drum track — per-step hits for each instrument."""
    kick: List[int] = field(default_factory=list)    # 0 or velocity
    snare: List[int] = field(default_factory=list)
    hihat: List[int] = field(default_factory=list)
    # Micro-timing offsets in quarter-note units (positive = late, negative = early)
    kick_timing: List[float] = field(default_factory=list)
    snare_timing: List[float] = field(default_factory=list)
    hihat_timing: List[float] = field(default_factory=list)
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

    Responds to both rhythmic density and intensity curve:
    - density modulates how many hits survive per bar
    - intensity modulates velocity, fill probability, and pattern complexity
    """
    sequences = [[(k, s, h) for k, s, h in pat] for pat in TRIP_HOP_DRUM_PATTERNS]
    model = build_model_from_sequences(sequences, order=2)

    track = DrumTrack(num_bars=num_bars)

    densities = _get_density_per_bar(analysis, num_bars)
    intensities = _get_intensity_per_bar(analysis, num_bars)

    for bar_idx in range(num_bars):
        bar_density = densities[bar_idx] * density_scale
        bar_intensity = intensities[bar_idx]

        # Intensity-adjusted temperature: hotter at climax for more variation
        bar_temp = temperature + (bar_intensity - 0.5) * 0.3

        # Generate 16 steps via second-order Markov chain
        bar_steps: List[Tuple[int, int, int]] = model.generate_sequence(
            length=STEPS_PER_BAR,
            rng=rng,
            temperature=bar_temp,
            start=[(1, 0, 1), (0, 0, 0)],
        )

        # Intensity-scaled velocity base: ppp(40) at calm → fff(120) at climax
        vel_base = int(40 + 80 * bar_intensity)  # 40–120 range
        vel_base = max(35, min(120, vel_base))

        for step_idx, (k, s, h) in enumerate(bar_steps):
            # Density gating — relaxed at high intensity (more hits survive)
            gate_threshold = bar_density * 0.6 + 0.4 + bar_intensity * 0.2
            gate = rng.random()
            if gate > gate_threshold:
                if step_idx not in (0, 8):
                    k, s, h = 0, 0, min(h, 1)

            # At high intensity: add extra kick hits on syncopated positions
            if bar_intensity > 0.7 and k == 0 and step_idx in (3, 6, 10, 13):
                if rng.random() < (bar_intensity - 0.7) * 1.5:  # 0–45% chance
                    k = 1

            # At high intensity: busier hi-hat (fill gaps)
            if bar_intensity > 0.6 and h == 0:
                if rng.random() < (bar_intensity - 0.6) * 0.5:
                    h = 1

            # Velocity with intensity scaling
            vel_var = int(rng.integers(-8, 9))
            kick_vel = (vel_base + 15 + vel_var) if k else 0
            snare_vel = (vel_base + 5 + vel_var) if s else 0
            hat_vel = (vel_base - 15 + vel_var) if h else 0

            # At low intensity: further soften everything
            if bar_intensity < 0.3:
                softening = 1.0 - (0.3 - bar_intensity) * 1.5  # 0.55–1.0
                kick_vel = int(kick_vel * softening)
                snare_vel = int(snare_vel * softening)
                hat_vel = int(hat_vel * softening)

            track.kick.append(max(0, min(127, kick_vel)))
            track.snare.append(max(0, min(127, snare_vel)))
            track.hihat.append(max(0, min(127, hat_vel)))

            # Micro-timing: tighter at high intensity, lazier at low
            timing_scale = 1.0 - bar_intensity * 0.4  # 0.6–1.0 (tighter when intense)
            track.kick_timing.append(rng.uniform(0.005, 0.035) * timing_scale if k else 0.0)
            track.snare_timing.append(rng.uniform(-0.025, 0.005) * timing_scale if s else 0.0)
            track.hihat_timing.append(rng.uniform(-0.008, 0.008) if h else 0.0)

        # Snare fill at high intensity: last 4 steps of the bar become a fill
        if bar_intensity > 0.75 and rng.random() < (bar_intensity - 0.5) * 0.8:
            _insert_snare_fill(track, bar_idx, rng, bar_intensity)

    # Post-process: ghost notes (more likely at higher intensity bars)
    _add_ghost_notes(track, rng, intensities)

    return track


def _insert_snare_fill(
    track: DrumTrack, bar_idx: int, rng: np.random.Generator, intensity: float
) -> None:
    """Replace the last 4 steps of a bar with a snare fill (building energy)."""
    start_step = bar_idx * STEPS_PER_BAR + 12  # last 4 sixteenths
    fill_vels = [70, 80, 95, 110]  # crescendo fill

    for i, fill_vel in enumerate(fill_vels):
        idx = start_step + i
        if idx < len(track.snare):
            vel = int(fill_vel * (0.8 + intensity * 0.3))
            track.snare[idx] = max(1, min(127, vel + int(rng.integers(-5, 6))))
            # Fills are tight — minimal timing offset
            if idx < len(track.snare_timing):
                track.snare_timing[idx] = rng.uniform(-0.005, 0.005)


def _add_ghost_notes(
    track: DrumTrack, rng: np.random.Generator, intensities: List[float]
) -> None:
    """Add ghost snare hits — more frequent during high-intensity bars."""
    total_steps = len(track.snare)
    ghost_vel_lo, ghost_vel_hi = 30, 48

    for i in range(total_steps):
        if track.snare[i] > 0:
            continue

        bar_idx = i // STEPS_PER_BAR
        bar_intensity = intensities[bar_idx] if bar_idx < len(intensities) else 0.5

        near_main = any(
            0 <= i + d < total_steps and track.snare[i + d] > 60
            for d in (-2, -1, 1, 2)
        )

        # Ghost probability scales with intensity
        base_prob = 0.20 if near_main else 0.06
        prob = base_prob * (0.5 + bar_intensity)  # ×0.5 at calm, ×1.5 at climax

        if rng.random() < prob:
            track.snare[i] = int(rng.integers(ghost_vel_lo, ghost_vel_hi + 1))
            if track.snare_timing:
                track.snare_timing[i] = rng.uniform(-0.015, 0.015)


def _get_density_per_bar(analysis: Analysis, num_bars: int) -> List[float]:
    """Map analysis rhythmic density to 0–1 range for each output bar."""
    if not analysis.rhythmic_density:
        return [0.5] * num_bars

    max_d = max(analysis.rhythmic_density) or 1.0
    normed = [d / max_d for d in analysis.rhythmic_density]

    result: List[float] = []
    for i in range(num_bars):
        src_idx = int(i * len(normed) / num_bars) % len(normed)
        result.append(normed[src_idx])
    return result


def _get_intensity_per_bar(analysis: Analysis, num_bars: int) -> List[float]:
    """Map intensity curve to output bars."""
    if not analysis.intensity_curve:
        return [0.5] * num_bars

    curve = analysis.intensity_curve
    result: List[float] = []
    for i in range(num_bars):
        src_idx = int(i * len(curve) / num_bars) % len(curve)
        result.append(curve[src_idx])
    return result
