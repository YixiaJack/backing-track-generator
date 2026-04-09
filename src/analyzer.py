"""Musical analysis: rhythmic density, chord inference, intensity curve, summary printing.

Intensity curve computation is based on research from:
- TenseMusic (Goebl et al., 2024): loudness, pitch height, onset frequency as tension predictors
- jSymbolic feature extraction: note density, pitch range, polyphony
- Rhythm-Based Attention Analysis (2025): rhythmic density change rate

In our symbolic (no audio) context, we approximate with:
  note_density + mean_pitch_height + pitch_range + polyphony + density_change_rate
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from src.parser import Analysis


# Pitch-class → note name mapping
_PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def analyze(analysis: Analysis) -> Analysis:
    """Fill in derived fields on an existing Analysis object."""
    _compute_rhythmic_density(analysis)
    _infer_chords(analysis)
    _compute_intensity_curve(analysis)
    return analysis


# ── Rhythmic density ─────────────────────────────────────────────────
def _compute_rhythmic_density(a: Analysis) -> None:
    """Notes-per-beat for each measure (higher = denser melody)."""
    beats_per_measure = a.time_sig_num
    counts: Counter[int] = Counter()
    for evt in a.note_events:
        if evt.pitch is not None:
            counts[evt.measure] += 1

    densities: List[float] = []
    for m in range(1, a.num_measures + 1):
        densities.append(counts[m] / max(beats_per_measure, 1))
    a.rhythmic_density = densities


# ── Chord inference ──────────────────────────────────────────────────
def _infer_chords(a: Analysis) -> None:
    """Simple per-measure chord: most common pitch-class = root, guess quality."""
    progression: List[Tuple[int, str]] = []
    for m in range(1, a.num_measures + 1):
        pcs: List[int] = [
            evt.pitch % 12
            for evt in a.note_events
            if evt.measure == m and evt.pitch is not None
        ]
        if not pcs:
            # Carry previous or default to D
            prev = progression[-1] if progression else (2, "min")
            progression.append(prev)
            continue

        counter = Counter(pcs)
        root = counter.most_common(1)[0][0]

        # Check for minor 3rd (root+3) vs major 3rd (root+4)
        has_minor_3rd = ((root + 3) % 12) in counter
        has_major_3rd = ((root + 4) % 12) in counter
        quality = "min" if has_minor_3rd and not has_major_3rd else "maj"
        progression.append((root, quality))

    a.chord_progression = progression


# ── Intensity curve ──────────────────────────────────────────────────

# Feature weights (inspired by TenseMusic: loudness highest, pitch & onset medium)
# In our symbolic proxy: note_density ≈ loudness/onset, pitch_height ≈ pitch,
# pitch_range ≈ roughness, polyphony ≈ texture, density_change ≈ tempo change
_W_DENSITY = 0.30       # note density (≈ loudness/onset frequency)
_W_PITCH_HEIGHT = 0.20  # mean pitch height
_W_PITCH_RANGE = 0.15   # pitch range within measure
_W_POLYPHONY = 0.20     # number of simultaneous voices
_W_DENSITY_CHANGE = 0.15  # rate of density change (acceleration)


def _compute_intensity_curve(a: Analysis) -> None:
    """Compute a per-measure intensity value (0.0–1.0).

    Combines five symbolic features, each normalised to 0–1, then
    weighted and smoothed to produce a curve that identifies climactic sections.
    """
    n = a.num_measures
    if n == 0:
        a.intensity_curve = []
        return

    # ── Extract raw features per measure ──────────────────────────
    note_density: List[float] = []
    mean_pitch: List[float] = []
    pitch_range: List[float] = []
    polyphony: List[float] = []

    for m in range(1, n + 1):
        pitches = [e.pitch for e in a.note_events if e.measure == m and e.pitch is not None]

        # Note density (count of pitched notes)
        note_density.append(float(len(pitches)))

        if pitches:
            mean_pitch.append(sum(pitches) / len(pitches))
            pitch_range.append(float(max(pitches) - min(pitches)))
        else:
            mean_pitch.append(0.0)
            pitch_range.append(0.0)

        # Polyphony: count of unique offsets with overlapping notes (proxy for simultaneous voices)
        offsets = [e.offset for e in a.note_events if e.measure == m and e.pitch is not None]
        if offsets:
            offset_counts = Counter(offsets)
            polyphony.append(float(max(offset_counts.values())))
        else:
            polyphony.append(0.0)

    # Density change rate: |density[m] - density[m-1]|
    density_change: List[float] = [0.0]
    for i in range(1, n):
        density_change.append(abs(note_density[i] - note_density[i - 1]))

    # ── Normalise each feature to 0–1 ────────────────────────────
    def _norm(vals: List[float]) -> List[float]:
        mx = max(vals) if vals else 1.0
        if mx == 0:
            return [0.0] * len(vals)
        return [v / mx for v in vals]

    nd = _norm(note_density)
    mp = _norm(mean_pitch)
    pr = _norm(pitch_range)
    po = _norm(polyphony)
    dc = _norm(density_change)

    # ── Weighted combination ──────────────────────────────────────
    raw: List[float] = []
    for i in range(n):
        val = (_W_DENSITY * nd[i]
               + _W_PITCH_HEIGHT * mp[i]
               + _W_PITCH_RANGE * pr[i]
               + _W_POLYPHONY * po[i]
               + _W_DENSITY_CHANGE * dc[i])
        raw.append(val)

    # ── Smooth with a moving average (window=5) to avoid jitter ──
    smoothed = _smooth(raw, window=5)

    # ── Final normalise to 0–1 ────────────────────────────────────
    a.intensity_curve = _norm(smoothed)


def _smooth(values: List[float], window: int = 5) -> List[float]:
    """Simple centered moving average smoothing."""
    n = len(values)
    half = window // 2
    result: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def print_summary(a: Analysis) -> None:
    """Print a human-readable analysis summary to stdout."""
    print("=" * 50)
    print("  Analysis Summary")
    print("=" * 50)
    print(f"  Key:            {a.key_name or 'D Hungarian Minor'}")
    print(f"  Tempo:          {a.tempo} BPM")
    print(f"  Time Signature: {a.time_sig_num}/{a.time_sig_den}")
    print(f"  Measures:       {a.num_measures}")
    print(f"  Total notes:    {sum(1 for e in a.note_events if e.pitch is not None)}")

    # Top pitch classes
    top_pcs = sorted(range(12), key=lambda pc: a.pitch_histogram[pc], reverse=True)[:5]
    top_str = ", ".join(f"{_PC_NAMES[pc]}({a.pitch_histogram[pc]})" for pc in top_pcs)
    print(f"  Top pitches:    {top_str}")

    # Average density
    if a.rhythmic_density:
        avg = sum(a.rhythmic_density) / len(a.rhythmic_density)
        print(f"  Avg density:    {avg:.2f} notes/beat")

    # First 8 chords
    if a.chord_progression:
        chords = [f"{_PC_NAMES[r]}{q}" for r, q in a.chord_progression[:8]]
        print(f"  Chords (1-8):   {' | '.join(chords)}")

    # Intensity curve summary
    if a.intensity_curve:
        peak_m = max(range(len(a.intensity_curve)), key=lambda i: a.intensity_curve[i])
        peak_val = a.intensity_curve[peak_m]
        avg_int = sum(a.intensity_curve) / len(a.intensity_curve)
        # Find climax regions (intensity > 0.75)
        climax_bars = [i + 1 for i, v in enumerate(a.intensity_curve) if v > 0.75]
        climax_str = _format_ranges(climax_bars) if climax_bars else "none"
        print(f"  Avg intensity:  {avg_int:.2f}")
        print(f"  Peak intensity: {peak_val:.2f} at measure {peak_m + 1}")
        print(f"  Climax bars:    {climax_str}")

    print("=" * 50)


def _format_ranges(bars: List[int]) -> str:
    """Format [1,2,3,7,8] as '1-3, 7-8'."""
    if not bars:
        return ""
    ranges: List[str] = []
    start = bars[0]
    end = bars[0]
    for b in bars[1:]:
        if b == end + 1:
            end = b
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = b
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)
