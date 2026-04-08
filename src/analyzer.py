"""Musical analysis: rhythmic density, chord inference, summary printing."""
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
    print("=" * 50)
