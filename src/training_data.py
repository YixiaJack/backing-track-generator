"""Hand-coded trip-hop drum and bass patterns as training corpus.

Each pattern is one bar of 4/4 at 16th-note resolution (16 steps).
Drum hits are encoded as tuples: (kick, snare, hihat) where 0=silent, 1=hit.
Bass intervals are semitone distances from the chord root.

Expanded corpus (20 drum + 16 bass patterns) for richer Markov model training.
Inspired by Massive Attack, Portishead, Tricky, DJ Shadow, UNKLE.
"""
from __future__ import annotations

from typing import List, Tuple

# ── Drum patterns ────────────────────────────────────────────────────
# Each pattern: list of 16 steps, each step = (kick, snare, hihat)

TRIP_HOP_DRUM_PATTERNS: List[List[Tuple[int, int, int]]] = [
    # Pattern 1: Classic trip-hop — heavy kick on 1, snare on 3, sparse hats
    [(1,0,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,1),(0,0,0),(0,0,1),(0,0,0),
     (0,1,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,1),(0,0,0),(0,0,1),(0,0,0)],

    # Pattern 2: Mezzanine-style — syncopated kick
    [(1,0,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,1),(0,0,0),(1,0,1),(0,0,0),
     (0,1,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,1),(0,0,0),(0,0,1),(1,0,0)],

    # Pattern 3: Portishead — ghost snares
    [(1,0,1),(0,0,0),(0,0,1),(0,0,1),  (0,0,1),(0,0,0),(0,0,1),(0,0,0),
     (0,1,1),(0,0,0),(0,0,1),(0,0,1),  (0,0,1),(0,0,0),(0,0,1),(0,0,0)],

    # Pattern 4: Minimal — kick and snare only
    [(1,0,0),(0,0,0),(0,0,0),(0,0,0),  (0,0,0),(0,0,0),(0,0,0),(0,0,0),
     (0,1,0),(0,0,0),(0,0,0),(0,0,0),  (0,0,0),(0,0,0),(0,0,0),(0,0,0)],

    # Pattern 5: Denser hats with kick push
    [(1,0,1),(0,0,1),(0,0,1),(0,0,1),  (0,0,1),(0,0,1),(0,0,1),(0,0,1),
     (0,1,1),(0,0,1),(0,0,1),(0,0,1),  (1,0,1),(0,0,1),(0,0,1),(0,0,1)],

    # Pattern 6: Half-time feel
    [(1,0,1),(0,0,0),(0,0,0),(0,0,0),  (0,0,1),(0,0,0),(0,0,0),(0,0,0),
     (0,0,1),(0,0,0),(0,0,0),(0,0,0),  (0,1,1),(0,0,0),(0,0,0),(0,0,0)],

    # Pattern 7: Double kick
    [(1,0,1),(0,0,0),(0,0,1),(1,0,0),  (0,0,1),(0,0,0),(0,0,1),(0,0,0),
     (0,1,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,1),(0,0,0),(1,0,1),(0,0,0)],

    # Pattern 8: Shuffle feel
    [(1,0,1),(0,0,0),(0,0,0),(0,0,1),  (0,0,1),(0,0,0),(0,0,0),(0,0,1),
     (0,1,1),(0,0,0),(0,0,0),(0,0,1),  (0,0,1),(0,0,0),(0,0,0),(0,0,1)],

    # ── New patterns ──────────────────────────────────────────────────

    # Pattern 9: "Angel" (Massive Attack) — lazy kick with open hat
    [(1,0,1),(0,0,0),(0,0,0),(0,0,0),  (0,0,1),(0,0,0),(1,0,0),(0,0,0),
     (0,1,1),(0,0,0),(0,0,0),(0,0,0),  (0,0,1),(0,0,0),(0,0,0),(0,0,0)],

    # Pattern 10: "Roads" (Portishead) — swung hi-hat
    [(1,0,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,0),(0,0,1),(0,0,1),(0,0,0),
     (0,1,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,0),(0,0,1),(0,0,1),(0,0,0)],

    # Pattern 11: Breakbeat slice — displaced kick
    [(0,0,1),(0,0,0),(1,0,1),(0,0,0),  (0,0,1),(0,0,0),(0,0,1),(1,0,0),
     (0,1,1),(0,0,0),(0,0,1),(0,0,0),  (1,0,1),(0,0,0),(0,0,1),(0,0,0)],

    # Pattern 12: Ride cymbal feel (hats = ride pattern)
    [(1,0,0),(0,0,1),(0,0,0),(0,0,1),  (0,0,0),(0,0,1),(0,0,0),(0,0,1),
     (0,1,0),(0,0,1),(0,0,0),(0,0,1),  (0,0,0),(0,0,1),(0,0,0),(0,0,1)],

    # Pattern 13: Tricky-style — dense kick and snare interplay
    [(1,0,1),(0,0,0),(0,1,1),(0,0,0),  (1,0,1),(0,0,0),(0,0,1),(0,0,0),
     (0,1,1),(0,0,0),(1,0,1),(0,0,0),  (0,0,1),(0,1,0),(0,0,1),(0,0,0)],

    # Pattern 14: DJ Shadow — boom-bap influenced trip-hop
    [(1,0,1),(0,0,0),(0,0,1),(0,0,0),  (1,0,1),(0,0,0),(0,0,1),(0,0,0),
     (0,1,1),(0,0,0),(0,0,1),(0,0,0),  (0,1,1),(0,0,0),(1,0,1),(0,0,0)],

    # Pattern 15: Fill pattern — snare roll approaching beat 1
    [(1,0,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,1),(0,0,0),(0,0,1),(0,0,0),
     (0,0,1),(0,0,0),(0,1,1),(0,0,0),  (0,1,1),(0,1,0),(0,1,0),(0,1,0)],

    # Pattern 16: UNKLE-style — heavy, spacious
    [(1,0,0),(0,0,0),(0,0,0),(0,0,0),  (0,0,0),(0,0,0),(0,0,0),(1,0,0),
     (0,1,0),(0,0,0),(0,0,0),(0,0,0),  (0,0,0),(0,0,0),(0,0,0),(0,0,0)],

    # Pattern 17: Cross-stick variation — snare on beat 4 instead of 3
    [(1,0,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,1),(0,0,0),(0,0,1),(0,0,0),
     (0,0,1),(0,0,0),(0,0,1),(0,0,0),  (0,1,1),(0,0,0),(0,0,1),(0,0,0)],

    # Pattern 18: Kick-heavy — two kicks per bar, sparse hats
    [(1,0,0),(0,0,0),(0,0,1),(0,0,0),  (1,0,0),(0,0,0),(0,0,1),(0,0,0),
     (0,1,0),(0,0,0),(0,0,1),(0,0,0),  (0,0,0),(0,0,0),(0,0,1),(0,0,0)],

    # Pattern 19: Triplet-feel adaptation (8th-note triplet approximation)
    [(1,0,1),(0,0,0),(0,0,0),(0,0,1),  (0,0,0),(0,0,1),(0,0,0),(0,0,0),
     (0,1,1),(0,0,0),(0,0,0),(0,0,1),  (0,0,0),(0,0,1),(0,0,0),(1,0,0)],

    # Pattern 20: "Teardrop" (Massive Attack) — iconic sparse groove
    [(1,0,1),(0,0,0),(0,0,1),(0,0,0),  (0,0,1),(0,0,0),(0,0,1),(0,0,0),
     (0,1,1),(0,0,0),(0,0,1),(0,0,0),  (1,0,1),(0,0,0),(0,0,1),(0,0,0)],
]


# ── Bass interval sequences ─────────────────────────────────────────
# Each sequence: intervals from chord root in semitones over 16 steps
# -1 = rest, 0 = root, 7 = fifth, etc.

TRIP_HOP_BASS_PATTERNS: List[List[int]] = [
    # Pattern 1: Root-fifth pulse
    [0, -1, -1, -1,  7, -1, -1, -1,  0, -1, -1, -1,  7, -1, -1, -1],

    # Pattern 2: Walking root-octave
    [0, -1, -1, 5,  7, -1, -1, -1,  12, -1, -1, 7,  5, -1, -1, -1],

    # Pattern 3: Syncopated root
    [0, -1, 0, -1,  -1, -1, 7, -1,  0, -1, -1, -1,  -1, 5, -1, -1],

    # Pattern 4: Minimal — root only
    [0, -1, -1, -1,  -1, -1, -1, -1,  0, -1, -1, -1,  -1, -1, -1, -1],

    # Pattern 5: Chromatic approach
    [0, -1, -1, -1,  6, 7, -1, -1,  0, -1, -1, -1,  11, 12, -1, -1],

    # Pattern 6: Octave bounce
    [0, -1, 12, -1,  0, -1, 12, -1,  0, -1, 12, -1,  7, -1, 5, -1],

    # Pattern 7: Sparse dub
    [0, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  7, -1, -1, -1],

    # Pattern 8: Descending line
    [12, -1, -1, 10,  7, -1, -1, 5,  3, -1, -1, 2,  0, -1, -1, -1],

    # ── New bass patterns ─────────────────────────────────────────────

    # Pattern 9: Massive Attack "Angel" — root pedal with fifth
    [0, -1, -1, -1,  -1, -1, 0, -1,  7, -1, -1, -1,  -1, -1, 0, -1],

    # Pattern 10: Portishead — chromatic approach from below
    [-1, -1, -1, -1,  -1, -1, -1, 11,  12, -1, -1, -1,  -1, -1, -1, -1],

    # Pattern 11: Dub-influenced — root with slide to 3rd
    [0, -1, -1, -1,  -1, 2, 3, -1,  0, -1, -1, -1,  -1, -1, -1, -1],

    # Pattern 12: Triplet-feel bass
    [0, -1, -1, 7,  -1, -1, 5, -1,  -1, 0, -1, -1,  7, -1, -1, -1],

    # Pattern 13: Syncopated funk bass
    [0, -1, 0, -1,  7, -1, -1, 5,  -1, 0, -1, -1,  -1, 7, -1, -1],

    # Pattern 14: Wide interval leap
    [0, -1, -1, -1,  12, -1, -1, -1,  7, -1, -1, -1,  0, -1, -1, -1],

    # Pattern 15: Pedal tone with upper movement
    [0, -1, 0, -1,  0, -1, 3, -1,  0, -1, 0, -1,  0, -1, 5, -1],

    # Pattern 16: Riff-based — ascending then descending
    [0, -1, 3, -1,  5, -1, 7, -1,  5, -1, 3, -1,  0, -1, -1, -1],
]
