"""Hand-coded trip-hop drum and bass patterns as training corpus.

Each pattern is one bar of 4/4 at 16th-note resolution (16 steps).
Drum hits are encoded as tuples: (kick, snare, hihat) where 0=silent, 1=hit.
Bass intervals are semitone distances from the chord root.
"""
from __future__ import annotations

from typing import List, Tuple

# ── Drum patterns ────────────────────────────────────────────────────
# Each pattern: list of 16 steps, each step = (kick, snare, hihat)
# Inspired by Massive Attack, Portishead, Tricky

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
]
