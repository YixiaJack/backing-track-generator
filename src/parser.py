"""MusicXML parsing via music21 — returns an Analysis dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import music21


@dataclass
class NoteEvent:
    """A single note or rest event extracted from the score."""
    pitch: Optional[int]  # MIDI pitch, None for rests
    duration: float       # in quarter-note lengths
    offset: float         # offset within measure (quarter-note lengths)
    measure: int          # 1-based measure number
    part_name: str = ""


@dataclass
class Analysis:
    """Container for everything extracted / derived from a MusicXML file."""
    # Basic metadata
    key_name: str = ""
    scale_pitches: List[int] = field(default_factory=list)  # MIDI pitch classes (0-11)
    tempo: float = 90.0
    time_sig_num: int = 4
    time_sig_den: int = 4

    # Note-level data
    note_events: List[NoteEvent] = field(default_factory=list)
    num_measures: int = 0

    # Per-measure analysis (filled by analyzer)
    rhythmic_density: List[float] = field(default_factory=list)   # notes-per-beat per measure
    pitch_histogram: List[int] = field(default_factory=lambda: [0] * 12)
    chord_progression: List[Tuple[int, str]] = field(default_factory=list)  # (root_pc, quality) per measure
    intensity_curve: List[float] = field(default_factory=list)    # 0.0–1.0 per measure, composite intensity


def parse_musicxml(path: Path) -> Analysis:
    """Parse a MusicXML / .mxl file and return a raw Analysis object."""
    score = music21.converter.parse(str(path))
    analysis = Analysis()

    # --- Key -----------------------------------------------------------
    keys = score.flatten().getElementsByClass(music21.key.KeySignature)
    if keys:
        ks = keys[0]
        analysis.key_name = str(ks)
    # Scale pitches will be auto-detected from the actual pitch histogram
    # after all notes are parsed (see below)

    # --- Tempo ---------------------------------------------------------
    tempos = score.flatten().getElementsByClass(music21.tempo.MetronomeMark)
    if tempos:
        analysis.tempo = tempos[0].number
    else:
        analysis.tempo = 90.0

    # --- Time Signature ------------------------------------------------
    time_sigs = score.flatten().getElementsByClass(music21.meter.TimeSignature)
    if time_sigs:
        ts = time_sigs[0]
        analysis.time_sig_num = ts.numerator
        analysis.time_sig_den = ts.denominator

    # --- Notes ---------------------------------------------------------
    max_measure = 0
    for part in score.parts:
        part_name = part.partName or ""
        for measure in part.getElementsByClass(music21.stream.Measure):
            m_num = measure.number
            if m_num > max_measure:
                max_measure = m_num
            for elem in measure.notesAndRests:
                if isinstance(elem, music21.note.Note):
                    evt = NoteEvent(
                        pitch=elem.pitch.midi,
                        duration=float(elem.quarterLength),
                        offset=float(elem.offset),
                        measure=m_num,
                        part_name=part_name,
                    )
                    analysis.note_events.append(evt)
                    analysis.pitch_histogram[elem.pitch.midi % 12] += 1
                elif isinstance(elem, music21.chord.Chord):
                    for p in elem.pitches:
                        evt = NoteEvent(
                            pitch=p.midi,
                            duration=float(elem.quarterLength),
                            offset=float(elem.offset),
                            measure=m_num,
                            part_name=part_name,
                        )
                        analysis.note_events.append(evt)
                        analysis.pitch_histogram[p.midi % 12] += 1
                elif isinstance(elem, music21.note.Rest):
                    evt = NoteEvent(
                        pitch=None,
                        duration=float(elem.quarterLength),
                        offset=float(elem.offset),
                        measure=m_num,
                        part_name=part_name,
                    )
                    analysis.note_events.append(evt)

    analysis.num_measures = max_measure

    # Auto-detect scale from actual pitch histogram: take the top 7 pitch classes
    # This avoids hardcoding a scale that doesn't match the actual composition
    if any(c > 0 for c in analysis.pitch_histogram):
        ranked_pcs = sorted(range(12), key=lambda pc: analysis.pitch_histogram[pc], reverse=True)
        analysis.scale_pitches = ranked_pcs[:7]
    else:
        # Fallback: D natural minor
        analysis.scale_pitches = [2, 4, 5, 7, 9, 10, 0]

    return analysis
