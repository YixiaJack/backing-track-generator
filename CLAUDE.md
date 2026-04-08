# CLAUDE.md — Snow Dreaming AI Backing Track Generator

## Project Purpose
A Python CLI tool that reads an existing MusicXML composition and generates a stylistically matching backing track (drums, bass, pad) using Markov Chain models. Built as a course assignment demonstrating AI/ML in music, and as a practical tool for the original composition *Snow Dreaming* .

## Musical Context
- **Piece**: Snow Dreaming, original composition by JK
- **Key**: D Hungarian Minor (D E F G# A Bb C# D)
- **Tempo**: 90 BPM
- **Style**: Trip-hop / Downtempo (Massive Attack, Portishead lineage)
- **Time Signature**: 4/4
- **Instruments in original**: Voice, violin, piano, electric guitar, classical guitar
- **Backing track target**: Drums (kick/snare/hi-hat), bass line, ambient pad

## Architecture

```
input/*.musicxml  →  [MusicXML Parser]  →  Analysis Object
                                              ↓
                     Trip-hop training data + Analysis Object
                                              ↓
                     [Markov Chain Generator]  →  Drum pattern
                     [Interval Markov Chain]   →  Bass line (scale-constrained)
                     [Note Density Model]      →  Pad/sustained chords
                                              ↓
                     [MIDI Writer]  →  output/*.mid
```

### Pipeline Steps
1. **Parse**: Extract key signature, tempo, time signature, note sequences, measure structure from MusicXML using `music21`
2. **Analyze**: Compute pitch histogram, rhythmic density per measure, phrase boundaries, implied chord progression
3. **Generate drums**: Markov Chain trained on trip-hop patterns, modulated by extracted rhythmic density (sparse melody → sparser drums, dense melody → fills)
4. **Generate bass**: Interval Markov Chain constrained to detected scale, following implied chord roots
5. **Generate pad**: Sustained chord tones derived from melody harmony analysis, with slow attack envelopes (via MIDI CC)
6. **Export**: Multi-track MIDI with proper GM instrument assignments

## Key Files
- `src/parser.py` — MusicXML parsing via music21, returns Analysis dataclass
- `src/analyzer.py` — Musical analysis: pitch histogram, rhythmic density, chord inference
- `src/markov.py` — Markov Chain model: build transition matrices, generate sequences, temperature sampling
- `src/drum_gen.py` — Trip-hop drum pattern generation with positional priors
- `src/bass_gen.py` — Scale-constrained bass line generation
- `src/pad_gen.py` — Chord-based pad generation
- `src/midi_writer.py` — MIDI export using midiutil
- `src/main.py` — CLI entry point
- `src/training_data.py` — Hand-coded trip-hop drum/bass patterns as training corpus

## Dependencies
- `music21` — MusicXML parsing and music theory
- `midiutil` — MIDI file creation
- `numpy` — Numerical operations for Markov matrices
- `click` — CLI interface

## Commands
```bash
# Install
pip install music21 midiutil numpy click

# Run
python src/main.py input/snow_dreaming.musicxml --output output/ --bars 16 --temperature 0.8

# Run with options
python src/main.py input/snow_dreaming.musicxml \
  --output output/ \
  --bars 32 \
  --temperature 1.0 \
  --no-pad \
  --drum-density 0.6
```

## Coding Conventions
- Type hints everywhere
- Dataclasses for structured data (Analysis, MarkovModel, GeneratedTrack)
- No classes where functions suffice
- All randomness seeded via `--seed` flag for reproducibility
- Print analysis summary to stdout during generation
- Windows-compatible (no `export`, use pathlib)

## ML Approach & References
The core ML technique is **Markov Chains** — probabilistic models where the next state depends only on the current state. This maps naturally to musical sequence generation.

### Key papers/references:
- Shapiro & Huber (2021), "Markov Chains for Computer Music Generation", J. Humanistic Mathematics — MusicXML parsing + Markov generation in Python
- Ding & Cui (2023), "MuseFlow: music accompaniment generation based on flow", Applied Intelligence — flow-based accompaniment
- Ren et al. (2020), "PopMAG: Pop Music Accompaniment Generation", ACM Multimedia — multi-track accompaniment from melody
- Fragnière et al. (2025), "Real-Time Symbolic Music Accompaniment Generation for Edge Devices", DCAI — GPT-2 based MIDI accompaniment with REMIBlock tokenization
- ReaLchords (2025), UCSD — RL-finetuned melody-to-chord accompaniment with online adaptation

### Why Markov over deep learning for this project:
1. Interpretable: transition matrices are inspectable, musically meaningful
2. No training data bottleneck: hand-coded trip-hop patterns are sufficient
3. Controllable: temperature, density parameters give direct creative control
4. Deterministic with seed: reproducible for academic submission
5. Runs instantly: no GPU needed

## Assignment Requirements
- Demonstrate relationship between ML/AI and music
- Upload: recording/score/notebook + description to Brightspace
- Submission: GitHub repo link + generated MIDI files + brief writeup in docs/
