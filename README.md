# AI Backing Track Generator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![music21](https://img.shields.io/badge/music21-9.1+-green.svg)](https://web.mit.edu/music21/)
[![MIDIUtil](https://img.shields.io/badge/MIDIUtil-1.2+-green.svg)](https://pypi.org/project/MIDIUtil/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange.svg)](https://numpy.org/)
[![ML: Markov Chain](https://img.shields.io/badge/ML-Markov_Chain-purple.svg)](#how-aiml-is-used)
[![Style: Trip-Hop](https://img.shields.io/badge/Style-Trip--Hop-black.svg)](#musical-context)

A Python CLI tool that reads an existing MusicXML composition and generates a stylistically matching **trip-hop backing track** (drums, bass, ambient pad) using **Markov Chain** models — a foundational probabilistic method in machine learning.

Built as a course assignment demonstrating the relationship between AI/ML and music, and as a practical tool for the original composition *Snow Dreaming*.

---

## Assignment Response

> **Prompt**: *Make something using existing tools that show a relationship to machine learning and/or artificial intelligence. This could be a small piece of music, an analysis process, sound synthesis, or anything else that demonstrates the impact of AI on a musical project.*

### What I Made

A **command-line Python tool** that takes my original composition *Snow Dreaming* — a trip-hop/downtempo piece in D Hungarian Minor at 90 BPM — as a MusicXML file, **analyzes its musical properties using computational methods**, and **generates a complementary backing track** (drums, bass, pad) as multi-track MIDI output. The generated backing track is musically aware: it follows the original piece's key, tempo, harmonic progression, and rhythmic density.

### How AI/ML Is Used

The core machine learning technique is **Markov Chain** modeling — a probabilistic model where the next state depends on a fixed window of preceding states. This maps naturally to musical sequence generation because music is inherently sequential: what note or drum hit comes next depends on what came before.

The system employs a **two-stage pipeline**:

**Stage 1 — Analysis (Computational Music Information Retrieval)**

The tool parses the MusicXML score using the `music21` library, extracting:
- Key signature and scale pitch classes
- Tempo and time signature
- Per-measure **rhythmic density** (notes per beat) — used to modulate backing track activity
- **Pitch histogram** across all 12 pitch classes
- **Implied chord progression** via per-measure pitch-class frequency analysis

**Stage 2 — Generation (Markov Chain Models)**

Three separate Markov Chain models generate the backing track components:

| Component | Model Type | What It Learns |
|-----------|-----------|---------------|
| **Drums** | 2nd-order Markov Chain | Transition probabilities between (kick, snare, hi-hat) states — e.g., "after kick+silence then silence, what comes next?" Trained on 20 hand-transcribed trip-hop patterns inspired by Massive Attack, Portishead, and Tricky. |
| **Bass** | 2nd-order Interval Markov Chain | Melodic interval movements (root, 5th, octave, rest) constrained to the D Hungarian Minor scale and locked to the inferred chord roots. Trained on 16 bass patterns. |
| **Pad** | Rule-based + Randomised Voicing | Extended chord voicings (sus2, sus4, add9, min7) sustained over 2–4 bars with slow CC envelope evolution. |

**Key ML concepts demonstrated:**

1. **Markov Property** — The memoryless property (future depends only on the present state) is relaxed via **second-order (bigram) models** where the next state depends on the previous *two* states, producing more coherent musical patterns.
2. **Temperature Sampling** — A single hyperparameter (0.1–2.0) controls the entropy of the probability distribution: low temperature → conservative, predictable grooves; high temperature → experimental, surprising patterns. This directly demonstrates how an ML hyperparameter shapes creative output.
3. **Laplace Smoothing** — Additive smoothing (α = 0.01) prevents zero-probability transitions, ensuring the model can generate novel combinations not seen in training data.
4. **Backoff** — When a second-order context is unseen, the model automatically falls back to first-order, then to uniform random — a standard technique in n-gram language models applied here to music.
5. **Density Modulation** — The analysis-derived rhythmic density dynamically gates the generative model: sparse melody sections get sparser drums, dense sections get busier patterns. This is an example of **conditional generation** where the output adapts to input features.

### Why Markov Chains Over Deep Learning?

| Criterion | Markov Chain | Deep Learning (RNN/Transformer) |
|-----------|-------------|-------------------------------|
| Interpretability | Transition matrices are directly inspectable and musically meaningful | Black-box weights |
| Training data | 20 hand-coded patterns are sufficient | Requires thousands of MIDI files |
| Controllability | Temperature, density, seed give direct creative control | Less intuitive parameter tuning |
| Reproducibility | Deterministic with `--seed` flag | Non-trivial to reproduce |
| Runtime | Instant (<1s) | Requires GPU for real-time |
| Academic fit | Demonstrates core ML concepts clearly | Harder to explain in a course context |

---

## Musical Context

- **Piece**: *Snow Dreaming* (雪中梦), original composition by JK
- **Key**: D Hungarian Minor (D E F G# A Bb C# D)
- **Tempo**: 90 BPM
- **Style**: Trip-hop / Downtempo (Massive Attack, Portishead, Tricky lineage)
- **Time Signature**: 4/4
- **Original instruments**: Voice, violin, piano, electric guitar, classical guitar
- **Generated backing**: Drums (kick/snare/hi-hat with ghost notes), bass line, ambient pad

---

## Architecture

```
input/*.musicxml  →  [MusicXML Parser]  →  Analysis Object
                          (music21)            ↓
                                    ┌──────────┴──────────┐
                                    │   Musical Analysis   │
                                    │  • Rhythmic density   │
                                    │  • Pitch histogram    │
                                    │  • Chord inference    │
                                    └──────────┬──────────┘
                                               ↓
              ┌────────────────────────────────┼────────────────────────────────┐
              ↓                                ↓                                ↓
   [2nd-Order Markov]              [2nd-Order Interval             [Rule-Based +
    20 trip-hop patterns            Markov] 16 bass patterns        Random Voicing]
    + Density modulation            + Scale constraint              + Extended chords
    + Ghost notes                   + Bass-kick sync                + Multi-bar sustain
    + Micro-timing                  + Voice leading                 + CC envelopes
              ↓                                ↓                                ↓
         Drum Track                       Bass Track                      Pad Track
              └────────────────────────────────┼────────────────────────────────┘
                                               ↓
                                      [MIDI Writer]  →  output/*.mid
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/YixiaJack/backing-track-generator.git
cd backing-track-generator
pip install -r requirements.txt
```

### Usage

```bash
# Basic usage (16 bars, temperature 0.8)
python src/main.py input/sn2.mxl --output output/ --bars 16 --temperature 0.8

# Full options
python src/main.py input/sn2.mxl \
  --output output/ \
  --bars 32 \
  --temperature 1.0 \
  --seed 42 \
  --drum-density 0.8

# Skip specific tracks
python src/main.py input/sn2.mxl --no-pad --bars 16
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output, -o` | `output/` | Output directory for MIDI files |
| `--bars, -b` | `16` | Number of bars to generate |
| `--temperature, -t` | `0.8` | Sampling temperature (0.1 = conservative, 2.0 = experimental) |
| `--seed, -s` | `None` | Random seed for reproducibility |
| `--no-drums` | `False` | Skip drum track generation |
| `--no-bass` | `False` | Skip bass track generation |
| `--no-pad` | `False` | Skip pad track generation |
| `--drum-density` | `1.0` | Drum density multiplier (0.0–2.0) |

---

## Project Structure

```
backing-track-generator/
├── src/
│   ├── main.py            # CLI entry point (Click)
│   ├── parser.py          # MusicXML parsing via music21
│   ├── analyzer.py        # Rhythmic density, pitch histogram, chord inference
│   ├── markov.py          # Markov Chain models (1st & 2nd order, Laplace, backoff)
│   ├── drum_gen.py        # Trip-hop drum generation + ghost notes + micro-timing
│   ├── bass_gen.py        # Scale-constrained bass generation + kick sync
│   ├── pad_gen.py         # Extended voicing pad generation + multi-bar sustain
│   ├── midi_writer.py     # Multi-track MIDI export with CC automation
│   └── training_data.py   # 20 drum + 16 bass hand-coded trip-hop patterns
├── input/                 # MusicXML source files
├── output/                # Generated MIDI files
├── docs/                  # Assignment writeup
├── requirements.txt
├── LICENSE
├── CLAUDE.md
└── README.md
```

---

## Technical Details

### Second-Order Markov Chain with Backoff

Unlike a simple first-order model where P(next | current), our second-order model computes P(next | previous, current), capturing two-step dependencies that produce more musically coherent sequences:

```
First-order:   Kick → ?        (many possibilities)
Second-order:  Kick → Silence → ?  (much more constrained, more musical)
```

When a bigram context hasn't been seen in training, the model **backs off** to unigram, then to uniform — the same strategy used in NLP n-gram language models (Katz, 1987).

### Humanisation Features

- **Micro-timing**: Kick drums arrive 5–35ms late (trip-hop "lazy" feel), snares 5–25ms early (tension), hi-hats stay on grid
- **Ghost notes**: Low-velocity snare hits (vel 30–48) inserted around main backbeats at 6–20% probability
- **Velocity variation**: ±10 velocity jitter on all instruments
- **Bass-kick lock**: Bass notes are biased toward kick positions (65% probability), creating the locked groove characteristic of trip-hop

### Extended Pad Voicings

Beyond basic triads, the pad generator randomly selects from 8 voicing types:

| Voicing | Intervals | Character |
|---------|-----------|-----------|
| Triad (min/maj) | 0-3-7 / 0-4-7 | Standard |
| sus2 | 0-2-7 | Open, ambiguous |
| sus4 | 0-5-7 | Suspended tension |
| add9 | 0-3/4-7-14 | Colour, shimmer |
| min7 | 0-3-7-10 | Jazz, warmth |
| open 5th | -12-0-7-12 | Power, cinematic |

Chords are sustained for 2–4 bars (trip-hop "big block" harmony) with CC11 (expression) attack ramps and CC1 (modulation) sine-wave texture evolution.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [music21](https://web.mit.edu/music21/) | ≥ 9.1 | MusicXML parsing and music theory |
| [MIDIUtil](https://pypi.org/project/MIDIUtil/) | ≥ 1.2.1 | MIDI file creation |
| [NumPy](https://numpy.org/) | ≥ 1.24 | Numerical operations for Markov matrices |
| [Click](https://click.palletsprojects.com/) | ≥ 8.1 | CLI interface |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## References

### Core Method

1. Shapiro, A. & Huber, D. (2021). "Markov Chains for Computer Music Generation." *Journal of Humanistic Mathematics*, 11(2), 167–195. https://scholarship.claremont.edu/jhm/vol11/iss2/8/

2. Katz, S. M. (1987). "Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer." *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 35(3), 400–401.

### Music Accompaniment Generation

3. Ren, Y., He, J., Tan, X., Qin, T., Zhao, Z., & Liu, T.-Y. (2020). "PopMAG: Pop Music Accompaniment Generation." *Proceedings of the 28th ACM International Conference on Multimedia*, 1198–1206.

4. Ding, L. & Cui, S. (2023). "MuseFlow: Music Accompaniment Generation Based on Flow." *Applied Intelligence*, 53, 9498–9514.

5. Fragnière, E., Briot, J.-P., & Music, A. (2025). "Real-Time Symbolic Music Accompaniment Generation for Edge Devices Using GPT-2 with REMIBlock Tokenization." *DCAI 2025*.

6. Haki, B. et al. (2024). "ReaLchords: Adaptive Melody-to-Chord Accompaniment with Reinforcement Learning." *UC San Diego*. https://music-cms.ucsd.edu/

### Markov Models in Music

7. Dubnov, S. et al. (2003). "Using Machine-Learning Methods for Musical Style Modeling." *IEEE Computer*, 36(10), 73–80.

8. Pachet, F. (2003). "The Continuator: Musical Interaction with Style." *Journal of New Music Research*, 32(3), 333–341.

9. Pearce, M. T. (2005). "The Construction and Evaluation of Statistical Models of Melodic Structure in Music Perception and Composition." PhD thesis, City University London. *(IDyOM / PPM variable-order Markov model)*

### Related Tools & Datasets

10. Cuthbert, M. S. & Ariza, C. (2010). "music21: A Toolkit for Computer-Aided Musicology and Music Theory." *Proceedings of the 11th ISMIR Conference*.

11. Raffel, C. (2016). "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching." PhD thesis, Columbia University. *(Lakh MIDI Dataset)*

12. mapio/markovdrummer — Algorithmic drum groove generation using Markov chains. https://github.com/mapio/markovdrummer

13. simonholliday/subsequence — Stateful algorithmic MIDI sequencer combining Markov chains with Euclidean rhythms. https://github.com/simonholliday/subsequence

### Trip-Hop Production References

14. Massive Attack — *Mezzanine* (1998). Virgin Records. *(Drum pattern and production style reference)*

15. Portishead — *Dummy* (1994). Go! Beat Records. *(Breakbeat sampling and ghost note patterns)*

16. Tricky — *Maxinquaye* (1995). Fourth & B'way Records. *(Lo-fi drum programming and bass textures)*
