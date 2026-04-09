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

A **command-line Python tool** that takes my original composition *Snow Dreaming* — a trip-hop/downtempo piece at 90 BPM — as a MusicXML file, **analyzes its musical properties using computational methods**, and **generates a complementary backing track** (drums, bass, pad) as multi-track MIDI output. The generated backing track is musically aware: it follows the original piece's key, tempo, harmonic progression, rhythmic density, and **intensity curve** — automatically detecting and reinforcing climax sections.

### How AI/ML Is Used

The core machine learning technique is **Markov Chain** modeling — a probabilistic model where the next state depends on a fixed window of preceding states. This maps naturally to musical sequence generation because music is inherently sequential: what note or drum hit comes next depends on what came before.

The system employs a **two-stage pipeline**:

**Stage 1 — Analysis (Computational Music Information Retrieval)**

The tool parses the MusicXML score using the `music21` library, extracting:
- **Auto-detected scale** from actual pitch histogram (top 7 pitch classes), not hardcoded
- Tempo and time signature
- Per-measure **rhythmic density** (notes per beat) — used to modulate backing track activity
- **Pitch histogram** across all 12 pitch classes
- **Implied chord progression** via per-measure pitch-class frequency analysis
- **Intensity curve** — a composite 0–1 value per measure combining 5 weighted features (note density, pitch height, pitch range, polyphony, density change rate), inspired by the TenseMusic tension model (Goebl et al., 2024). Used to drive dynamic climax response across all three tracks.

**Stage 2 — Generation (Markov Chain Models + Intensity-Aware Dynamics)**

Three separate Markov Chain models generate the backing track components:

| Component | Model Type | What It Learns |
|-----------|-----------|---------------|
| **Drums** | 2nd-order Markov Chain | Transition probabilities between (kick, snare, hi-hat) states. Trained on 20 hand-transcribed trip-hop patterns. **Intensity-aware**: velocity scales from ppp(40) to fff(120), snare fills at climax, busier hi-hat/kick, tighter micro-timing at high intensity. |
| **Bass** | 2nd-order Interval Markov Chain | Melodic interval movements constrained to the auto-detected scale and locked to inferred chord roots. **Intensity-aware**: higher octave at climax, velocity ppp(35)–fff(120), fewer rests at high intensity. |
| **Pad** | Rule-based + Consonance-Validated Voicing | Spread voicings in C3–C5 register with voice leading, staggered entry/exit, and ghost re-attacks. **Intensity-aware**: thicker voicing at climax, full ppp(20)–fff(120) dynamic range, more ghost re-attacks, faster chord changes. |

**Key ML concepts demonstrated:**

1. **Markov Property** — The memoryless property (future depends only on the present state) is relaxed via **second-order (bigram) models** where the next state depends on the previous *two* states, producing more coherent musical patterns.
2. **Temperature Sampling** — A single hyperparameter (0.1–2.0) controls the entropy of the probability distribution: low temperature → conservative, predictable grooves; high temperature → experimental, surprising patterns.
3. **Laplace Smoothing** — Additive smoothing (α = 0.01) prevents zero-probability transitions, ensuring the model can generate novel combinations not seen in training data.
4. **Backoff** — When a second-order context is unseen, the model automatically falls back to first-order, then to uniform random — a standard technique in n-gram language models.
5. **Conditional Generation** — The intensity curve dynamically gates all three generative models: calm sections get sparse/soft output, climax sections get dense/loud output with fills and extended techniques.

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

- **Piece**: *Snow Dreaming*, original composition by JK
- **Key**: Auto-detected from score (top 7 pitch classes from actual pitch histogram)
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
                                    │  • Auto-detect scale  │
                                    │  • Rhythmic density   │
                                    │  • Pitch histogram    │
                                    │  • Chord inference    │
                                    │  • Intensity curve    │
                                    └──────────┬──────────┘
                                               ↓
              ┌────────────────────────────────┼────────────────────────────────┐
              ↓                                ↓                                ↓
   [2nd-Order Markov]              [2nd-Order Interval             [Rule-Based +
    20 trip-hop patterns            Markov] 16 bass patterns        Consonance-Validated]
    + Intensity modulation          + Scale constraint              + Voice leading
    + Ghost notes                   + Bass-kick sync                + Spread voicing C3–C5
    + Micro-timing                  + Intensity dynamics            + Staggered entry/exit
    + Snare fills at climax         + Octave shift at climax        + Ghost re-attacks
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
python -m src.main input/sn2.mxl --output output/ --bars 16 --temperature 0.8

# Full composition (all measures, reproducible)
python -m src.main input/sn2.mxl --output output/ --bars 93 --temperature 0.8 --seed 42

# Full options
python -m src.main input/sn2.mxl \
  --output output/ \
  --bars 32 \
  --temperature 1.0 \
  --seed 42 \
  --drum-density 0.8

# Skip specific tracks
python -m src.main input/sn2.mxl --no-pad --bars 16
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
│   ├── parser.py          # MusicXML parsing via music21, auto scale detection
│   ├── analyzer.py        # Rhythmic density, chord inference, intensity curve
│   ├── markov.py          # Markov Chain models (1st & 2nd order, Laplace, backoff)
│   ├── drum_gen.py        # Trip-hop drums + ghost notes + intensity-driven fills
│   ├── bass_gen.py        # Scale-constrained bass + kick sync + intensity dynamics
│   ├── pad_gen.py         # Consonance-validated pad + voice leading + spread voicing
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

### Intensity Curve (Climax Detection)

The analyzer computes a per-measure intensity value (0.0–1.0) by combining five symbolic features, inspired by the TenseMusic tension prediction model:

| Feature | Weight | Musical Meaning |
|---------|--------|-----------------|
| Note density | 0.30 | Loudness / onset frequency proxy |
| Mean pitch height | 0.20 | Higher pitch → more tension |
| Polyphony | 0.20 | More simultaneous voices → thicker texture |
| Pitch range | 0.15 | Wider range → more dramatic |
| Density change rate | 0.15 | Sudden changes → transitions/build-ups |

The raw curve is smoothed with a 5-measure moving average to avoid jitter, then normalised to 0–1. All three generators respond to this curve:

- **Drums**: velocity ppp(40)–fff(120), relaxed density gating at climax, syncopated kick fills, snare rolls at bar ends, tighter micro-timing
- **Bass**: velocity ppp(35)–fff(120), octave shifts at climax, fewer rests, wider pitch range
- **Pad**: velocity ppp(20)–fff(120), thicker spread voicings at climax, more ghost re-attacks, faster chord changes, more tension tones

### Second-Order Markov Chain with Backoff

Unlike a simple first-order model where P(next | current), our second-order model computes P(next | previous, current), capturing two-step dependencies that produce more musically coherent sequences:

```
First-order:   Kick → ?        (many possibilities)
Second-order:  Kick → Silence → ?  (much more constrained, more musical)
```

When a bigram context hasn't been seen in training, the model **backs off** to unigram, then to uniform — the same strategy used in NLP n-gram language models (Katz, 1987).

### Humanisation Features

- **Micro-timing**: Kick drums arrive 5–35ms late (trip-hop "lazy" feel), snares 5–25ms early (tension), hi-hats stay on grid. Timing tightens at high intensity.
- **Ghost notes**: Low-velocity snare hits (vel 30–48) inserted around main backbeats at 6–20% probability, scaled by intensity.
- **Velocity variation**: ±10 velocity jitter on all instruments
- **Bass-kick lock**: Bass notes are biased toward kick positions (65% probability), creating the locked groove characteristic of trip-hop

### Pad Voicing & Consonance

The pad generator uses several techniques to ensure harmonic compatibility:

**Register placement** (Berklee "Writing String Pads" rules):
- Root sits in C3–G3 (MIDI 48–55) — the "warm pad" sweet spot
- Guide tones (3rd, 7th) stay above G2 (MIDI 43)
- Upper voices spread up to C5 (MIDI 72) — not clustered in one octave
- Total range: G2–C5 (MIDI 43–72)

**Spread voicing** (orchestration best practice):
- Voices distributed across 1–2 octaves instead of close position
- Low intensity: root + fifth only (thin, open)
- High intensity: spread triad + upper extensions

**Consonance validation** (from automatic harmonization research):
Each pad note is checked against the melody's pitch classes for the current measure:
1. Pad note is a melody tone → always kept (chord tone)
2. Pad note forms a consonant interval (3rd, 5th, 6th) with any melody note → kept
3. Pad note is in-scale and not a semitone from any melody note → kept
4. Otherwise → removed

**Voice leading** (Tymoczko, 2006): When chords change, each voice moves to the nearest available pitch class in the new chord, producing smooth transitions.

**Additional pad techniques**:
- Staggered note entry (lower voices first) and exit (upper voices first)
- Per-voice velocity LFO with phase offset for subtle movement
- Ghost re-attacks at low velocity on beats 2/3 for internal motion
- Tone subtraction (drop a voice) at low intensity, tension tone addition (9th, 11th) at high intensity

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [music21](https://web.mit.edu/music21/) | >= 9.1 | MusicXML parsing and music theory |
| [MIDIUtil](https://pypi.org/project/MIDIUtil/) | >= 1.2.1 | MIDI file creation |
| [NumPy](https://numpy.org/) | >= 1.24 | Numerical operations for Markov matrices |
| [Click](https://click.palletsprojects.com/) | >= 8.1 | CLI interface |

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

5. Fragniere, E., Briot, J.-P., & Music, A. (2025). "Real-Time Symbolic Music Accompaniment Generation for Edge Devices Using GPT-2 with REMIBlock Tokenization." *DCAI 2025*.

6. Haki, B. et al. (2024). "ReaLchords: Adaptive Melody-to-Chord Accompaniment with Reinforcement Learning." *UC San Diego*. https://music-cms.ucsd.edu/

### Intensity & Tension Analysis

7. Goebl, W. et al. (2024). "TenseMusic: An automatic prediction model for musical tension." *PLOS ONE*, 19(1). https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296385

8. Yeh, Y.-C. et al. (2021). "Automatic Melody Harmonization with Triad Chords: A Comparative Study." *Journal of New Music Research*, 50(1). https://arxiv.org/pdf/2001.02360

### Pad Voicing & Orchestration

9. Tymoczko, D. (2006). "The Geometry of Musical Chords." *Science*, 313(5783), 72–74. *(Voice leading via minimal pitch movement)*

10. Berklee College of Music (2018). "Writing String Pads." *Berklee Today*, Fall 2018. https://www.berklee.edu/berklee-today/fall-2018/writing-string-pads

11. Rogers, E. "Big Band Arranging: Open Voicings." https://www.evanrogersmusic.com/blog-contents/big-band-arranging/open-voicings *(Low register: root + fifth only, no 3rd)*

### Markov Models in Music

12. Dubnov, S. et al. (2003). "Using Machine-Learning Methods for Musical Style Modeling." *IEEE Computer*, 36(10), 73–80.

13. Pachet, F. (2003). "The Continuator: Musical Interaction with Style." *Journal of New Music Research*, 32(3), 333–341.

14. Pearce, M. T. (2005). "The Construction and Evaluation of Statistical Models of Melodic Structure in Music Perception and Composition." PhD thesis, City University London. *(IDyOM / PPM variable-order Markov model)*

### Related Tools & Datasets

15. Cuthbert, M. S. & Ariza, C. (2010). "music21: A Toolkit for Computer-Aided Musicology and Music Theory." *Proceedings of the 11th ISMIR Conference*.

16. Dannenberg, R. B. (2006). "The Interpretation of MIDI Velocity." *Proceedings of the International Computer Music Conference*. https://www.cs.cmu.edu/~rbd/papers/velocity-icmc2006.pdf

17. simonholliday/subsequence — Stateful algorithmic MIDI sequencer with voice leading and chord graphs. https://github.com/simonholliday/subsequence

18. ideoforms/isobar — Python library for algorithmic composition with pattern-based arpeggiation. https://github.com/ideoforms/isobar

### Trip-Hop Production References

19. Massive Attack — *Mezzanine* (1998). Virgin Records. *(Drum pattern and production style reference)*

20. Portishead — *Dummy* (1994). Go! Beat Records. *(Breakbeat sampling and ghost note patterns)*

21. Tricky — *Maxinquaye* (1995). Fourth & B'way Records. *(Lo-fi drum programming and bass textures)*
