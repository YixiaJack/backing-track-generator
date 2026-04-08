# AI-Generated Backing Track for *Snow Dreaming*

## What I Made
A Python command-line tool that reads my original composition *Snow Dreaming* (雪中梦) — a trip-hop/downtempo piece in D Hungarian Minor at 90 BPM — as a MusicXML file, analyzes its musical properties, and generates a complementary backing track (drums, bass, pad) as multi-track MIDI output.

## How AI/ML Was Used
The core technique is **Markov Chain** modeling, a foundational probabilistic method in machine learning. The system works in two stages:

**Analysis stage**: The tool parses the MusicXML score using the `music21` library, extracting key signature, tempo, time signature, per-measure rhythmic density, pitch histogram, and implied chord progressions.

**Generation stage**: Three Markov Chain models generate the backing track:

- **Drum model**: A **second-order Markov Chain** trained on 20 hand-transcribed trip-hop rhythm patterns (Massive Attack, Portishead, Tricky). It learns two-step transition probabilities — "after kick+silence then silence, what comes next?" — producing more coherent patterns than a first-order model. The output is enriched with **ghost snare notes** (low-velocity hits) and **micro-timing offsets** (kick arrives late, snare arrives early) for authentic trip-hop humanisation. Generation density adapts to the original melody's rhythmic activity.

- **Bass model**: A **second-order interval Markov Chain** (16 training patterns) generates melodic movements constrained to the D Hungarian Minor scale, biased toward chord roots on strong beats. The bass line **synchronises with the kick drum** — bass notes preferentially land on kick positions, creating the locked groove characteristic of trip-hop.

- **Pad model**: Infers chords from per-measure melody analysis and generates **extended voicings** (sus2, sus4, add9, min7) sustained over 2–4 bars with slow MIDI CC envelope evolution (CC11 expression ramp + CC1 modulation sweep).

**Key ML concepts demonstrated**: second-order Markov models, temperature-controlled sampling, Laplace smoothing, n-gram backoff, conditional generation (density-modulated output).

A **temperature parameter** (0.1–2.0) controls randomness: low temperature produces conservative, predictable grooves; high temperature yields experimental, surprising patterns — directly demonstrating how a single ML hyperparameter shapes creative output. A `--seed` flag ensures full reproducibility.

## Tools & References
Python, music21, MIDIUtil, NumPy, Click. Key reference: Shapiro & Huber (2021), "Markov Chains for Computer Music Generation," *Journal of Humanistic Mathematics*, 11(2), 167–195.

## Repository
GitHub: https://github.com/YixiaJack/backing-track-generator
