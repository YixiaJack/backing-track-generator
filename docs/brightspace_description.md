# AI-Generated Backing Track for *Snow Dreaming*

## What I Made
A Python command-line tool that reads my original composition *Snow Dreaming* (雪中梦) — a trip-hop/downtempo piece in D Hungarian Minor at 90 BPM — as a MusicXML file, analyzes its musical properties, and generates a complementary backing track (drums, bass, pad) as MIDI output.

## How AI/ML Was Used
The core technique is **Markov Chain** modeling, a foundational probabilistic method in machine learning. The system works in two stages:

**Analysis stage**: The tool parses the MusicXML score using the `music21` library, extracting key signature, tempo, time signature, per-measure rhythmic density, pitch histogram, and implied harmonic content.

**Generation stage**: Three separate Markov Chain models generate drum patterns, a bass line, and sustained pad chords:
- **Drum model**: Trained on hand-transcribed trip-hop rhythm patterns (kick, snare, hi-hat), it learns transition probabilities — e.g., "after a kick hit, how likely is the next 16th note to also be a kick?" The generation combines these learned transitions with positional priors (downbeats are more likely to have kicks) and adapts density based on the original melody's rhythmic activity.
- **Bass model**: An interval-based Markov Chain generates melodic movement constrained to the D Hungarian Minor scale, biased toward the root on strong beats.
- **Pad model**: Infers simple chords from per-measure melody content and sustains them as whole notes.

A **temperature parameter** controls randomness: low temperature produces conservative, predictable grooves; high temperature yields experimental, surprising patterns — directly demonstrating how a single ML hyperparameter shapes creative output.

## Tools & References
Python, music21, midiutil, numpy. Key reference: Shapiro & Huber (2021), "Markov Chains for Computer Music Generation," *Journal of Humanistic Mathematics*.

## Repository
GitHub: [link]
