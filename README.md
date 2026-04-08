# Snow Dreaming AI Backing Track Generator

Generate stylistically matching backing tracks from MusicXML scores using Markov Chain models.

## Quick Start

```bash
pip install -r requirements.txt
python src/main.py input/snow_dreaming.musicxml --output output/ --bars 16 --temperature 0.8
```

## How It Works

This tool parses a MusicXML composition using `music21`, analyzes its key, tempo, rhythmic density, and pitch content, then uses trained Markov Chain models to generate complementary drum, bass, and pad tracks exported as MIDI.

The drum generator learns from hand-transcribed trip-hop patterns. The bass generator uses interval-based Markov transitions constrained to the detected scale. The pad generator infers chords from per-measure melody analysis.

A temperature parameter (0.1–2.0) controls how conservative or experimental the output is.

## References

- Shapiro & Huber (2021). "Markov Chains for Computer Music Generation." *Journal of Humanistic Mathematics*, 11(2), 167–195.
- Ren et al. (2020). "PopMAG: Pop Music Accompaniment Generation." *ACM Multimedia*.
- Fragnière et al. (2025). "Real-Time Symbolic Music Accompaniment Generation for Edge Devices." *DCAI 2025*.
