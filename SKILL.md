# SKILL.md — AI Backing Track Generator

## Skill Description
Generate a stylistically matching backing track (drums, bass, pad) from an input MusicXML file using Markov Chain models. The tool parses the original composition, analyzes its musical properties, and produces complementary MIDI tracks.

## Prompts for Claude Code

### Prompt 1: Bootstrap the project
```
Read CLAUDE.md. Create the full project structure:
- src/parser.py: Use music21 to parse a MusicXML file. Return a dataclass `Analysis` containing: key_signature (str), scale_notes (list[int] as MIDI), tempo (int), time_signature (tuple), measures (list of measure data where each measure has notes with pitch, duration, offset), total_measures (int), pitch_histogram (dict[int, float]), rhythmic_density_per_measure (list[float] from 0-1).
- src/training_data.py: Define TRIP_HOP_KICK_PATTERNS, TRIP_HOP_SNARE_PATTERNS, TRIP_HOP_HH_PATTERNS as lists of 16-step binary sequences (8+ patterns each). Define TRIP_HOP_BASS_INTERVALS as lists of interval sequences. These should capture the characteristic trip-hop feel: sparse heavy kicks, half-time snare on beat 3, shuffled hi-hats, root-heavy bass with stepwise motion.
- src/markov.py: MarkovModel class with fit(patterns) and generate(num_steps, temperature, positional_prior) methods. Support both binary (drum) and multi-state (interval) transition matrices. Include temperature sampling.
- requirements.txt with music21, midiutil, numpy, click.
- .gitignore for Python project + output/*.mid
```

### Prompt 2: Build the generators
```
Read CLAUDE.md. Now build the three generators in separate files:

src/drum_gen.py:
- Function generate_drums(analysis: Analysis, num_bars: int, temperature: float, density: float) -> dict with keys 'kick', 'snare', 'hh' each being list[int] (0/1).
- Train 3 MarkovModels on the trip-hop patterns from training_data.py.
- Modulate generation using analysis.rhythmic_density_per_measure: where the melody is dense, slightly increase drum density; where melody is sparse, keep drums minimal.
- The density parameter (0-1) globally scales hit probability.

src/bass_gen.py:
- Function generate_bass(analysis: Analysis, num_bars: int, temperature: float) -> list[tuple[int, float]] (pitch, duration pairs).
- Train an interval MarkovModel on bass interval patterns.
- Constrain all output pitches to analysis.scale_notes in bass register (MIDI 36-60).
- Beat 1 of each bar should strongly prefer the root or fifth of the key.
- Use analysis.pitch_histogram to bias toward notes that appear in the melody (complementary, not unison).

src/pad_gen.py:
- Function generate_pad(analysis: Analysis, num_bars: int) -> list[tuple[list[int], float]] (chord as list of MIDI pitches, duration).
- Analyze melody notes per measure to infer simple chord (root + third + fifth from scale).
- Generate sustained whole-note chords, one per bar.
- Use MIDI program 89 (Pad 2 Warm) or similar.
```

### Prompt 3: Build the MIDI writer and CLI
```
Read CLAUDE.md. Build the final assembly:

src/midi_writer.py:
- Function write_backing_track(drums, bass, pad, analysis, output_path) that creates a multi-track MIDI file:
  - Track 0: Drums (channel 9, GM mapping: kick=36, snare=38, closed_hh=42)
  - Track 1: Bass (channel 0, program 33=Fingered Bass)
  - Track 2: Pad (channel 1, program 89=Pad 2 Warm)
- Add tempo from analysis. Add humanization: slight velocity variation (±10) on all notes, slight timing offset (±0.02 beats) on hi-hats for shuffle feel.

src/main.py:
- Click CLI with arguments: input_file (path to .musicxml), --output (directory), --bars (int, default=length of input), --temperature (float 0.1-2.0, default=0.8), --density (float 0.1-1.0, default=0.5), --seed (int), --no-pad (flag), --no-drums (flag), --no-bass (flag).
- Pipeline: parse → analyze → print summary → generate → write → print output path.
- Also generate a standalone Jupyter notebook (docs/analysis.ipynb) from the run showing: the analysis results, transition matrices as heatmaps, generated patterns visualized, and the description text for Brightspace submission.

Test by running: python src/main.py input/snow_dreaming.musicxml --output output/ --bars 16
```

### Prompt 4: Generate documentation and notebook
```
Create docs/README.md with:
1. Project title and one-line description
2. "How It Works" section explaining the Markov Chain approach (3-4 paragraphs)
3. "Musical Analysis" section showing what gets extracted from the MusicXML
4. "References" section citing:
   - Shapiro & Huber (2021), "Markov Chains for Computer Music Generation"
   - Ren et al. (2020), "PopMAG: Pop Music Accompaniment Generation"
   - Fragnière et al. (2025), "Real-Time Symbolic Music Accompaniment Generation for Edge Devices"
   - SuperWillow system (MC-based chord/rhythm modeling from MusicXML)
5. "How to Run" with install + usage commands
6. "Sample Output" section (placeholder for screenshots)

Also create docs/brightspace_description.md:
A 200-300 word description suitable for Brightspace submission explaining what was made and how AI/ML was used. Mention: Markov Chains, MusicXML parsing, trip-hop style, temperature parameter, and that the tool analyzes the original composition to generate complementary backing tracks.
```

### Prompt 5: Polish and test
```
Run the full pipeline end-to-end. Fix any import errors or bugs.
Make sure:
- The CLI prints a clear summary of the analysis (key, tempo, measures, density)
- The generated MIDI is listenable (no overlapping notes, reasonable velocities)
- The README has correct usage instructions
- All files have docstrings
- .gitignore includes __pycache__, *.pyc, .env, output/*.mid
Then initialize git repo, create initial commit.
```

## Workflow Summary
1. Place Snow Dreaming .musicxml file in `input/`
2. Run prompts 1-5 sequentially in Claude Code
3. Run the tool: `python src/main.py input/snow_dreaming.musicxml --output output/`
4. Upload to GitHub
5. Submit to Brightspace: GitHub link + output MIDI files + docs/brightspace_description.md
