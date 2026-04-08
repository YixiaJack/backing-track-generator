"""CLI entry point for the Snow Dreaming backing track generator."""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from src.parser import parse_musicxml
from src.analyzer import analyze, print_summary
from src.drum_gen import generate_drums
from src.bass_gen import generate_bass
from src.pad_gen import generate_pad
from src.midi_writer import write_midi


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("output"),
              help="Output directory for MIDI files.")
@click.option("--bars", "-b", type=int, default=16, help="Number of bars to generate.")
@click.option("--temperature", "-t", type=float, default=0.8,
              help="Sampling temperature (0.1=conservative, 2.0=experimental).")
@click.option("--seed", "-s", type=int, default=None, help="Random seed for reproducibility.")
@click.option("--no-drums", is_flag=True, help="Skip drum track generation.")
@click.option("--no-bass", is_flag=True, help="Skip bass track generation.")
@click.option("--no-pad", is_flag=True, help="Skip pad track generation.")
@click.option("--drum-density", type=float, default=1.0,
              help="Drum density multiplier (0.0–2.0).")
def main(
    input_file: Path,
    output: Path,
    bars: int,
    temperature: float,
    seed: int | None,
    no_drums: bool,
    no_bass: bool,
    no_pad: bool,
    drum_density: float,
) -> None:
    """Generate a trip-hop backing track from a MusicXML score.

    INPUT_FILE: Path to a .musicxml or .mxl file.
    """
    # Seed RNG
    rng = np.random.default_rng(seed)

    # 1. Parse
    click.echo(f"Parsing {input_file} ...")
    analysis = parse_musicxml(input_file)

    # 2. Analyze
    click.echo("Analyzing musical content ...")
    analysis = analyze(analysis)
    print_summary(analysis)

    # 3. Generate tracks
    drum_track = None
    bass_track = None
    pad_track = None

    if not no_drums:
        click.echo(f"Generating drum track ({bars} bars, temp={temperature}) ...")
        drum_track = generate_drums(analysis, bars, rng, temperature, drum_density)

    if not no_bass:
        click.echo(f"Generating bass line ({bars} bars, temp={temperature}) ...")
        bass_track = generate_bass(analysis, bars, rng, temperature, drum_track=drum_track)

    if not no_pad:
        click.echo(f"Generating pad chords ({bars} bars) ...")
        pad_track = generate_pad(analysis, bars, rng)

    # 4. Export MIDI
    stem = input_file.stem
    out_path = output / f"{stem}_backing.mid"
    click.echo(f"Writing MIDI to {out_path} ...")
    result_path = write_midi(
        drum_track=drum_track,
        bass_track=bass_track,
        pad_track=pad_track,
        tempo=analysis.tempo,
        time_sig_num=analysis.time_sig_num,
        time_sig_den=analysis.time_sig_den,
        output_path=out_path,
    )

    click.echo(f"\nDone! Backing track saved to: {result_path}")
    click.echo(f"  Tracks: {', '.join(t for t, skip in [('drums', no_drums), ('bass', no_bass), ('pad', no_pad)] if not skip)}")
    click.echo(f"  Bars: {bars} | Tempo: {analysis.tempo} BPM | Temperature: {temperature}")
    if seed is not None:
        click.echo(f"  Seed: {seed}")


if __name__ == "__main__":
    main()
