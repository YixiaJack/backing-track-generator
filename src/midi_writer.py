"""MIDI export using midiutil — multi-track with GM instrument assignments."""
from __future__ import annotations

from pathlib import Path

from midiutil import MIDIFile

from src.drum_gen import DrumTrack
from src.bass_gen import BassTrack
from src.pad_gen import PadTrack

# General MIDI drum note numbers
GM_KICK = 36
GM_SNARE = 38
GM_HIHAT_CLOSED = 42

# General MIDI program numbers (0-indexed)
GM_BASS = 33       # Electric Bass (finger)
GM_PAD = 89        # Pad 2 (warm)

DRUM_CHANNEL = 9   # GM drum channel (0-indexed)


def write_midi(
    drum_track: DrumTrack | None,
    bass_track: BassTrack | None,
    pad_track: PadTrack | None,
    tempo: float,
    time_sig_num: int,
    time_sig_den: int,
    output_path: Path,
) -> Path:
    """Write generated tracks to a multi-track MIDI file."""
    num_tracks = sum(1 for t in (drum_track, bass_track, pad_track) if t is not None)
    midi = MIDIFile(numTracks=max(num_tracks, 1), ticks_per_quarternote=480)

    track_idx = 0

    # ── Drums ─────────────────────────────────────────────────────
    if drum_track is not None:
        midi.addTrackName(track_idx, 0, "Drums")
        midi.addTempo(track_idx, 0, tempo)
        midi.addTimeSignature(track_idx, 0, time_sig_num, int.bit_length(time_sig_den) - 1, 24, 8)

        step_dur = 0.25  # 16th note in quarter-note units
        total_steps = len(drum_track.kick)

        for step in range(total_steps):
            time = step * step_dur
            if drum_track.kick[step] > 0:
                midi.addNote(track_idx, DRUM_CHANNEL, GM_KICK, time, step_dur, drum_track.kick[step])
            if drum_track.snare[step] > 0:
                midi.addNote(track_idx, DRUM_CHANNEL, GM_SNARE, time, step_dur, drum_track.snare[step])
            if drum_track.hihat[step] > 0:
                midi.addNote(track_idx, DRUM_CHANNEL, GM_HIHAT_CLOSED, time, step_dur * 0.8, drum_track.hihat[step])

        track_idx += 1

    # ── Bass ──────────────────────────────────────────────────────
    if bass_track is not None:
        bass_channel = 0
        midi.addTrackName(track_idx, 0, "Bass")
        midi.addTempo(track_idx, 0, tempo)
        midi.addProgramChange(track_idx, bass_channel, 0, GM_BASS)

        step_dur = 0.25
        total_steps = len(bass_track.pitches)

        i = 0
        while i < total_steps:
            pitch = bass_track.pitches[i]
            vel = bass_track.velocities[i]
            if pitch > 0 and vel > 0:
                # Find note duration: extend through subsequent rests
                dur = step_dur
                j = i + 1
                while j < total_steps and bass_track.pitches[j] == 0:
                    dur += step_dur
                    j += 1
                time = i * step_dur
                midi.addNote(track_idx, bass_channel, pitch, time, dur, vel)
            i += 1

        track_idx += 1

    # ── Pad ───────────────────────────────────────────────────────
    if pad_track is not None:
        pad_channel = 1
        midi.addTrackName(track_idx, 0, "Pad")
        midi.addTempo(track_idx, 0, tempo)
        midi.addProgramChange(track_idx, pad_channel, 0, GM_PAD)

        # Add slow attack via CC (expression)
        beats_per_bar = time_sig_num

        for bar_idx in range(pad_track.num_bars):
            bar_time = bar_idx * beats_per_bar
            chord = pad_track.chords[bar_idx]
            vel = pad_track.velocities[bar_idx]
            dur = pad_track.durations[bar_idx]

            # CC11 (expression) ramp for slow attack
            for cc_step in range(4):
                cc_time = bar_time + cc_step * 0.25
                cc_val = min(127, 30 + cc_step * 32)
                midi.addControllerEvent(track_idx, pad_channel, cc_time, 11, cc_val)

            for pitch in chord:
                midi.addNote(track_idx, pad_channel, pitch, bar_time, dur, vel)

    # ── Write file ────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return output_path
