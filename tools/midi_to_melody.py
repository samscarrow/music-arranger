#!/usr/bin/env python3
"""
Extract melody from MIDI file and convert to harmonizer format.

Takes a MIDI file, extracts the melody line, quantizes durations to valid
vocabulary values, and outputs as a Python list ready to paste into harmonize.py.

Usage:
  python tools/midi_to_melody.py input.mid
  python tools/midi_to_melody.py input.mid --quantize 0.25
  python tools/midi_to_melody.py input.mid --part 0 --min-pitch 48 --max-pitch 84
"""

import music21
import sys
from pathlib import Path

# Valid durations in our vocabulary (quarter note = 1.0)
VALID_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

# Default note range (skip notes outside this range)
DEFAULT_MIN_PITCH = 48   # C3 (low range)
DEFAULT_MAX_PITCH = 84   # C6 (high range)


def get_time_signature(score):
    """Extract time signature as (numerator, denominator) or (4, 4) default."""
    for ts in score.flatten().getElementsByClass('TimeSignature'):
        return (ts.numerator, ts.denominator)
    return (4, 4)


def quantize_duration(quarter_length, grid_size=0.25):
    """
    Snap duration to nearest valid value in our vocabulary.

    Args:
        quarter_length: Duration in quarter notes (from music21)
        grid_size: Minimum grid size for snapping (0.25 = sixteenth note)

    Returns:
        Snapped duration from VALID_DURATIONS
    """
    # Round to grid first
    rounded = round(quarter_length / grid_size) * grid_size

    # Find closest valid duration
    closest = min(VALID_DURATIONS, key=lambda x: abs(x - rounded))
    return closest


def extract_melody(midi_path, part_index=0, min_pitch=DEFAULT_MIN_PITCH, max_pitch=DEFAULT_MAX_PITCH, transpose_semitones=0):
    """
    Extract melody from MIDI file.

    Args:
        midi_path: Path to MIDI file
        part_index: Which part to extract (0 = first part)
        min_pitch: Skip notes below this MIDI number
        max_pitch: Skip notes above this MIDI number
        transpose_semitones: Number of semitones to transpose (positive = up, negative = down)

    Returns:
        Tuple of (melody_list, meter_string) where meter_string is "N/D" format
    """
    print(f"📂 Reading {midi_path}...")

    try:
        mf = music21.converter.parse(midi_path)
    except Exception as e:
        print(f"❌ Error loading MIDI: {e}")
        return [], "4/4"

    if not mf.parts:
        print(f"❌ No parts found in MIDI file")
        return [], "4/4"

    # Get the specified part (usually first part is melody)
    if part_index >= len(mf.parts):
        print(f"❌ Part {part_index} not found (file has {len(mf.parts)} parts)")
        return [], "4/4"

    part = mf.parts[part_index]

    # Extract time signature
    meter_num, meter_denom = get_time_signature(mf)
    meter_str = f"{meter_num}/{meter_denom}"

    # Flatten to get all notes in order (ignore nested structure)
    notes_and_rests = part.flatten().notesAndRests

    print(f"   Found {len(notes_and_rests)} events")
    print(f"   Extracting part {part_index}, pitch range {min_pitch}-{max_pitch}")
    if transpose_semitones != 0:
        print(f"   Transposing by {transpose_semitones:+d} semitones")

    melody = []
    skipped = 0

    for event in notes_and_rests:
        # Skip rests
        if event.isRest:
            skipped += 1
            continue

        # Get pitch (handle chords by taking highest note)
        if event.isChord:
            if not event.pitches:
                # Skip empty chords
                skipped += 1
                continue
            pitch = max(p.midi for p in event.pitches)
        else:
            pitch = event.pitch.midi

        # Apply transposition AFTER extraction but BEFORE range filtering
        pitch = pitch + transpose_semitones

        # Skip notes outside range (applied AFTER transposition)
        if pitch < min_pitch or pitch > max_pitch:
            skipped += 1
            continue

        # Quantize duration
        dur = quantize_duration(event.duration.quarterLength)

        melody.append((pitch, dur))

    print(f"   Extracted {len(melody)} notes (skipped {skipped})")

    return melody, meter_str


def format_melody_list(melody):
    """Format melody as Python list for easy copy-paste."""
    if not melody:
        return "[]"

    # Format nicely with comments showing note names
    lines = ["["]
    music21_obj = music21.note.Note()

    for i, (pitch, dur) in enumerate(melody):
        music21_obj.pitch.midi = pitch
        note_name = music21_obj.pitch.nameWithOctave

        if i < len(melody) - 1:
            lines.append(f"    ({pitch:2d}, {dur}),    # {note_name}")
        else:
            lines.append(f"    ({pitch:2d}, {dur}),    # {note_name} (last)")

    lines.append("]")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python midi_to_melody.py <file.mid> [--part N] [--min-pitch M] [--max-pitch M] [--transpose N]")
        print()
        print("Examples:")
        print("  python midi_to_melody.py song.mid")
        print("  python midi_to_melody.py song.mid --part 0 --min-pitch 48 --max-pitch 84")
        print("  python midi_to_melody.py song.mid --transpose -3  # Transpose down 3 semitones (Eb to C)")
        sys.exit(1)

    midi_file = sys.argv[1]

    # Parse optional arguments
    part_index = 0
    min_pitch = DEFAULT_MIN_PITCH
    max_pitch = DEFAULT_MAX_PITCH
    transpose_semitones = 0

    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--part" and i + 2 < len(sys.argv):
            part_index = int(sys.argv[i + 3])
        elif arg == "--min-pitch" and i + 2 < len(sys.argv):
            min_pitch = int(sys.argv[i + 3])
        elif arg == "--max-pitch" and i + 2 < len(sys.argv):
            max_pitch = int(sys.argv[i + 3])
        elif arg == "--transpose" and i + 2 < len(sys.argv):
            transpose_semitones = int(sys.argv[i + 3])

    # Extract melody
    melody, meter_str = extract_melody(midi_file, part_index, min_pitch, max_pitch, transpose_semitones)

    if not melody:
        print("❌ No notes extracted!")
        sys.exit(1)

    print()
    print("=" * 70)
    print("✅ Melody extracted successfully!")
    print("=" * 70)
    print()

    # Show statistics
    print(f"Total notes: {len(melody)}")
    print(f"Meter: {meter_str}")
    durations = [d for _, d in melody]
    pitches = [p for p, _ in melody]

    print(f"Pitch range: {min(pitches)} - {max(pitches)}")
    duration_counts = {}
    for d in durations:
        duration_counts[d] = duration_counts.get(d, 0) + 1

    print("Duration distribution:")
    for dur in sorted(duration_counts.keys()):
        count = duration_counts[dur]
        pct = (count / len(melody)) * 100
        print(f"  {dur:4.2f}: {count:3d} notes ({pct:5.1f}%)")

    print()
    print("=" * 70)
    print("📋 Copy this into tools/harmonize.py:")
    print("=" * 70)
    print()
    print(f"# Set METER constant (line ~133)")
    print(f'METER = "{meter_str}"')
    print()
    print(f"# Replace TEST_MELODY (line ~140)")
    print(f"TEST_MELODY = {format_melody_list(melody)}")
    print()
    print("=" * 70)
    print("Then run: python tools/harmonize.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
