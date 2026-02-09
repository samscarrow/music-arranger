#!/usr/bin/env python3
"""
Tokenize Barbershop Quartet Scores for Transformer Training

Converts 4-part SATB arrangements into vertical slices (chord-based events) with:
  - Transposition to C Major / A Minor (normalized key)
  - Per-voice MIDI pitches at each rhythmic event
  - Chord label using STYLE_MAP from analyze_style.py
  - Verbose, human-readable token format for debugging

Output: tools/barbershop_dataset/training_sequences.txt
"""

import music21
import glob
import json
import os
from collections import Counter
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = "tools/barbershop_dataset/processed/open_scores"
OUTPUT_DIR = "tools/barbershop_dataset"
OUTPUT_FILE = f"{OUTPUT_DIR}/training_sequences.txt"

# Import STYLE_MAP from analyze_style.py (reuse exact chord classifier)
STYLE_MAP = {
    # Major / Tonic function
    "major triad": "MAJOR_TRIAD",
    "major seventh chord": "MAJOR_TRIAD",
    "Major Third with octave doublings": "MAJOR_TRIAD",
    "enharmonic equivalent to major triad": "MAJOR_TRIAD",
    "major-second major tetrachord": "MAJOR_TRIAD",
    "incomplete major-seventh chord": "MAJOR_TRIAD",

    # Dominant function
    "dominant seventh chord": "DOM7",
    "incomplete dominant-seventh chord": "DOM7",
    "enharmonic to dominant seventh chord": "DOM7",
    "German augmented sixth chord": "DOM7",
    "diminished triad": "DOM7",
    "diminished seventh chord": "DOM7",
    "French augmented sixth chord": "DOM7",
    "Italian augmented sixth chord": "DOM7",

    # Color chords
    "minor seventh chord": "MINOR7",
    "incomplete minor-seventh chord": "MINOR7",
    "enharmonic equivalent to minor seventh chord": "MINOR7",

    "minor triad": "MINOR_TRIAD",
    "enharmonic equivalent to minor triad": "MINOR_TRIAD",

    "half-diminished seventh chord": "HALF_DIM",
    "enharmonic equivalent to half-diminished seventh chord": "HALF_DIM",

    "augmented triad": "AUG",
    "augmented seventh chord": "AUG",

    # Texture
    "note": "UNISON",
    "Perfect Octave": "UNISON",
    "Perfect Unison": "UNISON",
    "Perfect Fifth": "OPEN_5TH",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_chord(raw_name):
    """Map raw music21 chord name to normalized style label."""
    return STYLE_MAP.get(raw_name, "OTHER")


def infer_voice_names(score):
    """
    Infer voice names (Tenor, Lead, Bari, Bass) by average pitch.
    Returns list: [Tenor_part, Lead_part, Bari_part, Bass_part]
    """
    ranges = {}
    for i, part in enumerate(score.parts):
        notes = [n for n in part.flatten().notesAndRests if n.isNote]
        if notes:
            pitches = [n.pitch.midi for n in notes]
            ranges[i] = {
                "min": min(pitches),
                "max": max(pitches),
                "avg": sum(pitches) / len(pitches)
            }

    # Sort by descending average pitch (Tenor highest, Bass lowest)
    sorted_parts = sorted(ranges.items(), key=lambda x: -x[1]['avg'])
    voice_names = ['Tenor', 'Lead', 'Bari', 'Bass']

    # Build mapping: voice_name -> part_index
    voice_to_part = {voice_names[j]: score.parts[i] for j, (i, _) in enumerate(sorted_parts)}

    # Return as [Tenor_part, Lead_part, Bari_part, Bass_part]
    return [voice_to_part[name] for name in voice_names]


def get_key_and_tonic(score):
    """
    Extract key signature and determine tonic pitch.
    Returns (key_sig_obj, tonic_pitch_midi) or (None, 60) if not found.
    """
    for ks in score.flatten().getElementsByClass('KeySignature'):
        key = ks.asKey()
        if key:
            # Get tonic MIDI (middle C = 60 for C major/minor)
            tonic_pitch = key.tonic
            # Use pitch class (0-11), default to octave 4 for middle range
            pitch_class = tonic_pitch.pitchClass
            tonic_midi = pitch_class + (4 * 12)  # Octave 4 (middle C range)
            return ks, tonic_midi
    return None, 60  # Default to C4 if not found


def transpose_midi(midi_pitch, src_tonic, dst_tonic=60):
    """
    Transpose a MIDI pitch from src_tonic to dst_tonic.
    Both tonics are MIDI values.
    """
    interval = dst_tonic - src_tonic
    return midi_pitch + interval


def get_time_signature(score):
    """Extract time signature as (numerator, denominator) or (4, 4) default."""
    for ts in score.flatten().getElementsByClass('TimeSignature'):
        return (ts.numerator, ts.denominator)
    return (4, 4)


def extract_vertical_events(score, voices):
    """
    Extract vertical events (chords) from the score.

    For each unique time offset, capture:
      - MIDI pitches for each voice
      - Duration (in quarter notes)
      - Chord label from chordification

    Returns list of events: [
        {
            'offset': float,
            'bar': int,
            'voices': {'Tenor': midi or None, 'Lead': midi or None, ...},
            'duration': float,
            'chord_label': str
        },
        ...
    ]
    """
    voice_names = ['Tenor', 'Lead', 'Bari', 'Bass']

    # Chordify the ENTIRE score to get harmonic analysis
    flat = score.flatten()
    chords = score.chordify()

    # Build a mapping: offset -> chord object
    chord_by_offset = {}
    for c in chords.flatten().getElementsByClass('Chord'):
        offset = c.offset
        if offset not in chord_by_offset:
            chord_by_offset[offset] = c

    # Build a mapping: offset -> [(part_idx, note), ...]
    events = {}

    for part_idx, part in enumerate(voices):
        notes_and_rests = part.flatten().notesAndRests
        for item in notes_and_rests:
            offset = item.offset
            if offset not in events:
                events[offset] = {}

            if item.isNote:
                events[offset][part_idx] = item.pitch.midi
            else:
                # Rest
                events[offset][part_idx] = None

    # Convert to sorted list of events
    result = []

    for offset in sorted(events.keys()):
        # Calculate bar number (assuming 4/4, quarter note = 1 beat)
        bar_num = int(offset // 4) + 1

        voice_pitches = {}
        for voice_idx, voice_name in enumerate(voice_names):
            voice_pitches[voice_name] = events[offset].get(voice_idx, None)

        # Get duration: find next offset
        next_offset = min([o for o in events.keys() if o > offset], default=offset + 1)
        duration = next_offset - offset

        # Chord label: find closest chord at or before this offset
        chord_label = "OTHER"
        for chord_offset in sorted(chord_by_offset.keys(), reverse=True):
            if chord_offset <= offset:
                chord_label = normalize_chord(chord_by_offset[chord_offset].commonName)
                break

        result.append({
            'offset': offset,
            'bar': bar_num,
            'voices': voice_pitches,
            'duration': duration,
            'chord_label': chord_label
        })

    return result


def tokenize_score(file_path):
    """
    Tokenize a single score file.

    Returns (tokens_list, key_original, key_normalized) or None if parsing fails.
    """
    try:
        score = music21.converter.parse(file_path)
    except Exception as e:
        print(f"  FAILED to parse: {e}")
        return None

    # Get key info
    key_sig, tonic_midi = get_key_and_tonic(score)
    if key_sig:
        key_name = str(key_sig.asKey())
    else:
        key_name = "Unknown"

    # Infer voices by pitch
    try:
        voices = infer_voice_names(score)
    except Exception as e:
        print(f"  FAILED to infer voices: {e}")
        return None

    # Get time signature
    ts_num, ts_den = get_time_signature(score)
    meter_str = f"{ts_num}/{ts_den}"

    # Extract vertical events (before transposition)
    try:
        events = extract_vertical_events(score, voices)
    except Exception as e:
        print(f"  FAILED to extract events: {e}")
        return None

    if not events:
        print(f"  WARNING: No events extracted")
        return None

    # Transpose all pitches to C Major (tonic_midi = 60)
    transposition_amount = 60 - tonic_midi

    # Build tokens
    tokens = []
    tokens.append(f"[key:C] [meter:{meter_str}]")

    for event in events:
        bar = event['bar']
        duration = event['duration']
        chord = event['chord_label']

        # Transpose voices
        transposed_voices = {}
        all_rest = True
        for voice_name, midi in event['voices'].items():
            if midi is not None:
                transposed_voices[voice_name] = midi + transposition_amount
                all_rest = False
            else:
                transposed_voices[voice_name] = None

        # Skip if all voices are resting (no meaningful event)
        if all_rest:
            continue

        # Format pitches
        tokens_for_voices = []
        for voice_name in ['Bass', 'Bari', 'Lead', 'Tenor']:
            midi = transposed_voices[voice_name]
            if midi is not None:
                tokens_for_voices.append(f"[{voice_name.lower()}:{midi}]")
            else:
                tokens_for_voices.append(f"[{voice_name.lower()}:rest]")

        # Build token line
        token_line = f"[bar:{bar}] {' '.join(tokens_for_voices)} [dur:{duration:.1f}] [chord:{chord}]"
        tokens.append(token_line)

    tokens.append("[song_end]")

    return tokens, key_name, "C major"


def main():
    print("=" * 70)
    print("BARBERSHOP QUARTET TOKENIZER")
    print("=" * 70)
    print()

    # Find all score files
    files = (
        glob.glob(f"{DATA_DIR}/*.xml") +
        glob.glob(f"{DATA_DIR}/*.mxl") +
        glob.glob(f"{DATA_DIR}/*.musicxml")
    )

    if not files:
        print(f"ERROR: No files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} score files")
    print(f"Processing to: {OUTPUT_FILE}")
    print()

    # Statistics
    total_files = 0
    total_tokens = 0
    total_songs = 0
    failed_files = []
    vocab = Counter()
    key_distribution = Counter()

    # Open output file for writing
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_FILE, 'w') as out_f:
        for i, file_path in enumerate(sorted(files)):
            file_name = Path(file_path).stem

            result = tokenize_score(file_path)
            if result is None:
                failed_files.append(file_name)
                continue

            tokens, key_orig, key_norm = result

            # Write tokens
            out_f.write('\n'.join(tokens) + '\n\n')

            # Count statistics
            for token in tokens:
                # Extract chord from anywhere in the line (usually at end)
                if '[chord:' in token:
                    chord_part = token.split('[chord:')[1]
                    chord_name = chord_part.split(']')[0]
                    vocab[chord_name] += 1
                if token.startswith('[key:'):
                    key_distribution[key_orig] += 1

            total_tokens += len(tokens)
            total_songs += 1
            total_files += 1

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(files)}...")

    # Summary
    print()
    print("=" * 70)
    print("TOKENIZATION COMPLETE")
    print("=" * 70)
    print(f"\nFiles processed:     {total_files}")
    print(f"Files failed:        {len(failed_files)}")
    print(f"Total songs:         {total_songs}")
    print(f"Total tokens:        {total_tokens}")
    print(f"Vocabulary size:     {len(vocab)}")

    print("\n--- CHORD DISTRIBUTION ---")
    for chord_name, count in vocab.most_common():
        pct = (count / sum(vocab.values())) * 100 if vocab else 0
        print(f"  {chord_name:<16} : {count:>6,}  ({pct:5.1f}%)")

    print("\n--- ORIGINAL KEY DISTRIBUTION ---")
    for key_name, count in key_distribution.most_common():
        pct = (count / sum(key_distribution.values())) * 100 if key_distribution else 0
        print(f"  {key_name:<16} : {count:>6,}  ({pct:5.1f}%)")

    if failed_files:
        print(f"\n--- FAILED FILES ({len(failed_files)}) ---")
        for fname in failed_files[:10]:
            print(f"  {fname}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
    print()


if __name__ == "__main__":
    main()
