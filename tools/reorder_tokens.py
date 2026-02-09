#!/usr/bin/env python3
"""
Reorder training tokens for causal arrangement learning.

Original order (physically correct, algorithmically backwards):
  [bar:1] [bass:48] [bari:55] [lead:60] [tenor:64] [dur:1.0] [chord:MAJOR_TRIAD]

Reordered (causal - melody constraints first, then harmony):
  [bar:1] [chord:MAJOR_TRIAD] [lead:60] -> [bass:48] [bari:55] [tenor:64] [dur:1.0]

This allows a GPT model to:
  1. See the Chord (harmonic constraint)
  2. See the Lead (melody constraint)
  3. Generate Bass, Bari, Tenor (harmony given the constraints)

Which matches the arranger's mental model: "Given this melody and chord, what harmony fits?"
"""

import re
from pathlib import Path

INPUT_FILE = "tools/barbershop_dataset/training_sequences.txt"
OUTPUT_FILE = "tools/barbershop_dataset/training_sequences_causal.txt"


def parse_event_line(line):
    """
    Parse a token line into structured components.

    Format: [bar:N] [bass:M] [bari:M] [lead:M] [tenor:M] [dur:F] [chord:LABEL]

    Returns dict with keys: bar, bass, bari, lead, tenor, dur, chord
    Or None if not an event line (e.g., header or [song_end])
    """
    if not line.strip() or line.startswith('[key:') or line.startswith('[song_end]'):
        return None

    parts = {}

    # Extract bar number
    bar_match = re.search(r'\[bar:(\d+)\]', line)
    if bar_match:
        parts['bar'] = bar_match.group(1)
    else:
        return None

    # Extract voice pitches (or 'rest')
    for voice in ['bass', 'bari', 'lead', 'tenor']:
        match = re.search(rf'\[{voice}:([\d]+|rest)\]', line)
        if match:
            parts[voice] = match.group(1)

    # Extract duration
    dur_match = re.search(r'\[dur:([\d.]+)\]', line)
    if dur_match:
        parts['dur'] = dur_match.group(1)

    # Extract chord
    chord_match = re.search(r'\[chord:(\w+)\]', line)
    if chord_match:
        parts['chord'] = chord_match.group(1)

    return parts if len(parts) >= 6 else None  # Must have all key fields


def reorder_event_line(event):
    """
    Reorder tokens within an event line to causal order.

    Original:  [bar:N] [bass:B] [bari:BA] [lead:L] [tenor:T] [dur:D] [chord:C]
    Reordered: [bar:N] [chord:C] [lead:L] -> [bass:B] [bari:BA] [tenor:T] [dur:D]

    The arrow separator makes it clear: "Given chord and lead, generate the rest"
    """
    bar = event.get('bar', '?')
    chord = event.get('chord', 'OTHER')
    lead = event.get('lead', 'rest')
    bass = event.get('bass', 'rest')
    bari = event.get('bari', 'rest')
    tenor = event.get('tenor', 'rest')
    dur = event.get('dur', '1.0')

    # Format: constraints first, then arrow, then generation targets
    return f"[bar:{bar}] [chord:{chord}] [lead:{lead}] -> [bass:{bass}] [bari:{bari}] [tenor:{tenor}] [dur:{dur}]"


def reorder_tokens(input_path, output_path):
    """
    Reorder all tokens in the input file and write to output.
    """
    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")
    print()

    stats = {
        'header_lines': 0,
        'event_lines': 0,
        'song_end_lines': 0,
        'blank_lines': 0,
    }

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            line = line.rstrip('\n')

            # Preserve headers
            if line.startswith('[key:'):
                outfile.write(line + '\n')
                stats['header_lines'] += 1
                continue

            # Preserve song markers
            if line.startswith('[song_end]'):
                outfile.write(line + '\n')
                stats['song_end_lines'] += 1
                continue

            # Preserve blank lines
            if not line.strip():
                outfile.write('\n')
                stats['blank_lines'] += 1
                continue

            # Parse and reorder event lines
            event = parse_event_line(line)
            if event:
                reordered = reorder_event_line(event)
                outfile.write(reordered + '\n')
                stats['event_lines'] += 1
            else:
                # Line didn't parse; pass through as-is
                outfile.write(line + '\n')

    return stats


def main():
    print("=" * 70)
    print("TOKEN REORDERING FOR CAUSAL ARRANGEMENT LEARNING")
    print("=" * 70)
    print()

    # Check input exists
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"ERROR: {INPUT_FILE} not found")
        return

    # Reorder tokens
    stats = reorder_tokens(input_path, OUTPUT_FILE)

    # Summary
    print()
    print("=" * 70)
    print("REORDERING COMPLETE")
    print("=" * 70)
    print()
    print(f"Header lines:    {stats['header_lines']:,}")
    print(f"Event lines:     {stats['event_lines']:,}")
    print(f"Song end lines:  {stats['song_end_lines']:,}")
    print(f"Blank lines:     {stats['blank_lines']:,}")
    print()

    # Show sample
    print("SAMPLE OUTPUT:")
    print("-" * 70)
    with open(OUTPUT_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i < 25:
                print(line.rstrip('\n'))
            else:
                break
    print()
    print(f"Output: {OUTPUT_FILE}")
    output_size = Path(OUTPUT_FILE).stat().st_size / (1024*1024)
    print(f"File size: {output_size:.2f} MB")
    print()


if __name__ == "__main__":
    main()
