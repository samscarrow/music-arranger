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


def reverse_reorder_event_line(event):
    """
    Convert melody-first format back to physical ordering for detokenizer.

    Input (melody-first):  [bar:N] [lead:M] [dur:F] [chord:X] [bass:B] [bari:BA] [tenor:T]
    Output (physical):     [bar:N] [chord:X] [bass:B] [bari:BA] [lead:M] [tenor:T] [dur:F]
    """
    bar = event.get('bar', '?')
    chord = event.get('chord', 'OTHER')
    bass = event.get('bass', 'rest')
    bari = event.get('bari', 'rest')
    lead = event.get('lead', 'rest')
    tenor = event.get('tenor', 'rest')
    dur = event.get('dur', '1.0')

    # Physical order: bar, chord, bass, bari, lead, tenor, dur
    return f"[bar:{bar}] [chord:{chord}] [bass:{bass}] [bari:{bari}] [lead:{lead}] [tenor:{tenor}] [dur:{dur}]"


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


def reverse_reorder_tokens(input_path, output_path):
    """
    Reverse the reordering - convert melody-first back to physical order.

    This is used to convert harmonizer output (melody-first format)
    back to the physical format expected by the detokenizer.

    Handles both:
    - Multi-line format: one event per line
    - Single-line format: all tokens space-separated on one line
    """
    print(f"Converting from melody-first to physical order...")
    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")
    print()

    stats = {
        'header_lines': 0,
        'event_lines': 0,
        'song_end_lines': 0,
    }

    # Read entire file and tokenize
    with open(input_path, 'r') as f:
        content = f.read()

    # Extract all [token:value] patterns
    tokens = re.findall(r'\[([a-z_]+):([^\]]+)\]', content)

    # Build output in phases: headers, events, markers
    headers = []
    events = []
    song_end = False
    current_event = {}

    for token_type, token_value in tokens:
        if token_type == 'key':
            # Save any pending event before new header
            if current_event and 'bar' in current_event:
                events.append(current_event.copy())
                current_event = {}
            headers.append(f'[key:{token_value}]')
            stats['header_lines'] += 1

        elif token_type == 'meter':
            headers.append(f'[meter:{token_value}]')
            stats['header_lines'] += 1

        elif token_type == 'song_end':
            # Save any pending event before song_end marker
            if current_event and 'bar' in current_event:
                events.append(current_event.copy())
                current_event = {}
            song_end = True
            stats['song_end_lines'] += 1

        elif token_type == 'bar':
            # Starting a new event - save the old one first
            if current_event and 'bar' in current_event:
                events.append(current_event.copy())
            current_event = {'bar': token_value}

        else:
            # Add token to current event (only for voice/chord/dur tokens)
            if token_type in ['chord', 'bass', 'bari', 'lead', 'tenor', 'dur']:
                current_event[token_type] = token_value

    # Save last event
    if current_event and 'bar' in current_event:
        events.append(current_event)

    # Write output
    with open(output_path, 'w') as outfile:
        # Write headers
        for header in headers:
            outfile.write(header + '\n')

        # Write events with reordered tokens
        for event in events:
            reversed_line = reverse_reorder_event_line(event)
            outfile.write(reversed_line + '\n')
            stats['event_lines'] += 1

        # Write song end marker
        if song_end:
            outfile.write('[song_end]\n')

    return stats


def main():
    import sys

    # Check for --reverse flag
    if len(sys.argv) > 1 and sys.argv[1] == '--reverse':
        # Reverse mode: melody-first → physical order
        if len(sys.argv) < 4:
            print("Usage: python reorder_tokens.py --reverse <input> <output>")
            print()
            print("Convert melody-first token format (from harmonizer)")
            print("back to physical format (for detokenizer)")
            print()
            print("Example:")
            print("  python reorder_tokens.py --reverse harmonized_output.txt reordered.txt")
            sys.exit(1)

        input_file = sys.argv[2]
        output_file = sys.argv[3]

        # Check input exists
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"ERROR: {input_file} not found")
            sys.exit(1)

        print("=" * 70)
        print("TOKEN REVERSE REORDERING")
        print("=" * 70)
        print()

        stats = reverse_reorder_tokens(input_file, output_file)

        # Summary
        print()
        print("=" * 70)
        print("REVERSE REORDERING COMPLETE")
        print("=" * 70)
        print()
        print(f"Header lines:    {stats['header_lines']:,}")
        print(f"Event lines:     {stats['event_lines']:,}")
        print(f"Song end lines:  {stats['song_end_lines']:,}")
        print()

        # Show sample
        print("SAMPLE OUTPUT (first 10 event lines):")
        print("-" * 70)
        event_count = 0
        with open(output_file, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('[bar:'):
                    print(line)
                    event_count += 1
                    if event_count >= 10:
                        break
                elif event_count == 0 and line.startswith('[key:'):
                    print(line)
        print()
        print(f"Output: {output_file}")
        output_size = Path(output_file).stat().st_size / (1024*1024)
        print(f"File size: {output_size:.2f} MB")
        print()

    else:
        # Forward mode: physical → melody-first (existing behavior)
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
