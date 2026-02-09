#!/usr/bin/env python3
"""
Detokenize Barbershop Quartet Arrangements to MusicXML

Parses token sequences from AI-generated arrangements and reconstructs
a music21 Score object, then exports to MusicXML format.

Input: Token file (from generate.py output)
Output: MusicXML file (playable in MuseScore, Finale, etc.)

Token format (from tokenizer.py):
  [key:X] [meter:N/D]
  [bar:N] [chord:LABEL] [bass:M] [bari:M] [lead:M] [tenor:M] [dur:F]
  [song_end]

Usage:
  python tools/detokenize.py <token_file.txt> [output.xml]
"""

import sys
import re
from pathlib import Path
from collections import defaultdict

try:
    import music21
except ImportError:
    print("ERROR: music21 not installed. Install with: pip install music21")
    sys.exit(1)


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_OUTPUT = "arrangement.xml"


# ============================================================================
# TOKENIZER CLASS
# ============================================================================

class BarbershopDetokenizer:
    """Parse token sequences and reconstruct music21 Score."""

    def __init__(self):
        """Initialize detokenizer state."""
        self.score = None
        self.parts = {}  # voice_name -> Part
        self.current_key = None
        self.current_meter = None
        self.current_time_signature = None
        self.current_clef = None

        # State machine for vertical slices
        self.active_slice = {}  # voice -> MIDI pitch (or None for rest)
        self.current_dur = 1.0  # Default quarter note

        # Statistics
        self.events_processed = 0
        self.lines_processed = 0
        self.errors = []

    def initialize_score(self):
        """Create a music21 Score with 4 parts (Tenor, Lead, Bari, Bass)."""
        self.score = music21.stream.Score()

        # Define voices with proper clefs
        voice_config = {
            'Tenor': ('treble8vb', 'Treble8vb'),
            'Lead': ('treble8vb', 'Treble8vb'),
            'Bari': ('bass', 'Bass'),
            'Bass': ('bass', 'Bass'),
        }

        for voice_name, (clef_type, clef_class) in voice_config.items():
            part = music21.stream.Part()
            part.id = voice_name

            # Add clef
            if clef_class == 'Treble8vb':
                clef = music21.clef.Treble8vbClef()
            else:
                clef = music21.clef.BassClef()

            part.append(clef)

            self.parts[voice_name] = part
            self.score.append(part)

    def parse_token_line(self, line):
        """
        Extract tokens from a line using regex.
        Returns list of tuples: (token_type, token_value)
        E.g., [('key', 'C'), ('meter', '4/4'), ...]
        """
        tokens = []
        # Match [key:value] patterns
        pattern = r'\[([a-z_]+):([^\]]+)\]'
        matches = re.findall(pattern, line)
        for key, value in matches:
            tokens.append((key, value))
        return tokens

    def apply_key_signature(self, key_str):
        """Parse key string and apply to all parts."""
        try:
            # key_str is like "C", "G", "A", etc.
            key = music21.key.Key(key_str)
            for part in self.parts.values():
                part.append(key)
            self.current_key = key
        except Exception as e:
            self.errors.append(f"Failed to parse key '{key_str}': {e}")

    def apply_time_signature(self, meter_str):
        """Parse meter string (N/D) and apply to all parts."""
        try:
            # meter_str is like "4/4", "3/4", etc.
            parts = meter_str.split('/')
            numerator = int(parts[0])
            denominator = int(parts[1])
            ts = music21.meter.TimeSignature(f'{numerator}/{denominator}')
            for part in self.parts.values():
                part.append(ts)
            self.current_meter = meter_str
            self.current_time_signature = ts
        except Exception as e:
            self.errors.append(f"Failed to parse meter '{meter_str}': {e}")

    def commit_slice(self):
        """
        Commit the current vertical slice to all parts.
        Called when [dur:X] is encountered.
        """
        if not self.active_slice and self.current_dur > 0:
            # Empty slice - add rests to all voices
            for voice_name in ['Tenor', 'Lead', 'Bari', 'Bass']:
                rest = music21.note.Rest(quarterLength=self.current_dur)
                self.parts[voice_name].append(rest)
            self.events_processed += 1
            return

        # For each voice, create note or rest
        for voice_name in ['Tenor', 'Lead', 'Bari', 'Bass']:
            midi_pitch = self.active_slice.get(voice_name)

            try:
                if midi_pitch is None:
                    # Rest
                    n = music21.note.Rest(quarterLength=self.current_dur)
                else:
                    # Note from MIDI pitch
                    n = music21.note.Note(midi_pitch, quarterLength=self.current_dur)

                self.parts[voice_name].append(n)
            except Exception as e:
                self.errors.append(
                    f"Failed to add note to {voice_name} (MIDI={midi_pitch}, "
                    f"dur={self.current_dur}): {e}"
                )
                # Add rest as fallback
                rest = music21.note.Rest(quarterLength=self.current_dur)
                self.parts[voice_name].append(rest)

        self.events_processed += 1
        self.active_slice = {}

    def process_tokens(self, token_list):
        """
        Process a list of (key, value) tuples from a line.
        Updates state machine and commits slices as needed.
        """
        for token_key, token_value in token_list:
            if token_key == 'key':
                self.apply_key_signature(token_value)

            elif token_key == 'meter':
                self.apply_time_signature(token_value)

            elif token_key == 'bar':
                # Bar number - just informational, ignore
                pass

            elif token_key == 'chord':
                # Chord label - store for potential future use
                pass

            elif token_key == 'dur':
                # Duration - triggers slice commit
                try:
                    self.current_dur = float(token_value)
                except ValueError:
                    self.errors.append(f"Invalid duration: {token_value}")
                    self.current_dur = 1.0

                # Commit the slice
                self.commit_slice()

            elif token_key in ['tenor', 'lead', 'bari', 'bass']:
                # Voice pitch
                voice_name = token_key.capitalize()
                if token_value == 'rest':
                    self.active_slice[voice_name] = None
                else:
                    try:
                        midi_pitch = int(token_value)
                        self.active_slice[voice_name] = midi_pitch
                    except ValueError:
                        self.errors.append(
                            f"Invalid MIDI pitch for {voice_name}: {token_value}"
                        )
                        self.active_slice[voice_name] = None

            elif token_key == 'song_end':
                # End of song - finalize
                pass

    def process_file(self, file_path):
        """
        Read and process token file.
        Yields progress updates and errors.
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            return f"ERROR: Failed to read {file_path}: {e}"

        print(f"📖 Parsing {len(lines)} lines from {file_path}")

        self.initialize_score()

        for line_num, line in enumerate(lines, 1):
            self.lines_processed += 1
            line = line.strip()

            if not line:
                continue

            # Extract tokens from line
            token_list = self.parse_token_line(line)

            if not token_list:
                # Line has no tokens, skip
                continue

            # Process tokens
            self.process_tokens(token_list)

            if line_num % 100 == 0:
                print(f"  {line_num}/{len(lines)}... ({self.events_processed} events)")

        # Ensure final slice is committed
        if self.active_slice:
            self.commit_slice()

        return None  # Success

    def validate(self):
        """
        Validate the generated score.
        Returns list of validation issues (warnings/errors).
        """
        issues = []

        if not self.parts:
            issues.append("❌ No parts created")
            return issues

        # Check all parts have same total duration
        durations = {}
        for voice_name, part in self.parts.items():
            total_dur = part.duration.quarterLength
            durations[voice_name] = total_dur

        unique_durs = set(durations.values())
        if len(unique_durs) > 1:
            issues.append(
                f"⚠️  Parts have different durations: {durations}"
            )
        else:
            duration = list(unique_durs)[0]
            print(f"✅ All parts have duration {duration} quarter notes "
                  f"({duration // 4} bars in 4/4)")

        return issues

    def export(self, output_path):
        """Export score to MusicXML file."""
        try:
            self.score.write('musicxml', fp=output_path)
            file_size = Path(output_path).stat().st_size / 1024  # KB
            print(f"✅ Exported to {output_path} ({file_size:.1f} KB)")
            return None
        except Exception as e:
            return f"❌ Export failed: {e}"


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("BARBERSHOP QUARTET DETOKENIZER")
    print("=" * 70)
    print()

    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python tools/detokenize.py <token_file.txt> [output.xml]")
        print()
        print("Examples:")
        print("  python tools/detokenize.py generated.txt arrangement.xml")
        print("  python tools/detokenize.py output.txt  # saves to arrangement.xml")
        sys.exit(1)

    token_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    # Validate input
    if not Path(token_file).exists():
        print(f"❌ Token file not found: {token_file}")
        sys.exit(1)

    # Detokenize
    detokenizer = BarbershopDetokenizer()

    print(f"📂 Input:  {token_file}")
    print(f"📝 Output: {output_file}")
    print()

    # Process file
    error = detokenizer.process_file(token_file)
    if error:
        print(error)
        sys.exit(1)

    print()
    print(f"✅ Parsed {detokenizer.lines_processed} lines, "
          f"{detokenizer.events_processed} events")
    print()

    # Validate
    print("🔍 Validating score...")
    issues = detokenizer.validate()
    if issues:
        for issue in issues:
            print(f"  {issue}")

    print()

    # Export
    print("💾 Exporting to MusicXML...")
    error = detokenizer.export(output_file)
    if error:
        print(error)
        sys.exit(1)

    # Report errors
    if detokenizer.errors:
        print()
        print(f"⚠️  {len(detokenizer.errors)} warnings/errors during processing:")
        for err in detokenizer.errors[:10]:
            print(f"  • {err}")
        if len(detokenizer.errors) > 10:
            print(f"  ... and {len(detokenizer.errors) - 10} more")

    print()
    print("=" * 70)
    print("✅ DETOKENIZATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Next steps:")
    print(f"  1. Open {output_file} in MuseScore or Finale")
    print(f"  2. Play back to hear the arrangement")
    print(f"  3. Export to MP3: mscore {output_file} -o arrangement.mp3")
    print()


if __name__ == "__main__":
    main()
