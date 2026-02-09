#!/usr/bin/env python3
"""
Detokenize Barbershop Quartet Arrangements to MusicXML

Parses token sequences from AI-generated arrangements and reconstructs
a music21 Score object, then exports to MusicXML format.

Input: Token file (any format — canonical multi-line, legacy single-line, etc.)
Output: MusicXML file (playable in MuseScore, Finale, etc.)

Uses the canonical Event schema from event.py as the single source of truth
for parsing. Supports all token orderings (physical, melody-first, canonical).

Usage:
  python tools/detokenize.py <token_file.txt> [output.xml]
"""

import sys
from pathlib import Path

# Add tools dir to path for event import
sys.path.insert(0, str(Path(__file__).parent))
from event import parse_tokens, quarter_notes_per_bar, Header, Event

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
# DETOKENIZER CLASS
# ============================================================================

class BarbershopDetokenizer:
    """Parse token sequences and reconstruct music21 Score."""

    def __init__(self):
        """Initialize detokenizer state."""
        self.score = None
        self.parts = {}  # voice_name -> Part
        self.events_processed = 0
        self.errors = []

    def initialize_score(self):
        """Create a music21 Score with 4 parts (Tenor, Lead, Bari, Bass)."""
        self.score = music21.stream.Score()

        voice_config = {
            'Tenor': 'Treble8vb',
            'Lead': 'Treble8vb',
            'Bari': 'Bass',
            'Bass': 'Bass',
        }

        for voice_name, clef_class in voice_config.items():
            part = music21.stream.Part()
            part.id = voice_name

            if clef_class == 'Treble8vb':
                clef = music21.clef.Treble8vbClef()
            else:
                clef = music21.clef.BassClef()

            part.append(clef)
            self.parts[voice_name] = part
            self.score.append(part)

    def apply_key_signature(self, key_str):
        """Parse key string and apply to all parts."""
        try:
            key = music21.key.Key(key_str)
            for part in self.parts.values():
                part.append(key)
        except Exception as e:
            self.errors.append(f"Failed to parse key '{key_str}': {e}")

    def apply_time_signature(self, meter_str):
        """Parse meter string (N/D) and apply to all parts."""
        try:
            parts = meter_str.split('/')
            numerator = int(parts[0])
            denominator = int(parts[1])
            ts = music21.meter.TimeSignature(f'{numerator}/{denominator}')
            for part in self.parts.values():
                part.append(ts)
        except Exception as e:
            self.errors.append(f"Failed to parse meter '{meter_str}': {e}")

    def _write_event(self, event: Event):
        """Write one Event to all voice parts."""
        voice_fields = {
            'Tenor': event.tenor,
            'Lead': event.lead,
            'Bari': event.bari,
            'Bass': event.bass,
        }

        for voice_name, midi_pitch in voice_fields.items():
            try:
                if midi_pitch is None:
                    n = music21.note.Rest(quarterLength=event.dur)
                else:
                    n = music21.note.Note(midi_pitch, quarterLength=event.dur)
                self.parts[voice_name].append(n)
            except Exception as e:
                self.errors.append(
                    f"Failed to add note to {voice_name} (MIDI={midi_pitch}, "
                    f"dur={event.dur}): {e}"
                )
                rest = music21.note.Rest(quarterLength=event.dur)
                self.parts[voice_name].append(rest)

        self.events_processed += 1

    def process_file(self, file_path):
        """Read and process token file using canonical Event parser."""
        try:
            with open(file_path, 'r') as f:
                text = f.read()
        except Exception as e:
            return f"ERROR: Failed to read {file_path}: {e}"

        # Parse using canonical event schema
        header, events = parse_tokens(text)

        print(f"Parsed {len(events)} events from {file_path}")

        # Build score
        self.initialize_score()
        self.apply_key_signature(header.key)
        self.apply_time_signature(header.meter)

        for event in events:
            self._write_event(event)

        return None  # Success

    def validate(self, meter=None):
        """
        Validate the generated score.
        Returns list of validation issues (warnings/errors).
        """
        issues = []

        if not self.parts:
            issues.append("No parts created")
            return issues

        # Check all parts have same total duration
        durations = {}
        for voice_name, part in self.parts.items():
            total_dur = part.duration.quarterLength
            durations[voice_name] = total_dur

        unique_durs = set(durations.values())
        if len(unique_durs) > 1:
            issues.append(f"Parts have different durations: {durations}")
        else:
            duration = list(unique_durs)[0]
            if meter:
                qn_per_bar = quarter_notes_per_bar(meter)
                bars = duration / qn_per_bar
                print(f"All parts have duration {duration} quarter notes "
                      f"({bars:.1f} bars in {meter})")
            else:
                print(f"All parts have duration {duration} quarter notes")

        # Report voice ranges
        for voice_name, part in self.parts.items():
            pitches = []
            for n in part.flatten().notes:
                if hasattr(n, 'pitch'):
                    pitches.append(n.pitch.midi)
            if pitches:
                print(f"  {voice_name}: MIDI {min(pitches)}-{max(pitches)}")

        return issues

    def export(self, output_path):
        """Export score to MusicXML file."""
        try:
            self.score.write('musicxml', fp=output_path)
            file_size = Path(output_path).stat().st_size / 1024  # KB
            print(f"Exported to {output_path} ({file_size:.1f} KB)")
            return None
        except Exception as e:
            return f"Export failed: {e}"


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
        print(f"Token file not found: {token_file}")
        sys.exit(1)

    # Detokenize
    detokenizer = BarbershopDetokenizer()

    print(f"Input:  {token_file}")
    print(f"Output: {output_file}")
    print()

    # Process file
    error = detokenizer.process_file(token_file)
    if error:
        print(error)
        sys.exit(1)

    print(f"Processed {detokenizer.events_processed} events")
    print()

    # Validate
    print("Validating score...")
    issues = detokenizer.validate()
    if issues:
        for issue in issues:
            print(f"  WARNING: {issue}")

    print()

    # Export
    print("Exporting to MusicXML...")
    error = detokenizer.export(output_file)
    if error:
        print(error)
        sys.exit(1)

    # Report errors
    if detokenizer.errors:
        print()
        print(f"{len(detokenizer.errors)} warnings/errors during processing:")
        for err in detokenizer.errors[:10]:
            print(f"  - {err}")
        if len(detokenizer.errors) > 10:
            print(f"  ... and {len(detokenizer.errors) - 10} more")

    print()
    print("=" * 70)
    print("DETOKENIZATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Next steps:")
    print(f"  1. Open {output_file} in MuseScore or Finale")
    print(f"  2. Play back to hear the arrangement")
    print(f"  3. Export to MP3: mscore {output_file} -o arrangement.mp3")
    print()


if __name__ == "__main__":
    main()
