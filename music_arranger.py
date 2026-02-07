"""
Music Arranging Engine
Parses input melodies (MusicXML/MIDI), uses Claude to interpret natural language
harmony requests, maps them to constraint sets, feeds them into the CP-SAT solver,
and outputs valid MusicXML.
"""

import json
import os
from fractions import Fraction
from math import gcd

import anthropic
import music21

from solver_template import ArrangerSolver


# ---------------------------------------------------------------------------
# Tool schema sent to Claude for structured extraction
# ---------------------------------------------------------------------------
EXTRACT_TOOL = {
    "name": "apply_arrangement",
    "description": (
        "Extract structured musical arrangement parameters from a natural language request. "
        "Return the key, scale, cadence, chord assignments, and voice-leading preferences."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key_root": {
                "type": "integer",
                "description": "Root note as MIDI pitch class 0-11 (0=C, 1=C#, 2=D, 3=Eb, 4=E, 5=F, 6=F#, 7=G, 8=Ab, 9=A, 10=Bb, 11=B)",
            },
            "scale_type": {
                "type": "string",
                "enum": [
                    "major", "natural_minor", "harmonic_minor",
                    "melodic_minor_ascending", "pentatonic_major",
                    "pentatonic_minor", "whole_tone", "chromatic",
                ],
            },
            "cadence": {
                "type": "string",
                "enum": ["authentic", "plagal", "half", "deceptive", "perfect_authentic", "none"],
                "description": "Type of cadence to apply. Use 'none' if no cadence is requested.",
            },
            "cadence_position": {
                "type": "string",
                "enum": ["end", "middle"],
                "description": "Where to place the cadence in the arrangement.",
            },
            "chord_sequence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer", "description": "Zero-based time step index."},
                        "roman_numeral": {"type": "string", "description": "Roman numeral chord label (e.g. 'I', 'IV', 'V', 'vi')."},
                    },
                    "required": ["step", "roman_numeral"],
                },
                "description": "Optional explicit chord-per-step assignments using Roman numerals.",
            },
            "voice_leading_max_interval": {
                "type": "integer",
                "description": "Maximum semitone leap allowed between consecutive notes in a voice. Default 7.",
            },
            "per_voice_max_interval": {
                "type": "object",
                "description": "Optional per-voice leap limits (e.g. {\"alto\": 4, \"tenor\": 4}). Voices not listed use voice_leading_max_interval.",
                "additionalProperties": {"type": "integer"},
            },
            "enable_chord_completeness": {
                "type": "boolean",
                "description": "Ensure every chord tone appears in at least one voice. Default true.",
            },
            "enable_bass_restrictions": {
                "type": "boolean",
                "description": "Restrict bass to root/fifth (or chord-specific allowed tones). Default true.",
            },
        },
        "required": ["key_root", "scale_type"],
    },
}


class MelodyParser:
    """Parses MusicXML or MIDI files into discrete (step_index, midi_pitch) tuples."""

    def parse(self, filepath: str) -> tuple[list[tuple[int, int]], list[float], int]:
        """
        Parse a music file and return:
          - melody_steps: list of (step_index, midi_pitch) for notes only (no rests)
          - step_durations: list of float(grid) for every grid step
          - total_steps: total number of grid steps
        """
        score = music21.converter.parse(filepath)
        flat = score.flatten().notesAndRests  # Notes, Chords, and Rests

        elements = list(flat)
        if not elements:
            return [], [], 0

        # Compute grid resolution as GCD of all durations
        grid = self._compute_grid_from_elements(elements)

        # Total steps: from offset 0 to end of last element
        last = elements[-1]
        end_offset = Fraction(last.offset).limit_denominator(256) + \
                     Fraction(last.quarterLength).limit_denominator(256)
        total_steps = int(round(float(end_offset / grid)))

        melody_steps = []
        step_durations = [float(grid)] * total_steps

        for e in elements:
            dur = Fraction(e.quarterLength).limit_denominator(256)
            if dur <= 0:
                continue  # skip grace notes

            offset = Fraction(e.offset).limit_denominator(256)
            start_step = int(round(float(offset / grid)))
            span = int(round(float(dur / grid)))

            if isinstance(e, music21.note.Rest):
                continue  # rests produce no melody_steps entries

            if hasattr(e, 'pitch'):
                midi = e.pitch.midi
            else:
                # Chord — take the highest pitch (soprano line)
                midi = max(p.midi for p in e.pitches)

            # A note spanning multiple grid steps: repeat constraint at each sub-step
            for s in range(span):
                step_idx = start_step + s
                if step_idx < total_steps:
                    melody_steps.append((step_idx, midi))

        return melody_steps, step_durations, total_steps

    @staticmethod
    def _compute_grid_from_elements(elements) -> Fraction:
        """Compute the GCD of all element durations to determine grid resolution."""
        g = Fraction(0)
        for e in elements:
            dur = Fraction(e.quarterLength).limit_denominator(256)
            if dur > 0:
                if g == 0:
                    g = dur
                else:
                    # gcd of two fractions: gcd(a/b, c/d) = gcd(a*d, c*b) / (b*d)
                    # but simpler: use numerator/denominator properties
                    num = gcd(g.numerator * dur.denominator, dur.numerator * g.denominator)
                    den = g.denominator * dur.denominator
                    g = Fraction(num, den)
        return g if g > 0 else Fraction(1)


class RequestMapper:
    """Uses Claude with tool_use to map natural language requests to structured parameters."""

    def __init__(self, theory_file: str = 'theory_definitions.json'):
        self.client = anthropic.Anthropic()
        with open(theory_file, 'r') as f:
            self.theory = json.load(f)

    def map_request(self, request: str, num_steps: int = 0) -> dict:
        """
        Send a natural language arrangement request to Claude and get back
        structured constraint parameters via tool_use.
        """
        # Build context about available theory values
        available_scales = list(self.theory['scales'].keys())
        available_cadences = list(self.theory.get('cadences', {}).keys()) + ['none']
        diatonic_chords = self.theory.get('diatonic_chords', {})
        available_chords_info = {scale: list(chords.keys()) for scale, chords in diatonic_chords.items()}

        system_prompt = (
            "You are a music theory assistant. Extract structured arrangement parameters "
            "from the user's natural language request. Use the apply_arrangement tool to "
            "return the parameters.\n\n"
            f"Available scales: {available_scales}\n"
            f"Available cadences: {available_cadences}\n"
            f"Available diatonic chords per scale: {json.dumps(available_chords_info)}\n"
            f"Number of time steps in the melody: {num_steps}\n\n"
            "Pitch class mapping: 0=C, 1=C#/Db, 2=D, 3=Eb/D#, 4=E, 5=F, "
            "6=F#/Gb, 7=G, 8=Ab/G#, 9=A, 10=Bb/A#, 11=B\n\n"
            "NOTE: Cadences and chord_sequence are only supported for scales with "
            "diatonic chord definitions (major, natural_minor, harmonic_minor, "
            "melodic_minor_ascending). For other scales, set cadence to 'none' "
            "and leave chord_sequence empty."
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system=system_prompt,
            tools=[EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "apply_arrangement"},
            messages=[{"role": "user", "content": request}],
        )

        # Extract the tool_use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "apply_arrangement":
                params = block.input
                # Apply defaults
                params.setdefault('cadence', 'none')
                params.setdefault('cadence_position', 'end')
                params.setdefault('chord_sequence', [])
                params.setdefault('voice_leading_max_interval', 7)
                params.setdefault('per_voice_max_interval', None)
                params.setdefault('enable_chord_completeness', True)
                params.setdefault('enable_bass_restrictions', True)
                return params

        raise RuntimeError("Claude did not return an apply_arrangement tool call.")


class MusicXMLExporter:
    """Converts solver output to a music21 Score and writes MusicXML."""

    @staticmethod
    def _merge_consecutive(midi_notes: list[int], step_durations: list[float],
                           rest_steps: set[int]) -> list[tuple[int | None, float]]:
        """Merge consecutive same-pitch steps (or consecutive rests) into single entries.

        Returns list of (midi_or_None, total_duration) tuples.
        """
        merged: list[tuple[int | None, float]] = []
        for i, (midi_val, dur) in enumerate(zip(midi_notes, step_durations)):
            val = None if i in rest_steps else midi_val
            if merged and merged[-1][0] == val:
                merged[-1] = (val, merged[-1][1] + dur)
            else:
                merged.append((val, dur))
        return merged

    def export(
        self,
        solution: dict,
        step_durations: list[float],
        filepath: str,
        melody_voice: str | None = None,
        rest_steps: set[int] | None = None,
    ):
        """
        Write a MusicXML file from the solver solution.

        Args:
            solution: dict mapping voice_name -> list of MIDI note numbers
            step_durations: quarterLength per step
            filepath: output path (.musicxml)
            melody_voice: which voice carries the melody (for rest insertion)
            rest_steps: step indices where the melody has rests
        """
        if rest_steps is None:
            rest_steps = set()

        score = music21.stream.Score()

        for voice_name, midi_notes in solution.items():
            part = music21.stream.Part()
            part.partName = voice_name.capitalize()

            is_melody = (voice_name == melody_voice)
            merged = self._merge_consecutive(
                midi_notes, step_durations,
                rest_steps if is_melody else set(),
            )

            for midi_val, dur in merged:
                if midi_val is None:
                    n = music21.note.Rest()
                else:
                    n = music21.note.Note(midi_val)
                n.quarterLength = dur
                part.append(n)

            # Make measures from the flat note list
            part.makeMeasures(inPlace=True)
            score.append(part)

        score.write('musicxml', fp=filepath)
        return filepath


class MusicArranger:
    """Top-level orchestrator tying parser, LLM mapper, solver, and exporter together."""

    def __init__(self, theory_file: str = 'theory_definitions.json'):
        self.theory_file = theory_file
        self.parser = MelodyParser()
        self.mapper = RequestMapper(theory_file=theory_file)
        self.exporter = MusicXMLExporter()

        with open(theory_file, 'r') as f:
            self.theory = json.load(f)

    def arrange(
        self,
        melody_path: str,
        request: str,
        output_path: str = 'output.musicxml',
        voice_names: list[str] | None = None,
        melody_voice: str | None = None,
    ) -> str:
        """
        Full pipeline: parse melody -> interpret request -> solve -> export.

        Args:
            melody_path: path to MusicXML or MIDI file with the input melody
            request: natural language arrangement instruction
            output_path: where to write the output MusicXML
            voice_names: list of voice names (default: SATB)
            melody_voice: which voice carries the melody (default: first voice)

        Returns:
            Path to the written MusicXML file.
        """
        if voice_names is None:
            voice_names = ['soprano', 'alto', 'tenor', 'bass']

        # 1. Parse melody
        melody_steps, step_durations, num_steps = self.parser.parse(melody_path)
        note_step_set = {idx for idx, _ in melody_steps}
        rest_steps = set(range(num_steps)) - note_step_set
        note_count = len(note_step_set)
        print(f"Parsed {note_count} notes and {len(rest_steps)} rests from {melody_path}")

        # 2. Map NL request -> structured constraints via Claude
        constraints = self.mapper.map_request(request, num_steps=num_steps)
        print(f"Mapped constraints: {json.dumps(constraints, indent=2)}")

        # 3. Setup solver
        solver = ArrangerSolver(theory_file=self.theory_file)
        solver.setup_problem(num_steps, voice_names)

        # 4. Pin the melody
        if melody_voice is None:
            melody_voice = voice_names[0]
        elif melody_voice not in voice_names:
            raise ValueError(f"melody_voice '{melody_voice}' not in voice_names {voice_names}")
        for step_idx, midi_pitch in melody_steps:
            solver.add_melodic_constraint(melody_voice, step_idx, midi_pitch)

        # 5. Structural constraints
        solver.add_no_crossing_constraint()
        solver.add_voice_leading_constraint(
            constraints.get('voice_leading_max_interval', 7),
            per_voice_max=constraints.get('per_voice_max_interval'),
            exclude_voices=[melody_voice],
        )

        # 6. Scale constraint — keep all notes diatonic
        solver.add_scale_constraint(constraints['key_root'], constraints['scale_type'],
                                     exclude_voices=[melody_voice])

        # 7. Cadence
        cadence_type = constraints.get('cadence', 'none')
        if cadence_type and cadence_type != 'none':
            cadence_def = self.theory.get('cadences', {}).get(cadence_type)
            if cadence_def:
                prog_len = len(cadence_def['progression'])
                if constraints.get('cadence_position') == 'middle':
                    start = max(0, num_steps // 2 - prog_len // 2)
                else:
                    start = max(0, num_steps - prog_len)
                enable_comp = constraints.get('enable_chord_completeness', True)
                enable_bass = constraints.get('enable_bass_restrictions', True)
                # Skip bass restriction if melody voice is the bass
                if melody_voice == voice_names[-1]:
                    enable_bass = False
                solver.add_cadence_constraint(
                    constraints['key_root'], constraints['scale_type'],
                    cadence_type, start, melody_voice=melody_voice,
                    exclude_voices=[melody_voice],
                    ensure_completeness=enable_comp,
                    restrict_bass=enable_bass,
                )
                print(f"Applied {cadence_type} cadence at step {start}")

        # 8. Explicit chord sequence
        enable_comp = constraints.get('enable_chord_completeness', True)
        enable_bass = constraints.get('enable_bass_restrictions', True)
        if melody_voice == voice_names[-1]:
            enable_bass = False
        bass_voice = voice_names[-1]

        diatonic = self.theory.get('diatonic_chords', {}).get(constraints['scale_type'], {})
        for chord_spec in constraints.get('chord_sequence', []):
            step = chord_spec['step']
            numeral = chord_spec['roman_numeral']
            if step >= num_steps:
                continue
            chord_info = diatonic.get(numeral)
            if chord_info:
                absolute_root = (constraints['key_root'] + chord_info['root_degree']) % 12
                solver.add_harmonic_constraint(step, absolute_root, chord_info['quality'],
                                               exclude_voices=[melody_voice])
                if enable_comp:
                    solver.add_chord_completeness_constraint(
                        step, absolute_root, chord_info['quality'],
                        exclude_voices=[melody_voice])
                if enable_bass and bass_voice not in [melody_voice]:
                    solver.add_bass_restriction_constraint(
                        step, absolute_root, chord_info['quality'],
                        bass_voice=bass_voice)
                # Prefer root in bass
                solver.add_doubling_preference(step, absolute_root)

        # 9. Voicing quality: soft constraints
        solver.add_unison_penalty()
        solver.add_spacing_constraint()
        solver.add_parallel_octave_penalty()

        # 10. Solve
        print("Solving...")
        solution = solver.solve()
        if not solution:
            report = solver.get_diagnostic_report()
            print(report)
            raise RuntimeError("No feasible arrangement found. See diagnostics above.")

        print("Solution found!")
        for voice, notes in solution.items():
            names = []
            for m in notes:
                pname = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'][m % 12]
                octave = (m // 12) - 1
                names.append(f"{pname}{octave}")
            print(f"  {voice}: {names}")

        # 11. Export
        self.exporter.export(solution, step_durations, output_path,
                             melody_voice=melody_voice, rest_steps=rest_steps)
        print(f"Written to {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Music Arranging Engine')
    parser.add_argument('melody', help='Path to input melody file (MusicXML or MIDI)')
    parser.add_argument('request', help='Natural language arrangement request')
    parser.add_argument('-o', '--output', default='output.musicxml',
                        help='Output MusicXML path (default: output.musicxml)')
    parser.add_argument('-v', '--voices', nargs='+',
                        default=['soprano', 'alto', 'tenor', 'bass'],
                        help='Voice names (default: soprano alto tenor bass)')
    parser.add_argument('-m', '--melody-voice', default=None,
                        help='Which voice carries the melody (default: first voice)')
    parser.add_argument('-t', '--theory', default='theory_definitions.json',
                        help='Path to theory definitions JSON')
    args = parser.parse_args()

    arranger = MusicArranger(theory_file=args.theory)
    result = arranger.arrange(
        melody_path=args.melody,
        request=args.request,
        output_path=args.output,
        voice_names=args.voices,
        melody_voice=args.melody_voice,
    )
    print(f"Done: {result}")
