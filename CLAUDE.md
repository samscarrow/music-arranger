# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A CP-SAT constraint solver (Google OR-Tools) for SATB music arrangement. A Claude API layer (Sonnet 4.5) interprets natural language requests into structured constraints that feed the solver. Input is a melody file (MusicXML/MIDI) + natural language instruction; output is a fully-arranged MusicXML score.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install ortools anthropic music21
```

Requires `ANTHROPIC_API_KEY` env var for the full pipeline. The solver alone (`solver_template.py`) only needs `ortools`.

## Running

```bash
# Full pipeline (requires API key + melody file)
python music_arranger.py melody.xml "Arrange in C major with an authentic cadence" -o output.musicxml

# Solver only (no API key needed)
python solver_template.py

# Verification test for exclude_voices range logic
python verify_solver.py
```

## Architecture

Three-file system, no package structure:

- **`solver_template.py`** — `ArrangerSolver` class wrapping CP-SAT. Creates MIDI pitch integer variables per voice per time step, then applies constraints (harmonic, melodic, scale, cadence, voice-leading, doubling preference). Hard constraints use `AddAllowedAssignments`; soft constraints (doubling) use boolean objective terms.

- **`music_arranger.py`** — Four classes forming the pipeline:
  - `MelodyParser` — Parses MusicXML/MIDI via music21 into `(step_index, midi_pitch)` tuples. Computes grid resolution as GCD of all durations. Handles rests, tied notes, and chords.
  - `RequestMapper` — Sends NL request to Claude with `tool_use` (forced tool call to `apply_arrangement`) and extracts structured params: key_root, scale_type, cadence, chord_sequence, voice_leading_max_interval.
  - `MusicXMLExporter` — Converts solver output back to music21 Score. Merges consecutive same-pitch steps into single notes.
  - `MusicArranger` — Top-level orchestrator running parse → map → solve → export.

- **`theory_definitions.json`** — All music theory data: scales (as pitch class interval sets), chords (triads + sevenths), voice ranges (MIDI min/max + tessitura), cadence progressions (Roman numeral sequences), and diatonic chord tables per scale type.

## Critical Constraint Pattern: exclude_voices

When a melody voice is pinned via `add_melodic_constraint`, the pinned MIDI pitch may fall outside that voice's strict range in `theory_definitions.json`. If harmonic/scale/cadence constraints also enforce the strict range for that voice, the solver becomes **infeasible**.

The fix: `add_harmonic_constraint`, `add_scale_constraint`, and `add_cadence_constraint` all accept an `exclude_voices` parameter. Excluded voices use the wide variable domain (24-96) instead of the strict range from `theory_definitions.json`. The orchestrator in `music_arranger.py` always passes `exclude_voices=[melody_voice]`.

When adding new constraint methods to the solver, follow this pattern: accept `exclude_voices` and use range 24-96 for those voices.

## How the Solver Works

1. Variables are created with domain [24, 96] (wide MIDI range) via `NewIntVar`
2. `AddAllowedAssignments` narrows each variable's domain per constraint (scale membership, chord tones, voice range)
3. Multiple `AddAllowedAssignments` on the same variable intersect — the variable must satisfy ALL of them
4. This means constraint order doesn't matter, but conflicting constraints cause infeasibility
5. Soft constraints (doubling preference) use `BoolVar` indicators maximized in the objective

## Pitch Representation

All pitches are MIDI integers. Pitch class = `midi % 12` (0=C through 11=B). Octave = `(midi // 12) - 1`. Theory data uses semitone intervals from root (e.g., major chord = [0, 4, 7]).
