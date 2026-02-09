# Copilot Instructions

## Project Overview

A barbershop quartet arranger with two independent subsystems sharing `theory_definitions.json`:

1. **CP-SAT Constraint Solver** (`solver_template.py` + `music_arranger.py`) — Takes a melody file (MusicXML/MIDI) + natural language instruction, uses Claude API to extract structured constraints, feeds them into Google OR-Tools CP-SAT, and outputs a fully-arranged MusicXML score.

2. **Transformer Pipeline** (`tools/`) — Trains a NanoGPT-style Transformer on tokenized barbershop quartet scores, then generates harmonizations from a melody. Pipeline: `tokenizer.py` → `train.py` → `harmonize.py` → `detokenize.py` → MusicXML.

## Setup & Running

```bash
python -m venv venv && source venv/bin/activate
pip install ortools anthropic music21  # solver subsystem
pip install torch                      # transformer subsystem
```

```bash
# CP-SAT solver (full pipeline, needs ANTHROPIC_API_KEY)
python music_arranger.py melody.xml "Arrange in C major with an authentic cadence" -o output.musicxml

# CP-SAT solver only (no API key)
python solver_template.py

# Verification tests
python verify_solver.py              # exclude_voices range logic
python tools/test_event.py           # event schema round-trip (10+ tests)

# Transformer pipeline
python tools/tokenizer.py            # MusicXML → training tokens
python tools/train.py                # train model (~5min GPU, ~1hr CPU)
python tools/harmonize.py            # melody → harmonized tokens
python tools/detokenize.py tokens.txt output.xml  # tokens → MusicXML
```

## Architecture

### CP-SAT Subsystem

- `solver_template.py` — `ArrangerSolver` class. Creates MIDI pitch integer variables per voice per time step, applies constraints via `AddAllowedAssignments` (hard) and `BoolVar` objective terms (soft). Multiple `AddAllowedAssignments` on the same variable intersect — conflicting constraints cause infeasibility.
- `music_arranger.py` — Pipeline orchestrator with four classes: `MelodyParser` (MusicXML/MIDI → step tuples), `RequestMapper` (NL → Claude `tool_use` → structured params), `MusicXMLExporter` (solver output → music21 Score), `MusicArranger` (top-level).
- `theory_definitions.json` — All music theory data: scales (pitch class interval sets), chords (triads + sevenths), voice ranges (MIDI min/max + tessitura), cadence progressions, diatonic chord tables.

### Transformer Subsystem

- `tools/event.py` — **Single source of truth** for the token/event contract. `Header` and `Event` dataclasses, `parse_tokens()` (order-agnostic), `format_events()` (canonical output), `quarter_notes_per_bar()`. Round-trip property: `parse_tokens(format_events(h, e)) == (h, e)`.
- `tools/tokenizer.py` — MusicXML → training token sequences (transposed to C/Am).
- `tools/train.py` — 6-layer Transformer (384 dim, 6 heads). Melody-first causal ordering: model sees `[bar] [lead] [dur]`, predicts `[chord] [bass] [bari] [tenor]`.
- `tools/harmonize.py` — Force-feeds melody tokens, samples harmony from model, builds `Event` objects, serializes via `format_events()`.
- `tools/detokenize.py` — Consumes `Event` objects from `parse_tokens()`, writes MusicXML via music21.
- `tools/generate.py` — Free generation from checkpoint (architecture must match `train.py` exactly).
- `tools/test_event.py` — Round-trip tests, legacy format parsing, edge cases.

## Critical Patterns

### exclude_voices (CP-SAT)

When a melody voice is pinned via `add_melodic_constraint`, the pinned MIDI pitch may fall outside that voice's strict range in `theory_definitions.json`. If other constraints also enforce the strict range, the solver becomes **infeasible**.

**Every constraint method** must accept `exclude_voices` and use the wide domain (24–96) for those voices instead of the strict range. The orchestrator always passes `exclude_voices=[melody_voice]`.

### Event Contract (Transformer)

All token I/O flows through `tools/event.py`. Never parse tokens ad-hoc or commit slices on `[dur:]` — duration is a field on `Event`, not a commit trigger. The canonical format is one event per line; `parse_tokens()` also handles legacy single-line packed formats as a fallback.

### Pitch Representation

All pitches are MIDI integers. Pitch class = `midi % 12` (0=C through 11=B). Octave = `(midi // 12) - 1`. Theory data uses semitone intervals from root (e.g., major chord = `[0, 4, 7]`).

### Barbershop Voice Conventions

Voices are **TTBB** (Tenor, Lead, Baritone, Bass) — not SATB. Lead carries the melody; Tenor sits above Lead. Voice names in code: `tenor`/`tenor_barbershop`, `lead`, `baritone`/`bari`, `bass`/`bass_barbershop`. The solver also supports SATB (`soprano`, `alto`, `tenor`, `bass`).

### Model Architecture Duplication

`train.py`, `generate.py`, and `harmonize.py` each define the Transformer classes independently. They must match exactly (including dropout layers) or `load_state_dict()` will fail. This is a known tech debt — consolidation into a shared `model.py` is planned.

## Key Conventions

- `theory_definitions.json` is the single source of truth for scales, chords, voice ranges, and cadence data. Changes to music theory should go here, not be hardcoded in Python.
- `offset_qn` (cumulative quarter-note offset from song start) is the authoritative time coordinate; `bar` is derived from it via `int(offset_qn // qn_per_bar) + 1`.
- Bar math must use `quarter_notes_per_bar(meter)` — never hardcode `4.0`.
- Verification scripts (`verify_*.py`) are standalone — run them directly with `python verify_solver.py` etc.
- No package structure — files import each other via `sys.path` manipulation or direct import from project root.
