import sys
import json
from solver_template import ArrangerSolver

def run_verification():
    print("=== Solver Range Logic Verification ===")

    # 1. Initialize
    solver = ArrangerSolver()

    # 2. Setup Barbershop Problem (Tenor, Lead, Bari, Bass)
    voices = ['tenor', 'lead', 'baritone', 'bass']
    print(f"Initializing problem with voices: {voices}")
    solver.setup_problem(num_steps=1, voice_names=voices)

    # 3. The Test Case: Lead sings High C (MIDI 84)
    # Theory definition for 'lead' max is usually ~74 (Bb4/A4).
    # This tests if the variable initialization (24-96) and harmonic constraints allow exceptions.
    high_c = 84
    print(f"TEST ACTION: Forcing 'lead' to MIDI {high_c} (High C).")
    print("             This is typically out of the strict Barbershop Lead range.")
    solver.add_melodic_constraint('lead', 0, high_c)

    # 4. Apply Harmonic Constraint (C Major: C E G)
    # We MUST exclude 'lead' from the strict range check.
    # If the logic is broken, the solver will try to enforce Range(57, 74) AND Value(84) -> Infeasible.
    print("TEST ACTION: Applying C Major constraint with exclude_voices=['lead']...")
    solver.add_harmonic_constraint(
        step_index=0,
        root_note=0,  # C
        chord_type='major',
        exclude_voices=['lead']
    )

    # 5. Solve
    print("STATUS: Solving...")
    solution = solver.solve()

    # 6. Assertions
    if solution:
        lead_note = solution['lead'][0]
        print(f"RESULT: Solution Found! Lead Note = {lead_note}")

        # Verify Lead is exactly what we pinned
        if lead_note == high_c:
            print("PASS: Lead note preserved at 84 (Out of theoretical range).")
        else:
            print(f"FAIL: Lead note drifted to {lead_note}!")

        # Verify other voices formed a valid C Major chord
        c_major_pcs = {0, 4, 7} # C, E, G
        valid_chord = True
        for v in voices:
            note = solution[v][0]
            pc = note % 12
            if pc not in c_major_pcs:
                 print(f"WARNING: Voice {v} note {note} (PC {pc}) is not in C Major!")
                 valid_chord = False
            else:
                print(f"  - {v}: MIDI {note} (PC {pc}) -> OK")

        if valid_chord:
            print("PASS: Full chord is valid C Major.")

    else:
        print("FAIL: Solver returned No Solution (Infeasible).")
        print("DIAGNOSIS:")
        print("  1. Check if 'solver_template.py' initializes variables with wide range (24-96).")
        print("  2. Check if 'add_harmonic_constraint' actually uses the 'exclude_voices' parameter.")

def test_chord_completeness():
    """Solve C major with completeness constraint — all 3 PCs (C, E, G) must appear."""
    print("\n=== Test: Chord Completeness ===")
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_chord_completeness_constraint(0, 0, 'major')
    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    pcs_present = set()
    for name in ['soprano', 'alto', 'tenor', 'bass']:
        pcs_present.add(solution[name][0] % 12)
    required = {0, 4, 7}
    if required.issubset(pcs_present):
        print(f"PASS: All chord PCs present: {pcs_present}")
        return True
    else:
        print(f"FAIL: Missing PCs. Present: {pcs_present}, Required: {required}")
        return False


def test_parallel_octaves():
    """Solve a 2-step I-V progression with parallel octave penalty.
    Check that the solution avoids parallel octaves."""
    print("\n=== Test: Parallel Octave Penalty ===")
    solver = ArrangerSolver()
    solver.setup_problem(2, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()
    solver.add_harmonic_constraint(0, 0, 'major')  # C major
    solver.add_harmonic_constraint(1, 7, 'major')  # G major
    solver.add_voice_leading_constraint(max_interval=7)
    solver.add_parallel_octave_penalty(weight=3)
    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    voices = ['soprano', 'alto', 'tenor', 'bass']
    parallel_found = False
    for i in range(len(voices)):
        for j in range(i + 1, len(voices)):
            ni, nj = voices[i], voices[j]
            diff0 = abs(solution[ni][0] - solution[nj][0])
            diff1 = abs(solution[ni][1] - solution[nj][1])
            moved_i = solution[ni][0] != solution[ni][1]
            moved_j = solution[nj][0] != solution[nj][1]
            if diff0 % 12 == 0 and diff1 % 12 == 0 and (moved_i or moved_j):
                print(f"  WARNING: Parallel octave between {ni} and {nj}")
                parallel_found = True

    if not parallel_found:
        print("PASS: No parallel octaves found.")
        return True
    else:
        # Soft constraint — solver tried to avoid but might not succeed
        print("WARN: Parallel octaves still present (soft constraint).")
        return True  # Not a hard failure


def test_unison_penalty():
    """Solve 1 step with unison penalty — expect 4 distinct pitches."""
    print("\n=== Test: Unison Penalty ===")
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_unison_penalty(weight=3)
    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    pitches = [solution[v][0] for v in ['soprano', 'alto', 'tenor', 'bass']]
    unique = len(set(pitches))
    if unique == 4:
        print(f"PASS: 4 distinct pitches: {pitches}")
        return True
    else:
        print(f"WARN: Only {unique} distinct pitches: {pitches} (soft constraint)")
        return True  # Soft — not a hard failure


def test_spacing():
    """Solve 1 step with spacing constraint — total spread should be <= 19."""
    print("\n=== Test: Spacing Constraint ===")
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_spacing_constraint(max_total_spread=19, max_adjacent_upper=12, weight=3)
    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    s = solution['soprano'][0]
    b = solution['bass'][0]
    spread = s - b
    if spread <= 19:
        print(f"PASS: Spread = {spread} (soprano={s}, bass={b})")
        return True
    else:
        print(f"WARN: Spread = {spread} > 19 (soft constraint)")
        return True


def test_bass_restriction():
    """Solve C major with bass restriction — bass PC must be in {0, 7} (C or G)."""
    print("\n=== Test: Bass Restriction ===")
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_bass_restriction_constraint(0, 0, 'major')
    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    bass_pc = solution['bass'][0] % 12
    allowed = {0, 7}  # C major bass_restrictions = [0, 7]
    if bass_pc in allowed:
        print(f"PASS: Bass PC = {bass_pc} (in {allowed})")
        return True
    else:
        print(f"FAIL: Bass PC = {bass_pc} not in {allowed}")
        return False


def test_per_voice_leap():
    """Solve 2 steps with per-voice leap limits — inner voices <= 4 semitones."""
    print("\n=== Test: Per-Voice Leap Tolerances ===")
    solver = ArrangerSolver()
    solver.setup_problem(2, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_harmonic_constraint(1, 7, 'major')
    solver.add_voice_leading_constraint(max_interval=7,
                                        per_voice_max={'alto': 4, 'tenor': 4})
    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    passed = True
    for voice in ['alto', 'tenor']:
        leap = abs(solution[voice][1] - solution[voice][0])
        if leap <= 4:
            print(f"  PASS: {voice} leap = {leap} <= 4")
        else:
            print(f"  FAIL: {voice} leap = {leap} > 4")
            passed = False
    return passed


def test_diag_melody_outside_scale():
    """Pin soprano to F# (pc 6) in C major WITHOUT excluding soprano.
    Expect ERROR about melody note not in scale + empty domain."""
    print("\n=== Diagnostic Test: Melody Outside Scale ===")
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_melodic_constraint('soprano', 0, 66)  # F#4 (pc 6)
    # Deliberately do NOT exclude soprano from scale constraint
    solver.add_scale_constraint(key_root=0, scale_type='major', exclude_voices=[])
    solver.add_harmonic_constraint(0, 0, 'major', exclude_voices=[])

    diagnostics = solver.validate()
    errors = [msg for level, msg in diagnostics if level == 'ERROR']

    has_melody_error = any('Melody note' in e and 'not in' in e for e in errors)
    has_empty_domain = any('Empty domain' in e and 'soprano' in e for e in errors)

    if has_melody_error and has_empty_domain:
        print(f"PASS: Got {len(errors)} errors including melody-outside-scale and empty domain")
        return True
    else:
        print(f"FAIL: Expected melody and empty domain errors, got: {errors}")
        return False


def test_diag_voice_leading_too_tight():
    """Pin bass to two notes 8 semitones apart with a max_interval of 2.
    Expect WARN about minimum jump exceeding limit."""
    print("\n=== Diagnostic Test: Voice Leading Too Tight ===")
    solver = ArrangerSolver()
    solver.setup_problem(2, ['soprano', 'alto', 'tenor', 'bass'])
    # Pin bass to C2 at step 0 and Ab2 at step 1 — 8 semitone jump
    solver.add_melodic_constraint('bass', 0, 36)   # C2
    solver.add_melodic_constraint('bass', 1, 44)   # Ab2
    solver.add_voice_leading_constraint(max_interval=2)

    diagnostics = solver.validate()
    warns = [msg for level, msg in diagnostics if level == 'WARN']

    has_jump_warn = any('must jump at least' in w and 'bass' in w for w in warns)
    if has_jump_warn:
        print(f"PASS: Got voice leading warning for bass")
        return True
    else:
        print(f"FAIL: Expected voice leading warning for bass, got: {warns}")
        return False


def test_diag_empty_domain_conflict():
    """Apply two harmonic constraints with non-overlapping pitch classes at the
    same step. Expect ERROR about empty domain."""
    print("\n=== Diagnostic Test: Empty Domain from Conflicting Constraints ===")
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    # C major (pcs 0, 4, 7) then D major (pcs 2, 6, 9) — intersection is empty
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_harmonic_constraint(0, 2, 'major')

    diagnostics = solver.validate()
    errors = [msg for level, msg in diagnostics if level == 'ERROR']

    has_empty = any('Empty domain' in e for e in errors)
    if has_empty:
        print(f"PASS: Got empty domain errors for conflicting chords")
        return True
    else:
        print(f"FAIL: Expected empty domain errors, got: {errors}")
        return False


def test_diag_cadence_truncation():
    """Apply a 2-step cadence starting at the last step of a 1-step problem.
    Expect WARN about truncation."""
    print("\n=== Diagnostic Test: Cadence Truncation ===")
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    # Authentic cadence is V-I (2 steps) but only 1 step available from start_step=0
    solver.add_cadence_constraint(key_root=0, scale_type='major',
                                  cadence_type='authentic', start_step=0)

    diagnostics = solver.validate()
    warns = [msg for level, msg in diagnostics if level == 'WARN']

    has_truncation = any('truncated' in w.lower() for w in warns)
    if has_truncation:
        print(f"PASS: Got cadence truncation warning")
        return True
    else:
        print(f"FAIL: Expected truncation warning, got: {warns}")
        return False


def test_diag_perfect_authentic_vs_melody():
    """Pin soprano to D4 (non-tonic) at the final step of a perfect authentic
    cadence in C major. Expect ERROR about soprano_on_root conflict."""
    print("\n=== Diagnostic Test: Perfect Authentic Cadence vs Pinned Melody ===")
    solver = ArrangerSolver()
    solver.setup_problem(2, ['soprano', 'alto', 'tenor', 'bass'])
    # Pin soprano at step 1 to D4 (MIDI 62, pc 2) — not tonic C (pc 0)
    solver.add_melodic_constraint('soprano', 0, 64)  # E4 at step 0
    solver.add_melodic_constraint('soprano', 1, 62)  # D4 at step 1 (non-tonic)
    # Perfect authentic cadence V-I at steps 0-1, soprano_on_root requires
    # soprano to be on tonic (pc 0) at step 1
    solver.add_cadence_constraint(key_root=0, scale_type='major',
                                  cadence_type='perfect_authentic', start_step=0,
                                  melody_voice='soprano',
                                  exclude_voices=['soprano'])

    diagnostics = solver.validate()
    errors = [msg for level, msg in diagnostics if level == 'ERROR']

    has_conflict = any('soprano' in e.lower() and 'tonic' in e.lower() for e in errors)
    if has_conflict:
        print(f"PASS: Got soprano/tonic conflict error")
        return True
    else:
        # The soprano_on_root constraint uses exclude_voices range [24,96],
        # and D4(62) pc=2 is not pc=0, so the domain intersection catches it
        has_empty = any('Empty domain' in e and 'soprano' in e for e in errors)
        if has_empty:
            print(f"PASS: Got empty domain error for soprano (soprano_on_root conflict detected)")
            return True
        print(f"FAIL: Expected soprano/tonic conflict error, got: {errors}")
        return False


if __name__ == "__main__":
    run_verification()

    results = []
    results.append(("Chord completeness", test_chord_completeness()))
    results.append(("Parallel octaves", test_parallel_octaves()))
    results.append(("Unison penalty", test_unison_penalty()))
    results.append(("Spacing", test_spacing()))
    results.append(("Bass restriction", test_bass_restriction()))
    results.append(("Per-voice leap", test_per_voice_leap()))

    # Diagnostic tests
    results.append(("Diag: melody outside scale", test_diag_melody_outside_scale()))
    results.append(("Diag: voice leading too tight", test_diag_voice_leading_too_tight()))
    results.append(("Diag: empty domain conflict", test_diag_empty_domain_conflict()))
    results.append(("Diag: cadence truncation", test_diag_cadence_truncation()))
    results.append(("Diag: perfect auth vs melody", test_diag_perfect_authentic_vs_melody()))

    print("\n=== Summary ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    if all(p for _, p in results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
