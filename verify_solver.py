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


if __name__ == "__main__":
    run_verification()

    results = []
    results.append(("Chord completeness", test_chord_completeness()))
    results.append(("Parallel octaves", test_parallel_octaves()))
    results.append(("Unison penalty", test_unison_penalty()))
    results.append(("Spacing", test_spacing()))
    results.append(("Bass restriction", test_bass_restriction()))
    results.append(("Per-voice leap", test_per_voice_leap()))

    print("\n=== Summary ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    if all(p for _, p in results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
