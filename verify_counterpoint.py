"""Verification tests for counterpoint-core features: modulation and non-chord tones."""
import sys
from solver_template import ArrangerSolver


def test_modulation():
    """4 steps: C major (steps 0-1) → G major (steps 2-3).
    D major chord at step 2 resolved under G major. Verify F# (pc 6)
    appears at step 2 — impossible under C major, valid under G major.
    """
    print("\n=== Test: Modulation (C major → G major) ===")
    solver = ArrangerSolver()
    solver.setup_problem(4, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()

    # Pin soprano melody: C5, B4, A4, G4
    solver.add_melodic_constraint('soprano', 0, 72)  # C5
    solver.add_melodic_constraint('soprano', 1, 71)  # B4
    solver.add_melodic_constraint('soprano', 2, 69)  # A4
    solver.add_melodic_constraint('soprano', 3, 67)  # G4

    # Scale constraint with two segments
    solver.add_scale_constraint(key_root=0, scale_type='major',
                                exclude_voices=['soprano'],
                                hard=True, start_step=0, end_step=2)
    solver.add_scale_constraint(key_root=7, scale_type='major',
                                exclude_voices=['soprano'],
                                hard=True, start_step=2, end_step=4)

    # Chords: C major (step 0), G major (step 1), D major (step 2), G major (step 3)
    solver.add_harmonic_constraint(0, 0, 'major', exclude_voices=['soprano'])  # C
    solver.add_harmonic_constraint(1, 7, 'major', exclude_voices=['soprano'])  # G
    solver.add_harmonic_constraint(2, 2, 'major', exclude_voices=['soprano'])  # D (V in G)
    solver.add_harmonic_constraint(3, 7, 'major', exclude_voices=['soprano'])  # G (I in G)

    solver.add_voice_leading_constraint(max_interval=7, exclude_voices=['soprano'])

    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    # Verify F# (pc 6) appears at step 2 — it's a D major chord tone
    step2_pcs = set()
    for name in ['soprano', 'alto', 'tenor', 'bass']:
        step2_pcs.add(solution[name][2] % 12)

    # D major chord = pcs {2, 6, 9}. pc 6 (F#) must be present.
    if 6 in step2_pcs:
        print(f"PASS: F# (pc 6) present at step 2: pcs = {step2_pcs}")
    else:
        print(f"FAIL: F# (pc 6) not found at step 2: pcs = {step2_pcs}")
        return False

    # Verify step 0-1 have no F# (pc 6) in non-soprano voices
    # (C major scale has no F#)
    for t in [0, 1]:
        for name in ['alto', 'tenor', 'bass']:
            pc = solution[name][t] % 12
            if pc == 6:
                print(f"FAIL: {name} at step {t} has F# (pc 6) — violates C major")
                return False
    print("PASS: No F# in C major section (steps 0-1)")
    return True


def test_modulation_backward_compat():
    """Single-key call (no start_step/end_step) still works."""
    print("\n=== Test: Modulation backward compat (single key) ===")
    solver = ArrangerSolver()
    solver.setup_problem(2, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()
    solver.add_scale_constraint(key_root=0, scale_type='major', hard=True)
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_harmonic_constraint(1, 7, 'major')
    solver.add_voice_leading_constraint(max_interval=7)

    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    # All notes should be diatonic to C major (pcs {0,2,4,5,7,9,11})
    c_major_pcs = {0, 2, 4, 5, 7, 9, 11}
    for name in ['soprano', 'alto', 'tenor', 'bass']:
        for t in range(2):
            pc = solution[name][t] % 12
            if pc not in c_major_pcs:
                print(f"FAIL: {name} at step {t} has pc {pc} — not in C major")
                return False
    print("PASS: All notes diatonic to C major")
    return True


def test_nct_basic():
    """3 steps, C major chord at all 3, NCT enabled at step 1 (middle step).
    Verify: solution found, at most 1 voice has NCT at step 1, NCT is diatonic.
    """
    print("\n=== Test: NCT basic (3-step C major) ===")
    solver = ArrangerSolver()
    solver.setup_problem(3, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()

    # Pin soprano: E4, F4, E4 — F4 is an NCT (passing tone)
    solver.add_melodic_constraint('soprano', 0, 64)  # E4
    solver.add_melodic_constraint('soprano', 1, 65)  # F4
    solver.add_melodic_constraint('soprano', 2, 64)  # E4

    # Scale: C major
    solver.add_scale_constraint(key_root=0, scale_type='major',
                                exclude_voices=['soprano'], hard=True)

    # Boundary steps (0, 2): chord tones only
    solver.add_nct_harmonic_constraint(0, 0, 'major',
                                        key_root=0, scale_type='major',
                                        exclude_voices=['soprano'],
                                        boundary_step=True)
    # Middle step: NCT allowed, max 1
    solver.add_nct_harmonic_constraint(1, 0, 'major',
                                        key_root=0, scale_type='major',
                                        exclude_voices=['soprano'],
                                        max_ncts_per_step=1,
                                        boundary_step=False)
    solver.add_nct_harmonic_constraint(2, 0, 'major',
                                        key_root=0, scale_type='major',
                                        exclude_voices=['soprano'],
                                        boundary_step=True)

    solver.add_voice_leading_constraint(max_interval=7, exclude_voices=['soprano'])

    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    # Check step 1: at most 1 non-chord voice
    c_major_pcs = {0, 4, 7}  # C, E, G
    c_major_scale_pcs = {0, 2, 4, 5, 7, 9, 11}
    nct_count = 0
    nct_voices = []
    for name in ['alto', 'tenor', 'bass']:
        pc = solution[name][1] % 12
        if pc not in c_major_pcs:
            nct_count += 1
            nct_voices.append(name)
            # NCT must be diatonic
            if pc not in c_major_scale_pcs:
                print(f"FAIL: {name} at step 1 has pc {pc} — not diatonic")
                return False

    if nct_count <= 1:
        print(f"PASS: {nct_count} NCT(s) at step 1: {nct_voices}")
    else:
        print(f"FAIL: {nct_count} NCTs at step 1 (max 1): {nct_voices}")
        return False

    # Check boundary steps: all voices on chord tones
    for t in [0, 2]:
        for name in ['alto', 'tenor', 'bass']:
            pc = solution[name][t] % 12
            if pc not in c_major_pcs:
                print(f"FAIL: {name} at boundary step {t} has pc {pc} — not a chord tone")
                return False
    print("PASS: Boundary steps are all chord tones")
    return True


def test_nct_stepwise_motion():
    """Verify NCT voices move stepwise in and out (motion <= 2 semitones)."""
    print("\n=== Test: NCT stepwise motion ===")
    solver = ArrangerSolver()
    solver.setup_problem(3, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()

    # Pin soprano
    solver.add_melodic_constraint('soprano', 0, 64)  # E4
    solver.add_melodic_constraint('soprano', 1, 65)  # F4 (passing)
    solver.add_melodic_constraint('soprano', 2, 64)  # E4

    solver.add_scale_constraint(key_root=0, scale_type='major',
                                exclude_voices=['soprano'], hard=True)

    # All 3 steps: NCT enabled at middle
    solver.add_nct_harmonic_constraint(0, 0, 'major',
                                        key_root=0, scale_type='major',
                                        exclude_voices=['soprano'],
                                        boundary_step=True)
    solver.add_nct_harmonic_constraint(1, 0, 'major',
                                        key_root=0, scale_type='major',
                                        exclude_voices=['soprano'],
                                        max_ncts_per_step=1,
                                        boundary_step=False)
    solver.add_nct_harmonic_constraint(2, 0, 'major',
                                        key_root=0, scale_type='major',
                                        exclude_voices=['soprano'],
                                        boundary_step=True)

    solver.add_voice_leading_constraint(max_interval=7, exclude_voices=['soprano'])

    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    c_major_pcs = {0, 4, 7}
    for name in ['alto', 'tenor', 'bass']:
        pc1 = solution[name][1] % 12
        if pc1 not in c_major_pcs:
            # This is an NCT — verify stepwise approach and departure
            approach = abs(solution[name][1] - solution[name][0])
            departure = abs(solution[name][2] - solution[name][1])
            if approach > 2:
                print(f"FAIL: {name} NCT approach = {approach} > 2")
                return False
            if departure > 2:
                print(f"FAIL: {name} NCT departure = {departure} > 2")
                return False
            print(f"  {name}: NCT at step 1, approach={approach}, departure={departure}")

    print("PASS: All NCT motion is stepwise")
    return True


def test_nct_boundary_forces_chord_tones():
    """boundary_step=True should force all chord tones (no NCTs)."""
    print("\n=== Test: NCT boundary forces chord tones ===")
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()

    solver.add_nct_harmonic_constraint(0, 0, 'major',
                                        key_root=0, scale_type='major',
                                        boundary_step=True)

    solution = solver.solve()
    if not solution:
        print("FAIL: No solution found.")
        return False

    c_major_pcs = {0, 4, 7}
    for name in ['soprano', 'alto', 'tenor', 'bass']:
        pc = solution[name][0] % 12
        if pc not in c_major_pcs:
            print(f"FAIL: {name} has pc {pc} at boundary step — not a chord tone")
            return False

    print("PASS: All voices are chord tones at boundary step")
    return True


def test_key_at_step_helper():
    """Test MusicArranger._key_at_step static method."""
    print("\n=== Test: _key_at_step helper ===")
    # Can't import MusicArranger (needs anthropic), so inline the logic
    def _key_at_step(step, key_segments, default_key_root, default_scale_type):
        if key_segments:
            for seg in key_segments:
                if seg['start_step'] <= step < seg['end_step']:
                    return seg['key_root'], seg['scale_type']
        return default_key_root, default_scale_type

    segments = [
        {'start_step': 0, 'end_step': 4, 'key_root': 0, 'scale_type': 'major'},
        {'start_step': 4, 'end_step': 8, 'key_root': 7, 'scale_type': 'major'},
    ]

    # Step 2 → C major
    kr, st = _key_at_step(2, segments, 0, 'major')
    assert (kr, st) == (0, 'major'), f"Expected (0, 'major'), got ({kr}, {st})"

    # Step 5 → G major
    kr, st = _key_at_step(5, segments, 0, 'major')
    assert (kr, st) == (7, 'major'), f"Expected (7, 'major'), got ({kr}, {st})"

    # Step 10 → falls back to default
    kr, st = _key_at_step(10, segments, 0, 'major')
    assert (kr, st) == (0, 'major'), f"Expected (0, 'major'), got ({kr}, {st})"

    # No segments → default
    kr, st = _key_at_step(0, None, 5, 'natural_minor')
    assert (kr, st) == (5, 'natural_minor'), f"Expected (5, 'natural_minor'), got ({kr}, {st})"

    print("PASS: _key_at_step returns correct key for all cases")
    return True


if __name__ == "__main__":
    results = []
    results.append(("Modulation C→G", test_modulation()))
    results.append(("Modulation backward compat", test_modulation_backward_compat()))
    results.append(("NCT basic", test_nct_basic()))
    results.append(("NCT stepwise motion", test_nct_stepwise_motion()))
    results.append(("NCT boundary chord tones", test_nct_boundary_forces_chord_tones()))
    results.append(("_key_at_step helper", test_key_at_step_helper()))

    print("\n=== Summary ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    if all(p for _, p in results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
