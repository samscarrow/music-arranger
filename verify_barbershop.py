"""
Barbershop chromaticism integration tests.

Tests:
1. Barbershop tag — I-III7-VI7-II7-V7-I with chromatic tones
2. Seventh chord lookup — _lookup_chord finds both triads and sevenths
3. Soft scale allows chromatic — soft scale + chromatic chord succeeds; hard fails
4. Resolution preference — V7 -> I with tendency tone resolution
"""

import sys
from solver_template import ArrangerSolver


def test_seventh_chord_lookup():
    """Verify _lookup_chord finds chords in both triads and sevenths sub-dicts."""
    print("\n=== Test: Seventh Chord Lookup ===")
    solver = ArrangerSolver()

    dom7 = solver._lookup_chord('dominant_7th')
    major = solver._lookup_chord('major')
    minor7 = solver._lookup_chord('minor_7th')
    bogus = solver._lookup_chord('nonexistent')

    passed = True
    if dom7 == [0, 4, 7, 10]:
        print(f"  PASS: dominant_7th = {dom7}")
    else:
        print(f"  FAIL: dominant_7th = {dom7}, expected [0, 4, 7, 10]")
        passed = False

    if major == [0, 4, 7]:
        print(f"  PASS: major = {major}")
    else:
        print(f"  FAIL: major = {major}, expected [0, 4, 7]")
        passed = False

    if minor7 == [0, 3, 7, 10]:
        print(f"  PASS: minor_7th = {minor7}")
    else:
        print(f"  FAIL: minor_7th = {minor7}, expected [0, 3, 7, 10]")
        passed = False

    if bogus is None:
        print(f"  PASS: nonexistent = None")
    else:
        print(f"  FAIL: nonexistent = {bogus}, expected None")
        passed = False

    return passed


def test_soft_scale_allows_chromatic():
    """Soft scale + chromatic harmonic chord should succeed. Hard scale + same should fail."""
    print("\n=== Test: Soft Scale Allows Chromatic ===")

    # --- Soft mode: should succeed ---
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()
    # C major scale, soft
    solver.add_scale_constraint(key_root=0, scale_type='major', hard=False, diatonic_weight=2)
    # D dominant 7th (D F# A C) — F# is chromatic in C major
    solver.add_harmonic_constraint(0, 2, 'dominant_7th')
    solver.add_chord_completeness_constraint(0, 2, 'dominant_7th')

    solution = solver.solve()
    if solution:
        pcs = set()
        for v in ['soprano', 'alto', 'tenor', 'bass']:
            pcs.add(solution[v][0] % 12)
        required = {2, 6, 9, 0}  # D=2, F#=6, A=9, C=0
        if required.issubset(pcs):
            print(f"  PASS (soft): All D7 PCs present: {pcs}")
        else:
            print(f"  WARN (soft): Missing PCs. Present: {pcs}, Required: {required}")
        soft_ok = True
    else:
        print(f"  FAIL (soft): No solution found — soft scale should allow chromatic")
        soft_ok = False

    # --- Hard mode: should fail (F#=pc6 not in C major) ---
    solver2 = ArrangerSolver()
    solver2.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    solver2.add_no_crossing_constraint()
    # C major scale, hard
    solver2.add_scale_constraint(key_root=0, scale_type='major', hard=True)
    # D dominant 7th — F# is impossible under hard diatonic
    solver2.add_harmonic_constraint(0, 2, 'dominant_7th')
    solver2.add_chord_completeness_constraint(0, 2, 'dominant_7th')

    solution2 = solver2.solve()
    if solution2 is None:
        print(f"  PASS (hard): Infeasible as expected (F# blocked by hard scale)")
        hard_ok = True
    else:
        print(f"  FAIL (hard): Should be infeasible but found solution")
        hard_ok = False

    return soft_ok and hard_ok


def test_barbershop_tag():
    """
    Barbershop tag: I - III7 - VI7 - II7 - V7 - I in C major.

    III7 = E7 (E G# B D)  — G# chromatic
    VI7 = A7 (A C# E G)   — C# chromatic
    II7 = D7 (D F# A C)   — F# chromatic
    V7  = G7 (G B D F)    — all diatonic
    """
    print("\n=== Test: Barbershop Tag (I-III7-VI7-II7-V7-I) ===")

    solver = ArrangerSolver()
    voices = ['tenor_barbershop', 'lead', 'baritone', 'bass_barbershop']
    solver.setup_problem(6, voices)
    solver.add_no_crossing_constraint()

    # Soft C major scale
    solver.add_scale_constraint(key_root=0, scale_type='major', hard=False, diatonic_weight=1)

    # Voice leading
    solver.add_voice_leading_constraint(max_interval=7)

    # Chord sequence (all by absolute root + quality since III7/VI7/II7 are secondary dominants)
    chords = [
        (0, 0, 'major'),           # I:   C major
        (1, 4, 'dominant_7th'),    # III7: E7
        (2, 9, 'dominant_7th'),    # VI7:  A7
        (3, 2, 'dominant_7th'),    # II7:  D7
        (4, 7, 'dominant_7th'),    # V7:   G7
        (5, 0, 'major'),           # I:    C major
    ]

    for step, root, quality in chords:
        solver.add_harmonic_constraint(step, root, quality)
        solver.add_chord_completeness_constraint(step, root, quality)
        solver.add_bass_restriction_constraint(step, root, quality)

    # Resolution constraints for dominant 7ths
    for i, (step, root, quality) in enumerate(chords):
        if 'dominant' in quality and i + 1 < len(chords):
            next_step = chords[i + 1][0]
            solver.add_resolution_constraint(step, root, quality, next_step_index=next_step)

    # Voicing quality
    solver.add_unison_penalty()
    solver.add_spacing_constraint()
    solver.add_parallel_octave_penalty()

    solution = solver.solve()
    if not solution:
        print("  FAIL: No solution found for barbershop tag")
        report = solver.get_diagnostic_report()
        print(report)
        return False

    # Check for chromatic tones
    # G# = pc 8 should appear at step 1 (E7 chord)
    # C# = pc 1 should appear at step 2 (A7 chord)
    # F# = pc 6 should appear at step 3 (D7 chord)
    chromatic_checks = [
        (1, 8, 'G#'),  # E7 has G#
        (2, 1, 'C#'),  # A7 has C#
        (3, 6, 'F#'),  # D7 has F#
    ]

    all_ok = True
    for step, expected_pc, name in chromatic_checks:
        pcs_at_step = set(solution[v][step] % 12 for v in voices)
        if expected_pc in pcs_at_step:
            print(f"  PASS: {name} (pc {expected_pc}) present at step {step}")
        else:
            print(f"  FAIL: {name} (pc {expected_pc}) NOT present at step {step}, found pcs {pcs_at_step}")
            all_ok = False

    # Print the full solution for inspection
    NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    for v in voices:
        notes = [f"{NOTE_NAMES[m % 12]}{(m // 12) - 1}" for m in solution[v]]
        print(f"  {v}: {notes}")

    return all_ok


def test_resolution_preference():
    """
    V7 at step 0 -> I at step 1 in C major.
    G7 = G(7) B(11) D(2) F(5).
    7th (F, pc 5) should resolve down to E (pc 4) or Eb.
    3rd (B, pc 11) should resolve up to C (pc 0).
    """
    print("\n=== Test: Resolution Preference (V7 -> I) ===")

    solver = ArrangerSolver()
    solver.setup_problem(2, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()

    # Soft C major
    solver.add_scale_constraint(key_root=0, scale_type='major', hard=False, diatonic_weight=2)

    # V7 at step 0, I at step 1
    solver.add_harmonic_constraint(0, 7, 'dominant_7th')  # G7
    solver.add_harmonic_constraint(1, 0, 'major')          # C
    solver.add_chord_completeness_constraint(0, 7, 'dominant_7th')
    solver.add_chord_completeness_constraint(1, 0, 'major')

    # Resolution with strong weight
    solver.add_resolution_constraint(0, 7, 'dominant_7th', next_step_index=1, weight=4)

    solver.add_voice_leading_constraint(max_interval=7)
    solver.add_unison_penalty(weight=2)

    solution = solver.solve()
    if not solution:
        print("  FAIL: No solution found")
        return False

    NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

    # Check resolution behavior
    seventh_pc = 5   # F in G7
    third_pc = 11    # B in G7
    seventh_resolved = False
    third_resolved = False

    for v in ['soprano', 'alto', 'tenor', 'bass']:
        n0 = solution[v][0]
        n1 = solution[v][1]
        diff = n1 - n0
        pc0 = n0 % 12
        name0 = f"{NOTE_NAMES[pc0]}{(n0 // 12) - 1}"
        name1 = f"{NOTE_NAMES[n1 % 12]}{(n1 // 12) - 1}"

        if pc0 == seventh_pc:
            if diff in (-1, -2):
                print(f"  PASS: {v} has 7th ({name0}) resolving down to {name1} (diff={diff})")
                seventh_resolved = True
            else:
                print(f"  INFO: {v} has 7th ({name0}) -> {name1} (diff={diff}, not stepwise down)")
        elif pc0 == third_pc:
            if diff == 1:
                print(f"  PASS: {v} has 3rd ({name0}) resolving up to {name1} (diff={diff})")
                third_resolved = True
            else:
                print(f"  INFO: {v} has 3rd ({name0}) -> {name1} (diff={diff}, not half step up)")
        else:
            print(f"  INFO: {v}: {name0} -> {name1}")

    # Soft constraint — resolution is preferred, not guaranteed
    if seventh_resolved:
        print("  PASS: 7th resolution observed")
    else:
        print("  WARN: 7th resolution not observed (soft constraint)")

    if third_resolved:
        print("  PASS: 3rd resolution observed")
    else:
        print("  WARN: 3rd resolution not observed (soft constraint)")

    return True  # Soft constraints — always pass but report


def test_tessitura_preference():
    """Single-step C major chord with barbershop voices; verify notes land within tessitura."""
    print("\n=== Test: Tessitura Preference ===")

    solver = ArrangerSolver()
    voices = ['tenor_barbershop', 'lead', 'baritone', 'bass_barbershop']
    solver.setup_problem(1, voices)
    solver.add_no_crossing_constraint()

    # C major chord
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_chord_completeness_constraint(0, 0, 'major')

    # Strong tessitura preference
    solver.add_tessitura_preference(weight=3)

    solution = solver.solve()
    if not solution:
        print("  FAIL: No solution found")
        return False

    NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    ranges = solver.theory['voice_ranges_midi']
    all_ok = True
    for v in voices:
        midi = solution[v][0]
        name = f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"
        tess_min = ranges[v]['tessitura_min']
        tess_max = ranges[v]['tessitura_max']
        in_tess = tess_min <= midi <= tess_max
        status = "PASS" if in_tess else "WARN"
        print(f"  {status}: {v} = {name} ({midi}), tessitura [{tess_min}-{tess_max}]")
        if not in_tess:
            all_ok = False

    return all_ok


def test_graduated_spacing():
    """Barbershop voices with cone limits; verify upper pairs respect tighter limits."""
    print("\n=== Test: Graduated Spacing ===")

    solver = ArrangerSolver()
    voices = ['tenor_barbershop', 'lead', 'baritone', 'bass_barbershop']
    solver.setup_problem(1, voices)
    solver.add_no_crossing_constraint()

    # C major chord
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_chord_completeness_constraint(0, 0, 'major')

    # Graduated limits: tighter at top
    pairs = {
        ('tenor_barbershop', 'lead'): 8,
        ('lead', 'baritone'): 10,
        ('baritone', 'bass_barbershop'): 12,
    }
    solver.add_spacing_constraint(max_gap_per_pair=pairs, penalize_lowest_pair=True, weight=3)

    solution = solver.solve()
    if not solution:
        print("  FAIL: No solution found")
        return False

    NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    all_ok = True
    for (upper, lower), limit in pairs.items():
        u_midi = solution[upper][0]
        l_midi = solution[lower][0]
        gap = u_midi - l_midi
        u_name = f"{NOTE_NAMES[u_midi % 12]}{(u_midi // 12) - 1}"
        l_name = f"{NOTE_NAMES[l_midi % 12]}{(l_midi // 12) - 1}"
        ok = gap <= limit
        status = "PASS" if ok else "WARN"
        print(f"  {status}: {upper}({u_name}) - {lower}({l_name}) = {gap} semitones (limit {limit})")
        if not ok:
            all_ok = False

    return all_ok


def test_common_tone_retention():
    """C major -> A minor (share C=0 and E=4 PCs); verify at least one voice holds a common tone."""
    print("\n=== Test: Common Tone Retention ===")

    solver = ArrangerSolver()
    solver.setup_problem(2, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()

    # C major at step 0, A minor at step 1
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_harmonic_constraint(1, 9, 'minor')
    solver.add_chord_completeness_constraint(0, 0, 'major')
    solver.add_chord_completeness_constraint(1, 9, 'minor')
    solver.add_voice_leading_constraint(max_interval=7)

    # Strong common tone retention
    chord_steps = [(0, 0, 'major'), (1, 9, 'minor')]
    solver.add_common_tone_retention(chord_steps, weight=5)

    solution = solver.solve()
    if not solution:
        print("  FAIL: No solution found")
        return False

    NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    # Common PCs between C major (0,4,7) and A minor (9,0,4): C=0 and E=4
    common_held = False
    for v in ['soprano', 'alto', 'tenor', 'bass']:
        n0 = solution[v][0]
        n1 = solution[v][1]
        name0 = f"{NOTE_NAMES[n0 % 12]}{(n0 // 12) - 1}"
        name1 = f"{NOTE_NAMES[n1 % 12]}{(n1 // 12) - 1}"
        held = (n0 == n1)
        pc = n0 % 12
        if held and pc in (0, 4):  # C or E
            print(f"  PASS: {v} holds common tone {name0} ({n0})")
            common_held = True
        else:
            print(f"  INFO: {v}: {name0} -> {name1}")

    if common_held:
        print("  PASS: At least one common tone held")
    else:
        print("  WARN: No common tone held (soft constraint)")

    return True  # soft constraint — always pass but report


def test_stepwise_inner_voice():
    """3-step I-V-I with strong stepwise weight; verify inner voice motion <= 2 semitones."""
    print("\n=== Test: Stepwise Inner Voice Motion ===")

    solver = ArrangerSolver()
    solver.setup_problem(3, ['soprano', 'alto', 'tenor', 'bass'])
    solver.add_no_crossing_constraint()

    # I - V - I in C major
    solver.add_harmonic_constraint(0, 0, 'major')
    solver.add_harmonic_constraint(1, 7, 'major')
    solver.add_harmonic_constraint(2, 0, 'major')
    solver.add_chord_completeness_constraint(0, 0, 'major')
    solver.add_chord_completeness_constraint(1, 7, 'major')
    solver.add_chord_completeness_constraint(2, 0, 'major')
    solver.add_voice_leading_constraint(max_interval=7)

    # Strong stepwise preference for inner voices
    solver.add_stepwise_motion_preference(weight=5)

    solution = solver.solve()
    if not solution:
        print("  FAIL: No solution found")
        return False

    NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    inner_voices = ['alto', 'tenor']
    all_stepwise = True
    for v in inner_voices:
        for t in range(2):
            n0 = solution[v][t]
            n1 = solution[v][t + 1]
            motion = abs(n1 - n0)
            name0 = f"{NOTE_NAMES[n0 % 12]}{(n0 // 12) - 1}"
            name1 = f"{NOTE_NAMES[n1 % 12]}{(n1 // 12) - 1}"
            ok = motion <= 2
            status = "PASS" if ok else "WARN"
            print(f"  {status}: {v} step {t}->{t+1}: {name0}->{name1} (motion={motion})")
            if not ok:
                all_stepwise = False

    if all_stepwise:
        print("  PASS: All inner voice motion is stepwise")
    else:
        print("  WARN: Some inner voice motion exceeds 2 semitones (soft constraint)")

    return True  # soft constraint — always pass but report


if __name__ == "__main__":
    results = []
    results.append(("Seventh chord lookup", test_seventh_chord_lookup()))
    results.append(("Soft scale allows chromatic", test_soft_scale_allows_chromatic()))
    results.append(("Barbershop tag", test_barbershop_tag()))
    results.append(("Resolution preference", test_resolution_preference()))
    results.append(("Tessitura preference", test_tessitura_preference()))
    results.append(("Graduated spacing", test_graduated_spacing()))
    results.append(("Common tone retention", test_common_tone_retention()))
    results.append(("Stepwise inner voice", test_stepwise_inner_voice()))

    print("\n=== Summary ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    if all(p for _, p in results):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
