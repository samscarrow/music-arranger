from solver_template import ArrangerSolver

def test_specific_bass():
    print("Testing Explicit Bass Constraint...")
    solver = ArrangerSolver()
    solver.setup_problem(1, ['soprano', 'alto', 'tenor', 'bass'])
    
    # Setup: Step 0, C Major. Bass usually wants Root (C).
    # We force Bass to E (pc 4).
    
    # Soprano pin (high C)
    solver.add_melodic_constraint('soprano', 0, 72)
    
    # Harmonic constraint: C Major
    solver.add_harmonic_constraint(0, 0, 'major', exclude_voices=['soprano'])
    
    # Explicit Bass Constraint: E (First inversion)
    solver.add_specific_bass_constraint(0, 4) # 4 = E
    
    # Basic structural
    solver.add_no_crossing_constraint()
    
    solution = solver.solve()
    if solution:
        bass_note = solution['bass'][0]
        print(f"  Bass Note: {bass_note} (PC {bass_note % 12})")
        if bass_note % 12 == 4:
            print("  SUCCESS: Bass is E (pc 4).")
        else:
            print(f"  FAILURE: Bass is {bass_note % 12}, expected 4.")
            exit(1)
    else:
        print("  FAILURE: No solution found.")
        exit(1)

def test_rhythmic_stagger():
    print("Testing Rhythmic Staggering...")
    solver = ArrangerSolver()
    solver.setup_problem(2, ['soprano', 'alto']) 
    # Using 2 voices for simplicity: Soprano (Melody), Alto (Inner)
    
    # Melody is STATIC: C5 -> C5
    solver.add_melodic_constraint('soprano', 0, 72)
    solver.add_melodic_constraint('soprano', 1, 72)
    
    # Alto setup
    # Allow Alto to be C4 or D4 at both steps.
    # C4 = 60, D4 = 62.
    # We want to see it MOVE (e.g., 60->62 or 62->60) instead of Stay (60->60).
    
    # Harmonic constraint: Just allow white keys or specifically C and D
    # Let's just give explicit domains for Alto to ensure it has options.
    # Actually simpler: Harmonic constraint C Major (t0) -> G Major (t1)?
    # No, we want to test the Stagger preference specifically.
    # If I use harmonic constraints, the harmony might force movement.
    # I want the harmony to allow *either* moving or staying, and Stagger to force moving.
    # Let's use a trivial harmonic constraint that allows C and D at both steps.
    
    # Custom "Cluster" chord 0, 2 (C, D)
    # solver._lookup_chord would fail. Let's just manually AddAllowedAssignments for Alto.
    
    # Manually open Alto domain to {60, 62}
    for t in range(2):
        solver.model.AddAllowedAssignments([solver.voices['alto'][t]], [[60], [62]])
        
    # Add Stagger Preference
    # inner_voices=['alto'], melody_voice='soprano'
    solver.add_rhythmic_stagger_preference('soprano', ['alto'], weight=10)
    
    # IMPORTANT: Ensure NO other conflicting preferences (like common tone) are active.
    # Default Solver init has empty _objective_terms.
    
    solution = solver.solve()
    if solution:
        alto_notes = solution['alto']
        print(f"  Melody: Static (C5->C5)")
        print(f"  Alto: {alto_notes}")
        
        if alto_notes[0] != alto_notes[1]:
            print("  SUCCESS: Alto moved.")
        else:
            print("  FAILURE: Alto did not move (Stagger preference failed).")
            exit(1)
    else:
        print("  FAILURE: No solution found.")
        exit(1)

if __name__ == "__main__":
    test_specific_bass()
    test_rhythmic_stagger()
