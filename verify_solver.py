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

if __name__ == "__main__":
    run_verification()
