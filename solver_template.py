"""
Music Arranging Engine Skeleton
Uses Google OR-Tools (CP-SAT) to treat musical arranging as a Constraint Satisfaction Problem.

This template defines the architecture:
1. Setup specific voices (variables).
2. Load theory definitions (constants).
3. Apply constraints (logic).
4. Solve and output.
"""

import json
import sys
from ortools.sat.python import cp_model

class ArrangerSolver:
    def __init__(self, theory_file='theory_definitions.json'):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        with open(theory_file, 'r') as f:
            self.theory = json.load(f)
        
        # Variables storage
        # Structure: self.voices[voice_name][step_index] = cp_variable
        self.voices = {}
        self.steps = 0
        # Objective terms for soft constraints (list of BoolVars to maximize)
        self._objective_terms = []

    def setup_problem(self, num_steps, voice_names=['soprano', 'alto', 'tenor', 'bass']):
        """
        Initializes the variables (MIDI pitch integers) for each voice at each time step.
        """
        self.steps = num_steps
        self.voice_names = voice_names
        
        for name in voice_names:
            self.voices[name] = []
            # Get range from theory dict, default to wide range if missing
            ranges = self.theory['voice_ranges_midi'].get(name, {'min': 36, 'max': 84})
            
            for t in range(num_steps):
                # Create a variable for Note at Time t with domain [min, max]
                var = self.model.NewIntVar(24, 96, f'{name}_t{t}')
                self.voices[name].append(var)

        print(f"Problem Initialized: {len(voice_names)} voices, {num_steps} steps.")

    def add_harmonic_constraint(self, step_index, root_note, chord_type, exclude_voices=None):
        """
        Constraints all voices at a specific step to belong to a specific chord.
        Voices in exclude_voices use a wide MIDI range (24-96) instead of the
        strict voice range, preventing infeasibility when a melody note is pinned
        outside the voice's strict range.
        """
        if exclude_voices is None:
            exclude_voices = []
        chord_intervals = self.theory['chords'].get(chord_type) or self.theory['chords']['triads'].get(chord_type)
        if not chord_intervals:
            print(f"Warning: Chord type '{chord_type}' not found.")
            return

        # Calculate allowed MIDI notes (Pitch Classes)
        # We use Modulo 12 logic.
        # Note % 12 must equal (Root + Interval) % 12
        allowed_pcs = [(root_note + interval) % 12 for interval in chord_intervals]

        for name in self.voice_names:
            note_var = self.voices[name][step_index]
            # Pre-compute all MIDI values in the voice's range whose pitch class is allowed
            if name in exclude_voices:
                lo, hi = 24, 96
            else:
                ranges = self.theory['voice_ranges_midi'].get(name, {'min': 36, 'max': 84})
                lo, hi = ranges['min'], ranges['max']
            allowed_values = [m for m in range(lo, hi + 1)
                              if m % 12 in allowed_pcs]
            self.model.AddAllowedAssignments([note_var], [[v] for v in allowed_values])

    def add_melodic_constraint(self, voice_name, step_index, midi_pitch):
        """
        Hard constraint: Forces a specific voice to sing a specific pitch (e.g., the Input Melody).
        """
        if voice_name in self.voices:
            self.model.Add(self.voices[voice_name][step_index] == midi_pitch)

    def add_no_crossing_constraint(self):
        """
        Ensures voices stay in order (Soprano > Alto > Tenor > Bass).
        """
        for t in range(self.steps):
            # Iterate through voices by index
            for i in range(len(self.voice_names) - 1):
                upper = self.voices[self.voice_names[i]][t]
                lower = self.voices[self.voice_names[i+1]][t]
                # Upper voice must be strictly greater than lower voice (or >= if unisons allowed)
                self.model.Add(upper >= lower)

    def add_voice_leading_constraint(self, max_interval=7, exclude_voices=None):
        """
        Minimizes big jumps. |Note_t - Note_t+1| <= max_interval.
        Voices in exclude_voices are skipped (e.g. pinned melody voice).
        """
        if exclude_voices is None:
            exclude_voices = []
        for name in self.voice_names:
            if name in exclude_voices:
                continue
            for t in range(self.steps - 1):
                current_note = self.voices[name][t]
                next_note = self.voices[name][t+1]
                
                # Absolute difference constraint
                dist = self.model.NewIntVar(0, max_interval, f'{name}_dist_t{t}')
                self.model.AddAbsEquality(dist, current_note - next_note)

    def add_cadence_constraint(self, key_root, scale_type, cadence_type, start_step,
                               melody_voice='soprano', exclude_voices=None):
        """
        Applies a cadence (chord progression) starting at start_step.
        Looks up the cadence progression, resolves each Roman numeral to an
        absolute root + chord quality, and applies harmonic constraints.
        Voices in exclude_voices use a wide MIDI range (24-96) in harmonic
        and soprano_on_root constraints.
        """
        if exclude_voices is None:
            exclude_voices = []
        cadence = self.theory['cadences'].get(cadence_type)
        if not cadence:
            print(f"Warning: Cadence type '{cadence_type}' not found.")
            return

        diatonic = self.theory['diatonic_chords'].get(scale_type)
        if not diatonic:
            print(f"Warning: Scale type '{scale_type}' not in diatonic_chords.")
            return

        progression = cadence['progression']
        for i, numeral in enumerate(progression):
            step = start_step + i
            if step >= self.steps:
                break

            # Wildcard '*' means any chord — skip constraining this step
            if numeral == '*':
                continue

            chord_info = diatonic.get(numeral)
            if not chord_info:
                print(f"Warning: Roman numeral '{numeral}' not found in {scale_type} diatonic chords.")
                continue

            absolute_root = (key_root + chord_info['root_degree']) % 12
            self.add_harmonic_constraint(step, absolute_root, chord_info['quality'],
                                         exclude_voices=exclude_voices)

        # Handle soprano_on_root for perfect authentic cadence
        if cadence.get('soprano_on_root') and melody_voice in self.voices:
            last_step = start_step + len(progression) - 1
            if last_step < self.steps:
                # Resolve the tonic root pitch class
                tonic_info = diatonic.get(progression[-1])
                if tonic_info:
                    tonic_pc = (key_root + tonic_info['root_degree']) % 12
                    mel_var = self.voices[melody_voice][last_step]
                    if melody_voice in exclude_voices:
                        lo, hi = 24, 96
                    else:
                        ranges = self.theory['voice_ranges_midi'].get(melody_voice, {'min': 60, 'max': 84})
                        lo, hi = ranges['min'], ranges['max']
                    allowed = [m for m in range(lo, hi + 1)
                               if m % 12 == tonic_pc]
                    self.model.AddAllowedAssignments([mel_var], [[v] for v in allowed])

    def add_scale_constraint(self, key_root, scale_type, exclude_voices=None):
        """
        Constrains all notes across all steps to belong to a specific scale.
        Voices in exclude_voices use a wide MIDI range (24-96) instead of the
        strict voice range, preventing infeasibility when a melody note is pinned
        outside the voice's strict range.
        """
        if exclude_voices is None:
            exclude_voices = []
        scale_intervals = self.theory['scales'].get(scale_type)
        if not scale_intervals:
            print(f"Warning: Scale type '{scale_type}' not found.")
            return

        scale_pcs = set((key_root + degree) % 12 for degree in scale_intervals)

        for name in self.voice_names:
            if name in exclude_voices:
                lo, hi = 24, 96
            else:
                ranges = self.theory['voice_ranges_midi'].get(name, {'min': 36, 'max': 84})
                lo, hi = ranges['min'], ranges['max']
            allowed_values = [m for m in range(lo, hi + 1)
                              if m % 12 in scale_pcs]
            for t in range(self.steps):
                self.model.AddAllowedAssignments(
                    [self.voices[name][t]], [[v] for v in allowed_values])

    def add_doubling_preference(self, step_index, root_pc, bass_voice=None):
        """
        Soft constraint (objective) preferring the root pitch class to appear
        in the bass voice — standard SATB doubling practice.
        Adds a boolean term to the objective; call before solve().
        """
        if bass_voice is None:
            bass_voice = self.voice_names[-1]
        if bass_voice not in self.voices:
            return

        bass_var = self.voices[bass_voice][step_index]
        ranges = self.theory['voice_ranges_midi'].get(bass_voice, {'min': 36, 'max': 62})
        root_values = [m for m in range(ranges['min'], ranges['max'] + 1)
                       if m % 12 == root_pc]

        if not root_values:
            return

        # Create a boolean indicator: is the bass on the root pitch class?
        is_root = self.model.NewBoolVar(f'{bass_voice}_root_t{step_index}')
        # If is_root is true, bass must be one of the root values
        self.model.AddAllowedAssignments([bass_var], [[v] for v in root_values]).OnlyEnforceIf(is_root)
        # If is_root is false, bass must NOT be one of the root values
        non_root = [m for m in range(ranges['min'], ranges['max'] + 1)
                    if m % 12 != root_pc]
        self.model.AddAllowedAssignments([bass_var], [[v] for v in non_root]).OnlyEnforceIf(is_root.Not())
        self._objective_terms.append(is_root)

    def solve(self):
        """
        Runs the solver and returns a dictionary of results.
        """
        # Apply accumulated soft-constraint objective
        if self._objective_terms:
            self.model.Maximize(sum(self._objective_terms))

        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            result = {}
            for name in self.voice_names:
                result[name] = [self.solver.Value(v) for v in self.voices[name]]
            return result
        else:
            return None

if __name__ == "__main__":
    # Example Usage: C major melody with plagal cadence
    solver = ArrangerSolver()
    solver.setup_problem(4, ['soprano', 'alto', 'tenor', 'bass'])

    # Soprano melody: E4, D4, F4, E4 — compatible with I, V, IV, I
    # E(pc4) in I(C,E,G), D(pc2) in V(G,B,D), F(pc5) in IV(F,A,C), E(pc4) in I(C,E,G)
    solver.add_melodic_constraint('soprano', 0, 64)  # E4 over I
    solver.add_melodic_constraint('soprano', 1, 62)  # D4 over V
    solver.add_melodic_constraint('soprano', 2, 65)  # F4 over IV
    solver.add_melodic_constraint('soprano', 3, 64)  # E4 over I

    # Structural constraints
    solver.add_no_crossing_constraint()
    solver.add_voice_leading_constraint(max_interval=7, exclude_voices=['soprano'])

    # Keep all notes diatonic to C major
    solver.add_scale_constraint(key_root=0, scale_type='major', exclude_voices=['soprano'])

    # Apply chord constraints: I - V - IV - I
    solver.add_harmonic_constraint(0, 0, 'major', exclude_voices=['soprano'])   # C major (I)
    solver.add_harmonic_constraint(1, 7, 'major', exclude_voices=['soprano'])   # G major (V)

    # Plagal cadence at the end (steps 2-3): IV -> I
    solver.add_cadence_constraint(key_root=0, scale_type='major',
                                  cadence_type='plagal', start_step=2,
                                  exclude_voices=['soprano'])

    # Prefer root in bass (use correct root per chord)
    solver.add_doubling_preference(0, 0)   # C for I
    solver.add_doubling_preference(1, 7)   # G for V
    solver.add_doubling_preference(2, 5)   # F for IV
    solver.add_doubling_preference(3, 0)   # C for I

    solution = solver.solve()
    if solution:
        print("Solution found:")
        for voice, notes in solution.items():
            note_names = []
            for m in notes:
                name = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'][m % 12]
                octave = (m // 12) - 1
                note_names.append(f"{name}{octave}")
            print(f"  {voice}: {notes} ({note_names})")
    else:
        print("No solution found.")
