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

        # Diagnostic tracking
        self._constraint_log = []       # list of dicts describing each applied constraint
        self._domain_tracker = {}       # {voice_name: {step: set(midi_values)}}
        self._melody_pins = {}          # {step_index: midi_pitch}
        self._melody_voice = None
        self._scale_segments = []       # list of (key_root, scale_type, scale_pcs, exclude_voices, hard, start_step, end_step)
        self._diagnostics = []          # populated by validate()
        self._failure_stats = None      # populated by solve() on failure
        self._moved_vars = {}           # cached per-voice "moved" BoolVars

    def _lookup_chord(self, chord_type):
        """Look up chord intervals by type, searching triads then sevenths.

        Returns intervals sorted ascending so positional indexing
        (e.g. [1] = 3rd, [3] = 7th) is reliable regardless of JSON order.
        """
        chords = self.theory['chords']
        intervals = (chords.get('triads', {}).get(chord_type)
                     or chords.get('sevenths', {}).get(chord_type))
        return sorted(intervals) if intervals else None

    def setup_problem(self, num_steps, voice_names=['soprano', 'alto', 'tenor', 'bass']):
        """
        Initializes the variables (MIDI pitch integers) for each voice at each time step.
        """
        self.steps = num_steps
        self.voice_names = voice_names
        
        for name in voice_names:
            self.voices[name] = []
            self._domain_tracker[name] = {}
            # Get range from theory dict, default to wide range if missing
            ranges = self.theory['voice_ranges_midi'].get(name, {'min': 36, 'max': 84})

            for t in range(num_steps):
                # Create a variable for Note at Time t with domain [min, max]
                var = self.model.NewIntVar(24, 96, f'{name}_t{t}')
                self.voices[name].append(var)
                self._domain_tracker[name][t] = set(range(24, 97))

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
        chord_intervals = self._lookup_chord(chord_type)
        if not chord_intervals:
            print(f"Warning: Chord type '{chord_type}' not found.")
            return

        # Calculate allowed MIDI notes (Pitch Classes)
        # We use Modulo 12 logic.
        # Note % 12 must equal (Root + Interval) % 12
        allowed_pcs = [(root_note + interval) % 12 for interval in chord_intervals]

        self._constraint_log.append({
            'type': 'harmonic', 'step': step_index,
            'root': root_note, 'chord': chord_type,
            'exclude_voices': list(exclude_voices),
        })

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
            self._domain_tracker[name][step_index] &= set(allowed_values)

    def add_nct_harmonic_constraint(self, step_index, root_note, chord_type,
                                     key_root, scale_type,
                                     exclude_voices=None,
                                     max_ncts_per_step=1,
                                     nct_penalty_weight=0,
                                     boundary_step=False):
        """
        Harmonic constraint that allows non-chord tones (passing/neighbor notes).

        Replaces add_harmonic_constraint at steps where NCTs are desired.
        Non-excluded voices may play either a chord tone (CT) or a diatonic
        non-chord tone (NCT). NCT voices must move stepwise (<=2 semitones)
        into and out of the NCT.

        boundary_step: if True, forces all chord tones (no NCTs allowed).
        max_ncts_per_step: cap on how many voices can be NCTs simultaneously.
        nct_penalty_weight: if >0, add soft preference for chord tones.
        """
        if exclude_voices is None:
            exclude_voices = []

        chord_intervals = self._lookup_chord(chord_type)
        if not chord_intervals:
            print(f"Warning: Chord type '{chord_type}' not found (NCT).")
            return

        chord_pcs = set((root_note + iv) % 12 for iv in chord_intervals)
        scale_intervals = self.theory['scales'].get(scale_type)
        if not scale_intervals:
            print(f"Warning: Scale type '{scale_type}' not found (NCT).")
            return
        scale_pcs = set((key_root + deg) % 12 for deg in scale_intervals)
        nct_pcs = scale_pcs - chord_pcs

        self._constraint_log.append({
            'type': 'nct_harmonic', 'step': step_index,
            'root': root_note, 'chord': chord_type,
            'boundary': boundary_step,
            'max_ncts': max_ncts_per_step,
            'exclude_voices': list(exclude_voices),
        })

        nct_indicators = []  # collect is_ct.Not() BoolVars to cap NCTs

        for name in self.voice_names:
            if name in exclude_voices:
                continue
            note_var = self.voices[name][step_index]

            if name in exclude_voices:
                lo, hi = 24, 96
            else:
                ranges = self.theory['voice_ranges_midi'].get(
                    name, {'min': 36, 'max': 84})
                lo, hi = ranges['min'], ranges['max']

            ct_values = [m for m in range(lo, hi + 1) if m % 12 in chord_pcs]
            nct_values = [m for m in range(lo, hi + 1) if m % 12 in nct_pcs]

            # Boundary steps or no NCT values: chord tones only
            if boundary_step or not nct_values:
                self.model.AddAllowedAssignments(
                    [note_var], [[v] for v in ct_values])
                self._domain_tracker[name][step_index] &= set(ct_values)
                continue

            # Create CT/NCT switch
            is_ct = self.model.NewBoolVar(
                f'nct_isct_{name}_t{step_index}')
            self.model.AddAllowedAssignments(
                [note_var], [[v] for v in ct_values]
            ).OnlyEnforceIf(is_ct)
            self.model.AddAllowedAssignments(
                [note_var], [[v] for v in nct_values]
            ).OnlyEnforceIf(is_ct.Not())

            self._domain_tracker[name][step_index] &= set(ct_values) | set(nct_values)
            nct_indicators.append(is_ct.Not())

            # Stepwise approach: |note[t] - note[t-1]| <= 2 when NCT
            if step_index > 0:
                prev_var = self.voices[name][step_index - 1]
                approach_dist = self.model.NewIntVar(
                    0, 72, f'nct_appr_{name}_t{step_index}')
                self.model.AddAbsEquality(
                    approach_dist, note_var - prev_var)
                self.model.Add(approach_dist <= 2).OnlyEnforceIf(is_ct.Not())

            # Stepwise departure: |note[t+1] - note[t]| <= 2 when NCT
            if step_index < self.steps - 1:
                next_var = self.voices[name][step_index + 1]
                depart_dist = self.model.NewIntVar(
                    0, 72, f'nct_dept_{name}_t{step_index}')
                self.model.AddAbsEquality(
                    depart_dist, next_var - note_var)
                self.model.Add(depart_dist <= 2).OnlyEnforceIf(is_ct.Not())

            # Optional soft preference for chord tones
            if nct_penalty_weight > 0:
                for _ in range(nct_penalty_weight):
                    self._objective_terms.append(is_ct)

        # Cap NCTs per step
        if nct_indicators and max_ncts_per_step < len(nct_indicators):
            self.model.Add(
                sum(nct_indicators) <= max_ncts_per_step)

    def add_melodic_constraint(self, voice_name, step_index, midi_pitch):
        """
        Hard constraint: Forces a specific voice to sing a specific pitch (e.g., the Input Melody).
        """
        if voice_name in self.voices:
            self.model.Add(self.voices[voice_name][step_index] == midi_pitch)
            self._melody_pins[step_index] = midi_pitch
            self._melody_voice = voice_name
            self._domain_tracker[voice_name][step_index] &= {midi_pitch}
            self._constraint_log.append({
                'type': 'melodic', 'step': step_index,
                'voice': voice_name, 'pitch': midi_pitch,
            })

    def add_no_crossing_constraint(self):
        """
        Ensures voices stay in order (Soprano > Alto > Tenor > Bass).
        """
        self._constraint_log.append({'type': 'no_crossing'})
        for t in range(self.steps):
            # Iterate through voices by index
            for i in range(len(self.voice_names) - 1):
                upper = self.voices[self.voice_names[i]][t]
                lower = self.voices[self.voice_names[i+1]][t]
                # Upper voice must be strictly greater than lower voice (or >= if unisons allowed)
                self.model.Add(upper >= lower)

    def add_voice_leading_constraint(self, max_interval=7, per_voice_max=None,
                                      exclude_voices=None):
        """
        Minimizes big jumps. |Note_t - Note_t+1| <= limit.
        per_voice_max is an optional dict mapping voice name -> max semitone leap.
        Voices in that dict use their specific limit; others fall back to max_interval.
        Voices in exclude_voices are skipped (e.g. pinned melody voice).
        """
        if exclude_voices is None:
            exclude_voices = []
        self._constraint_log.append({
            'type': 'voice_leading', 'max_interval': max_interval,
            'per_voice_max': per_voice_max,
            'exclude_voices': list(exclude_voices),
        })
        for name in self.voice_names:
            if name in exclude_voices:
                continue
            limit = per_voice_max.get(name, max_interval) if per_voice_max else max_interval
            for t in range(self.steps - 1):
                current_note = self.voices[name][t]
                next_note = self.voices[name][t+1]

                # Absolute difference constraint
                dist = self.model.NewIntVar(0, limit, f'{name}_dist_t{t}')
                self.model.AddAbsEquality(dist, current_note - next_note)

    def add_cadence_constraint(self, key_root, scale_type, cadence_type, start_step,
                               melody_voice='soprano', exclude_voices=None,
                               ensure_completeness=False, restrict_bass=False):
        """
        Applies a cadence (chord progression) starting at start_step.
        Looks up the cadence progression, resolves each Roman numeral to an
        absolute root + chord quality, and applies harmonic constraints.
        Voices in exclude_voices use a wide MIDI range (24-96) in harmonic
        and soprano_on_root constraints.

        If ensure_completeness is True, also applies chord completeness at each step.
        If restrict_bass is True, also applies bass note restrictions at each step.
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

        bass_voice = self.voice_names[-1] if self.voice_names else None

        progression = cadence['progression']
        self._constraint_log.append({
            'type': 'cadence', 'cadence_type': cadence_type,
            'start_step': start_step, 'progression_len': len(progression),
            'available_steps': self.steps - start_step,
            'melody_voice': melody_voice,
            'exclude_voices': list(exclude_voices),
            'ensure_completeness': ensure_completeness,
            'restrict_bass': restrict_bass,
        })
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

            if ensure_completeness:
                self.add_chord_completeness_constraint(
                    step, absolute_root, chord_info['quality'],
                    exclude_voices=exclude_voices)

            if restrict_bass and bass_voice and bass_voice not in exclude_voices:
                self.add_bass_restriction_constraint(
                    step, absolute_root, chord_info['quality'],
                    bass_voice=bass_voice)

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
                    self._domain_tracker[melody_voice][last_step] &= set(allowed)
                    self._constraint_log.append({
                        'type': 'soprano_on_root', 'step': last_step,
                        'voice': melody_voice, 'tonic_pc': tonic_pc,
                    })

    def add_scale_constraint(self, key_root, scale_type, exclude_voices=None,
                             hard=True, diatonic_weight=2,
                             start_step=None, end_step=None):
        """
        Constrains notes to belong to a specific scale over a step range.

        hard=True (default): AddAllowedAssignments forces diatonic pitches only.
        hard=False: soft preference — chromatic notes allowed but diatonic rewarded.
            diatonic_weight controls the objective bonus per diatonic note.

        start_step/end_step define the half-open range [start, end). Defaults
        to the full problem (0, self.steps). Call multiple times with different
        ranges to support modulation.

        Voices in exclude_voices use a wide MIDI range (24-96) instead of the
        strict voice range, preventing infeasibility when a melody note is pinned
        outside the voice's strict range.
        """
        if exclude_voices is None:
            exclude_voices = []
        if start_step is None:
            start_step = 0
        if end_step is None:
            end_step = self.steps
        scale_intervals = self.theory['scales'].get(scale_type)
        if not scale_intervals:
            print(f"Warning: Scale type '{scale_type}' not found.")
            return

        scale_pcs = set((key_root + degree) % 12 for degree in scale_intervals)
        self._scale_segments.append(
            (key_root, scale_type, scale_pcs, list(exclude_voices), hard,
             start_step, end_step))
        self._constraint_log.append({
            'type': 'scale', 'key_root': key_root, 'scale_type': scale_type,
            'exclude_voices': list(exclude_voices), 'hard': hard,
            'start_step': start_step, 'end_step': end_step,
        })

        for name in self.voice_names:
            if name in exclude_voices:
                lo, hi = 24, 96
            else:
                ranges = self.theory['voice_ranges_midi'].get(name, {'min': 36, 'max': 84})
                lo, hi = ranges['min'], ranges['max']

            diatonic_values = [m for m in range(lo, hi + 1) if m % 12 in scale_pcs]
            diatonic_set = set(diatonic_values)

            if hard:
                for t in range(start_step, end_step):
                    self.model.AddAllowedAssignments(
                        [self.voices[name][t]], [[v] for v in diatonic_values])
                    self._domain_tracker[name][t] &= diatonic_set
            else:
                # Soft mode: reward diatonic, allow chromatic
                chromatic_values = [m for m in range(lo, hi + 1) if m % 12 not in scale_pcs]
                for t in range(start_step, end_step):
                    var = self.voices[name][t]
                    is_diatonic = self.model.NewBoolVar(
                        f'diatonic_{name}_t{t}_k{key_root}')
                    self.model.AddAllowedAssignments(
                        [var], [[v] for v in diatonic_values]
                    ).OnlyEnforceIf(is_diatonic)
                    if chromatic_values:
                        self.model.AddAllowedAssignments(
                            [var], [[v] for v in chromatic_values]
                        ).OnlyEnforceIf(is_diatonic.Not())
                    else:
                        # All values in range are diatonic — force true
                        self.model.Add(is_diatonic == 1)
                    for _ in range(diatonic_weight):
                        self._objective_terms.append(is_diatonic)
                    # Don't narrow domain tracker — chromatic notes are allowed

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

    def add_unison_penalty(self, weight=1):
        """
        Soft constraint penalising two voices landing on the exact same MIDI pitch
        at any step.  Each unison occurrence reduces the objective by `weight`.
        """
        for t in range(self.steps):
            for i in range(len(self.voice_names)):
                for j in range(i + 1, len(self.voice_names)):
                    vi = self.voices[self.voice_names[i]][t]
                    vj = self.voices[self.voice_names[j]][t]
                    is_unison = self.model.NewBoolVar(
                        f'unison_{self.voice_names[i]}_{self.voice_names[j]}_t{t}')
                    # is_unison == 1  =>  vi == vj
                    self.model.Add(vi == vj).OnlyEnforceIf(is_unison)
                    # is_unison == 0  =>  vi != vj
                    self.model.Add(vi != vj).OnlyEnforceIf(is_unison.Not())
                    for _ in range(weight):
                        self._objective_terms.append(is_unison.Not())

    def add_spacing_constraint(self, max_total_spread=19, max_adjacent_upper=12,
                               weight=1, max_gap_per_pair=None,
                               penalize_lowest_pair=False):
        """
        Soft constraint penalising wide voicings.

        - Total spread: penalise when highest – lowest voice > max_total_spread
        - Adjacent voices: penalise when the gap exceeds the limit.
        - max_gap_per_pair: optional dict mapping (upper_voice, lower_voice) tuple
          to max semitones, overriding max_adjacent_upper for that pair.
        - penalize_lowest_pair: if True, apply gap penalty to the lowest adjacent
          pair too (normally skipped).
        Relies on the no-crossing constraint ensuring voices[i] >= voices[i+1].
        """
        if max_gap_per_pair is None:
            max_gap_per_pair = {}

        last_pair_idx = len(self.voice_names) - 2
        for t in range(self.steps):
            top = self.voices[self.voice_names[0]][t]
            bot = self.voices[self.voice_names[-1]][t]
            spread = self.model.NewIntVar(0, 72, f'spread_t{t}')
            self.model.Add(spread == top - bot)
            is_wide = self.model.NewBoolVar(f'wide_spread_t{t}')
            self.model.Add(spread > max_total_spread).OnlyEnforceIf(is_wide)
            self.model.Add(spread <= max_total_spread).OnlyEnforceIf(is_wide.Not())
            for _ in range(weight):
                self._objective_terms.append(is_wide.Not())

            # Adjacent voice pairs
            for i in range(len(self.voice_names) - 1):
                # Skip lowest pair unless penalize_lowest_pair is set
                if i == last_pair_idx and not penalize_lowest_pair:
                    continue
                upper_name = self.voice_names[i]
                lower_name = self.voice_names[i + 1]
                upper = self.voices[upper_name][t]
                lower = self.voices[lower_name][t]
                # Per-pair limit or fallback
                limit = max_gap_per_pair.get((upper_name, lower_name), max_adjacent_upper)
                gap = self.model.NewIntVar(0, 72,
                    f'adj_gap_{upper_name}_{lower_name}_t{t}')
                self.model.Add(gap == upper - lower)
                is_far = self.model.NewBoolVar(
                    f'adj_far_{upper_name}_{lower_name}_t{t}')
                self.model.Add(gap > limit).OnlyEnforceIf(is_far)
                self.model.Add(gap <= limit).OnlyEnforceIf(is_far.Not())
                for _ in range(weight):
                    self._objective_terms.append(is_far.Not())

    def add_chord_completeness_constraint(self, step_index, root_note, chord_type,
                                          exclude_voices=None):
        """
        Hard constraint: every pitch class in the chord must appear in at least one
        voice at the given step.

        exclude_voices affects range selection only — excluded voices still
        participate in completeness but use range [24, 96].
        """
        if exclude_voices is None:
            exclude_voices = []
        self._constraint_log.append({
            'type': 'chord_completeness', 'step': step_index,
            'root': root_note, 'chord': chord_type,
            'exclude_voices': list(exclude_voices),
        })
        chord_intervals = self._lookup_chord(chord_type)
        if not chord_intervals:
            print(f"Warning: Chord type '{chord_type}' not found (completeness).")
            return

        for interval in chord_intervals:
            pc = (root_note + interval) % 12
            indicators = []
            for name in self.voice_names:
                var = self.voices[name][step_index]
                if name in exclude_voices:
                    lo, hi = 24, 96
                else:
                    ranges = self.theory['voice_ranges_midi'].get(
                        name, {'min': 36, 'max': 84})
                    lo, hi = ranges['min'], ranges['max']
                matching = [m for m in range(lo, hi + 1) if m % 12 == pc]
                non_matching = [m for m in range(lo, hi + 1) if m % 12 != pc]

                has_pc = self.model.NewBoolVar(
                    f'has_pc{pc}_{name}_t{step_index}')
                if matching:
                    self.model.AddAllowedAssignments(
                        [var], [[v] for v in matching]).OnlyEnforceIf(has_pc)
                else:
                    # Voice can never cover this PC — force indicator false
                    self.model.Add(has_pc == 0)
                if non_matching:
                    self.model.AddAllowedAssignments(
                        [var], [[v] for v in non_matching]).OnlyEnforceIf(has_pc.Not())
                else:
                    # Voice must cover this PC (only matching values exist)
                    self.model.Add(has_pc == 1)
                indicators.append(has_pc)

            # At least one voice must cover this pitch class
            self.model.AddBoolOr(indicators)

    def add_bass_restriction_constraint(self, step_index, root_note, chord_type,
                                        bass_voice=None):
        """
        Hard constraint: the bass voice may only play certain chord tones
        (typically root or fifth), as defined in theory_definitions.json
        under bass_restrictions.
        """
        if bass_voice is None:
            bass_voice = self.voice_names[-1] if self.voice_names else None
        if bass_voice is None or bass_voice not in self.voices:
            return

        restrictions = self.theory.get('bass_restrictions', {}).get(chord_type)
        if not restrictions:
            return

        chord_intervals = self._lookup_chord(chord_type)
        if not chord_intervals:
            return

        allowed_pcs = set((root_note + iv) % 12 for iv in restrictions)

        ranges = self.theory['voice_ranges_midi'].get(
            bass_voice, {'min': 36, 'max': 62})
        lo, hi = ranges['min'], ranges['max']
        allowed = [m for m in range(lo, hi + 1) if m % 12 in allowed_pcs]
        self._constraint_log.append({
            'type': 'bass_restriction', 'step': step_index,
            'root': root_note, 'chord': chord_type,
            'bass_voice': bass_voice, 'allowed_pcs': list(allowed_pcs),
        })
        if allowed:
            self.model.AddAllowedAssignments(
                [self.voices[bass_voice][step_index]],
                [[v] for v in allowed])
            self._domain_tracker[bass_voice][step_index] &= set(allowed)

    def add_parallel_octave_penalty(self, weight=1):
        """
        Soft constraint penalising parallel octaves (or unisons) between
        consecutive steps.  A parallel octave is when two voices are an octave
        (or unison) apart at step t AND at step t+1, and at least one of
        them moved.
        """
        num_voices = len(self.voice_names)
        if self.steps < 2:
            return

        # Use shared moved BoolVars (cached via _get_moved_boolvar)

        # Octave multiples within MIDI range [24, 96] => max diff 72
        octave_values = [v for v in range(0, 73) if v % 12 == 0]

        for i in range(num_voices):
            for j in range(i + 1, num_voices):
                ni, nj = self.voice_names[i], self.voice_names[j]
                for t in range(self.steps - 1):
                    # --- is_octave at step t ---
                    diff_t = self.model.NewIntVar(0, 72,
                        f'pdiff_{ni}_{nj}_t{t}')
                    self.model.AddAbsEquality(
                        diff_t, self.voices[ni][t] - self.voices[nj][t])
                    is_oct_t = self.model.NewBoolVar(
                        f'isoct_{ni}_{nj}_t{t}')
                    self.model.AddAllowedAssignments(
                        [diff_t], [[v] for v in octave_values]
                    ).OnlyEnforceIf(is_oct_t)
                    non_octave = [v for v in range(0, 73) if v % 12 != 0]
                    self.model.AddAllowedAssignments(
                        [diff_t], [[v] for v in non_octave]
                    ).OnlyEnforceIf(is_oct_t.Not())

                    # --- is_octave at step t+1 ---
                    diff_t1 = self.model.NewIntVar(0, 72,
                        f'pdiff_{ni}_{nj}_t{t + 1}_from{t}')
                    self.model.AddAbsEquality(
                        diff_t1,
                        self.voices[ni][t + 1] - self.voices[nj][t + 1])
                    is_oct_t1 = self.model.NewBoolVar(
                        f'isoct_{ni}_{nj}_t{t + 1}_from{t}')
                    self.model.AddAllowedAssignments(
                        [diff_t1], [[v] for v in octave_values]
                    ).OnlyEnforceIf(is_oct_t1)
                    self.model.AddAllowedAssignments(
                        [diff_t1], [[v] for v in non_octave]
                    ).OnlyEnforceIf(is_oct_t1.Not())

                    # --- either voice moved ---
                    mi = self._get_moved_boolvar(ni, t)
                    mj = self._get_moved_boolvar(nj, t)
                    either_moved = self.model.NewBoolVar(
                        f'emoved_{ni}_{nj}_t{t}')
                    self.model.AddBoolOr([mi, mj]).OnlyEnforceIf(either_moved)
                    self.model.AddBoolAnd([mi.Not(), mj.Not()]
                                          ).OnlyEnforceIf(either_moved.Not())

                    # --- is_parallel_octave = oct_t AND oct_t1 AND either_moved ---
                    is_par = self.model.NewBoolVar(
                        f'par_oct_{ni}_{nj}_t{t}')
                    self.model.AddBoolAnd([is_oct_t, is_oct_t1, either_moved]
                                          ).OnlyEnforceIf(is_par)
                    # Negation: at least one is false => not parallel
                    self.model.AddBoolOr([is_oct_t.Not(), is_oct_t1.Not(),
                                          either_moved.Not()]
                                         ).OnlyEnforceIf(is_par.Not())

                    for _ in range(weight):
                        self._objective_terms.append(is_par.Not())

    def add_resolution_constraint(self, step_index, root_note, chord_type,
                                   next_step_index=None, hard=False, weight=2,
                                   exclude_voices=None):
        """
        Tendency tone resolution: reward (or enforce) the 7th resolving down
        by step and the 3rd (leading tone) resolving up by half step.

        Works between step_index and next_step_index (defaults to step_index+1).
        """
        if exclude_voices is None:
            exclude_voices = []
        if next_step_index is None:
            next_step_index = step_index + 1
        if next_step_index >= self.steps:
            return

        chord_intervals = self._lookup_chord(chord_type)
        if not chord_intervals or len(chord_intervals) < 3:
            return

        # Identify the 3rd and 7th intervals of the chord
        third_interval = chord_intervals[1]  # 3rd is index 1 (e.g. 4 for major/dom7)
        seventh_interval = chord_intervals[3] if len(chord_intervals) >= 4 else None

        third_pc = (root_note + third_interval) % 12
        seventh_pc = (root_note + seventh_interval) % 12 if seventh_interval is not None else None

        self._constraint_log.append({
            'type': 'resolution', 'step': step_index,
            'root': root_note, 'chord': chord_type,
            'next_step': next_step_index,
            'exclude_voices': list(exclude_voices),
        })

        for name in self.voice_names:
            if name in exclude_voices:
                continue
            note_var = self.voices[name][step_index]
            next_var = self.voices[name][next_step_index]

            if name in exclude_voices:
                lo, hi = 24, 96
            else:
                ranges = self.theory['voice_ranges_midi'].get(name, {'min': 36, 'max': 84})
                lo, hi = ranges['min'], ranges['max']

            # --- 7th resolution: resolve down by 1 or 2 semitones ---
            if seventh_pc is not None:
                seventh_pitches = [m for m in range(lo, hi + 1) if m % 12 == seventh_pc]
                non_seventh = [m for m in range(lo, hi + 1) if m % 12 != seventh_pc]

                if seventh_pitches:
                    has_seventh = self.model.NewBoolVar(f'has7th_{name}_t{step_index}')
                    self.model.AddAllowedAssignments(
                        [note_var], [[v] for v in seventh_pitches]
                    ).OnlyEnforceIf(has_seventh)
                    if non_seventh:
                        self.model.AddAllowedAssignments(
                            [note_var], [[v] for v in non_seventh]
                        ).OnlyEnforceIf(has_seventh.Not())
                    else:
                        self.model.Add(has_seventh == 1)

                    # diff = next - current; want diff in {-1, -2}
                    diff7 = self.model.NewIntVar(-72, 72, f'res7diff_{name}_t{step_index}')
                    self.model.Add(diff7 == next_var - note_var)
                    resolves_down = self.model.NewBoolVar(f'res7ok_{name}_t{step_index}')
                    self.model.AddAllowedAssignments(
                        [diff7], [[-1], [-2]]
                    ).OnlyEnforceIf(resolves_down)
                    non_resolve = [v for v in range(-72, 73) if v not in (-1, -2)]
                    self.model.AddAllowedAssignments(
                        [diff7], [[v] for v in non_resolve]
                    ).OnlyEnforceIf(resolves_down.Not())

                    # Combine: reward when has_seventh AND resolves_down
                    both7 = self.model.NewBoolVar(f'res7both_{name}_t{step_index}')
                    self.model.AddBoolAnd([has_seventh, resolves_down]).OnlyEnforceIf(both7)
                    self.model.AddBoolOr([has_seventh.Not(), resolves_down.Not()]).OnlyEnforceIf(both7.Not())

                    if hard:
                        # If has_seventh, must resolve down
                        self.model.AddImplication(has_seventh, resolves_down)
                    else:
                        for _ in range(weight):
                            self._objective_terms.append(both7)

            # --- 3rd (leading tone) resolution: resolve up by 1 semitone ---
            third_pitches = [m for m in range(lo, hi + 1) if m % 12 == third_pc]
            non_third = [m for m in range(lo, hi + 1) if m % 12 != third_pc]

            if third_pitches:
                has_third = self.model.NewBoolVar(f'has3rd_{name}_t{step_index}')
                self.model.AddAllowedAssignments(
                    [note_var], [[v] for v in third_pitches]
                ).OnlyEnforceIf(has_third)
                if non_third:
                    self.model.AddAllowedAssignments(
                        [note_var], [[v] for v in non_third]
                    ).OnlyEnforceIf(has_third.Not())
                else:
                    self.model.Add(has_third == 1)

                diff3 = self.model.NewIntVar(-72, 72, f'res3diff_{name}_t{step_index}')
                self.model.Add(diff3 == next_var - note_var)
                resolves_up = self.model.NewBoolVar(f'res3ok_{name}_t{step_index}')
                self.model.AddAllowedAssignments([diff3], [[1]]).OnlyEnforceIf(resolves_up)
                non_up = [v for v in range(-72, 73) if v != 1]
                self.model.AddAllowedAssignments(
                    [diff3], [[v] for v in non_up]
                ).OnlyEnforceIf(resolves_up.Not())

                both3 = self.model.NewBoolVar(f'res3both_{name}_t{step_index}')
                self.model.AddBoolAnd([has_third, resolves_up]).OnlyEnforceIf(both3)
                self.model.AddBoolOr([has_third.Not(), resolves_up.Not()]).OnlyEnforceIf(both3.Not())

                if hard:
                    self.model.AddImplication(has_third, resolves_up)
                else:
                    for _ in range(weight):
                        self._objective_terms.append(both3)

    def _get_moved_boolvar(self, name, t):
        """Lazy-create and cache a BoolVar indicating voice `name` moved between t and t+1."""
        key = (name, t)
        if key not in self._moved_vars:
            m = self.model.NewBoolVar(f'moved_{name}_t{t}')
            self.model.Add(
                self.voices[name][t] != self.voices[name][t + 1]
            ).OnlyEnforceIf(m)
            self.model.Add(
                self.voices[name][t] == self.voices[name][t + 1]
            ).OnlyEnforceIf(m.Not())
            self._moved_vars[key] = m
        return self._moved_vars[key]

    def add_common_tone_retention(self, chord_steps, weight=2):
        """
        Soft constraint rewarding voices that hold a common tone at the same
        MIDI pitch across consecutive chord changes.

        chord_steps: list of (step_index, root_pc, chord_type) sorted by step.
        """
        if len(chord_steps) < 2:
            return

        for ci in range(len(chord_steps) - 1):
            step_t, root_t, quality_t = chord_steps[ci]
            step_t1, root_t1, quality_t1 = chord_steps[ci + 1]

            intervals_t = self._lookup_chord(quality_t)
            intervals_t1 = self._lookup_chord(quality_t1)
            if not intervals_t or not intervals_t1:
                continue

            pcs_t = set((root_t + iv) % 12 for iv in intervals_t)
            pcs_t1 = set((root_t1 + iv) % 12 for iv in intervals_t1)
            common_pcs = pcs_t & pcs_t1
            if not common_pcs:
                continue

            for pc in common_pcs:
                for name in self.voice_names:
                    var_t = self.voices[name][step_t]
                    var_t1 = self.voices[name][step_t1]

                    # has_pc: voice has this PC at step t
                    matching = [m for m in range(24, 97) if m % 12 == pc]
                    non_matching = [m for m in range(24, 97) if m % 12 != pc]
                    has_pc = self.model.NewBoolVar(
                        f'ct_haspc{pc}_{name}_s{step_t}')
                    self.model.AddAllowedAssignments(
                        [var_t], [[v] for v in matching]
                    ).OnlyEnforceIf(has_pc)
                    self.model.AddAllowedAssignments(
                        [var_t], [[v] for v in non_matching]
                    ).OnlyEnforceIf(has_pc.Not())

                    # stays: note unchanged between step t and t+1
                    stays = self.model.NewBoolVar(
                        f'ct_stays_{name}_s{step_t}_{step_t1}_pc{pc}')
                    self.model.Add(var_t == var_t1).OnlyEnforceIf(stays)
                    self.model.Add(var_t != var_t1).OnlyEnforceIf(stays.Not())

                    # held: has_pc AND stays
                    held = self.model.NewBoolVar(
                        f'ct_held_{name}_s{step_t}_{step_t1}_pc{pc}')
                    self.model.AddBoolAnd([has_pc, stays]).OnlyEnforceIf(held)
                    self.model.AddBoolOr(
                        [has_pc.Not(), stays.Not()]
                    ).OnlyEnforceIf(held.Not())

                    for _ in range(weight):
                        self._objective_terms.append(held)

    def add_tessitura_preference(self, weight=1):
        """
        Soft constraint rewarding notes within the voice's tessitura sub-range.
        Voices without tessitura_min/tessitura_max in theory data are skipped.
        """
        for name in self.voice_names:
            ranges = self.theory['voice_ranges_midi'].get(name, {})
            tess_min = ranges.get('tessitura_min')
            tess_max = ranges.get('tessitura_max')
            if tess_min is None or tess_max is None:
                continue
            tessitura_values = list(range(tess_min, tess_max + 1))
            for t in range(self.steps):
                var = self.voices[name][t]
                in_tess = self.model.NewBoolVar(f'tess_{name}_t{t}')
                self.model.AddAllowedAssignments(
                    [var], [[v] for v in tessitura_values]
                ).OnlyEnforceIf(in_tess)
                outside = [m for m in range(24, 97) if m < tess_min or m > tess_max]
                self.model.AddAllowedAssignments(
                    [var], [[v] for v in outside]
                ).OnlyEnforceIf(in_tess.Not())
                for _ in range(weight):
                    self._objective_terms.append(in_tess)

    def add_stepwise_motion_preference(self, inner_voices=None, max_stepwise=2, weight=1):
        """
        Soft constraint rewarding stepwise motion (0, 1, or 2 semitones) in
        inner voices. If inner_voices is None, auto-detects as all voices
        except the first and last.
        """
        if self.steps < 2:
            return
        if inner_voices is None:
            if len(self.voice_names) <= 2:
                return
            inner_voices = self.voice_names[1:-1]

        stepwise_values = list(range(0, max_stepwise + 1))
        non_stepwise = list(range(max_stepwise + 1, 73))

        for name in inner_voices:
            if name not in self.voices:
                continue
            for t in range(self.steps - 1):
                var_t = self.voices[name][t]
                var_t1 = self.voices[name][t + 1]
                abs_motion = self.model.NewIntVar(0, 72, f'sw_abs_{name}_t{t}')
                self.model.AddAbsEquality(abs_motion, var_t1 - var_t)
                is_stepwise = self.model.NewBoolVar(f'sw_ok_{name}_t{t}')
                self.model.AddAllowedAssignments(
                    [abs_motion], [[v] for v in stepwise_values]
                ).OnlyEnforceIf(is_stepwise)
                self.model.AddAllowedAssignments(
                    [abs_motion], [[v] for v in non_stepwise]
                ).OnlyEnforceIf(is_stepwise.Not())
                for _ in range(weight):
                    self._objective_terms.append(is_stepwise)

    def add_contrary_motion_preference(self, melody_voice=None, bass_voice=None, weight=1):
        """
        Soft constraint rewarding contrary motion between melody and bass voices.
        When one moves up and the other moves down, reward. Stationary = neutral.
        """
        if melody_voice is None:
            melody_voice = self.voice_names[0]
        if bass_voice is None:
            bass_voice = self.voice_names[-1]
        if melody_voice not in self.voices or bass_voice not in self.voices:
            return
        if self.steps < 2:
            return

        self._constraint_log.append({
            'type': 'contrary_motion',
            'melody_voice': melody_voice, 'bass_voice': bass_voice,
        })

        for t in range(self.steps - 1):
            mel_curr = self.voices[melody_voice][t]
            mel_next = self.voices[melody_voice][t + 1]
            bass_curr = self.voices[bass_voice][t]
            bass_next = self.voices[bass_voice][t + 1]

            # Melody direction: up (mel_next > mel_curr)
            mel_up = self.model.NewBoolVar(f'mel_up_t{t}')
            self.model.Add(mel_next > mel_curr).OnlyEnforceIf(mel_up)
            self.model.Add(mel_next <= mel_curr).OnlyEnforceIf(mel_up.Not())

            mel_down = self.model.NewBoolVar(f'mel_down_t{t}')
            self.model.Add(mel_next < mel_curr).OnlyEnforceIf(mel_down)
            self.model.Add(mel_next >= mel_curr).OnlyEnforceIf(mel_down.Not())

            # Bass direction
            bass_up = self.model.NewBoolVar(f'bass_up_t{t}')
            self.model.Add(bass_next > bass_curr).OnlyEnforceIf(bass_up)
            self.model.Add(bass_next <= bass_curr).OnlyEnforceIf(bass_up.Not())

            bass_down = self.model.NewBoolVar(f'bass_down_t{t}')
            self.model.Add(bass_next < bass_curr).OnlyEnforceIf(bass_down)
            self.model.Add(bass_next >= bass_curr).OnlyEnforceIf(bass_down.Not())

            # Contrary: mel_up AND bass_down, or mel_down AND bass_up
            contrary1 = self.model.NewBoolVar(f'contrary1_t{t}')
            self.model.AddBoolAnd([mel_up, bass_down]).OnlyEnforceIf(contrary1)
            self.model.AddBoolOr([mel_up.Not(), bass_down.Not()]).OnlyEnforceIf(contrary1.Not())

            contrary2 = self.model.NewBoolVar(f'contrary2_t{t}')
            self.model.AddBoolAnd([mel_down, bass_up]).OnlyEnforceIf(contrary2)
            self.model.AddBoolOr([mel_down.Not(), bass_up.Not()]).OnlyEnforceIf(contrary2.Not())

            is_contrary = self.model.NewBoolVar(f'contrary_t{t}')
            self.model.AddBoolOr([contrary1, contrary2]).OnlyEnforceIf(is_contrary)
            self.model.AddBoolAnd([contrary1.Not(), contrary2.Not()]).OnlyEnforceIf(is_contrary.Not())

            for _ in range(weight):
                self._objective_terms.append(is_contrary)

    def add_specific_bass_constraint(self, step_index, pitch_class, bass_voice=None):
        """
        Hard constraint: Forces the bass voice to a specific pitch class.
        Overrides general bass restrictions (like root/fifth only) if they conflict,
        though usually used in conjunction with a chord that contains this PC.
        """
        if bass_voice is None:
            bass_voice = self.voice_names[-1] if self.voice_names else None
        if bass_voice is None or bass_voice not in self.voices:
            return

        self._constraint_log.append({
            'type': 'specific_bass', 'step': step_index,
            'pitch_class': pitch_class, 'bass_voice': bass_voice
        })

        bass_var = self.voices[bass_voice][step_index]
        ranges = self.theory['voice_ranges_midi'].get(bass_voice, {'min': 36, 'max': 62})
        lo, hi = ranges['min'], ranges['max']
        
        allowed_values = [m for m in range(lo, hi + 1) if m % 12 == pitch_class]
        
        if allowed_values:
            self.model.AddAllowedAssignments([bass_var], [[v] for v in allowed_values])
            self._domain_tracker[bass_voice][step_index] &= set(allowed_values)
        else:
            print(f"Warning: Specific bass PC {pitch_class} has no valid notes in range {lo}-{hi} for {bass_voice}")
            # Ensure failure if impossible
            self.model.Add(bass_var == -1) 

    def add_rhythmic_stagger_preference(self, melody_voice, inner_voices, weight=1):
        """
        Soft constraint: When the melody holds a note (static from t to t+1),
        reward at least one inner voice for moving. Creates a more active texture.
        """
        if self.steps < 2:
            return
        if melody_voice not in self.voices:
            return
        
        valid_inner = [v for v in inner_voices if v in self.voices]
        if not valid_inner:
            return

        for t in range(self.steps - 1):
            # 1. Check if melody is static
            mel_moved = self._get_moved_boolvar(melody_voice, t)
            mel_static = self.model.NewBoolVar(f'mel_static_t{t}')
            self.model.Add(mel_moved == 0).OnlyEnforceIf(mel_static)
            self.model.Add(mel_moved == 1).OnlyEnforceIf(mel_static.Not())

            # 2. Check if ANY inner voice moved
            inner_moves = [self._get_moved_boolvar(v, t) for v in valid_inner]
            any_inner_moved = self.model.NewBoolVar(f'any_inner_moved_t{t}')
            self.model.AddBoolOr(inner_moves).OnlyEnforceIf(any_inner_moved)
            self.model.AddBoolAnd([m.Not() for m in inner_moves]).OnlyEnforceIf(any_inner_moved.Not())

            # 3. Reward (Melody Static AND Inner Moved)
            staggered = self.model.NewBoolVar(f'staggered_t{t}')
            self.model.AddBoolAnd([mel_static, any_inner_moved]).OnlyEnforceIf(staggered)
            self.model.AddBoolOr([mel_static.Not(), any_inner_moved.Not()]).OnlyEnforceIf(staggered.Not())

            for _ in range(weight):
                self._objective_terms.append(staggered)

    def validate(self):
        """
        Pre-solve validation. Returns a list of (level, message) tuples
        where level is 'ERROR', 'WARN', or 'INFO'.
        """
        results = []
        NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

        def pitch_str(midi):
            return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"

        # Check A: Empty domain intersection
        for name in self.voice_names:
            for t in range(self.steps):
                domain = self._domain_tracker.get(name, {}).get(t, set())
                if not domain:
                    # Find which constraints touched this voice/step
                    relevant = [c for c in self._constraint_log
                                if c.get('step') == t or c['type'] in ('scale', 'no_crossing', 'voice_leading')]
                    constraint_types = [c['type'] for c in relevant]
                    results.append(('ERROR',
                        f"Empty domain for {name} at step {t}. "
                        f"Constraints applied: {', '.join(constraint_types)}"))

        # Check B: Melody note outside scale (iterate all segments)
        if self._scale_segments and self._melody_voice:
            for step, pitch in self._melody_pins.items():
                # Find the segment covering this step
                for seg in self._scale_segments:
                    key_root, scale_type, scale_pcs, scale_excludes, scale_hard, s_start, s_end = seg
                    if s_start <= step < s_end and self._melody_voice not in scale_excludes:
                        if pitch % 12 not in scale_pcs:
                            level = 'ERROR' if scale_hard else 'INFO'
                            results.append((level,
                                f"Melody note {pitch_str(pitch)} (pc {pitch % 12}) "
                                f"at step {step} not in {NOTE_NAMES[key_root]} {scale_type} "
                                f"scale (pcs {sorted(scale_pcs)})"))
                        break  # first matching segment wins

        # Check C: soprano_on_root vs melody pin conflict
        soprano_root_entries = [c for c in self._constraint_log
                                if c['type'] == 'soprano_on_root']
        for entry in soprano_root_entries:
            step = entry['step']
            tonic_pc = entry['tonic_pc']
            if step in self._melody_pins:
                pinned_pc = self._melody_pins[step] % 12
                if pinned_pc != tonic_pc:
                    results.append(('ERROR',
                        f"Perfect authentic cadence requires {entry['voice']} on "
                        f"tonic (pc {tonic_pc}/{NOTE_NAMES[tonic_pc]}) at step {step}, "
                        f"but melody is pinned to {pitch_str(self._melody_pins[step])} "
                        f"(pc {pinned_pc}/{NOTE_NAMES[pinned_pc]})"))

        # Check D: Cadence truncation
        cadence_entries = [c for c in self._constraint_log if c['type'] == 'cadence']
        for entry in cadence_entries:
            if entry['progression_len'] > entry['available_steps']:
                results.append(('WARN',
                    f"{entry['cadence_type']} cadence has {entry['progression_len']} "
                    f"chords but only {entry['available_steps']} steps available "
                    f"(starting at step {entry['start_step']}). "
                    f"Cadence will be truncated."))

        # Check E: Voice leading feasibility estimate
        vl_entries = [c for c in self._constraint_log if c['type'] == 'voice_leading']
        for vl in vl_entries:
            per_voice = vl.get('per_voice_max') or {}
            default_max = vl['max_interval']
            excludes = vl.get('exclude_voices', [])
            for name in self.voice_names:
                if name in excludes:
                    continue
                limit = per_voice.get(name, default_max)
                for t in range(self.steps - 1):
                    domain_t = self._domain_tracker.get(name, {}).get(t, set())
                    domain_t1 = self._domain_tracker.get(name, {}).get(t + 1, set())
                    if not domain_t or not domain_t1:
                        continue  # already caught by Check A
                    min_jump = min(abs(a - b) for a in domain_t for b in domain_t1)
                    if min_jump > limit:
                        results.append(('WARN',
                            f"{name} must jump at least {min_jump} semitones "
                            f"between steps {t}-{t+1}, but limit is {limit}"))

        # Check F: CP-SAT model validation
        validation_msg = self.model.Validate()
        if validation_msg:
            results.append(('ERROR', f"CP-SAT model validation: {validation_msg}"))

        return results

    def get_diagnostic_report(self):
        """
        Formats a human-readable diagnostic report from validate() results,
        solver failure stats, and the constraint log summary.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("SOLVER DIAGNOSTIC REPORT")
        lines.append("=" * 60)

        # Diagnostics from validate()
        if self._diagnostics:
            lines.append("")
            lines.append("Pre-solve checks:")
            for level, msg in self._diagnostics:
                lines.append(f"  [{level}] {msg}")
        else:
            lines.append("")
            lines.append("Pre-solve checks: all passed")

        # Failure stats from solver
        if self._failure_stats:
            lines.append("")
            lines.append("Solver stats:")
            lines.append(f"  Status: {self._failure_stats['status']}")
            lines.append(f"  Wall time: {self._failure_stats['wall_time']:.3f}s")
            lines.append(f"  Conflicts: {self._failure_stats['num_conflicts']}")
            lines.append(f"  Branches: {self._failure_stats['num_branches']}")

        # Constraint log summary
        if self._constraint_log:
            lines.append("")
            lines.append("Constraints applied:")
            type_counts = {}
            for entry in self._constraint_log:
                t = entry['type']
                type_counts[t] = type_counts.get(t, 0) + 1
            for ctype, count in type_counts.items():
                lines.append(f"  {ctype}: {count}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def solve(self):
        """
        Runs the solver and returns a dictionary of results.
        """
        # Run pre-solve validation
        self._diagnostics = self.validate()
        for level, msg in self._diagnostics:
            if level in ('ERROR', 'WARN'):
                print(f"[{level}] {msg}")

        # Apply accumulated soft-constraint objective
        if self._objective_terms:
            self.model.Maximize(sum(self._objective_terms))

        status = self.solver.Solve(self.model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result = {}
            for name in self.voice_names:
                result[name] = [self.solver.Value(v) for v in self.voices[name]]
            return result
        else:
            self._failure_stats = {
                'status': self.solver.status_name(status),
                'wall_time': self.solver.wall_time,
                'num_conflicts': self.solver.num_conflicts,
                'num_branches': self.solver.num_branches,
            }
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
    solver.add_voice_leading_constraint(max_interval=7,
                                        per_voice_max={'alto': 4, 'tenor': 4},
                                        exclude_voices=['soprano'])

    # Keep all notes diatonic to C major
    solver.add_scale_constraint(key_root=0, scale_type='major', exclude_voices=['soprano'])

    # Apply chord constraints: I - V - IV - I
    solver.add_harmonic_constraint(0, 0, 'major', exclude_voices=['soprano'])   # C major (I)
    solver.add_harmonic_constraint(1, 7, 'major', exclude_voices=['soprano'])   # G major (V)

    # Chord completeness for explicit chords
    solver.add_chord_completeness_constraint(0, 0, 'major', exclude_voices=['soprano'])
    solver.add_chord_completeness_constraint(1, 7, 'major', exclude_voices=['soprano'])

    # Bass restrictions for explicit chords
    solver.add_bass_restriction_constraint(0, 0, 'major')
    solver.add_bass_restriction_constraint(1, 7, 'major')

    # Plagal cadence at the end (steps 2-3): IV -> I
    solver.add_cadence_constraint(key_root=0, scale_type='major',
                                  cadence_type='plagal', start_step=2,
                                  exclude_voices=['soprano'],
                                  ensure_completeness=True, restrict_bass=True)

    # Prefer root in bass (use correct root per chord)
    solver.add_doubling_preference(0, 0)   # C for I
    solver.add_doubling_preference(1, 7)   # G for V
    solver.add_doubling_preference(2, 5)   # F for IV
    solver.add_doubling_preference(3, 0)   # C for I

    # Voicing quality: soft constraints
    solver.add_unison_penalty()
    solver.add_spacing_constraint()
    solver.add_parallel_octave_penalty()

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
