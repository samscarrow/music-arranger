"""
Musical Arrangement Validation & Scoring Harness

Validates (Header, list[Event]) arrangements against hard constraints (errors)
and soft constraints (warnings + scores). Produces a scorecard for comparing
models and decoding settings.

Usage:
    python tools/validate_arrangement.py <token_file>
    
API:
    from validate_arrangement import validate
    scorecard = validate(header, events)
"""

import sys
from itertools import combinations
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))
from event import Header, Event, parse_tokens


@dataclass
class Issue:
    """A single validation issue (error or warning)."""
    level: str          # 'error' or 'warning'
    check: str          # check name, e.g. 'voice_range'
    event_idx: int      # which event (0-based), or -1 for global
    message: str


VOICE_RANGES = {
    'tenor': (46, 69),
    'lead':  (57, 74),
    'bari':  (50, 72),
    'bass':  (36, 62),
}

VOICES = ('bass', 'bari', 'lead', 'tenor')  # bottom to top

CHORD_PC_SETS = {
    'MAJOR_TRIAD': {0, 4, 7},
    'DOM7':        {0, 4, 7, 10},
    'MINOR7':      {0, 3, 7, 10},
    'MINOR_TRIAD': {0, 3, 7},
    'HALF_DIM':    {0, 3, 6, 10},
    'AUG':         {0, 4, 8},
    'UNISON':      None,
    'OPEN_5TH':    None,
    'OTHER':       None,
}


# ---------------------------------------------------------------------------
# Hard constraint checks
# ---------------------------------------------------------------------------

def check_voice_ranges(events: list[Event]) -> list[Issue]:
    """Check that each non-rest voice pitch falls within its allowed range."""
    issues: list[Issue] = []
    for idx, ev in enumerate(events):
        for voice in VOICES:
            pitch = getattr(ev, voice)
            if pitch is None:
                continue
            lo, hi = VOICE_RANGES[voice]
            if pitch < lo or pitch > hi:
                issues.append(Issue(
                    level='error',
                    check='voice_range',
                    event_idx=idx,
                    message=f"{voice} pitch {pitch} outside range [{lo}, {hi}]",
                ))
    return issues


def check_voice_crossing(events: list[Event]) -> list[Issue]:
    """Check that voices do not cross: bass ≤ bari ≤ lead ≤ tenor."""
    issues: list[Issue] = []
    pairs = list(zip(VOICES, VOICES[1:]))  # (bass,bari), (bari,lead), (lead,tenor)
    for idx, ev in enumerate(events):
        for lower, upper in pairs:
            lo_pitch = getattr(ev, lower)
            hi_pitch = getattr(ev, upper)
            if lo_pitch is None or hi_pitch is None:
                continue
            if lo_pitch > hi_pitch:
                issues.append(Issue(
                    level='error',
                    check='voice_crossing',
                    event_idx=idx,
                    message=f"{lower}({lo_pitch}) > {upper}({hi_pitch})",
                ))
    return issues


def check_durations_offsets(events: list[Event]) -> list[Issue]:
    """Check positive durations and monotonically non-decreasing offsets."""
    issues: list[Issue] = []
    for idx, ev in enumerate(events):
        if ev.dur <= 0:
            issues.append(Issue(
                level='error',
                check='duration',
                event_idx=idx,
                message=f"dur={ev.dur} is not positive",
            ))
        if idx > 0 and ev.offset_qn < events[idx - 1].offset_qn:
            issues.append(Issue(
                level='error',
                check='offset_order',
                event_idx=idx,
                message=(
                    f"offset {ev.offset_qn} < previous offset "
                    f"{events[idx - 1].offset_qn}"
                ),
            ))
    return issues


def check_spacing(events: list[Event]) -> list[Issue]:
    """Check adjacent-voice and total-spread spacing limits."""
    issues: list[Issue] = []
    pairs = list(zip(VOICES, VOICES[1:]))
    for idx, ev in enumerate(events):
        pitches = {v: getattr(ev, v) for v in VOICES if getattr(ev, v) is not None}
        # Adjacent voice spacing ≤ 19 semitones
        for lower, upper in pairs:
            if lower in pitches and upper in pitches:
                gap = pitches[upper] - pitches[lower]
                if abs(gap) > 19:
                    issues.append(Issue(
                        level='error',
                        check='spacing',
                        event_idx=idx,
                        message=(
                            f"{lower}-{upper} gap {abs(gap)} semitones > 19"
                        ),
                    ))
        # Total spread ≤ 36 semitones
        if len(pitches) >= 2:
            spread = max(pitches.values()) - min(pitches.values())
            if spread > 36:
                issues.append(Issue(
                    level='error',
                    check='total_spread',
                    event_idx=idx,
                    message=f"total spread {spread} semitones > 36",
                ))
    return issues


# ---------------------------------------------------------------------------
# Soft constraint checks
# ---------------------------------------------------------------------------

def _get_pitch(event: Event, voice: str) -> int | None:
    return getattr(event, voice)


def check_voice_leaps(events: list[Event]) -> list[Issue]:
    """Warn on voice leaps > 12 semitones between consecutive events."""
    issues: list[Issue] = []
    for i in range(len(events) - 1):
        for v in VOICES:
            p1 = _get_pitch(events[i], v)
            p2 = _get_pitch(events[i + 1], v)
            if p1 is None or p2 is None:
                continue
            interval = abs(p2 - p1)
            if interval > 12:
                issues.append(Issue(
                    level='warning', check='voice_leaps', event_idx=i + 1,
                    message=f'{v} leaps {interval} semitones between events {i} and {i + 1}',
                ))
    return issues


def calc_leap_rate(events: list[Event]) -> float:
    """Fraction of voice movements > 7 semitones."""
    leaps = total = 0
    for i in range(len(events) - 1):
        for v in VOICES:
            p1 = _get_pitch(events[i], v)
            p2 = _get_pitch(events[i + 1], v)
            if p1 is None or p2 is None:
                continue
            total += 1
            if abs(p2 - p1) > 7:
                leaps += 1
    return leaps / total if total > 0 else 0.0


def check_parallel_fifths_octaves(events: list[Event]) -> list[Issue]:
    """Detect parallel perfect 5ths and octaves between voice pairs."""
    issues: list[Issue] = []
    for i in range(len(events) - 1):
        for va, vb in combinations(VOICES, 2):
            a1, a2 = _get_pitch(events[i], va), _get_pitch(events[i + 1], va)
            b1, b2 = _get_pitch(events[i], vb), _get_pitch(events[i + 1], vb)
            if any(p is None for p in (a1, a2, b1, b2)):
                continue
            if a1 == a2 and b1 == b2:
                continue
            if (a2 - a1) != (b2 - b1):
                continue
            interval_mod = abs(a2 - b2) % 12
            if interval_mod in (0, 7):
                kind = 'octave' if interval_mod == 0 else 'fifth'
                issues.append(Issue(
                    level='warning', check='parallel_fifths_octaves', event_idx=i + 1,
                    message=f'Parallel {kind} between {va} and {vb} at events {i}-{i + 1}',
                ))
    return issues


def calc_parallel_rate(events: list[Event]) -> float:
    """Fraction of voice-pair movements that are parallel 5ths/octaves."""
    parallels = total = 0
    for i in range(len(events) - 1):
        for va, vb in combinations(VOICES, 2):
            a1, a2 = _get_pitch(events[i], va), _get_pitch(events[i + 1], va)
            b1, b2 = _get_pitch(events[i], vb), _get_pitch(events[i + 1], vb)
            if any(p is None for p in (a1, a2, b1, b2)):
                continue
            total += 1
            if a1 == a2 and b1 == b2:
                continue
            if (a2 - a1) == (b2 - b1):
                interval_mod = abs(a2 - b2) % 12
                if interval_mod in (0, 7):
                    parallels += 1
    return parallels / total if total > 0 else 0.0


def _check_coverage_for_event(event: Event, chord_pcs: set[int]) -> float:
    """Best coverage ratio across all 12 possible chord roots."""
    voiced_pcs = {_get_pitch(event, v) % 12
                  for v in VOICES if _get_pitch(event, v) is not None}
    if not voiced_pcs:
        return 0.0
    best = 0.0
    for root in range(12):
        transposed = {(root + pc) % 12 for pc in chord_pcs}
        ratio = len(voiced_pcs & transposed) / len(chord_pcs)
        if ratio > best:
            best = ratio
            if best == 1.0:
                break
    return best


def check_chord_coverage(events: list[Event]) -> list[Issue]:
    """Warn when voiced pitch classes don't cover the labeled chord."""
    issues: list[Issue] = []
    for i, ev in enumerate(events):
        if ev.chord is None or ev.chord not in CHORD_PC_SETS:
            continue
        chord_pcs = CHORD_PC_SETS[ev.chord]
        if chord_pcs is None:
            continue
        coverage = _check_coverage_for_event(ev, chord_pcs)
        threshold = 0.75 if len(chord_pcs) == 4 else 1.0
        if coverage < threshold:
            issues.append(Issue(
                level='warning', check='chord_coverage', event_idx=i,
                message=f'Chord {ev.chord} coverage {coverage:.0%} < {threshold:.0%}',
            ))
    return issues


def calc_chord_coverage_rate(events: list[Event]) -> float:
    """Average chord-tone coverage across events with known chords."""
    total = count = 0
    for ev in events:
        if ev.chord is None or ev.chord not in CHORD_PC_SETS:
            continue
        chord_pcs = CHORD_PC_SETS[ev.chord]
        if chord_pcs is None:
            continue
        total += _check_coverage_for_event(ev, chord_pcs)
        count += 1
    return total / count if count > 0 else 1.0


def check_harmonic_rhythm(events: list[Event]) -> list[Issue]:
    """Warn on chord changes too fast (< 0.5 qn) or too static (> 8 qn)."""
    issues: list[Issue] = []
    if len(events) < 2:
        return issues

    prev_chord = events[0].chord
    chord_start = events[0].offset_qn

    for i in range(1, len(events)):
        cur = events[i].chord
        if cur != prev_chord:
            span = events[i].offset_qn - chord_start
            if span < 0.5:
                issues.append(Issue(
                    level='warning', check='harmonic_rhythm', event_idx=i,
                    message=f'Chord changes too fast ({span:.2f} qn)',
                ))
            if span > 8.0:
                issues.append(Issue(
                    level='warning', check='harmonic_rhythm', event_idx=i,
                    message=f'Chord {prev_chord} static for {span:.1f} qn',
                ))
            prev_chord = cur
            chord_start = events[i].offset_qn

    # Check final span
    if prev_chord is not None:
        final_span = (events[-1].offset_qn + events[-1].dur) - chord_start
        if final_span > 8.0:
            issues.append(Issue(
                level='warning', check='harmonic_rhythm',
                event_idx=len(events) - 1,
                message=f'Chord {prev_chord} static for {final_span:.1f} qn (final)',
            ))
    return issues


def calc_harmonic_rhythm_score(events: list[Event]) -> float:
    """1.0 minus penalties for rhythm issues, clamped to [0, 1]."""
    if len(events) < 2:
        return 1.0
    penalty = 0.0
    prev_chord = events[0].chord
    chord_start = events[0].offset_qn
    for i in range(1, len(events)):
        cur = events[i].chord
        if cur != prev_chord:
            span = events[i].offset_qn - chord_start
            if span < 0.5:
                penalty += 0.05
            elif span > 8.0:
                penalty += 0.1
            prev_chord = cur
            chord_start = events[i].offset_qn
    if prev_chord is not None:
        final_span = (events[-1].offset_qn + events[-1].dur) - chord_start
        if final_span > 8.0:
            penalty += 0.1
    return max(0.0, min(1.0, 1.0 - penalty))


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

@dataclass
class Scorecard:
    """Aggregated validation results for an arrangement."""
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    passed: bool = True

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def overall_score(self) -> float:
        """0–100 quality score. Errors = 0, otherwise weighted soft scores."""
        if self.errors:
            return 0.0
        weights = {
            'leap_rate': 20,
            'parallel_rate': 20,
            'chord_coverage': 30,
            'harmonic_rhythm': 30,
        }
        total = 0.0
        for key, weight in weights.items():
            val = self.scores.get(key, 1.0)
            if key in ('leap_rate', 'parallel_rate'):
                val = 1.0 - val
            total += val * weight
        return round(total, 1)


def validate(header: Header, events: list[Event]) -> Scorecard:
    """Run all hard and soft checks, return a Scorecard."""
    sc = Scorecard()

    for check_fn in (check_voice_ranges, check_voice_crossing,
                     check_durations_offsets, check_spacing):
        for issue in check_fn(events):
            sc.errors.append(issue)

    sc.passed = len(sc.errors) == 0

    for check_fn in (check_voice_leaps, check_parallel_fifths_octaves,
                     check_chord_coverage, check_harmonic_rhythm):
        for issue in check_fn(events):
            sc.warnings.append(issue)

    sc.scores['leap_rate'] = calc_leap_rate(events)
    sc.scores['parallel_rate'] = calc_parallel_rate(events)
    sc.scores['chord_coverage'] = calc_chord_coverage_rate(events)
    sc.scores['harmonic_rhythm'] = calc_harmonic_rhythm_score(events)

    return sc


def print_scorecard(sc: Scorecard) -> None:
    """Print a human-readable scorecard to stdout."""
    print("=" * 60)
    print("  ARRANGEMENT SCORECARD")
    print("=" * 60)
    print()

    status = "✅ PASS" if sc.passed else "❌ FAIL"
    print(f"  Status:         {status}")
    print(f"  Overall Score:  {sc.overall_score}/100")
    print(f"  Errors:         {sc.error_count}")
    print(f"  Warnings:       {sc.warning_count}")
    print()

    if sc.errors:
        print("  ERRORS:")
        for issue in sc.errors[:20]:
            print(f"    [{issue.check}] event {issue.event_idx}: {issue.message}")
        if len(sc.errors) > 20:
            print(f"    ... and {len(sc.errors) - 20} more")
        print()

    print("  SCORES:")
    print(f"    Leap rate:         {sc.scores.get('leap_rate', 0):.1%}  (lower is better)")
    print(f"    Parallel rate:     {sc.scores.get('parallel_rate', 0):.1%}  (lower is better)")
    print(f"    Chord coverage:    {sc.scores.get('chord_coverage', 0):.1%}  (higher is better)")
    print(f"    Harmonic rhythm:   {sc.scores.get('harmonic_rhythm', 0):.1%}  (higher is better)")
    print()

    if sc.warnings:
        print(f"  WARNINGS (first 10 of {len(sc.warnings)}):")
        for issue in sc.warnings[:10]:
            print(f"    [{issue.check}] event {issue.event_idx}: {issue.message}")
        print()

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <token_file>")
        print(f"  Validates an arrangement token file and prints a scorecard.")
        sys.exit(1)

    token_file = sys.argv[1]
    if not Path(token_file).exists():
        print(f"❌ File not found: {token_file}")
        sys.exit(1)

    text = Path(token_file).read_text()
    header, events = parse_tokens(text)

    print(f"Loaded {len(events)} events from {token_file}")
    print(f"  Key: {header.key}, Meter: {header.meter}")
    print()

    sc = validate(header, events)
    print_scorecard(sc)

    sys.exit(0 if sc.passed else 1)


if __name__ == '__main__':
    main()
