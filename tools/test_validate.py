#!/usr/bin/env python3
"""Tests for validate_arrangement.py"""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from event import Header, Event
from validate_arrangement import (
    validate, check_voice_ranges, check_voice_crossing,
    check_durations_offsets, check_spacing, check_voice_leaps,
    check_parallel_fifths_octaves, check_chord_coverage,
    check_harmonic_rhythm, calc_leap_rate, calc_parallel_rate,
    calc_chord_coverage_rate, calc_harmonic_rhythm_score,
)

passed = failed = 0


def _assert(cond, msg=""):
    global passed, failed
    if cond:
        passed += 1
        return True
    failed += 1
    print(f"  FAILED: {msg}")
    return False


def _make_header():
    return Header(key="C", meter="4/4")


def _good_events():
    """4 events: in-range, no crossing, valid chords, smooth motion."""
    return [
        Event(bar=1, offset_qn=0.0, lead=60, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
        Event(bar=1, offset_qn=1.0, lead=62, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
        Event(bar=1, offset_qn=2.0, lead=64, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
        Event(bar=1, offset_qn=3.0, lead=65, tenor=69, bari=57, bass=50, dur=1.0, chord="MAJOR_TRIAD"),
    ]


# ============================================================================
# Test 1: Perfect arrangement passes
# ============================================================================
print("Test 1: Perfect arrangement passes")
sc = validate(_make_header(), _good_events())
_assert(sc.passed is True, "expected passed=True")
_assert(sc.overall_score > 80, f"expected overall_score > 80, got {sc.overall_score}")
_assert(sc.error_count == 0, f"expected 0 errors, got {sc.error_count}")

# ============================================================================
# Test 2: Voice range violation detected
# ============================================================================
print("Test 2: Voice range violation detected")
events = [Event(bar=1, offset_qn=0.0, lead=80, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD")]
issues = check_voice_ranges(events)
_assert(len(issues) > 0, "expected at least one range issue")
_assert(issues[0].check == 'voice_range', f"expected check='voice_range', got '{issues[0].check}'")
_assert(issues[0].level == 'error', f"expected level='error', got '{issues[0].level}'")

# ============================================================================
# Test 3: Voice crossing detected
# ============================================================================
print("Test 3: Voice crossing detected")
events = [Event(bar=1, offset_qn=0.0, lead=67, tenor=55, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD")]
issues = check_voice_crossing(events)
_assert(len(issues) > 0, "expected at least one crossing issue")
_assert(issues[0].check == 'voice_crossing', f"expected check='voice_crossing', got '{issues[0].check}'")
_assert(issues[0].level == 'error', f"expected level='error', got '{issues[0].level}'")

# ============================================================================
# Test 4: Zero duration error
# ============================================================================
print("Test 4: Zero duration error")
events = [Event(bar=1, offset_qn=0.0, lead=60, tenor=67, bari=55, bass=48, dur=0.0, chord="MAJOR_TRIAD")]
issues = check_durations_offsets(events)
_assert(any(i.check == 'duration' for i in issues), "expected duration error")
_assert(all(i.level == 'error' for i in issues), "expected level='error'")

# ============================================================================
# Test 5: Non-monotonic offset error
# ============================================================================
print("Test 5: Non-monotonic offset error")
events = [
    Event(bar=1, offset_qn=2.0, lead=60, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
    Event(bar=1, offset_qn=1.0, lead=62, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
]
issues = check_durations_offsets(events)
_assert(any(i.check == 'offset_order' for i in issues), "expected offset_order error")

# ============================================================================
# Test 6: Spacing violation
# ============================================================================
print("Test 6: Spacing violation")
events = [Event(bar=1, offset_qn=0.0, lead=64, tenor=67, bari=60, bass=36, dur=1.0, chord="MAJOR_TRIAD")]
issues = check_spacing(events)
_assert(any(i.check == 'spacing' for i in issues), "expected spacing error")
_assert(any(i.level == 'error' for i in issues), "expected level='error'")

# ============================================================================
# Test 7: Voice leaps warning
# ============================================================================
print("Test 7: Voice leaps warning")
events = [
    Event(bar=1, offset_qn=0.0, lead=60, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
    Event(bar=1, offset_qn=1.0, lead=80, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
]
issues = check_voice_leaps(events)
_assert(any(i.check == 'voice_leaps' for i in issues), "expected voice_leaps warning")
_assert(all(i.level == 'warning' for i in issues), "expected level='warning'")

# ============================================================================
# Test 8: Parallel fifths detected
# ============================================================================
print("Test 8: Parallel fifths detected")
# bass and tenor move by same interval, landing a 5th apart
# Event 0: bass=48(C3), tenor=55(G3) — interval=7 (perfect 5th)
# Event 1: bass=50(D3), tenor=57(A3) — interval=7 (perfect 5th), both moved +2
events = [
    Event(bar=1, offset_qn=0.0, lead=60, tenor=55, bari=52, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
    Event(bar=1, offset_qn=1.0, lead=62, tenor=57, bari=54, bass=50, dur=1.0, chord="MAJOR_TRIAD"),
]
issues = check_parallel_fifths_octaves(events)
_assert(any(i.check == 'parallel_fifths_octaves' for i in issues), "expected parallel_fifths_octaves warning")
_assert(all(i.level == 'warning' for i in issues), "expected level='warning'")

# ============================================================================
# Test 9: Chord coverage warning
# ============================================================================
print("Test 9: Chord coverage warning")
# MAJOR_TRIAD needs {0,4,7}. All voices on C (pitch class 0) → coverage = 1/3
events = [Event(bar=1, offset_qn=0.0, lead=60, tenor=72, bari=48, bass=36, dur=1.0, chord="MAJOR_TRIAD")]
issues = check_chord_coverage(events)
_assert(any(i.check == 'chord_coverage' for i in issues), "expected chord_coverage warning")
_assert(all(i.level == 'warning' for i in issues), "expected level='warning'")

# ============================================================================
# Test 10: Harmonic rhythm too fast
# ============================================================================
print("Test 10: Harmonic rhythm too fast")
events = [
    Event(bar=1, offset_qn=0.0, lead=60, tenor=67, bari=55, bass=48, dur=0.25, chord="MAJOR_TRIAD"),
    Event(bar=1, offset_qn=0.25, lead=62, tenor=67, bari=55, bass=48, dur=0.25, chord="DOM7"),
    Event(bar=1, offset_qn=0.50, lead=64, tenor=67, bari=55, bass=48, dur=0.25, chord="MINOR_TRIAD"),
    Event(bar=1, offset_qn=0.75, lead=65, tenor=67, bari=55, bass=48, dur=0.25, chord="MAJOR_TRIAD"),
]
issues = check_harmonic_rhythm(events)
_assert(any(i.check == 'harmonic_rhythm' for i in issues), "expected harmonic_rhythm warning")

# ============================================================================
# Test 11: Errors penalize score (5 points each, floor 0)
# ============================================================================
print("Test 11: Errors penalize score")
events = [Event(bar=1, offset_qn=0.0, lead=80, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD")]
sc = validate(_make_header(), events)
_assert(sc.overall_score < 100.0, f"expected score < 100 when errors present, got {sc.overall_score}")
_assert(sc.passed is False, "expected passed=False when errors present")

# ============================================================================
# Test 12: All rests handled gracefully
# ============================================================================
print("Test 12: All rests handled gracefully")
events = [
    Event(bar=1, offset_qn=0.0, lead=None, tenor=None, bari=None, bass=None, dur=1.0, chord=None),
    Event(bar=1, offset_qn=1.0, lead=None, tenor=None, bari=None, bass=None, dur=1.0, chord=None),
]
sc = validate(_make_header(), events)
_assert(sc.error_count == 0, f"rests should produce no errors, got {sc.error_count}")
# Ensure no crossing or range errors from None pitches
range_issues = check_voice_ranges(events)
crossing_issues = check_voice_crossing(events)
_assert(len(range_issues) == 0, "rests should not trigger range errors")
_assert(len(crossing_issues) == 0, "rests should not trigger crossing errors")

# ============================================================================
# Test 13: CLI exists
# ============================================================================
print("Test 13: CLI exits with code 1 without args")
result = subprocess.run(
    [sys.executable, str(Path(__file__).parent / "validate_arrangement.py")],
    capture_output=True, text=True,
)
_assert(result.returncode == 1, f"expected exit code 1, got {result.returncode}")

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 60)
print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
