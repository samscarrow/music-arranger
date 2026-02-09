#!/usr/bin/env python3
"""
Round-trip tests for the canonical event schema.

Tests:
1. quarter_notes_per_bar — common meters
2. Round-trip synthetic — Header + events, format→parse→assert equal
3. Round-trip real training data — first song from training_sequences.txt
4. Parse legacy single-line — multiple [bar:N] groups on one line
5. Parse legacy harmonizer output — harmonized_output.txt event count
6. Missing voices — only lead+dur+chord → bass/bari/tenor are None
7. Explicit rests — [bass:rest] parses as None, round-trips to [bass:rest]
8. offset_qn accumulation — 3 events with known durations → correct offsets

Usage:
    python tools/test_event.py
"""

import os
import sys

# Add tools dir to path so we can import event
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from event import Header, Event, parse_tokens, format_events, quarter_notes_per_bar


def _assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}\n  expected: {expected!r}\n  actual:   {actual!r}")


def _assert_float_eq(actual, expected, msg="", tol=1e-6):
    if abs(actual - expected) > tol:
        raise AssertionError(f"{msg}\n  expected: {expected}\n  actual:   {actual}")


class AssertionError(Exception):
    pass


# ============================================================================
# Test 1: quarter_notes_per_bar
# ============================================================================

def test_quarter_notes_per_bar():
    print("Test 1: quarter_notes_per_bar")

    _assert_float_eq(quarter_notes_per_bar("4/4"), 4.0, "4/4")
    _assert_float_eq(quarter_notes_per_bar("3/4"), 3.0, "3/4")
    _assert_float_eq(quarter_notes_per_bar("6/8"), 3.0, "6/8")
    _assert_float_eq(quarter_notes_per_bar("12/8"), 6.0, "12/8")
    _assert_float_eq(quarter_notes_per_bar("2/2"), 4.0, "2/2")
    _assert_float_eq(quarter_notes_per_bar("2/4"), 2.0, "2/4")

    # Invalid meter should raise
    try:
        quarter_notes_per_bar("invalid")
        raise AssertionError("Expected ValueError for 'invalid'")
    except ValueError:
        pass

    print("  PASSED")


# ============================================================================
# Test 2: Round-trip synthetic
# ============================================================================

def test_round_trip_synthetic():
    print("Test 2: Round-trip synthetic")

    header = Header(key="G", meter="3/4")
    events = [
        Event(bar=1, offset_qn=0.0, lead=60, tenor=67, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
        Event(bar=1, offset_qn=1.0, lead=62, tenor=None, bari=55, bass=48, dur=0.5, chord="DOM7"),
        Event(bar=1, offset_qn=1.5, lead=None, tenor=None, bari=None, bass=None, dur=1.5, chord=None),
    ]

    text = format_events(header, events)
    header2, events2 = parse_tokens(text)

    _assert_eq(header2, header, "Header round-trip")
    _assert_eq(len(events2), len(events), "Event count round-trip")

    for i, (e1, e2) in enumerate(zip(events, events2)):
        _assert_eq(e2, e1, f"Event {i} round-trip")

    # Verify offset_qn is correct
    _assert_float_eq(events2[0].offset_qn, 0.0, "Event 0 offset")
    _assert_float_eq(events2[1].offset_qn, 1.0, "Event 1 offset")
    _assert_float_eq(events2[2].offset_qn, 1.5, "Event 2 offset")

    print("  PASSED")


# ============================================================================
# Test 3: Round-trip real training data
# ============================================================================

def test_round_trip_training_data():
    print("Test 3: Round-trip real training data")

    training_path = os.path.join(
        os.path.dirname(__file__), "barbershop_dataset", "training_sequences.txt"
    )
    if not os.path.exists(training_path):
        print("  SKIPPED (training_sequences.txt not found)")
        return

    # Read first song (up to first [song_end])
    with open(training_path, 'r') as f:
        lines = []
        for line in f:
            lines.append(line)
            if '[song_end]' in line:
                break

    first_song = ''.join(lines)

    # Parse
    header, events = parse_tokens(first_song)

    _assert_eq(header.key, "C", "Training data key")
    _assert_eq(header.meter, "4/4", "Training data meter")
    assert len(events) > 0, "Training data should have events"

    # Format and re-parse
    text = format_events(header, events)
    header2, events2 = parse_tokens(text)

    _assert_eq(header2, header, "Training data header round-trip")
    _assert_eq(len(events2), len(events), f"Training data event count: {len(events)}")

    for i, (e1, e2) in enumerate(zip(events, events2)):
        _assert_eq(e2, e1, f"Training data event {i} round-trip")

    print(f"  PASSED ({len(events)} events)")


# ============================================================================
# Test 4: Parse legacy single-line
# ============================================================================

def test_parse_legacy_single_line():
    print("Test 4: Parse legacy single-line")

    # Simulate harmonizer output: all on one line, melody-first format
    text = (
        "[key:C] [meter:4/4] "
        "[bar:1] [lead:60] [dur:1.0] [chord:MAJOR_TRIAD] [bass:48] [bari:55] [tenor:64] "
        "[lead:62] [dur:0.5] [chord:DOM7] [bass:50] [bari:57] [tenor:65] "
        "[bar:2] [lead:64] [dur:2.0] [chord:MINOR_TRIAD] [bass:52] [bari:55] [tenor:67] "
        "[song_end]"
    )

    header, events = parse_tokens(text)

    _assert_eq(header.key, "C", "Single-line key")
    _assert_eq(header.meter, "4/4", "Single-line meter")
    _assert_eq(len(events), 3, f"Single-line event count (got {len(events)})")

    # Event 0: bar 1
    _assert_eq(events[0].bar, 1, "Event 0 bar")
    _assert_eq(events[0].lead, 60, "Event 0 lead")
    _assert_eq(events[0].bass, 48, "Event 0 bass")
    _assert_eq(events[0].bari, 55, "Event 0 bari")
    _assert_eq(events[0].tenor, 64, "Event 0 tenor")
    _assert_eq(events[0].chord, "MAJOR_TRIAD", "Event 0 chord")
    _assert_float_eq(events[0].dur, 1.0, "Event 0 dur")
    _assert_float_eq(events[0].offset_qn, 0.0, "Event 0 offset")

    # Event 1: still bar 1 (no new [bar:] token)
    _assert_eq(events[1].bar, 1, "Event 1 bar")
    _assert_eq(events[1].lead, 62, "Event 1 lead")
    _assert_float_eq(events[1].offset_qn, 1.0, "Event 1 offset")

    # Event 2: bar 2
    _assert_eq(events[2].bar, 2, "Event 2 bar")
    _assert_eq(events[2].lead, 64, "Event 2 lead")
    _assert_float_eq(events[2].offset_qn, 1.5, "Event 2 offset")

    print("  PASSED")


# ============================================================================
# Test 5: Parse legacy harmonizer output
# ============================================================================

def test_parse_harmonizer_output():
    print("Test 5: Parse legacy harmonizer output")

    harmonizer_path = os.path.join(
        os.path.dirname(__file__), "..", "harmonized_output.txt"
    )
    if not os.path.exists(harmonizer_path):
        print("  SKIPPED (harmonized_output.txt not found)")
        return

    with open(harmonizer_path, 'r') as f:
        text = f.read()

    header, events = parse_tokens(text)

    _assert_eq(header.key, "C", "Harmonizer key")
    _assert_eq(header.meter, "12/8", "Harmonizer meter")

    # The harmonizer output has ~400 melody notes, each producing an event
    # with [lead:X] [dur:X] [chord:X] [bass:X] [bari:X] and sometimes [tenor:X]
    print(f"  Parsed {len(events)} events from harmonizer output")
    assert len(events) > 100, f"Expected >100 events, got {len(events)}"

    # Check first event
    _assert_eq(events[0].lead, 55, "First event lead")
    _assert_float_eq(events[0].dur, 0.5, "First event dur")

    print(f"  PASSED ({len(events)} events)")


# ============================================================================
# Test 6: Missing voices
# ============================================================================

def test_missing_voices():
    print("Test 6: Missing voices")

    text = (
        "[key:C] [meter:4/4]\n"
        "[bar:1] [lead:60] [dur:1.0] [chord:MAJOR_TRIAD]\n"
        "[song_end]\n"
    )

    header, events = parse_tokens(text)
    _assert_eq(len(events), 1, "Missing voices event count")

    e = events[0]
    _assert_eq(e.lead, 60, "Lead present")
    _assert_eq(e.bass, None, "Bass missing → None")
    _assert_eq(e.bari, None, "Bari missing → None")
    _assert_eq(e.tenor, None, "Tenor missing → None")

    print("  PASSED")


# ============================================================================
# Test 7: Explicit rests
# ============================================================================

def test_explicit_rests():
    print("Test 7: Explicit rests")

    header = Header(key="C", meter="4/4")
    events = [
        Event(bar=1, offset_qn=0.0, lead=60, tenor=None, bari=None, bass=None, dur=1.0, chord="MAJOR_TRIAD"),
    ]

    text = format_events(header, events)

    # Check that None voices are formatted as [voice:rest]
    assert "[bass:rest]" in text, f"Expected [bass:rest] in output:\n{text}"
    assert "[bari:rest]" in text, f"Expected [bari:rest] in output:\n{text}"
    assert "[tenor:rest]" in text, f"Expected [tenor:rest] in output:\n{text}"

    # Round-trip
    header2, events2 = parse_tokens(text)
    _assert_eq(events2[0], events[0], "Explicit rest round-trip")

    print("  PASSED")


# ============================================================================
# Test 8: offset_qn accumulation
# ============================================================================

def test_offset_accumulation():
    print("Test 8: offset_qn accumulation")

    # Build events without [offset:] tokens — offsets should be computed from durations
    text = (
        "[key:C] [meter:4/4]\n"
        "[bar:1] [lead:60] [dur:1.0] [chord:MAJOR_TRIAD]\n"
        "[bar:1] [lead:62] [dur:0.5] [chord:DOM7]\n"
        "[bar:1] [lead:64] [dur:2.0] [chord:MINOR_TRIAD]\n"
        "[song_end]\n"
    )

    header, events = parse_tokens(text)
    _assert_eq(len(events), 3, "Accumulation event count")

    _assert_float_eq(events[0].offset_qn, 0.0, "Event 0 offset")
    _assert_float_eq(events[1].offset_qn, 1.0, "Event 1 offset (0.0 + 1.0)")
    _assert_float_eq(events[2].offset_qn, 1.5, "Event 2 offset (1.0 + 0.5)")

    print("  PASSED")


# ============================================================================
# Test 9: Format includes [song_end]
# ============================================================================

def test_format_song_end():
    print("Test 9: format_events includes [song_end]")

    header = Header(key="C", meter="4/4")
    events = [
        Event(bar=1, offset_qn=0.0, lead=60, tenor=64, bari=55, bass=48, dur=1.0, chord="MAJOR_TRIAD"),
    ]

    text = format_events(header, events)
    assert text.strip().endswith("[song_end]"), f"Expected [song_end] at end:\n{text}"

    print("  PASSED")


# ============================================================================
# Test 10: No-chord event round-trips
# ============================================================================

def test_no_chord_round_trip():
    print("Test 10: No-chord event round-trip")

    header = Header(key="C", meter="4/4")
    events = [
        Event(bar=1, offset_qn=0.0, lead=60, tenor=64, bari=55, bass=48, dur=1.0, chord=None),
    ]

    text = format_events(header, events)
    assert "[chord:" not in text.split('\n')[1], "No chord should not produce [chord:] token"

    header2, events2 = parse_tokens(text)
    _assert_eq(events2[0].chord, None, "No-chord round-trip")

    print("  PASSED")


# ============================================================================
# Test 11: Splitting determinism — event count == raw [dur:] count
# ============================================================================

def test_splitting_determinism():
    print("Test 11: Splitting determinism — event count == raw [dur:] count")

    import re

    # Construct a single-line input with exactly 7 events, mixed formats
    text = (
        "[key:C] [meter:4/4] "
        "[bar:1] [lead:60] [dur:1.0] [chord:MAJOR_TRIAD] [bass:48] [bari:55] [tenor:64] "
        "[lead:62] [dur:0.5] [chord:DOM7] [bass:50] [bari:57] [tenor:65] "
        "[lead:64] [dur:0.5] [chord:MINOR_TRIAD] [bass:52] [bari:55] [tenor:67] "
        "[lead:65] [dur:1.0] [chord:MAJOR_TRIAD] [bass:53] [bari:57] "
        "[bar:2] [lead:67] [dur:2.0] [chord:DOM7] [bass:55] [bari:59] [tenor:71] "
        "[lead:69] [dur:0.5] [bass:57] [bari:60] "
        "[lead:60] [dur:1.5] [chord:MAJOR_TRIAD] [bass:48] [bari:55] [tenor:64] "
        "[song_end]"
    )

    # Count raw [dur:] tokens
    raw_dur_count = len(re.findall(r'\[dur:', text))

    header, events = parse_tokens(text)

    _assert_eq(len(events), raw_dur_count,
               f"Event count ({len(events)}) must equal raw [dur:] count ({raw_dur_count})")

    # Verify each event has the correct lead
    expected_leads = [60, 62, 64, 65, 67, 69, 60]
    for i, (event, expected_lead) in enumerate(zip(events, expected_leads)):
        _assert_eq(event.lead, expected_lead, f"Event {i} lead")

    print(f"  PASSED ({len(events)} events == {raw_dur_count} [dur:] tokens)")


# ============================================================================
# Test 12: Bar derivation in compound meter (12/8)
# ============================================================================

def test_bar_derivation_compound_meter():
    print("Test 12: Bar derivation in compound meter (12/8)")

    # In 12/8, qn_per_bar = 6.0
    # Events at offsets 0.0, 3.0, 5.5, 6.0, 11.0, 12.0
    # Expected bars:     1,    1,    1,    2,     2,     3
    header = Header(key="C", meter="12/8")
    events = [
        Event(bar=1, offset_qn=0.0, lead=60, tenor=None, bari=None, bass=None, dur=3.0, chord=None),
        Event(bar=1, offset_qn=3.0, lead=62, tenor=None, bari=None, bass=None, dur=2.5, chord=None),
        Event(bar=1, offset_qn=5.5, lead=64, tenor=None, bari=None, bass=None, dur=0.5, chord=None),
        Event(bar=2, offset_qn=6.0, lead=65, tenor=None, bari=None, bass=None, dur=5.0, chord=None),
        Event(bar=2, offset_qn=11.0, lead=67, tenor=None, bari=None, bass=None, dur=1.0, chord=None),
        Event(bar=3, offset_qn=12.0, lead=69, tenor=None, bari=None, bass=None, dur=2.0, chord=None),
    ]

    # Round-trip: format → parse, check bars survive
    text = format_events(header, events)
    header2, events2 = parse_tokens(text)

    _assert_eq(len(events2), len(events), "12/8 event count")
    for i, (e1, e2) in enumerate(zip(events, events2)):
        _assert_eq(e2.bar, e1.bar, f"12/8 event {i} bar")
        _assert_float_eq(e2.offset_qn, e1.offset_qn, f"12/8 event {i} offset_qn")

    # Also verify: parse without [bar:] tokens derives correct bars from offset
    text_no_bar = (
        "[key:C] [meter:12/8]\n"
        "[offset:0.0] [lead:60] [dur:3.0]\n"
        "[offset:3.0] [lead:62] [dur:2.5]\n"
        "[offset:5.5] [lead:64] [dur:0.5]\n"
        "[offset:6.0] [lead:65] [dur:5.0]\n"
        "[offset:11.0] [lead:67] [dur:1.0]\n"
        "[offset:12.0] [lead:69] [dur:2.0]\n"
        "[song_end]\n"
    )
    _, events_derived = parse_tokens(text_no_bar)
    expected_bars = [1, 1, 1, 2, 2, 3]
    for i, (e, expected_bar) in enumerate(zip(events_derived, expected_bars)):
        _assert_eq(e.bar, expected_bar, f"Derived bar for offset {e.offset_qn} in 12/8")

    print("  PASSED")


# ============================================================================
# Test 13: Missing dur warns loudly
# ============================================================================

def test_missing_dur_warns():
    print("Test 13: Missing dur warns loudly")

    import warnings

    # A line with voice tokens but no [dur:] — should warn and produce no event
    text = (
        "[key:C] [meter:4/4]\n"
        "[bar:1] [lead:60] [bass:48] [chord:MAJOR_TRIAD]\n"
        "[bar:1] [lead:62] [dur:1.0] [chord:DOM7]\n"
        "[song_end]\n"
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        header, events = parse_tokens(text)

    # Should produce exactly 1 event (the one with dur), not 2
    _assert_eq(len(events), 1, f"Missing dur should drop event (got {len(events)})")
    _assert_eq(events[0].lead, 62, "Surviving event should be the one with dur")

    # Should have emitted a warning about the dropped group
    dur_warnings = [x for x in w if "no [dur:]" in str(x.message).lower()]
    assert len(dur_warnings) >= 1, (
        f"Expected warning about missing [dur:], got {len(dur_warnings)} warnings: "
        f"{[str(x.message) for x in w]}"
    )

    print(f"  PASSED (1 event, {len(dur_warnings)} warning(s))")


# ============================================================================
# Test 14: Harmonizer event count == melody length (structural)
# ============================================================================

def test_harmonizer_event_count_matches_melody():
    print("Test 14: Harmonizer output event count == TEST_MELODY length")

    harmonizer_path = os.path.join(
        os.path.dirname(__file__), "..", "harmonized_output.txt"
    )
    if not os.path.exists(harmonizer_path):
        print("  SKIPPED (harmonized_output.txt not found)")
        return

    import re

    with open(harmonizer_path, 'r') as f:
        text = f.read()

    # Count raw tokens to verify the splitting isn't adding/losing events
    raw_dur_count = len(re.findall(r'\[dur:', text))
    raw_lead_count = len(re.findall(r'\[lead:', text))

    header, events = parse_tokens(text)

    _assert_eq(len(events), raw_dur_count,
               f"Parsed events ({len(events)}) must equal raw [dur:] count ({raw_dur_count})")
    _assert_eq(raw_dur_count, raw_lead_count,
               f"Raw [dur:] count ({raw_dur_count}) must equal raw [lead:] count ({raw_lead_count})")

    # The known ground truth: TEST_MELODY has 400 notes, harmonizer produced 400 events
    _assert_eq(len(events), 400,
               f"Event count should match TEST_MELODY length (400)")

    # Extra sanity: every event must have a lead pitch (melody was force-fed)
    events_with_lead = sum(1 for e in events if e.lead is not None)
    _assert_eq(events_with_lead, 400,
               f"All 400 events should have a lead pitch, got {events_with_lead}")

    # Every event must have dur > 0
    zero_dur = [i for i, e in enumerate(events) if e.dur <= 0]
    _assert_eq(len(zero_dur), 0, f"Events with dur <= 0 at indices: {zero_dur[:10]}")

    print(f"  PASSED ({len(events)} events, {raw_dur_count} [dur:], {raw_lead_count} [lead:])")


# ============================================================================
# RUNNER
# ============================================================================

def main():
    print("=" * 70)
    print("EVENT SCHEMA ROUND-TRIP TESTS")
    print("=" * 70)
    print()

    tests = [
        test_quarter_notes_per_bar,
        test_round_trip_synthetic,
        test_round_trip_training_data,
        test_parse_legacy_single_line,
        test_parse_harmonizer_output,
        test_missing_voices,
        test_explicit_rests,
        test_offset_accumulation,
        test_format_song_end,
        test_no_chord_round_trip,
        test_splitting_determinism,
        test_bar_derivation_compound_meter,
        test_missing_dur_warns,
        test_harmonizer_event_count_matches_melody,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
