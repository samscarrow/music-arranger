#!/usr/bin/env python3
"""
Canonical Event Schema for Barbershop Token Pipeline

Single source of truth for what an "event" is. All consumers (tokenizer,
detokenizer, harmonizer, reorder script) should use these types and functions.

An Event is one vertical time-slice: all four voices sounding simultaneously
for a given duration at a given time offset.

Token format (canonical, one event per line):
  [key:C] [meter:4/4]
  [bar:1] [offset:0.0] [bass:48] [bari:55] [lead:60] [tenor:64] [dur:1.0] [chord:MAJOR_TRIAD]
  [bar:1] [offset:1.0] [bass:48] [bari:55] [lead:62] [tenor:64] [dur:0.5] [chord:MAJOR_TRIAD]
  [song_end]
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field


@dataclass
class Header:
    key: str       # "C", "Bb", etc.
    meter: str     # "4/4", "12/8", etc.

    def __eq__(self, other):
        if not isinstance(other, Header):
            return NotImplemented
        return self.key == other.key and self.meter == other.meter


@dataclass
class Event:
    bar: int
    offset_qn: float       # cumulative quarter-note offset from song start
    lead: int | None        # MIDI pitch or None (rest)
    tenor: int | None
    bari: int | None
    bass: int | None
    dur: float
    chord: str | None       # "MAJOR_TRIAD", "DOM7", etc.

    def __eq__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return (
            self.bar == other.bar
            and _float_eq(self.offset_qn, other.offset_qn)
            and self.lead == other.lead
            and self.tenor == other.tenor
            and self.bari == other.bari
            and self.bass == other.bass
            and _float_eq(self.dur, other.dur)
            and self.chord == other.chord
        )


def _float_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def _fmt_float(f: float) -> str:
    """Format a float for token output, avoiding ugly repr issues."""
    rounded = round(f, 4)
    s = f"{rounded:g}"
    # Ensure there's always a decimal point for consistency
    if '.' not in s:
        s += '.0'
    return s


def quarter_notes_per_bar(meter: str) -> float:
    """
    Compute quarter-note duration of one bar from a meter string.

    Formula: numerator * (4.0 / denominator)

    Examples:
        "4/4" -> 4.0, "3/4" -> 3.0, "6/8" -> 3.0, "12/8" -> 6.0, "2/2" -> 4.0
    """
    parts = meter.split('/')
    if len(parts) != 2:
        raise ValueError(f"Invalid meter: {meter!r}")
    try:
        numerator = int(parts[0])
        denominator = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid meter: {meter!r}")
    if denominator == 0:
        raise ValueError(f"Invalid meter: {meter!r}")
    return numerator * (4.0 / denominator)


# Token regex: matches [type:value] patterns
_TOKEN_RE = re.compile(r'\[([a-z_]+):([^\]]+)\]')


def _parse_voice(value: str) -> int | None:
    """Parse a voice token value to MIDI pitch or None."""
    if value == 'rest':
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_event_tokens(tokens: list[tuple[str, str]], cumulative_offset: float, meter: str) -> Event | None:
    """
    Build an Event from a list of (type, value) token pairs.

    Uses cumulative_offset as default for offset_qn if [offset:] not present.
    Uses meter to derive bar number if [bar:] not present.
    """
    fields: dict = {}

    for token_type, token_value in tokens:
        if token_type == 'bar':
            fields['bar'] = int(token_value)
        elif token_type == 'offset':
            fields['offset_qn'] = float(token_value)
        elif token_type == 'lead':
            fields['lead'] = _parse_voice(token_value)
        elif token_type == 'tenor':
            fields['tenor'] = _parse_voice(token_value)
        elif token_type == 'bari':
            fields['bari'] = _parse_voice(token_value)
        elif token_type == 'bass':
            fields['bass'] = _parse_voice(token_value)
        elif token_type == 'dur':
            fields['dur'] = float(token_value)
        elif token_type == 'chord':
            fields['chord'] = token_value

    # Must have a duration to be a valid event — warn loudly on missing dur
    if 'dur' not in fields:
        token_summary = [(t, v) for t, v in tokens]
        warnings.warn(
            f"Dropping token group with no [dur:] — tokens: {token_summary}",
            stacklevel=2,
        )
        return None

    # Default offset from cumulative
    offset_qn = fields.get('offset_qn', cumulative_offset)

    # Default bar from offset
    qn_per_bar = quarter_notes_per_bar(meter)
    bar = fields.get('bar', int(offset_qn // qn_per_bar) + 1)

    return Event(
        bar=bar,
        offset_qn=offset_qn,
        lead=fields.get('lead'),
        tenor=fields.get('tenor'),
        bari=fields.get('bari'),
        bass=fields.get('bass'),
        dur=fields['dur'],
        chord=fields.get('chord'),
    )


def parse_tokens(text: str) -> tuple[Header, list[Event]]:
    """
    Parse token text into a Header and list of Events.

    Two-tier strategy:
    1. Primary: line-by-line — each non-empty line with event tokens becomes one Event.
    2. Fallback: if input is a single line (legacy harmonizer output), split on [bar:N]
       boundaries within the line.

    Header tokens ([key:], [meter:]) are extracted first regardless of position.
    """
    # Extract all tokens from entire text for header
    all_tokens = _TOKEN_RE.findall(text)

    # Extract header
    key = 'C'
    meter = '4/4'
    for token_type, token_value in all_tokens:
        if token_type == 'key':
            key = token_value
        elif token_type == 'meter':
            meter = token_value

    header = Header(key=key, meter=meter)

    # Decide strategy: multi-line or single-line
    # A line with multiple [dur:] tokens is really multiple events packed on one line
    lines = text.strip().split('\n')
    non_empty_lines = [l.strip() for l in lines if l.strip()]

    # Filter to lines that have event tokens (not just header or song_end)
    event_lines = []
    has_packed_line = False
    for line in non_empty_lines:
        tokens = _TOKEN_RE.findall(line)
        # Skip pure header lines (only key/meter)
        token_types = {t for t, v in tokens}
        if token_types <= {'key', 'meter'}:
            continue
        if token_types == {'song_end'} or (len(tokens) == 1 and tokens[0][0] == 'song_end'):
            continue
        # Count [dur:] tokens — multiple means packed events on one line
        dur_count = sum(1 for t, v in tokens if t == 'dur')
        if dur_count > 1:
            has_packed_line = True
            break
        # Must contain a dur token to be an event
        if dur_count == 1:
            event_lines.append(tokens)
        elif any(t in ('lead', 'tenor', 'bari', 'bass') for t, v in tokens):
            # Line has voice tokens but no dur — warn about dropped data
            warnings.warn(
                f"Dropping line with no [dur:] — tokens: {tokens}",
                stacklevel=2,
            )

    if not has_packed_line and event_lines:
        # Multi-line mode: one event per line
        return header, _parse_events_from_token_lists(event_lines, meter)

    # Single-line (or packed) mode — split by [dur:] tokens
    return header, _parse_single_line(all_tokens, meter)


def _parse_events_from_token_lists(
    token_lists: list[list[tuple[str, str]]],
    meter: str,
) -> list[Event]:
    """Parse a list of per-line token lists into Events."""
    events = []
    cumulative_offset = 0.0

    for tokens in token_lists:
        # Filter out header/meta tokens
        event_tokens = [
            (t, v) for t, v in tokens
            if t not in ('key', 'meter', 'song_end')
        ]
        if not event_tokens:
            continue

        event = _parse_event_tokens(event_tokens, cumulative_offset, meter)
        if event is not None:
            events.append(event)
            cumulative_offset = event.offset_qn + event.dur

    return events


def _parse_single_line(
    all_tokens: list[tuple[str, str]],
    meter: str,
) -> list[Event]:
    """
    Parse a single-line token stream into Events.

    Strategy: find each [dur:] token (exactly one per event), then gather
    surrounding tokens to form each event's group.

    For each pair of consecutive dur tokens at indices i and j, the event
    for dur[i] gets tokens from the previous boundary up to just before
    the boundary of dur[j]'s event. The boundary between events is found
    by looking for repeated token types (a voice token type that already
    appeared in the current group signals a new event's start).

    This handles both physical and melody-first token orderings.
    """
    # Filter out header tokens, find dur positions
    filtered: list[tuple[str, str]] = []
    for token_type, token_value in all_tokens:
        if token_type in ('key', 'meter', 'song_end'):
            continue
        filtered.append((token_type, token_value))

    if not filtered:
        return []

    # Find indices of all dur tokens
    dur_indices = [i for i, (t, v) in enumerate(filtered) if t == 'dur']
    if not dur_indices:
        return []

    # Split into groups: each group is centered around one [dur:] token.
    # The boundary between groups is where a voice/bar token type repeats
    # or a [bar:] appears after a dur.
    # Simple approach: walk forward from each dur to find where the next
    # event's tokens begin.

    # Build groups by walking through tokens and using [dur:] as commit points.
    # Key insight: tokens AFTER a [dur:] but BEFORE the next event-start token
    # still belong to the current event (e.g., [chord:X] [bass:48] after [dur:]).
    # A new event starts when we see a token type that would conflict with an
    # existing token in the current accumulator.

    events = []
    cumulative_offset = 0.0
    current_tokens: list[tuple[str, str]] = []
    current_types: set[str] = set()
    has_dur = False

    for token_type, token_value in filtered:
        # Check if this token would conflict (duplicate type) in current event
        is_duplicate = token_type in current_types and token_type != 'bar'
        # [bar:] after a dur always starts a new event
        is_new_bar = token_type == 'bar' and has_dur

        if (is_duplicate or is_new_bar) and has_dur:
            # Commit the current event
            event = _parse_event_tokens(current_tokens, cumulative_offset, meter)
            if event is not None:
                events.append(event)
                cumulative_offset = event.offset_qn + event.dur
            current_tokens = []
            current_types = set()
            has_dur = False

        current_tokens.append((token_type, token_value))
        current_types.add(token_type)
        if token_type == 'dur':
            has_dur = True

    # Flush remaining
    if current_tokens and has_dur:
        event = _parse_event_tokens(current_tokens, cumulative_offset, meter)
        if event is not None:
            events.append(event)

    return events


def _fmt_voice(name: str, pitch: int | None) -> str:
    """Format a voice token."""
    if pitch is None:
        return f"[{name}:rest]"
    return f"[{name}:{pitch}]"


def format_events(header: Header, events: list[Event]) -> str:
    """
    Serialize a Header + Events to canonical multi-line token format.

    Line 1: [key:X] [meter:N/D]
    One event per line:
      [bar:N] [offset:F] [bass:X] [bari:Y] [lead:Z] [tenor:W] [dur:D] [chord:LABEL]
    Last line: [song_end]
    """
    lines = []

    # Header
    lines.append(f"[key:{header.key}] [meter:{header.meter}]")

    # Events
    for event in events:
        parts = [
            f"[bar:{event.bar}]",
            f"[offset:{_fmt_float(event.offset_qn)}]",
            _fmt_voice('bass', event.bass),
            _fmt_voice('bari', event.bari),
            _fmt_voice('lead', event.lead),
            _fmt_voice('tenor', event.tenor),
            f"[dur:{_fmt_float(event.dur)}]",
        ]
        if event.chord is not None:
            parts.append(f"[chord:{event.chord}]")
        lines.append(' '.join(parts))

    # Song end
    lines.append('[song_end]')

    return '\n'.join(lines) + '\n'
