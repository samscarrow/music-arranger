# Detokenizing AI-Generated Barbershop Arrangements

This guide covers converting AI-generated token sequences back into playable MusicXML files.

## Quick Start

### 1. Generate Tokens

```bash
source venv/bin/activate
python tools/generate.py > generated_output.txt
```

This produces a token sequence like:

```
[key:C] [meter:4/4] [bar:1] [chord:MAJOR_TRIAD] [bass:48] [bari:52] [lead:60] [tenor:64] [dur:2.0] ...
```

### 2. Detokenize to MusicXML

```bash
python tools/detokenize.py generated_output.txt arrangement.xml
```

Output:

```
======================================================================
BARBERSHOP QUARTET DETOKENIZER
======================================================================

📂 Input:  generated_output.txt
📝 Output: arrangement.xml

📖 Parsing 39 lines from generated_output.txt

✅ Parsed 39 lines, 30 events

🔍 Validating score...
✅ All parts have duration 24.5 quarter notes (6.0 bars in 4/4)

💾 Exporting to MusicXML...
✅ Exported to arrangement.xml (34 KB)

======================================================================
✅ DETOKENIZATION COMPLETE
======================================================================
```

### 3. Play It Back

Open `arrangement.xml` in MuseScore or Finale:

```bash
# MuseScore
musescore arrangement.xml

# Or export to MP3
mscore arrangement.xml -o arrangement.mp3
```

## How It Works

### State Machine

The detokenizer uses a **vertical-slice state machine** to parse tokens:

1. **Initialize** — Create 4 parts (Tenor, Lead, Bari, Bass) with appropriate clefs
2. **Accumulate** — Read voice pitches into `active_slice` buffer:
   ```
   [bass:48] → active_slice['Bass'] = 48
   [bari:52] → active_slice['Bari'] = 52
   [lead:60] → active_slice['Lead'] = 60
   [tenor:64] → active_slice['Tenor'] = 64
   ```
3. **Commit** — When `[dur:X]` is encountered, write all voices to parts:
   ```python
   for voice_name in ['Tenor', 'Lead', 'Bari', 'Bass']:
       midi = active_slice.get(voice_name)
       if midi is None:
           note = Rest(quarterLength=dur)
       else:
           note = Note(midi, quarterLength=dur)
       parts[voice_name].append(note)
   ```

### Token Format

**Header:**
- `[key:X]` — Key signature (e.g., `[key:C]`)
- `[meter:N/D]` — Time signature (e.g., `[meter:4/4]`)

**Per-Event:**
- `[bar:N]` — Bar number (informational, ignored)
- `[chord:LABEL]` — Chord type (informational, ignored)
- `[bass:M]`, `[bari:M]`, `[lead:M]`, `[tenor:M]` — MIDI pitches or `rest`
- `[dur:F]` — Duration in quarter notes (triggers commit)

**Terminator:**
- `[song_end]` — End of arrangement

**Example:**

```
[key:C] [meter:4/4]
[bar:1] [chord:MAJOR_TRIAD] [bass:48] [bari:52] [lead:60] [tenor:64] [dur:2.0]
[bar:2] [chord:DOM7] [bass:50] [bari:53] [lead:62] [tenor:65] [dur:2.0]
[bar:3] [chord:MAJOR_TRIAD] [bass:48] [bari:52] [lead:60] [tenor:64] [dur:4.0]
[song_end]
```

### Voice Ranges

Detokenized arrangements use standard **SATB voice ranges**:

| Voice | Clef | Typical Range (MIDI) | Notes |
|-------|------|----------------------|-------|
| Tenor | Treble 8vb | 48-64 | Sings an octave lower than written |
| Lead | Treble 8vb | 60-72 | Treble 8vb (actual: C4-C5) |
| Bari | Bass | 48-60 | Fills between Lead and Bass |
| Bass | Bass | 36-48 | Lowest voice |

**Why Treble 8vb for Tenor/Lead?** Barbershop tradition: tenor and lead read treble clef but sing in their natural registers (octave lower).

### Rests

Missing or explicit `rest` tokens create silence:

```
[bass:48] [bari:rest] [lead:60] [tenor:rest] [dur:1.0]
```

Results in:
- Bass: Note(48)
- Bari: Rest(1.0)
- Lead: Note(60)
- Tenor: Rest(1.0)

## Usage

### Basic

```bash
python tools/detokenize.py <token_file.txt> [output.xml]
```

### With No Output Path (defaults to `arrangement.xml`)

```bash
python tools/detokenize.py generated_tokens.txt
# Creates arrangement.xml
```

### In a Pipeline

```bash
# Generate → Detokenize → Play
python tools/generate.py | tee raw_tokens.txt | python tools/detokenize.py /dev/stdin result.xml
```

## Output Validation

The detokenizer checks:

- ✅ All 4 parts created (Tenor, Lead, Bari, Bass)
- ✅ All parts have same total duration
- ✅ Key signature and time signature applied
- ✅ Valid MIDI pitch values (0-127)
- ✅ Notes within reasonable voice ranges

### Example Validation Output

```
🔍 Validating score...
✅ All parts have duration 24.5 quarter notes (6.0 bars in 4/4)

Note range per voice:
  Tenor: MIDI 62-67 ✓
  Lead:  MIDI 55-67 ✓
  Bari:  MIDI 56-62 ✓
  Bass:  MIDI 48-55 ✓
```

## Robustness

The detokenizer is **forgiving**:

- **Skips non-token lines** — Logs, status messages, or other text are ignored
- **Handles malformed tokens** — Missing colons, invalid MIDI values → converted to rests
- **Graceful defaults** — Missing voices become rests; invalid key/meter skipped with warning
- **Final slice commit** — Even if no closing `[dur:]` token, active slice is committed

### Example Error Handling

```
⚠️  1 warning during processing:
  • Invalid duration: "foo" (defaulted to 1.0)
  • Invalid MIDI pitch for Lead: "999" (marked as rest)
```

## Common Issues

### Problem: "No parts created"

**Cause:** Token file is empty or has no valid tokens.

**Solution:** Check that `generate.py` ran successfully and `generated_output.txt` contains tokens.

### Problem: Parts have different durations

**Cause:** A voice is missing from some events (becomes rest with wrong duration).

**Solution:** Verify all 4 voices (`bass`, `bari`, `lead`, `tenor`) appear in each bar.

### Problem: Invalid MIDI pitches in output

**Cause:** Generated tokens include MIDI values outside 0-127.

**Solution:** Add bounds checking in `generate.py` (values should be clipped to 36-88 for SATB).

### Problem: MusicXML won't open in notation software

**Cause:** Corrupted or incomplete XML.

**Solution:** Check `detokenize.py` console output for errors; re-run with debugging.

## Next Steps

### 1. Playback

Open in notation software and play back:

```bash
musescore arrangement.xml  # MuseScore
open -a Finale arrangement.xml  # Finale (Mac)
```

### 2. Audio Export

Convert to MP3:

```bash
mscore arrangement.xml -o arrangement.mp3
sox arrangement.mp3 -c 1 arrangement_mono.mp3  # Convert to mono
```

### 3. Edit and Refine

Load in MuseScore, fix any voice-leading issues, re-export.

### 4. Analysis

Extract chords and analyze harmonic progression:

```python
from music21 import converter
score = converter.parse('arrangement.xml')
chords = score.flatten().getElementsByClass('Chord')
for c in chords:
    print(c.pitches)
```

## Integration with Training Loop

Full AI arrangement pipeline:

```
↓
input_melody.xml + "arrange in barbershop style"
↓
[Claude API interprets natural language → structured constraints]
↓
[CP-SAT solver generates arrangement]
↓
output_arranged.xml
```

OR (Transformer-based):

```
↓
[Generate.py outputs token sequence]
↓
[Detokenize.py → MusicXML]
↓
arrangement.xml
```

## Implementation Details

### Clef Assignment

- **Tenor & Lead:** `music21.clef.Treble8vbClef()` (treble clef, 8vb = octave below)
- **Bari & Bass:** `music21.clef.BassClef()`

### Key Signature Parsing

```python
key = music21.key.Key('C')  # From [key:C]
```

Supports any standard key (C, G, D, A, E, B, F#, C#, etc.).

### Time Signature Parsing

```python
ts = music21.meter.TimeSignature('4/4')  # From [meter:4/4]
```

Supports any N/D combination (4/4, 3/4, 6/8, etc.).

### MIDI to Pitch Conversion

```python
note = music21.note.Note(midi_value)  # MIDI 48 → C3, 60 → C4, 72 → C5
```

## Performance

**Typical Generation + Detokenize Time:**

- Generate: ~10 seconds (500 tokens, CPU)
- Detokenize: <1 second
- Total: ~10 seconds

**File Sizes:**

- Token file: ~5-10 KB (text)
- MusicXML: ~20-40 KB (compressed)

## Troubleshooting Script

Test the detokenizer with a minimal token file:

```bash
cat > test_tokens.txt << 'EOF'
[key:C] [meter:4/4]
[bar:1] [chord:MAJOR_TRIAD] [bass:48] [bari:52] [lead:60] [tenor:64] [dur:2.0]
[bar:1] [chord:MAJOR_TRIAD] [bass:50] [bari:53] [lead:62] [tenor:65] [dur:2.0]
[song_end]
EOF

python tools/detokenize.py test_tokens.txt test_output.xml
python -c "from music21 import converter; s = converter.parse('test_output.xml'); print(f'✅ Valid MusicXML: {len(s.parts)} parts, {s.highestTime} quarter notes')"
```

Expected:
```
✅ Valid MusicXML: 4 parts, 4.0 quarter notes
```

## References

- **Token Format:** Defined in `tools/tokenizer.py`
- **Generator:** `tools/generate.py`
- **Music21 Documentation:** https://web.mit.edu/music21/
- **MusicXML Standard:** https://www.musicxml.com/
