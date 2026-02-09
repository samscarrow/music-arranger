# Pure Arranger Mode - Quick Start Guide

## What Changed?

The Transformer model architecture has been transformed from:
- **Old (Songwriter)**: Given chord labels, harmonize a melody
- **New (Pure Arranger)**: Given ONLY a melody, predict chords AND harmonize

This matches what a real arranger does: look at a song, decide what chords fit, and write out the harmony.

## The Recipe

### Step 1: Delete Old Model
```bash
rm tools/barbershop_dataset/arranger_model.pt
```

Old model learned songwriter mode (chord input → harmony output). New mode is different, so retrain needed.

### Step 2: Retrain (5-10 minutes on GPU)
```bash
python tools/train.py
```

Expected output:
```
🎹 Barbershop Quartet Arranger - Training
══════════════════════════════════════════════════════════════════════════
Device: cuda
Batch Size: 32
Block Size: 256
Model: 6-layer Transformer (384 dims, 6 heads)
══════════════════════════════════════════════════════════════════════════

Loading tools/barbershop_dataset/training_sequences.txt...
Optimizing token order for Melody->Harmony generation...
Vocabulary Size: 250
Total Tokens: 106146

Model Parameters: 10,934,018
══════════════════════════════════════════════════════════════════════════

🚀 Training for 5000 iterations...

Step    0 / 5000: loss = 5.6932
Step  100 / 5000: loss = 2.1543
Step  200 / 5000: loss = 1.8765
...
Step 5000 / 5000: loss = 1.0234

✅ Training Complete!
Model saved to: tools/barbershop_dataset/arranger_model.pt
```

**Key indicator**: Loss should drop from ~5.7 to ~1.0-2.0. If not, training failed.

### Step 3: Harmonize!
```bash
python tools/harmonize.py
```

Expected output:
```
══════════════════════════════════════════════════════════════════════════
🎹 BARBERSHOP MELODY HARMONIZER
══════════════════════════════════════════════════════════════════════════

✅ Model loaded from tools/barbershop_dataset/arranger_model.pt
   Vocab size: 250
   Device: cuda

🎹 Harmonizing 17-note melody...

  Note  1/17: MIDI 60, dur 0.75
      Predicted chord: [chord:MAJOR_TRIAD]
      Predicted bass:  [bass:48]
      Predicted bari:  [bari:52]
      Predicted tenor: [tenor:64]

  Note  2/17: MIDI 60, dur 0.25
      Predicted chord: [chord:MAJOR_TRIAD]
      Predicted bass:  [bass:48]
      Predicted bari:  [bari:55]
      Predicted tenor: [tenor:67]

  ...

══════════════════════════════════════════════════════════════════════════
✅ Harmonization Complete
══════════════════════════════════════════════════════════════════════════

Output tokens:
[key:C] [meter:4/4] [bar:1] [lead:60] [dur:0.75] [chord:MAJOR_TRIAD] [bass:48] [bari:52] [tenor:64] ...

💾 Saved to harmonized_output.txt

Statistics:
  Total tokens:       145
  Bars:               4
  Chord predictions:  17

  Chord Distribution:
    [chord:MAJOR_TRIAD]          : 12
    [chord:DOM7]                 : 3
    [chord:MINOR_TRIAD]          : 2

Next steps:
  1. Detokenize to MusicXML:
     python tools/detokenize.py harmonized_output.txt final_arrangement.xml

  2. Play in MuseScore:
     musescore final_arrangement.xml
```

### Step 4: Convert to MusicXML
```bash
python tools/detokenize.py harmonized_output.txt final_arrangement.xml
```

Output:
```
📝 Barbershop Detokenizer
══════════════════════════════════════════════════════════════════════════

Reading harmonized_output.txt...
✅ Found 4 voices per bar, all durations match
✅ Valid structure, creating MusicXML...

Voice Ranges:
  Lead:   MIDI 60-72 (C4-C5)
  Tenor:  MIDI 64-79 (E4-G5)
  Bari:   MIDI 52-67 (E3-G4)
  Bass:   MIDI 36-55 (C2-G3)

💾 Saved to final_arrangement.xml
```

### Step 5: Play It!
```bash
musescore final_arrangement.xml
```

(Or open in Finale, MuseScore, or any notation software)

---

## How It Works

### Old (Songwriter) Token Order
```
[bar:1] [chord:MAJOR_TRIAD] [lead:60] [bass:48] [bari:52] [tenor:64] [dur:1.0]
         ← INPUT              ← INPUT   ← OUTPUTS (predicted) ←
```
Model sees: "I'm told it's MAJOR_TRIAD and the melody is C (60). What bass/bari/tenor?"

### New (Pure Arranger) Token Order
```
[bar:1] [lead:60] [dur:1.0] [chord:MAJOR_TRIAD] [bass:48] [bari:52] [tenor:64]
         ← INPUTS ←           ← OUTPUTS (all predicted) ←
```
Model sees: "The melody is C (60), duration is 1 beat. What chord? What harmony?"

---

## What Gets Locked

**Melody is ALWAYS locked** (you can't change it):
- We **force-feed** `[lead:60]` and `[dur:1.0]` into the model
- Model can't change them
- The output will have exactly the same melody as input

**Chords and harmony are predicted**:
- `[chord:X]` — AI chooses the chord
- `[bass:M]`, `[bari:M]`, `[tenor:M]` — AI chooses the voices
- Learned from training data (real barbershop arrangements)

---

## Customizing the Melody

The test melody is hardcoded in `tools/harmonize.py` (lines 128-155):

```python
TEST_MELODY = [
    (60, 0.75),   # C, dotted eighth
    (60, 0.25),   # C, sixteenth
    (62, 1.0),    # D, quarter
    (60, 1.0),    # C, quarter
    (65, 1.0),    # F, quarter
    (64, 2.0),    # E, half
    # ... Happy Birthday continues ...
]
```

**To harmonize a different melody:**
1. Edit `TEST_MELODY` list (MIDI pitch, duration in beats)
2. Run `python tools/harmonize.py` again
3. It generates the harmonization

**MIDI pitch reference:**
- 60 = Middle C (C4)
- 48 = C3 (octave lower)
- 72 = C5 (octave higher)
- Each semitone = ±1 in MIDI

**Duration format:**
- 0.25 = sixteenth note
- 0.5 = eighth note
- 0.75 = dotted eighth
- 1.0 = quarter note
- 2.0 = half note
- 4.0 = whole note

---

## Troubleshooting

### Training loss doesn't drop
- **Cause**: Token reordering failed
- **Fix**: Check train.py lines 151-159, verify order is: bar(0), lead(1), dur(2), chord(3), bass(4), bari(5), tenor(6)

### Harmonizer crashes on unknown tokens
- **Cause**: Melody notes or durations not in training data vocab
- **Fix**: Use common MIDI pitches (48-84) and durations (0.25, 0.5, 0.75, 1.0, 2.0, 4.0)

### Detokenizer produces invalid XML
- **Cause**: Harmonizer didn't generate all 4 voices per bar
- **Fix**: Check harmonized_output.txt has exactly 4 tokens (chord, bass, bari, tenor) per melody note

### Melody doesn't match input
- **Cause**: Force-feed mechanism failed
- **Fix**: Check line 251-258 in harmonize.py, verify `[lead:]` and `[dur:]` tokens added to sequence

---

## What's Different from Before?

| Aspect | Before | After |
|--------|--------|-------|
| **Model training** | Chord is INPUT | Chord is OUTPUT |
| **Use case** | "Harmonize with chord constraints" | "Harmonize from melody alone" |
| **User provides** | Melody + chord progression | Melody only |
| **AI decides** | Harmony only | Chord + harmony |
| **Token order** | `[bar] [chord] [lead] [bass] [bari] [tenor] [dur]` | `[bar] [lead] [dur] [chord] [bass] [bari] [tenor]` |
| **Real-world use** | Constrained arrangement | True arranging (AI chooses chords) |

---

## Key Files

- **`tools/train.py`** — Training script (modified)
  - Lines 11: New training strategy doc
  - Lines 151-159: New token order_priority

- **`tools/harmonize.py`** — NEW harmonizer script
  - Loads trained model
  - Takes melody input
  - Generates chords + harmony
  - Outputs token sequence

- **`tools/detokenize.py`** — Unchanged
  - Converts token sequence to MusicXML
  - Works with any token order

- **`tools/generate.py`** — Unchanged
  - Free-form generation (no melody constraint)
  - Still works with retrained model

---

## Next Steps

1. ✅ Train the new model (`python tools/train.py`)
2. ✅ Harmonize "Happy Birthday" (`python tools/harmonize.py`)
3. ✅ Convert to MusicXML (`python tools/detokenize.py harmonized_output.txt out.xml`)
4. ✅ Play in MuseScore (`musescore out.xml`)

Then customize:
- Edit TEST_MELODY in harmonize.py
- Harmonize different melodies
- Compare with real barbershop versions

---

## Expected Behavior

✅ **Training**: Loss drops from ~5.7 to ~1.0-2.0 in 5000 iterations
✅ **Harmonization**: Chords are musically sensible (I, IV, V, etc. in C major)
✅ **Voice leading**: Minimal jumps, singable ranges
✅ **Melody**: Preserved exactly (matches input)
✅ **Output**: Valid MusicXML playable in MuseScore

---

## Questions?

- **Why retrain?** Token positions changed; old model position embeddings are invalid
- **Why not just update embeddings?** Easier to retrain; model has learned the data well
- **Can I change the key?** Currently hardcoded to C major; future enhancement
- **Can I change the meter?** Currently hardcoded to 4/4; future enhancement
- **Will old code still work?** `generate.py` will work with new model; `music_arranger.py` unaffected
