# Implementation Summary: Pure Arranger Model

## Overview

Successfully implemented the "Pure Arranger" architecture transformation. The model now:
- **Input**: Melody note + duration
- **Output**: Chord label + 3 harmony voices
- **Purpose**: Given a melody (like "Happy Birthday"), the AI figures out the chords AND generates full SATB harmony

This transforms the model from "songwriter constrained by input chords" to "true arranger who chooses the chords."

## Files Modified

### 1. `tools/train.py` (Token Reordering)

**Changed lines 81-159:** Token ordering in `BarbershopDataset` class

**Old token order (songwriter mode):**
```
[bar:N] [chord:X] [lead:M] [bass:M] [bari:M] [tenor:M] [dur:F]
         ^^^^^^^^^ INPUT    ^^^^^^^^^ INPUT  ^^^^^^^^^^^^^^^^^^^^ OUTPUT
```

**New token order (pure arranger mode):**
```
[bar:N] [lead:M] [dur:F] [chord:X] [bass:M] [bari:M] [tenor:M]
         ^^^^^^^^^^^^^^^^^ INPUTS    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ OUTPUTS
```

**Changes made:**
- Lines 11-17: Updated docstring explaining new strategy
- Lines 48-53: Updated BarbershopDataset class docstring
- Lines 81-96: Updated preprocess_and_reorder() docstring with new format
- Lines 133-159: Updated _reorder_event() with new order_priority dictionary

**Why it works:**
- The model's attention mechanism now sees melody+duration BEFORE making chord predictions
- This matches human arranger intuition: "Given this melody note, what chord works best?"
- Position embeddings adapt to the new token positions during retraining

### 2. `tools/harmonize.py` (New Harmonizer Script)

**Created complete new file (~350 lines)**

**Key components:**

1. **Model Architecture** (lines 17-114): Copy of Transformer from `generate.py`
   - AttentionHead, MultiHeadAttention, FeedForward, TransformerBlock, BarbershopTransformer
   - All classes match train.py exactly for compatibility

2. **Configuration** (lines 118-168):
   - MODEL_PATH, DEVICE, TEMPERATURE constants
   - TEST_MELODY: Hardcoded "Happy Birthday" snippet (17 notes)
   - Format: (MIDI_pitch, duration_in_beats)

3. **load_model()** (lines 171-192):
   - Loads checkpoint from arranger_model.pt
   - Sets global N_EMBD, N_HEAD, N_LAYER from config
   - Returns (model, stoi, itos) tuple

4. **harmonize_melody()** (lines 195-273):
   - Iterates through melody notes
   - Force-feeds constraint tokens: `[lead:pitch]`, `[dur:duration]`
   - Generates up to 4 harmony tokens: chord, bass, bari, tenor
   - **Stopping logic**: Stop if model predicts `[bar:]` or `[song_end]` (prevents runaway)
   - Tracks bar numbers and beat positions
   - Pretty-prints generated chords

5. **main()** (lines 276-345):
   - Loads model and harmonizes TEST_MELODY
   - Prints tokens to stdout AND saves to harmonized_output.txt
   - Shows statistics (chord distribution, bar count)
   - Displays next steps (detokenize → MusicXML → MuseScore)

**Design principles:**
- ✅ Hardcoded melody (prototyping, no file I/O complexity)
- ✅ Hardcoded key/meter (C major, 4/4)
- ✅ Simple CLI: `python tools/harmonize.py` (no arguments)
- ✅ Both stdout and file output for debugging
- ✅ Clear separation of constraints (force-fed) vs. predictions (generated)

## How to Use

### 1. Delete the old model
```bash
rm tools/barbershop_dataset/arranger_model.pt
```
(The old model learned songwriter mode; we need to retrain for arranger mode)

### 2. Retrain with new token ordering
```bash
python tools/train.py
```

Expected:
- Loss starts ~6.0 (random)
- Drops to ~1.0-2.0 (learned chord + harmony patterns)
- Training time: ~10 min on GPU, ~1 hour on CPU

### 3. Harmonize a melody
```bash
python tools/harmonize.py
```

Output:
- Prints tokens to screen
- Saves to `harmonized_output.txt`
- Shows chord distribution

### 4. Convert to MusicXML
```bash
python tools/detokenize.py harmonized_output.txt final_arrangement.xml
```

### 5. Play in MuseScore
```bash
musescore final_arrangement.xml
```

## What's Happening Under the Hood

### Training Phase
1. Training sequences reordered: `[bar] [chord] [lead]...` → `[bar] [lead] [dur] [chord] [bass]...`
2. Model trains to predict next token based on context
3. At inference, melody tokens are fixed; chord+harmony are auto-generated

### Generation Phase
1. **Constraint tokens** (forced): `[lead:60]`, `[dur:1.0]`
   - These are NOT generated; we force them into the sequence
   - They provide context about the melody for the model
2. **Prediction tokens** (generated): `[chord:MAJOR_TRIAD]`, `[bass:48]`, `[bari:52]`, `[tenor:64]`
   - Model learns: "When I see melody C (60), duration 1.0, what chord/harmony?"
3. **Stop condition**: After 4 predictions OR when model tries `[bar:]`
   - Prevents infinite generation
   - Ensures melody alignment

## Verification Plan

### Test 1: Token reordering correctness
```python
from tools.train import BarbershopDataset
ds = BarbershopDataset('tools/barbershop_dataset/training_sequences.txt', 256)
print(' '.join(ds.tokens[:50]))
# Should see: [key:C] [meter:4/4] [bar:1] [lead:XX] [dur:X.X] [chord:XXX] [bass:XX] ...
```

### Test 2: Training convergence
```bash
python tools/train.py
# Watch loss: should drop from ~6.0 to ~1.0-2.0
# If loss stays high (>3.0), token reordering failed
```

### Test 3: Harmonizer output
```bash
python tools/harmonize.py
# Should show:
#   Note 1/17: MIDI 60, dur 0.75
#   Predicted chord: [chord:MAJOR_TRIAD]
#   Predicted bass: [bass:48]
#   ...
```

### Test 4: Full pipeline
```bash
python tools/harmonize.py > tokens.txt
python tools/detokenize.py tokens.txt out.xml
musescore out.xml
# Should be recognizable "Happy Birthday" with valid harmony
```

## Key Architectural Decisions

### Why Force-Feed Melody?
- Melody is REQUIRED to stay the same (user's input)
- Forcing tokens locks melody to exact pitch and duration
- Model can't drift or modify the melody

### Why Stop After 4 Tokens?
- 1 chord + 3 harmony voices = complete event
- Stopping prevents model from generating extra tokens
- If model tries `[bar:]`, it means "next event"; we comply and advance

### Why Not Use Existing generate.py?
- `generate.py` generates from scratch (free form)
- We need constraint enforcement (melody locked)
- Harmonizer needs special stopping logic (after 4 voices)
- Different use case = different script

### Token Format Compatibility
- Detokenizer (`tools/detokenize.py`) is format-agnostic
- It just reconstructs MusicXML from `[bar:] [chord:] [bass:] [bari:] [lead:] [tenor:]` tokens
- Works with ANY token ordering (old or new)
- No changes to detokenizer needed!

## Performance Expectations

### Training
- **Loss curve**: ~5.70 → ~1.0-2.0 (slightly higher than before because chord is now an output)
- **Convergence**: Should be faster than 5000 iterations (good training data, simpler task)
- **Vocabulary**: Unchanged (~250 tokens)

### Harmonization Quality
- **Chord choice**: Should prefer common progressions (I, IV, V, etc.) in C major
- **Voice leading**: Minimal movement (learned from training data)
- **Parallel fifths**: May still occur (model saw them in barbershop data)
- **Range**: Voices should stay within barbershop ranges

### Musical Output
- Melody is 100% preserved (forced)
- Chords should sound "barbershop-like"
- Harmony should be singable (close harmony, ringing voicing)

## Future Enhancements

1. **Melody file input**: `python harmonize.py --melody song.xml --key Bb`
2. **Key detection**: Analyze melody, infer key automatically
3. **Chord hints**: Allow user to suggest some chords, model fills gaps
4. **Style transfer**: Fine-tune on different barbershop eras
5. **Interactive UI**: Web interface for real-time harmonization

## Summary of Changes

| File | Change | Lines | Impact |
|------|--------|-------|--------|
| `tools/train.py` | Token order: `[chord] [lead] -> ...` to `[lead] [dur] [chord] -> ...` | 11-159 | **CRITICAL**: Retraining required; old model incompatible |
| `tools/harmonize.py` | NEW: Melody harmonizer script | All | **NEW**: Enables pure arranger mode |
| Memory notes | Updated documentation | — | Reference for future work |

## What's NOT Changed

- ✅ Model architecture (still 6-layer Transformer)
- ✅ Training parameters (batch size, learning rate, etc.)
- ✅ Detokenizer (format-agnostic)
- ✅ generate.py (free-form generation still works)
- ✅ Tokenizer (training_sequences.txt format unchanged)

## Backward Compatibility

⚠️ **WARNING**: Old model (`arranger_model.pt`) is incompatible with new token ordering.

**Before running harmonize.py**, must:
1. Delete old model: `rm tools/barbershop_dataset/arranger_model.pt`
2. Retrain: `python tools/train.py`
3. THEN use `python tools/harmonize.py`

After retraining, the new model works with both:
- `python tools/harmonize.py` (melody → chord + harmony)
- `python tools/generate.py` (free-form generation from scratch)

Both work on the same retrained model; they just use different inference strategies.

## Notes for Future Debugging

If harmonization produces bad chords:
1. Check training loss converged (<2.0)
2. Verify token order in first 50 tokens: `[lead:] [dur:]` should appear before `[chord:]`
3. Check vocab includes all chord types and MIDI pitches
4. Try lower TEMPERATURE (0.5) in harmonize.py for more deterministic results

If melody drifts:
- The force-feed mechanism should prevent this
- If it happens, check stopping logic (line 254 in harmonize.py)

If detokenize fails:
- Verify harmonized_output.txt has all 4 voices per event
- Check that `[dur:]` tokens are present (signals end of event)
- Print first 100 tokens and verify format
