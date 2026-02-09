# Training the Barbershop Arranger AI

This guide walks through training a Transformer model to generate barbershop quartet arrangements given a melody and chord progression.

## Quick Start

### 1. Install PyTorch (one-time)

```bash
source venv/bin/activate
pip install torch torchvision torchaudio
```

Or for CPU-only (faster installation, slower training):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Prepare Data

The training data is already generated in the repo:

- **`tools/barbershop_dataset/training_sequences.txt`** — Original flat format
- **`tools/barbershop_dataset/training_sequences_causal.txt`** — Causal order (optional, if you want to use pre-reordered data)

The training script automatically reorders tokens from physical order to causal order:

```
Original (physical):  [bass:48] [bari:55] [lead:60] [tenor:64] [chord:C]
Causal (learning):    [chord:C] [lead:60] [bass:48] [bari:55] [tenor:64]
```

### 3. Train the Model

```bash
source venv/bin/activate
python tools/train.py
```

This will:
- Load 585 barbershop quartet songs (~16,668 tokens)
- Train a 6-layer Transformer with melody-first causal ordering
- Save checkpoints every 500 iterations
- Output final model to `tools/barbershop_dataset/arranger_model.pt`

Expected output:

```
🎹 Barbershop Quartet Arranger - Training
======================================================================
Device: cuda
Batch Size: 32
Block Size: 256
Model: 6-layer Transformer (384 dims, 6 heads)
======================================================================

Model Parameters: 2,567,680
======================================================================

🚀 Training for 5000 iterations...

Step    0 / 5000: loss = 6.8732
Step  100 / 5000: loss = 4.2156
Step  200 / 5000: loss = 3.1847
...
✅ Training Complete!
Model saved to: tools/barbershop_dataset/arranger_model.pt
```

## Model Architecture

### NanoGPT (Lightweight Transformer)

**Embeddings:**
- Token embeddings: vocab_size → 384 dims
- Position embeddings: 256 positions → 384 dims

**Transformer Blocks (×6):**
- Multi-head self-attention (6 heads, 64 dims each)
- Position-wise feed-forward (384 → 1536 → 384)
- LayerNorm + residual connections
- Dropout: 0.2

**Output:**
- Linear projection: 384 → vocab_size
- Cross-entropy loss on next-token prediction

**Total Parameters:** ~2.5M

### Why This Architecture Works for Barbershop

1. **Causal ordering** — Chord and melody first, then predict harmony
2. **Multi-head attention** — Each head can attend to different harmonic/melodic relationships
3. **Small model** — Barbershop has strong rules and patterns; overfitting to 585 songs is unlikely
4. **256-token context** — Enough to capture full chord progressions (typically 8-16 bars)

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch Size | 32 | Random song chunks per batch |
| Block Size | 256 | Tokens per sequence |
| Max Iterations | 5000 | ~80 epochs over the data |
| Learning Rate | 3e-4 | Adam optimizer |
| Embedding Dims | 384 | Per token |
| Attention Heads | 6 | Each 64 dims |
| Transformer Layers | 6 | Depth of model |
| Dropout | 0.2 | Regularization |
| Checkpoint Interval | 500 iters | Save state for resume |

## Data Format

### Input Tokens (Example)

```
[key:C] [meter:4/4]
[bar:1] [chord:MAJOR_TRIAD] [lead:63] [bass:53] [bari:60] [tenor:69] [dur:1.0]
[bar:2] [chord:DOM7] [lead:67] [bass:51] [bari:63] [tenor:70] [dur:2.0]
...
[song_end]
```

### Vocabulary (~500 tokens)

- **Metadata:** `[key:X]`, `[meter:N/D]`, `[bar:N]`, `[dur:F]`, `[song_end]`
- **Chords (9 types):** `[chord:MAJOR_TRIAD]`, `[chord:DOM7]`, `[chord:MINOR7]`, etc.
- **Voice pitches:** `[lead:60]`, `[bass:48]`, `[bari:55]`, `[tenor:69]` (MIDI 0-127)
- **Rests:** `[lead:rest]`, etc.

## Causal Learning Strategy

The key insight: **melody and chord constraints come first, then harmony.**

```
Tokens 1-3 (Constraints):
  [chord:MAJOR_TRIAD] [lead:67] [bass:?]

Tokens 4-6 (Predictions):
  [bari:?] [tenor:?] [dur:1.0]
```

This matches how a human arranger thinks:
1. "What chord am I in?"
2. "What's the melody note?"
3. "What bass, baritone, and tenor notes fit?"

By putting constraints first, the Transformer can attend to them when predicting the harmony.

## Monitoring Training

Watch the loss curve in the terminal:
- **Good:** Loss decreases from ~6.8 → ~2.0-3.0
- **Bad:** Loss plateaus or increases (learning rate too high, or data issue)
- **Overfitting:** Train loss low, but generation is garbage (rare with 585 songs)

## Resuming Training

To train longer, edit `tools/train.py`:

```python
MAX_ITERS = 10000  # Instead of 5000
```

The model will load the latest checkpoint and continue from there.

## Next Steps

Once trained, use the model for:

1. **Generation** — Given a melody and chord progression, sample harmony
2. **Fine-tuning** — Continue training on new arrangements
3. **Analysis** — Attention visualization to understand learned patterns
4. **Integration** — Use as a constraint in the CP-SAT solver

## Troubleshooting

### Out of Memory (OOM)

Reduce `BATCH_SIZE`:

```python
BATCH_SIZE = 16  # or 8
```

### Training is too slow

- Ensure `DEVICE = 'cuda'` if you have a GPU
- Reduce `MAX_ITERS` to 2000 for quick testing
- Use smaller embedding dims (256 instead of 384)

### Loss not decreasing

- Check that `training_sequences.txt` exists and is not empty
- Try increasing `LEARNING_RATE` slightly (3e-4 to 5e-4)
- Verify the data is being loaded correctly (check dataset vocab size)

## References

- **NanoGPT:** Andrej Karpathy's minimal Transformer implementation
- **Barbershop Theory:** Prietto, "Arranging Barbershop Harmony" (2023)
- **Token Format:** Custom vertical-slice format for homophonic music
