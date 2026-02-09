#!/usr/bin/env python3
"""
Train a NanoGPT Transformer on Barbershop Quartet Arrangements

Architecture:
  - 6-layer Transformer with multi-head attention
  - 384 embedding dimensions, 6 attention heads
  - ~500-token vocabulary (chord types + voice pitches + metadata)

Training Strategy:
  - Melody-first causal ordering: [bar] [lead] [dur] -> [chord] [bass] [bari] [tenor]
  - Model sees melody + duration, predicts chord choice + harmony voices
  - Random song chunks (256 tokens) with next-token prediction loss
  - Adam optimizer with learning rate 3e-4
  - Checkpoint every 500 iterations

Dataset assumes: training_sequences.txt (original flat format)
  - Automatically reorders tokens during loading for pure arranger mode
"""

import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re

# Add tools dir to path for model import
sys.path.insert(0, str(Path(__file__).parent))
from model import build_model, save_checkpoint

# ============================================================================
# CONFIG
# ============================================================================
BATCH_SIZE = 32
BLOCK_SIZE = 256  # Sequence length (tokens)
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

DATA_FILE = "tools/barbershop_dataset/training_sequences.txt"
MODEL_PATH = "tools/barbershop_dataset/arranger_model.pt"

# ============================================================================
# DATASET
# ============================================================================
class BarbershopDataset(Dataset):
    """
    Load and preprocess barbershop quartet training sequences.

    Automatically reorders tokens from physical order (bass, bari, lead, tenor)
    to causal order (lead, dur -> chord, bass, bari, tenor) for pure arranger mode.

    In this mode, the model sees ONLY melody + duration, then predicts chord + harmony.
    """

    def __init__(self, text_file, block_size):
        print(f"Loading {text_file}...")
        with open(text_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Parse and reorder tokens for melody-first arrangement
        print("Optimizing token order for Melody->Harmony generation...")
        self.tokens = self.preprocess_and_reorder(raw_text)

        # Build vocabulary
        self.vocab = sorted(list(set(self.tokens)))
        self.vocab_size = len(self.vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}

        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Total Tokens: {len(self.tokens)}")

        # Encode tokens to integers
        self.data = torch.tensor(
            [self.stoi[t] for t in self.tokens],
            dtype=torch.long
        )
        self.block_size = block_size

    def preprocess_and_reorder(self, raw_text):
        """
        Parse and reorder tokens for melody-first causal learning.

        Original format per event:
          [bar:N] [bass:M] [bari:M] [lead:M] [tenor:M] [dur:F] [chord:LABEL]

        Reordered (melody first, then harmony + chord):
          [bar:N] [lead:M] [dur:F] [chord:LABEL] [bass:M] [bari:M] [tenor:M]

        This way, the model sees:
          1. Bar number (position)
          2. Melody note (constraint)
          3. Duration (constraint)
          Then predicts: chord, bass, bari, tenor (chord choice + harmony)
        """
        tokens = raw_text.split()
        reordered = []
        event_buffer = []

        for token in tokens:
            # Start of new event or metadata line
            if token.startswith('[bar:'):
                # Flush previous event
                if event_buffer:
                    reordered.extend(self._reorder_event(event_buffer))
                    event_buffer = []
                event_buffer.append(token)

            # Preserve metadata headers and end markers
            elif token.startswith('[key:') or token.startswith('[meter:'):
                if event_buffer:
                    reordered.extend(self._reorder_event(event_buffer))
                    event_buffer = []
                reordered.append(token)

            elif token == '[song_end]':
                if event_buffer:
                    reordered.extend(self._reorder_event(event_buffer))
                    event_buffer = []
                reordered.append(token)

            # Accumulate event tokens
            else:
                event_buffer.append(token)

        # Flush final event
        if event_buffer:
            reordered.extend(self._reorder_event(event_buffer))

        return reordered

    def _reorder_event(self, event_tokens):
        """
        Reorder tokens within a single event to melody-first order.

        Priority:
          1. [bar:N]      - position
          2. [lead:M]     - melodic constraint (INPUT)
          3. [dur:F]      - duration (INPUT)
          4. [chord:X]    - chord to predict (OUTPUT)
          5. [bass:M]     - bass to predict (OUTPUT)
          6. [bari:M]     - baritone to predict (OUTPUT)
          7. [tenor:M]    - tenor to predict (OUTPUT)

        Returns reordered list of tokens.
        """
        order_priority = {
            'bar': 0,
            'lead': 1,
            'dur': 2,
            'chord': 3,
            'bass': 4,
            'bari': 5,
            'tenor': 6,
        }

        def get_sort_key(token):
            for key, priority in order_priority.items():
                if f'[{key}:' in token:
                    return priority
            return 99  # Unknown token to end

        return sorted(event_tokens, key=get_sort_key)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Return a random chunk of BLOCK_SIZE tokens.
        Input: tokens[idx:idx+BLOCK_SIZE]
        Target: tokens[idx+1:idx+BLOCK_SIZE+1]
        """
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ============================================================================
# TRAINING
# ============================================================================
def train():
    """Main training loop."""
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found. Run tokenize.py first.")
        return

    print(f"🎹 Barbershop Quartet Arranger - Training")
    print(f"{'=' * 70}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Block Size: {BLOCK_SIZE}")
    print(f"Model: {N_LAYER}-layer Transformer ({N_EMBD} dims, {N_HEAD} heads)")
    print(f"{'=' * 70}\n")

    # Load dataset
    dataset = BarbershopDataset(DATA_FILE, BLOCK_SIZE)

    # Initialize model
    config = {
        'n_embd': N_EMBD,
        'n_head': N_HEAD,
        'n_layer': N_LAYER,
        'block_size': BLOCK_SIZE,
        'dropout': DROPOUT,
        'vocab_size': dataset.vocab_size,
    }
    model = build_model(dataset.vocab_size, config, device=DEVICE).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'=' * 70}\n")
    print(f"🚀 Training for {MAX_ITERS} iterations...\n")

    # Training loop
    for iteration in range(MAX_ITERS):
        # Sample batch
        batch_indices = torch.randint(0, len(dataset), (BATCH_SIZE,))
        xb = torch.stack([dataset[i][0] for i in batch_indices]).to(DEVICE)
        yb = torch.stack([dataset[i][1] for i in batch_indices]).to(DEVICE)

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Logging
        if iteration % 100 == 0:
            print(f"Step {iteration:4d} / {MAX_ITERS}: loss = {loss.item():.4f}")

        # Checkpoint
        if iteration % EVAL_INTERVAL == 0 and iteration > 0:
            save_checkpoint(MODEL_PATH, model, dataset.stoi, dataset.itos, config)
            print(f"   💾 Checkpoint saved to {MODEL_PATH}")

    # Final save
    save_checkpoint(MODEL_PATH, model, dataset.stoi, dataset.itos, config)
    print(f"\n✅ Training Complete!")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
