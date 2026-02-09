#!/usr/bin/env python3
"""
Generate Barbershop Arrangements from Trained Model

Loads the trained Transformer checkpoint and generates a complete
barbershop arrangement sequence starting from a key signature.

This demonstrates whether the model actually learned harmonic relationships
or just memorized token sequences.
"""

import sys
from pathlib import Path

import torch
from torch.nn import functional as F
import os

# Add tools dir to path for model import
sys.path.insert(0, str(Path(__file__).parent))
from model import load_checkpoint


# ============================================================================
# GENERATION
# ============================================================================

MODEL_PATH = "tools/barbershop_dataset/arranger_model.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_TOKENS = 500
TEMPERATURE = 0.8  # 1.0 = normal, <1.0 = more deterministic
TOP_K = None  # None = sample from all, else top-k only


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    print(f"🎹 Barbershop Arranger - Generation Mode")
    print(f"{'=' * 70}")
    print(f"Device: {DEVICE}")
    print(f"Loading model from: {MODEL_PATH}")
    print()

    # Load checkpoint
    print(f"⏳ Loading model...")
    model, stoi, itos, config = load_checkpoint(MODEL_PATH, device=DEVICE)
    vocab_size = len(stoi)
    BLOCK_SIZE = config.get('block_size', 256)

    print(f"   Model Config: {config}")
    print(f"   Vocabulary Size: {vocab_size} tokens")
    print(f"✅ Model loaded successfully!")
    print()

    # --- GENERATION LOOP ---
    print(f"{'=' * 70}")
    print(f"🎵 Generating Barbershop Arrangement")
    print(f"{'=' * 70}")
    print()

    # Start with standard barbershop key signature
    start_prompt = "[key:C] [meter:4/4]"
    print(f"Starting with: {start_prompt}\n")

    # Encode prompt
    start_tokens = [stoi.get(s, stoi.get('[key:C]', 0)) for s in start_prompt.split()]
    idx = torch.tensor(start_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    print("Generated sequence:")
    print("-" * 70)
    print(start_prompt, end=" ")

    # Generate tokens
    with torch.no_grad():
        for step in range(MAX_TOKENS):
            # Crop to block size
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Get logits from model
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]  # Last position only

            # Apply temperature
            logits = logits / TEMPERATURE
            probs = F.softmax(logits, dim=-1)

            # Top-k filtering (optional)
            if TOP_K is not None:
                v, _ = torch.topk(probs, min(TOP_K, probs.size(-1)))
                probs[probs < v[:, [-1]]] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Decode and print
            token = itos[idx_next.item()]
            print(token, end=" ")

            # Stop at song end
            if token == "[song_end]":
                break

    print()
    print()
    print("-" * 70)
    print(f"Generated {idx.shape[1]} tokens")
    print(f"🏁 Generation Complete")
    print()

    # Statistics
    token_list = [itos[idx[0, i].item()] for i in range(idx.shape[1])]
    chord_tokens = [t for t in token_list if t.startswith('[chord:')]
    bar_tokens = [t for t in token_list if t.startswith('[bar:')]

    print(f"{'=' * 70}")
    print("Statistics:")
    print(f"  Total tokens:      {len(token_list)}")
    print(f"  Chord tokens:      {len(chord_tokens)}")
    print(f"  Bar markers:       {len(bar_tokens)}")
    print(f"  Unique tokens:     {len(set(token_list))}")
    print()

    # Show chord distribution
    if chord_tokens:
        from collections import Counter
        chord_dist = Counter(chord_tokens)
        print("Chord Distribution:")
        for chord, count in chord_dist.most_common():
            pct = (count / len(chord_tokens)) * 100
            print(f"  {chord:<25} : {count:3d} ({pct:5.1f}%)")
    print()


if __name__ == "__main__":
    main()
