#!/usr/bin/env python3
"""
Generate Barbershop Arrangements from Trained Model

Loads the trained Transformer checkpoint and generates a complete
barbershop arrangement sequence starting from a key signature.

This demonstrates whether the model actually learned harmonic relationships
or just memorized token sequences.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ============================================================================
# ARCHITECTURE (Must match train.py exactly)
# ============================================================================

# Config will be loaded from checkpoint
N_EMBD = None
N_HEAD = None
N_LAYER = None
BLOCK_SIZE = 256


class AttentionHead(nn.Module):
    """Single attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer block."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BarbershopTransformer(nn.Module):
    """NanoGPT Transformer for barbershop arrangement generation."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[TransformerBlock(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


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
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Extract config
    config = checkpoint['config']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = len(stoi)

    # Set global dimensions
    global N_EMBD, N_HEAD, N_LAYER
    N_EMBD = config['n_embd']
    N_HEAD = config['n_head']
    N_LAYER = config['n_layer']

    print(f"   Model Config: {config}")
    print(f"   Vocabulary Size: {vocab_size} tokens")

    # Initialize model
    model = BarbershopTransformer(vocab_size)
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
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
            logits = model(idx_cond)
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
