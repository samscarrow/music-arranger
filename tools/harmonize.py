#!/usr/bin/env python3
"""
Harmonize a Melody Using the Trained Transformer Model

Given a melody (pitch, duration pairs), uses the trained model to:
  1. See each melody note + duration
  2. Predict the best chord to harmonize it
  3. Generate harmonizing bass, baritone, tenor voices

This is a "pure arranger" mode: melody is fixed (INPUT), chords and harmony are generated (OUTPUT).

Usage:
  python tools/harmonize.py
    (Harmonizes TEST_MELODY hardcoded below)

The output is saved to harmonized_output.txt and can be detokenized to MusicXML:
  python tools/detokenize.py harmonized_output.txt final_arrangement.xml
  musescore final_arrangement.xml
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ============================================================================
# MODEL ARCHITECTURE (Must match train.py exactly)
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
# CONFIG
# ============================================================================

MODEL_PATH = "tools/barbershop_dataset/arranger_model.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEMPERATURE = 0.8  # <1.0 = more deterministic, >1.0 = more random
OUTPUT_FILE = "harmonized_output.txt"
METER = "12/8"  # Change this to match your melody's time signature

# ============================================================================
# TEST MELODY: "She's Always a Woman" (Billy Joel) - Transposed to C major
# Extracted from shes-always-a-woman-billy-joel-Voice.mxl (400 notes, -3 semitones)
# Format: (MIDI pitch, duration in beats)
# ============================================================================

TEST_MELODY = [
    (55, 0.25),    # G3
    (55, 0.25),    # G3
    (55, 0.25),    # G3
    (50, 0.25),    # D3
    (55, 0.25),    # G3
    (57, 0.25),    # A3
    (55, 0.25),    # G3
    (50, 0.25),    # D3
    (62, 0.25),    # D4
    (62, 0.25),    # D4
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 1.0),    # E4
    (62, 0.25),    # D4
    (60, 0.25),    # C4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 0.75),    # E4
    (62, 0.25),    # D4
    (60, 0.5),    # C4
    (60, 0.5),    # C4
    (67, 0.25),    # G4
    (67, 0.25),    # G4
    (67, 0.5),    # G4
    (69, 0.5),    # A4
    (72, 0.5),    # C5
    (48, 0.25),    # C3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (65, 0.5),    # F4
    (52, 0.25),    # E3
    (67, 0.5),    # G4
    (69, 0.5),    # A4
    (72, 0.5),    # C5
    (48, 0.25),    # C3
    (69, 1.5),    # A4
    (52, 0.25),    # E3
    (62, 0.5),    # D4
    (62, 0.5),    # D4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 0.5),    # E4
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (59, 0.25),    # B3
    (64, 1.5),    # E4
    (62, 0.5),    # D4
    (52, 0.25),    # E3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (52, 0.25),    # E3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (48, 0.25),    # C3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (59, 0.5),    # B3
    (55, 0.25),    # G3
    (60, 3.0),    # C4
    (55, 0.25),    # G3
    (55, 0.25),    # G3
    (62, 0.25),    # D4
    (62, 0.25),    # D4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 1.0),    # E4
    (62, 0.25),    # D4
    (60, 0.25),    # C4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 0.5),    # E4
    (60, 1.0),    # C4
    (67, 0.25),    # G4
    (67, 0.25),    # G4
    (67, 0.5),    # G4
    (69, 0.5),    # A4
    (72, 0.5),    # C5
    (48, 0.25),    # C3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (65, 0.5),    # F4
    (57, 0.25),    # A3
    (67, 0.5),    # G4
    (69, 0.5),    # A4
    (72, 0.5),    # C5
    (50, 0.25),    # D3
    (69, 0.5),    # A4
    (48, 0.25),    # C3
    (67, 1.0),    # G4
    (62, 0.25),    # D4
    (50, 0.25),    # D3
    (62, 0.25),    # D4
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 0.5),    # E4
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (53, 0.25),    # F3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (59, 0.25),    # B3
    (64, 1.5),    # E4
    (62, 0.25),    # D4
    (62, 0.25),    # D4
    (52, 0.25),    # E3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (52, 0.25),    # E3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (48, 0.25),    # C3
    (62, 0.5),    # D4
    (60, 0.75),    # C4
    (64, 0.25),    # E4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (60, 0.75),    # C4
    (59, 0.25),    # B3
    (55, 0.25),    # G3
    (60, 3.0),    # C4
    (53, 0.25),    # F3
    (55, 0.25),    # G3
    (55, 0.25),    # G3
    (52, 0.25),    # E3
    (72, 3.0),    # C5
    (52, 0.25),    # E3
    (50, 0.25),    # D3
    (72, 0.5),    # C5
    (69, 0.5),    # A4
    (71, 0.5),    # B4
    (57, 0.25),    # A3
    (72, 0.5),    # C5
    (74, 0.5),    # D5
    (69, 0.5),    # A4
    (50, 0.25),    # D3
    (72, 3.0),    # C5
    (50, 0.25),    # D3
    (50, 0.25),    # D3
    (72, 0.5),    # C5
    (69, 0.5),    # A4
    (71, 0.5),    # B4
    (50, 0.25),    # D3
    (72, 0.5),    # C5
    (74, 0.5),    # D5
    (67, 0.5),    # G4
    (48, 0.25),    # C3
    (69, 3.0),    # A4
    (60, 0.25),    # C4
    (57, 0.25),    # A3
    (65, 0.5),    # F4
    (67, 0.5),    # G4
    (50, 0.25),    # D3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (65, 0.5),    # F4
    (55, 0.25),    # G3
    (67, 3.0),    # G4
    (55, 0.25),    # G3
    (55, 0.25),    # G3
    (67, 1.5),    # G4
    (55, 0.25),    # G3
    (75, 3.0),    # E-5
    (55, 0.25),    # G3
    (53, 0.25),    # F3
    (75, 0.5),    # E-5
    (72, 0.5),    # C5
    (74, 0.5),    # D5
    (48, 0.25),    # C3
    (75, 0.5),    # E-5
    (77, 0.5),    # F5
    (72, 0.5),    # C5
    (53, 0.25),    # F3
    (74, 3.0),    # D5
    (53, 0.25),    # F3
    (51, 0.25),    # E-3
    (74, 0.5),    # D5
    (70, 0.5),    # B-4
    (72, 0.5),    # C5
    (58, 0.25),    # B-3
    (74, 0.5),    # D5
    (75, 0.5),    # E-5
    (70, 0.5),    # B-4
    (51, 0.25),    # E-3
    (72, 3.0),    # C5
    (51, 0.25),    # E-3
    (51, 0.25),    # E-3
    (69, 0.5),    # A4
    (71, 0.5),    # B4
    (48, 0.25),    # C3
    (72, 0.5),    # C5
    (71, 0.5),    # B4
    (69, 0.5),    # A4
    (50, 0.25),    # D3
    (67, 3.0),    # G4
    (52, 0.25),    # E3
    (53, 0.25),    # F3
    (62, 0.25),    # D4
    (62, 0.25),    # D4
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 1.0),    # E4
    (62, 0.25),    # D4
    (60, 0.25),    # C4
    (53, 0.25),    # F3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 0.5),    # E4
    (60, 1.0),    # C4
    (58, 0.25),    # B-3
    (67, 0.25),    # G4
    (67, 0.25),    # G4
    (67, 0.5),    # G4
    (69, 0.5),    # A4
    (72, 0.5),    # C5
    (48, 0.25),    # C3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (65, 0.5),    # F4
    (52, 0.25),    # E3
    (67, 0.5),    # G4
    (69, 0.5),    # A4
    (72, 0.5),    # C5
    (50, 0.25),    # D3
    (69, 0.5),    # A4
    (67, 1.0),    # G4
    (51, 0.25),    # E-3
    (62, 0.25),    # D4
    (53, 0.25),    # F3
    (62, 0.25),    # D4
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 0.5),    # E4
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (59, 0.25),    # B3
    (64, 1.5),    # E4
    (62, 0.25),    # D4
    (62, 0.25),    # D4
    (52, 0.25),    # E3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (52, 0.25),    # E3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (48, 0.25),    # C3
    (62, 0.5),    # D4
    (60, 0.75),    # C4
    (64, 0.25),    # E4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (60, 0.75),    # C4
    (59, 0.25),    # B3
    (55, 0.25),    # G3
    (60, 3.0),    # C4
    (55, 0.25),    # G3
    (55, 0.25),    # G3
    (60, 0.5),    # C4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 1.0),    # E4
    (62, 0.5),    # D4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (59, 0.25),    # B3
    (64, 1.5),    # E4
    (64, 1.0),    # E4
    (64, 0.5),    # E4
    (52, 0.25),    # E3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (64, 0.5),    # E4
    (52, 0.25),    # E3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (64, 0.5),    # E4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (59, 0.5),    # B3
    (55, 0.25),    # G3
    (60, 3.0),    # C4
    (55, 0.25),    # G3
    (55, 0.25),    # G3
    (62, 0.25),    # D4
    (52, 0.25),    # E3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (67, 0.5),    # G4
    (52, 0.25),    # E3
    (67, 0.5),    # G4
    (67, 0.5),    # G4
    (72, 0.5),    # C5
    (48, 0.25),    # C3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (64, 0.5),    # E4
    (48, 0.25),    # C3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (57, 0.25),    # A3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (53, 0.25),    # F3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (59, 0.5),    # B3
    (55, 0.25),    # G3
    (60, 3.0),    # C4
    (55, 0.25),    # G3
    (55, 0.25),    # G3
    (60, 0.5),    # C4
    (62, 0.5),    # D4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (55, 0.25),    # G3
    (64, 1.0),    # E4
    (62, 0.5),    # D4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (64, 0.5),    # E4
    (67, 0.5),    # G4
    (59, 0.25),    # B3
    (64, 1.5),    # E4
    (64, 1.0),    # E4
    (64, 0.5),    # E4
    (52, 0.25),    # E3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (64, 0.5),    # E4
    (52, 0.25),    # E3
    (69, 0.5),    # A4
    (67, 0.5),    # G4
    (64, 0.5),    # E4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (64, 0.5),    # E4
    (50, 0.25),    # D3
    (62, 0.5),    # D4
    (60, 0.5),    # C4
    (59, 0.5),    # B3
    (55, 0.25),    # G3
    (60, 3.0),    # C4
    (55, 0.25),    # G3
    (55, 0.25),    # G3 (last)
]


def load_model():
    """Load trained model from checkpoint."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print(f"   Run 'python tools/train.py' first to train the model.")
        return None, None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    config = checkpoint['config']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']

    # Set global dimensions
    global N_EMBD, N_HEAD, N_LAYER
    N_EMBD = config['n_embd']
    N_HEAD = config['n_head']
    N_LAYER = config['n_layer']

    model = BarbershopTransformer(len(stoi))
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()

    return model, stoi, itos


def format_duration(dur):
    """Convert numeric duration to string format for tokens."""
    # Common durations: 0.25 (16th), 0.5 (8th), 0.75 (dotted 8th), 1.0 (quarter), 2.0 (half)
    dur_str = str(dur)
    # Remove trailing zeros
    if '.' in dur_str:
        dur_str = dur_str.rstrip('0').rstrip('.')
    return dur_str


def harmonize_melody(melody_notes, model, stoi, itos):
    """
    Harmonize a melody by feeding notes one-by-one to the model.

    Args:
        melody_notes: List of (midi_pitch, duration) tuples
        model: Loaded Transformer model
        stoi: String-to-index vocab dict
        itos: Index-to-string vocab dict

    Returns:
        List of tokens (strings)
    """
    print(f"🎹 Harmonizing {len(melody_notes)}-note melody...")
    print()

    # Start with key/meter context (use METER constant)
    context_tokens = ["[key:C]", f"[meter:{METER}]"]
    output_tokens = list(context_tokens)

    # Convert to indices for initial context
    idx = torch.tensor(
        [stoi.get(t, 0) for t in context_tokens],
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    # CRITICAL: Bar tracking uses 4.0 beats regardless of actual meter
    # This matches the tokenizer bug (line 192 in tokenizer.py: bar_num = int(offset // 4) + 1)
    # The training data has this pattern, so the model expects it.
    # TODO: Fix tokenizer to use actual meter, then retrain model.
    bar_num = 1
    beat_in_bar = 0.0
    beats_per_bar = 4.0

    for i, (pitch, duration) in enumerate(melody_notes, 1):
        print(f"  Note {i:2d}/{len(melody_notes)}: MIDI {pitch:2d}, dur {duration}")

        # Check if we need a new bar marker
        if beat_in_bar == 0.0:
            bar_token = f"[bar:{bar_num}]"
            if bar_token in stoi:
                output_tokens.append(bar_token)
                token_idx = torch.tensor([[stoi[bar_token]]], device=DEVICE)
                idx = torch.cat((idx, token_idx), dim=1)

        # 1. FORCE-FEED melody constraint tokens
        constraint_tokens = [
            f"[lead:{pitch}]",
            f"[dur:{format_duration(duration)}]"
        ]

        for token in constraint_tokens:
            if token not in stoi:
                print(f"    ⚠️ Warning: Token '{token}' not in vocab, skipping")
                continue

            output_tokens.append(token)
            token_idx = torch.tensor([[stoi[token]]], device=DEVICE)
            idx = torch.cat((idx, token_idx), dim=1)

        # 2. GENERATE harmony (chord + bass + bari + tenor)
        harmony_tokens = []
        max_harmony_tokens = 4  # chord, bass, bari, tenor

        while len(harmony_tokens) < max_harmony_tokens:
            # Crop context to block size
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Get logits
            logits = model(idx_cond)
            logits = logits[:, -1, :] / TEMPERATURE
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            token = itos[idx_next.item()]

            # STOP if model tries to start a new bar or end song
            if token.startswith("[bar:") or token == "[song_end]":
                print(f"    Model signaled next event, stopping generation")
                break

            # Accept harmony token
            output_tokens.append(token)
            harmony_tokens.append(token)
            idx = torch.cat((idx, idx_next), dim=1)

            # Debug: show what was generated
            if token.startswith("[chord:"):
                print(f"      Predicted chord: {token}")
            elif token.startswith("[bass:"):
                print(f"      Predicted bass:  {token}")
            elif token.startswith("[bari:"):
                print(f"      Predicted bari:  {token}")
            elif token.startswith("[tenor:"):
                print(f"      Predicted tenor: {token}")

        # Update bar tracking
        beat_in_bar += duration
        if beat_in_bar >= beats_per_bar:
            beat_in_bar -= beats_per_bar
            bar_num += 1

        print()

    # Add end marker
    output_tokens.append("[song_end]")

    return output_tokens


def main():
    print("=" * 70)
    print("🎹 BARBERSHOP MELODY HARMONIZER")
    print("=" * 70)
    print()

    # Load model
    model, stoi, itos = load_model()
    if model is None:
        return

    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   Vocab size: {len(stoi)}")
    print(f"   Device: {DEVICE}")
    print(f"   Meter: {METER}")

    # Warn about non-4/4 meters
    if METER != "4/4":
        print()
        print("⚠️  WARNING: Model quality may be poor for non-4/4 meters!")
        print(f"   Training data: 98.6% are 4/4 (577/585 songs)")
        print(f"   Only 1 song in 6/8, 5 songs in 3/4")
        print(f"   Model may generate 4/4-style harmonies even with [meter:{METER}] token")
        print()

    print()

    # Harmonize
    tokens = harmonize_melody(TEST_MELODY, model, stoi, itos)

    # Output results
    print()
    print("=" * 70)
    print("✅ Harmonization Complete")
    print("=" * 70)
    print()

    print("Output tokens:")
    print("-" * 70)
    token_str = " ".join(tokens)
    print(token_str)
    print()

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(token_str)

    print(f"💾 Saved to {OUTPUT_FILE}")
    print()

    # Statistics
    chord_tokens = [t for t in tokens if t.startswith('[chord:')]
    bar_tokens = [t for t in tokens if t.startswith('[bar:')]
    print("=" * 70)
    print("Statistics:")
    print(f"  Total tokens:       {len(tokens)}")
    print(f"  Bars:               {len(bar_tokens)}")
    print(f"  Chord predictions:  {len(chord_tokens)}")
    if chord_tokens:
        from collections import Counter
        chord_dist = Counter(chord_tokens)
        print()
        print("  Chord Distribution:")
        for chord, count in chord_dist.most_common():
            print(f"    {chord:<25} : {count}")
    print()

    # Next steps
    print("Next steps:")
    print(f"  1. Detokenize to MusicXML:")
    print(f"     python tools/detokenize.py {OUTPUT_FILE} final_arrangement.xml")
    print()
    print(f"  2. Play in MuseScore:")
    print(f"     musescore final_arrangement.xml")
    print()


if __name__ == "__main__":
    main()
