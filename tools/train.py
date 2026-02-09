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

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re
from pathlib import Path

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
# MODEL (NanoGPT)
# ============================================================================
class AttentionHead(nn.Module):
    """Single attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        # Causal mask: prevent attending to future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer block: attention + feed-forward with residuals and LayerNorm."""

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
    """NanoGPT: Lightweight Transformer for barbershop arrangement generation."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # Output logits
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.

        Args:
            idx: (B, T) tensor of token indices to start with
            max_new_tokens: number of tokens to generate
            temperature: controls randomness (1.0 = normal, <1.0 = more deterministic)
            top_k: if set, only sample from top-k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop idx to BLOCK_SIZE if needed
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for next token (last position)
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# Override block name
Block = TransformerBlock


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
    model = BarbershopTransformer(dataset.vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'=' * 70}\n")
    print(f"🚀 Training for {MAX_ITERS} iterations...\n")

    # Training loop
    model.train()
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
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'stoi': dataset.stoi,
                    'itos': dataset.itos,
                    'config': {
                        'n_embd': N_EMBD,
                        'n_head': N_HEAD,
                        'n_layer': N_LAYER,
                        'vocab_size': dataset.vocab_size,
                    }
                },
                MODEL_PATH
            )
            print(f"   💾 Checkpoint saved to {MODEL_PATH}")

    # Final save
    torch.save(
        {
            'model_state': model.state_dict(),
            'stoi': dataset.stoi,
            'itos': dataset.itos,
            'config': {
                'n_embd': N_EMBD,
                'n_head': N_HEAD,
                'n_layer': N_LAYER,
                'vocab_size': dataset.vocab_size,
            }
        },
        MODEL_PATH
    )
    print(f"\n✅ Training Complete!")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
