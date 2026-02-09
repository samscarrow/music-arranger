#!/usr/bin/env python3
"""
Shared Transformer Model for Barbershop Quartet Arrangement

Single source of truth for model architecture. Used by train.py, generate.py,
and harmonize.py to ensure identical module trees and checkpoint compatibility.

Architecture: NanoGPT-style — 6-layer Transformer with multi-head attention.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# Default hyperparameters (overridden by checkpoint config at load time)
_DEFAULTS = {
    'n_embd': 384,
    'n_head': 6,
    'n_layer': 6,
    'block_size': 256,
    'dropout': 0.2,
}


class AttentionHead(nn.Module):
    """Single attention head."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer block: attention + feed-forward with residuals and LayerNorm."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BarbershopTransformer(nn.Module):
    """NanoGPT: Lightweight Transformer for barbershop arrangement generation."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, device='cpu'):
        super().__init__()
        self.block_size = block_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def build_model(vocab_size, config=None, device='cpu'):
    """
    Build a BarbershopTransformer from a config dict.

    Config keys: n_embd, n_head, n_layer, block_size, dropout.
    Missing keys use defaults.
    """
    if config is None:
        config = {}
    cfg = {**_DEFAULTS, **config}
    model = BarbershopTransformer(
        vocab_size=vocab_size,
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_layer=cfg['n_layer'],
        block_size=cfg['block_size'],
        dropout=cfg['dropout'],
        device=device,
    )
    return model


def save_checkpoint(path, model, stoi, itos, config):
    """Save model checkpoint with vocab and config."""
    torch.save({
        'model_state': model.state_dict(),
        'stoi': stoi,
        'itos': itos,
        'config': config,
    }, path)


def load_checkpoint(path, device='cpu'):
    """
    Load a model checkpoint.

    Returns (model, stoi, itos, config).

    Backward compatible: fills in defaults for keys missing from old checkpoints
    (block_size, dropout).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']

    # Backward compat: old checkpoints lack block_size and dropout
    config.setdefault('block_size', _DEFAULTS['block_size'])
    config.setdefault('dropout', _DEFAULTS['dropout'])

    vocab_size = config.get('vocab_size', len(stoi))
    model = build_model(vocab_size, config, device=device)

    # Old checkpoints may lack dropout layers — load with strict=False
    # and log any missing keys for transparency
    result = model.load_state_dict(checkpoint['model_state'], strict=False)
    if result.missing_keys:
        print(f"   ℹ️  Initialized missing checkpoint keys (expected for old checkpoints):")
        for k in result.missing_keys:
            print(f"      {k}")

    model.to(device)
    model.eval()
    return model, stoi, itos, config
