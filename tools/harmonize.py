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

import sys
import os
from pathlib import Path
import re

import torch
from torch.nn import functional as F

# Add tools dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from event import Header, Event, format_events, quarter_notes_per_bar
from model import load_checkpoint

# ============================================================================
# CONFIG
# ============================================================================

MODEL_PATH = "tools/barbershop_dataset/arranger_model.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEMPERATURE = 0.7  # <1.0 = more deterministic, >1.0 = more random
TOP_K = 16         # Sample from top-k tokens within allowed set
MAX_RETRIES = 3    # Retry sampling with higher temperature on failure
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

    model, stoi, itos, _config = load_checkpoint(MODEL_PATH, device=DEVICE)
    model.eval()
    return model, stoi, itos


def format_duration(dur):
    """Convert numeric duration to string format matching tokenizer output.

    Tokenizer uses f'{dur:.1f}' — always one decimal place.
    """
    return f"{dur:.1f}"


def _quantize_duration(dur, stoi):
    """Snap a duration to the nearest [dur:*] token in the vocabulary.

    Returns the formatted duration string (e.g. '1.0') or raises ValueError
    if no dur tokens exist in the vocab.
    """
    formatted = format_duration(dur)
    if f"[dur:{formatted}]" in stoi:
        return formatted

    # Collect all available durations from vocab
    avail = []
    for tok in stoi:
        if tok.startswith("[dur:"):
            val_str = tok[5:-1]  # strip [dur: and ]
            try:
                avail.append(float(val_str))
            except ValueError:
                continue

    if not avail:
        raise ValueError("No [dur:*] tokens in vocabulary")

    nearest = min(avail, key=lambda v: abs(v - dur))
    return format_duration(nearest)


def _parse_harmony_token(token: str) -> tuple[str, str] | None:
    """Extract (type, value) from a token string like '[chord:MAJOR_TRIAD]'."""
    m = re.match(r'\[([a-z_]+):([^\]]+)\]', token)
    if m:
        return m.group(1), m.group(2)
    return None


# Expected generation order after force-feeding [bar] [lead] [dur]
_GENERATION_ORDER = ('chord', 'bass', 'bari', 'tenor')


def build_token_masks(stoi, device='cpu'):
    """Precompute boolean masks and allowed index tensors per token type.

    Returns dict mapping prefix name → (bool_mask, allowed_indices).
    Bool mask has shape (vocab_size,); allowed_indices is a LongTensor of
    valid token positions. Called once at harmonization start.
    """
    vocab_size = max(stoi.values()) + 1
    masks = {}
    for prefix in _GENERATION_ORDER:
        key = f'[{prefix}:'
        m = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        for tok, idx in stoi.items():
            if tok.startswith(key):
                m[idx] = True
        masks[prefix] = (m, m.nonzero(as_tuple=True)[0])
    return masks


def constrained_sample(model, idx, allowed_idx, itos,
                       temperature=TEMPERATURE, top_k=TOP_K,
                       max_retries=MAX_RETRIES):
    """Sample one token, constrained to allowed_idx positions.

    Uses reduced-space top-k: extracts only allowed logits, applies top-k and
    softmax in that subspace, then maps back to vocab indices.

    Returns (token_string, updated_idx) or (None, idx) if all retries fail.
    """
    for attempt in range(max_retries):
        t = temperature * (1.0 + 0.5 * attempt)  # ramp: 0.8 → 1.2 → 1.6

        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[0, -1]  # (vocab_size,)

        # Extract logits for allowed tokens only
        sub_logits = logits[allowed_idx] / t  # (n_allowed,)

        # Top-k within allowed set
        k = min(top_k, sub_logits.size(0)) if top_k else sub_logits.size(0)
        topk_vals, topk_pos = torch.topk(sub_logits, k)

        probs = F.softmax(topk_vals, dim=-1)

        if torch.isnan(probs).any() or probs.max().item() < 1e-8:
            continue

        sampled_pos = torch.multinomial(probs, num_samples=1)
        vocab_idx = allowed_idx[topk_pos[sampled_pos.item()]]

        idx_next = vocab_idx.view(1, 1)
        idx = torch.cat((idx, idx_next), dim=1)
        return itos[vocab_idx.item()], idx

    return None, idx


def harmonize_melody(melody_notes, model, stoi, itos, meter="4/4", quiet=False,
                     temperature=TEMPERATURE, top_k=TOP_K):
    """
    Harmonize a melody by feeding notes one-by-one to the model.

    Args:
        melody_notes: List of (midi_pitch, duration) tuples
        model: Loaded Transformer model
        stoi: String-to-index vocab dict
        itos: Index-to-string vocab dict
        meter: Time signature string (e.g., "4/4", "12/8")
        quiet: If True, suppress per-note progress output
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Restrict sampling to top-k tokens within allowed set

    Returns:
        (Header, list[Event])
    """
    _log = (lambda *a, **kw: None) if quiet else print

    _log(f"Harmonizing {len(melody_notes)}-note melody...")
    _log()

    header = Header(key="C", meter=meter)

    # Build constrained decoding masks (once)
    token_masks = build_token_masks(stoi, device=DEVICE)
    for prefix, (mask, aidx) in token_masks.items():
        _log(f"  Token mask [{prefix}:*]: {int(mask.sum().item())} allowed tokens")
    _log()

    # Start with key/meter context
    meter_token = f"[meter:{meter}]"
    if meter_token not in stoi:
        avail = [t for t in stoi if t.startswith("[meter:")]
        _log(f"    Warning: {meter_token} not in vocab, available: {avail}")
        # Fall back to 4/4 as closest universal default
        meter_token = "[meter:4/4]" if "[meter:4/4]" in stoi else avail[0]
        _log(f"    Using {meter_token} instead")
    context_tokens = ["[key:C]", meter_token]

    missing = [t for t in context_tokens if t not in stoi]
    if missing:
        raise ValueError(f"Missing context tokens in vocab: {missing}")

    idx = torch.tensor(
        [[stoi[t] for t in context_tokens]],
        dtype=torch.long,
        device=DEVICE
    )

    model_bar_num = 1
    model_beat_in_bar = 0.0
    model_beats_per_bar = quarter_notes_per_bar(meter)

    cumulative_offset = 0.0
    real_qn_per_bar = model_beats_per_bar

    events = []

    with torch.inference_mode():
        for i, (pitch, duration) in enumerate(melody_notes, 1):
            _log(f"  Note {i:3d}/{len(melody_notes)}: MIDI {pitch:2d}, dur {duration}")

            # Feed [bar:N] token to model for EVERY event
            bar_token = f"[bar:{model_bar_num}]"
            if bar_token in stoi:
                token_idx = torch.tensor([[stoi[bar_token]]], device=DEVICE)
                idx = torch.cat((idx, token_idx), dim=1)

            # 1. FORCE-FEED melody constraint tokens
            dur_str = _quantize_duration(duration, stoi)
            if dur_str != format_duration(duration):
                _log(f"    Quantized dur {format_duration(duration)} → {dur_str}")

            constraint_tokens = [
                f"[lead:{pitch}]",
                f"[dur:{dur_str}]"
            ]

            for token in constraint_tokens:
                if token not in stoi:
                    _log(f"    Warning: Token '{token}' not in vocab, skipping note")
                    break
                token_idx = torch.tensor([[stoi[token]]], device=DEVICE)
                idx = torch.cat((idx, token_idx), dim=1)
            else:
                # All constraint tokens fed successfully — generate harmony
                chord_val = None
                bass_val = None
                bari_val = None
                tenor_val = None

                # 2. GENERATE harmony: chord → bass → bari → tenor (constrained)
                for token_type in _GENERATION_ORDER:
                    _mask, allowed_idx = token_masks[token_type]
                    token, idx = constrained_sample(
                        model, idx, allowed_idx, itos,
                        temperature=temperature, top_k=top_k
                    )

                    if token is None:
                        _log(f"      {token_type:5s}: FAILED (all retries exhausted, using rest)")
                        continue

                    parsed = _parse_harmony_token(token)
                    if parsed is None:
                        continue
                    tt, tv = parsed

                    if tt == 'chord':
                        chord_val = tv
                        _log(f"      Predicted chord: {token}")
                    elif tt == 'bass':
                        bass_val = int(tv) if tv != 'rest' else None
                        _log(f"      Predicted bass:  {token}")
                    elif tt == 'bari':
                        bari_val = int(tv) if tv != 'rest' else None
                        _log(f"      Predicted bari:  {token}")
                    elif tt == 'tenor':
                        tenor_val = int(tv) if tv != 'rest' else None
                        _log(f"      Predicted tenor: {token}")

                # 3. BUILD Event from generated tokens
                real_bar = int(cumulative_offset // real_qn_per_bar) + 1

                event = Event(
                    bar=real_bar,
                    offset_qn=cumulative_offset,
                    lead=pitch,
                    tenor=tenor_val,
                    bari=bari_val,
                    bass=bass_val,
                    dur=duration,
                    chord=chord_val,
                )
                events.append(event)

            # Update tracking
            cumulative_offset += duration
            model_beat_in_bar += duration
            if model_beat_in_bar >= model_beats_per_bar:
                model_beat_in_bar -= model_beats_per_bar
                model_bar_num += 1

            _log()

    return header, events


def main():
    from collections import Counter

    print("=" * 70)
    print("BARBERSHOP MELODY HARMONIZER")
    print("=" * 70)
    print()

    # Load model
    model, stoi, itos = load_model()
    if model is None:
        return

    print(f"Model loaded from {MODEL_PATH}")
    print(f"  Vocab size: {len(stoi)}")
    print(f"  Device: {DEVICE}")
    print(f"  Meter: {METER}")

    # Warn about non-4/4 meters
    if METER != "4/4":
        print()
        print(f"WARNING: Model quality may be poor for non-4/4 meters!")
        print(f"  Training data: 98.6% are 4/4 (577/585 songs)")
        print(f"  Only 1 song in 6/8, 5 songs in 3/4")
        print(f"  Model may generate 4/4-style harmonies even with [meter:{METER}] token")
        print()

    print()

    # Harmonize — returns (Header, list[Event])
    header, events = harmonize_melody(TEST_MELODY, model, stoi, itos, meter=METER)

    # Serialize to canonical format
    output_text = format_events(header, events)

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(output_text)

    # Output results
    print()
    print("=" * 70)
    print("Harmonization Complete")
    print("=" * 70)
    print()
    print(f"Saved to {OUTPUT_FILE}")
    print()

    # Statistics
    chord_dist = Counter(e.chord for e in events if e.chord)
    print("Statistics:")
    print(f"  Events:             {len(events)}")
    print(f"  Melody notes input: {len(TEST_MELODY)}")
    if chord_dist:
        print()
        print("  Chord Distribution:")
        for chord, count in chord_dist.most_common():
            pct = 100.0 * count / len(events)
            print(f"    {chord:<25} : {count} ({pct:.1f}%)")
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
