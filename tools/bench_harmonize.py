#!/usr/bin/env python3
"""
Benchmark runner for the barbershop harmonizer.

Sweeps parameter combinations (temperature, top_k) over a standard melody suite,
runs harmonize_melody + validate per config, outputs JSONL scorecards and a
summary table.

Usage:
    python tools/bench_harmonize.py                           # default sweep, all melodies
    python tools/bench_harmonize.py --melody hymn_4_4         # single melody
    python tools/bench_harmonize.py --melody stress_12_8 --notes 50
    python tools/bench_harmonize.py --temps 0.6 0.8 --topk 20
    python tools/bench_harmonize.py --repeat 3 --seed 42
    python tools/bench_harmonize.py --tag baseline --model old_model.pt
"""

import argparse
import json
import subprocess
import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmonize import harmonize_melody, TEST_MELODY, MODEL_PATH, DEVICE
from model import load_checkpoint
from validate_arrangement import validate


# ---------------------------------------------------------------------------
# Standard melody suite
# Each entry: (melody_list, meter_string)
#   melody_list = [(midi_pitch, duration_in_beats), ...]
# ---------------------------------------------------------------------------

# Clean C-major diatonic hymn in 4/4, lead range (60–72), 32 notes
_HYMN_4_4 = [
    (60, 1.0), (62, 1.0), (64, 1.0), (65, 1.0),  # C D E F
    (67, 2.0), (65, 2.0),                          # G-- F--
    (64, 1.0), (62, 1.0), (60, 1.0), (62, 1.0),  # E D C D
    (64, 2.0), (62, 2.0),                          # E-- D--
    (64, 1.0), (65, 1.0), (67, 1.0), (69, 1.0),  # E F G A
    (72, 2.0), (69, 2.0),                          # C5-- A--
    (67, 1.0), (65, 1.0), (64, 1.0), (62, 1.0),  # G F E D
    (60, 4.0),                                      # C----
    (67, 1.0), (67, 1.0), (69, 1.0), (67, 1.0),  # G G A G
    (65, 2.0), (64, 2.0),                          # F-- E--
    (62, 1.0), (64, 1.0), (62, 1.0), (60, 1.0),  # D E D C
    (60, 4.0),                                      # C----
]

# Barbershop tag with held notes, 4/4, 16 notes
_TAG_4_4 = [
    (67, 2.0), (65, 1.0), (64, 1.0),              # G-- F E
    (62, 2.0), (64, 2.0),                          # D-- E--
    (67, 1.0), (69, 1.0), (67, 1.0), (65, 1.0),  # G A G F
    (64, 4.0),                                      # E----
    (65, 2.0), (67, 2.0),                          # F-- G--
    (69, 1.0), (67, 1.0), (65, 2.0),              # A G F--
    (64, 4.0),                                      # E----
]

# Waltz in 3/4, G major feel, 24 notes
_WALTZ_3_4 = [
    (67, 1.5), (69, 0.5), (71, 1.0),              # G A. B
    (72, 1.5), (71, 0.5), (69, 1.0),              # C5 B. A
    (67, 1.0), (64, 1.0), (62, 1.0),              # G E D
    (60, 3.0),                                      # C---
    (64, 1.5), (65, 0.5), (67, 1.0),              # E F. G
    (69, 1.5), (67, 0.5), (65, 1.0),              # A G. F
    (64, 1.0), (62, 1.0), (60, 1.0),              # E D C
    (62, 3.0),                                      # D---
    (67, 1.0), (67, 1.0), (69, 1.0),              # G G A
    (71, 3.0),                                      # B---
]

STANDARD_MELODIES = {
    'hymn_4_4':     (_HYMN_4_4,    '4/4'),
    'tag_4_4':      (_TAG_4_4,     '4/4'),
    'waltz_3_4':    (_WALTZ_3_4,   '3/4'),
    'stress_12_8':  (TEST_MELODY,  '12/8'),
}


def _git_short_sha():
    """Return short git SHA of HEAD, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_single(melody, melody_name, model, stoi, itos, meter,
               temperature, top_k, tag, device, model_path,
               chord_biases=None, seed=None):
    """Run one harmonization + validation. Returns result dict."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    header, events = harmonize_melody(
        melody, model, stoi, itos, meter=meter, quiet=True,
        temperature=temperature, top_k=top_k,
        chord_biases=chord_biases
    )
    elapsed = time.time() - t0

    card = validate(header, events, exclude_voices=('lead',))

    # Count issues by check type
    from collections import Counter
    err_types = dict(Counter(i.check for i in card.errors))
    warn_types = dict(Counter(i.check for i in card.warnings))

    return {
        "tag": tag,
        "model": model_path,
        "meter": meter,
        "device": device,
        "melody": melody_name,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "elapsed_sec": round(elapsed, 2),
        "events": len(events),
        "score": round(card.overall_score, 1),
        "errors": card.error_count,
        "warnings": card.warning_count,
        "passed": card.passed,
        "scores": {k: round(v * 100, 1) for k, v in card.scores.items()},
        "error_types": err_types,
        "warning_types": warn_types,
    }


def print_summary(results, tag, model_path, device):
    """Print a summary table to stdout."""
    print()
    print("=" * 95)
    print("  BENCHMARK SUMMARY")
    print("=" * 95)
    print(f"  Tag:    {tag}")
    print(f"  Model:  {model_path}")
    print(f"  Device: {device}")
    print()

    header_fmt = (
        f"  {'melody':>16s}  {'temp':>5s}  {'top_k':>5s}  {'score':>6s}  {'err':>4s}  "
        f"{'warn':>5s}  {'chord%':>6s}  {'leap%':>6s}  {'par%':>5s}  "
        f"{'harm%':>6s}  {'time':>6s}"
    )
    print(header_fmt)
    print("  " + "-" * 91)

    for r in results:
        s = r["scores"]
        mel_short = r['melody'][:16]
        print(
            f"  {mel_short:>16s}  {r['temperature']:5.2f}  {r['top_k']:5d}  "
            f"{r['score']:6.1f}  {r['errors']:4d}  {r['warnings']:5d}  "
            f"{s.get('chord_coverage', 0):6.1f}  "
            f"{s.get('leap_rate', 0):6.1f}  "
            f"{s.get('parallel_rate', 0):5.1f}  "
            f"{s.get('harmonic_rhythm', 0):6.1f}  "
            f"{r['elapsed_sec']:5.1f}s"
        )

    print()

    # Aggregates
    scores = [r["score"] for r in results]
    errors = [r["errors"] for r in results]
    print(f"  Runs:        {len(results)}")
    print(f"  Mean score:  {sum(scores) / len(scores):.1f}")
    print(f"  Best score:  {max(scores):.1f}")
    print(f"  Mean errors: {sum(errors) / len(errors):.1f}")
    print(f"  Total time:  {sum(r['elapsed_sec'] for r in results):.1f}s")

    # Per-melody breakdown
    melody_names = list(dict.fromkeys(r['melody'] for r in results))
    if len(melody_names) > 1:
        print()
        print("  Per-melody:")
        for name in melody_names:
            mr = [r for r in results if r['melody'] == name]
            ms = sum(r['score'] for r in mr) / len(mr)
            me = sum(r['errors'] for r in mr) / len(mr)
            print(f"    {name:20s}  score={ms:5.1f}  errors={me:4.1f}  (n={len(mr)})")

    # Top offenders: aggregate error_types and warning_types
    from collections import Counter
    err_agg = Counter()
    warn_agg = Counter()
    for r in results:
        for k, v in r.get('error_types', {}).items():
            err_agg[k] += v
        for k, v in r.get('warning_types', {}).items():
            warn_agg[k] += v
    if err_agg:
        print("  Top error types:")
        for k, v in err_agg.most_common(5):
            print(f"    {k:30s}  {v:4d}")
    if warn_agg:
        print("  Top warning types:")
        for k, v in warn_agg.most_common(5):
            print(f"    {k:30s}  {v:4d}")

    print("=" * 95)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark harmonizer across parameter sweeps"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Label for this run (default: git short SHA)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Model checkpoint path (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--meter", type=str, default=None,
        help="Override time signature (default: per-melody)"
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda"], default=None,
        help=f"Force device (default: auto = {DEVICE})"
    )
    parser.add_argument(
        "--melody", type=str, nargs="+", default=None,
        help=f"Melodies to run: {', '.join(STANDARD_MELODIES)} or 'all' (default: all)"
    )
    parser.add_argument(
        "--temps", type=float, nargs="+",
        default=[0.6, 0.8, 1.0],
        help="Temperature values to sweep (default: 0.6 0.8 1.0)"
    )
    parser.add_argument(
        "--topk", type=int, nargs="+",
        default=[10, 20, 40],
        help="Top-k values to sweep (default: 10 20 40)"
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Runs per config (default: 1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base seed (increments per run, default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default="bench_results.jsonl",
        help="JSONL output file (default: bench_results.jsonl)"
    )
    parser.add_argument(
        "--notes", type=int, default=None,
        help="Use only first N melody notes (faster iteration)"
    )
    parser.add_argument(
        "--no-bias", action="store_true",
        help="Disable chord logit biases (default: no biases unless --bias used)"
    )
    parser.add_argument(
        "--bias", action="store_true",
        help="Enable default chord biases (+1.0 MAJOR_TRIAD, +0.5 DOM7, etc.)"
    )
    args = parser.parse_args()

    # Resolve defaults
    tag = args.tag or _git_short_sha()
    model_path = args.model or MODEL_PATH
    device = args.device or DEVICE
    if args.bias and not args.no_bias:
        biases = {
            '[chord:MAJOR_TRIAD]': 1.0,
            '[chord:DOM7]': 0.5,
            '[chord:MINOR_TRIAD]': 0.0,
            '[chord:MINOR7]': 0.0,
            '[chord:HALF_DIM]': -0.5,
            '[chord:AUG]': -1.0,
            '[chord:OTHER]': -2.0,
            '[chord:UNISON]': -1.0,
            '[chord:OPEN_5TH]': -0.5,
        }
    else:
        biases = None

    # Select melodies
    if args.melody is None or args.melody == ['all']:
        melody_names = list(STANDARD_MELODIES.keys())
    else:
        for name in args.melody:
            if name not in STANDARD_MELODIES:
                print(f"ERROR: Unknown melody '{name}'. "
                      f"Choose from: {', '.join(STANDARD_MELODIES)}")
                return 1
        melody_names = args.melody

    # Load model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return 1
    model, stoi, itos, _config = load_checkpoint(model_path, device=device)
    model.eval()

    # Build run configs
    configs = [
        (t, k)
        for t in args.temps
        for k in args.topk
    ]

    # Prepare melodies (apply --notes truncation and --meter override)
    melodies = []
    for name in melody_names:
        mel, default_meter = STANDARD_MELODIES[name]
        meter = args.meter or default_meter
        display = name
        if args.notes is not None:
            mel = mel[:args.notes]
            display = f"{name}[:{args.notes}]"
        melodies.append((display, mel, meter))

    total_runs = len(configs) * args.repeat * len(melodies)
    print(f"Tag: {tag} | Model: {os.path.basename(model_path)} | Device: {device}")
    print(f"Melodies: {', '.join(n for n, _, _ in melodies)}")
    print(f"Biases: {'disabled' if biases is None else 'enabled'}")
    print(f"Running {total_runs} benchmarks "
          f"({len(configs)} configs × {args.repeat} repeats × "
          f"{len(melodies)} melodies)")
    print()

    results = []
    seed_counter = args.seed
    run_num = 0

    for mel_name, mel_notes, mel_meter in melodies:
        for temp, top_k in configs:
            for rep in range(args.repeat):
                run_num += 1
                print(f"  [{run_num}/{total_runs}] {mel_name} "
                      f"temp={temp:.2f} top_k={top_k}"
                      f"{f' rep={rep+1}' if args.repeat > 1 else ''} ...",
                      end="", flush=True)

                result = run_single(
                    mel_notes, mel_name, model, stoi, itos,
                    meter=mel_meter,
                    temperature=temp,
                    top_k=top_k,
                    tag=tag,
                    device=device,
                    model_path=model_path,
                    chord_biases=biases,
                    seed=seed_counter,
                )
                results.append(result)
                seed_counter += 1

                print(f" score={result['score']:.1f} "
                      f"err={result['errors']} "
                      f"({result['elapsed_sec']:.1f}s)")

    # Write JSONL
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults written to {args.output}")

    print_summary(results, tag, model_path, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
