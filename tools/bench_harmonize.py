#!/usr/bin/env python3
"""
Benchmark runner for the barbershop harmonizer.

Sweeps parameter combinations (temperature, top_k) over a fixed melody set,
runs harmonize_melody + validate per config, outputs JSONL scorecards and a
summary table.

Usage:
    python tools/bench_harmonize.py                           # default sweep
    python tools/bench_harmonize.py --temps 0.6 0.8 --topk 20
    python tools/bench_harmonize.py --repeat 3 --seed 42
    python tools/bench_harmonize.py --output results.jsonl
    python tools/bench_harmonize.py --notes 50                # first 50 notes only
"""

import argparse
import json
import sys
import os
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harmonize import harmonize_melody, load_model, TEST_MELODY, METER
from validate_arrangement import validate


def run_single(melody, melody_name, model, stoi, itos, meter,
               temperature, top_k, seed=None):
    """Run one harmonization + validation. Returns result dict."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    header, events = harmonize_melody(
        melody, model, stoi, itos, meter=meter, quiet=True,
        temperature=temperature, top_k=top_k
    )
    elapsed = time.time() - t0

    card = validate(header, events)

    return {
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
    }


def print_summary(results):
    """Print a summary table to stdout."""
    print()
    print("=" * 90)
    print("  BENCHMARK SUMMARY")
    print("=" * 90)

    header_fmt = (
        f"  {'temp':>5s}  {'top_k':>5s}  {'score':>6s}  {'err':>4s}  "
        f"{'warn':>5s}  {'chord%':>6s}  {'leap%':>6s}  {'par%':>5s}  "
        f"{'harm%':>6s}  {'time':>6s}  {'seed':>5s}"
    )
    print(header_fmt)
    print("  " + "-" * 86)

    for r in results:
        s = r["scores"]
        print(
            f"  {r['temperature']:5.2f}  {r['top_k']:5d}  "
            f"{r['score']:6.1f}  {r['errors']:4d}  {r['warnings']:5d}  "
            f"{s.get('chord_coverage', 0):6.1f}  "
            f"{s.get('leap_rate', 0):6.1f}  "
            f"{s.get('parallel_rate', 0):5.1f}  "
            f"{s.get('harmonic_rhythm', 0):6.1f}  "
            f"{r['elapsed_sec']:5.1f}s  "
            f"{str(r['seed']) if r['seed'] is not None else '-':>5s}"
        )

    print()

    # Aggregates
    scores = [r["score"] for r in results]
    errors = [r["errors"] for r in results]
    print(f"  Runs:       {len(results)}")
    print(f"  Mean score: {sum(scores) / len(scores):.1f}")
    print(f"  Best score: {max(scores):.1f}")
    print(f"  Mean errors: {sum(errors) / len(errors):.1f}")
    print(f"  Total time: {sum(r['elapsed_sec'] for r in results):.1f}s")
    print("=" * 90)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark harmonizer across parameter sweeps"
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
    args = parser.parse_args()

    # Load model once
    print("Loading model...")
    model, stoi, itos = load_model()
    if model is None:
        print("ERROR: Model not found. Run 'python tools/train.py' first.")
        return 1

    # Prepare melody
    melody = TEST_MELODY
    melody_name = "test_melody"
    if args.notes is not None:
        melody = melody[:args.notes]
        melody_name = f"test_melody[:{args.notes}]"

    # Build run configs
    configs = [
        (t, k)
        for t in args.temps
        for k in args.topk
    ]

    total_runs = len(configs) * args.repeat
    print(f"Running {total_runs} benchmarks "
          f"({len(configs)} configs × {args.repeat} repeats) "
          f"on {len(melody)}-note melody")
    print()

    results = []
    seed_counter = args.seed
    run_num = 0

    for temp, top_k in configs:
        for rep in range(args.repeat):
            run_num += 1
            print(f"  [{run_num}/{total_runs}] temp={temp:.2f} top_k={top_k}"
                  f"{f' rep={rep+1}' if args.repeat > 1 else ''} ...",
                  end="", flush=True)

            result = run_single(
                melody, melody_name, model, stoi, itos,
                meter=METER,
                temperature=temp,
                top_k=top_k,
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

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
