#!/usr/bin/env python3
"""
Verify that Pure Arranger implementation is correctly set up.

This script checks:
1. Token reordering in train.py is correct
2. harmonize.py exists and has required components
3. Key constants match between train.py and harmonize.py
4. Training data file exists
"""

import os
import re
from pathlib import Path

def check_token_order():
    """Verify token ordering in train.py."""
    print("🔍 Checking token ordering in train.py...")

    with open("tools/train.py", "r") as f:
        content = f.read()

    # Look for order_priority dictionary
    match = re.search(r"order_priority = \{([^}]+)\}", content)
    if not match:
        print("  ❌ FAILED: order_priority dictionary not found")
        return False

    order_dict = match.group(1)

    # Expected order: bar(0), lead(1), dur(2), chord(3), bass(4), bari(5), tenor(6)
    expected = [
        ("'bar'", "0"),
        ("'lead'", "1"),
        ("'dur'", "2"),
        ("'chord'", "3"),
        ("'bass'", "4"),
        ("'bari'", "5"),
        ("'tenor'", "6"),
    ]

    for token, priority in expected:
        if f"{token}: {priority}" not in order_dict:
            print(f"  ❌ FAILED: Missing {token}: {priority}")
            return False

    print("  ✅ PASSED: Token order is correct")
    print("     Order: bar(0), lead(1), dur(2), chord(3), bass(4), bari(5), tenor(6)")
    return True

def check_harmonize_exists():
    """Verify harmonize.py exists and is complete."""
    print("\n🔍 Checking harmonize.py...")

    if not os.path.exists("tools/harmonize.py"):
        print("  ❌ FAILED: harmonize.py not found")
        return False

    print("  ✅ File exists")

    with open("tools/harmonize.py", "r") as f:
        content = f.read()

    # Check for required components
    required = [
        ("AttentionHead class", "class AttentionHead"),
        ("BarbershopTransformer class", "class BarbershopTransformer"),
        ("load_model function", "def load_model"),
        ("harmonize_melody function", "def harmonize_melody"),
        ("TEST_MELODY constant", "TEST_MELODY = ["),
        ("Force-feed logic", "constraint_tokens = ["),
        ("Generation loop", "while len(harmony_tokens)"),
        ("Stop condition", 'if token.startswith("[bar:")'),
    ]

    all_found = True
    for name, pattern in required:
        if pattern in content:
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")
            all_found = False

    return all_found

def check_constants():
    """Verify key constants match between files."""
    print("\n🔍 Checking constants...")

    with open("tools/train.py", "r") as f:
        train_content = f.read()

    with open("tools/harmonize.py", "r") as f:
        harmonize_content = f.read()

    # BLOCK_SIZE is hardcoded in both
    block_size_match = re.search(r"BLOCK_SIZE = 256", train_content) and \
                       re.search(r"BLOCK_SIZE = 256", harmonize_content)
    if block_size_match:
        print("  ✅ BLOCK_SIZE = 256 (in both files)")
    else:
        print("  ❌ BLOCK_SIZE mismatch")
        return False

    # N_EMBD, N_HEAD, N_LAYER are defined in train.py as constants
    # In harmonize.py they're global and set from checkpoint, which is fine
    if re.search(r"N_EMBD = 384", train_content):
        print("  ✅ N_EMBD = 384 (in train.py, loaded from checkpoint in harmonize.py)")
    else:
        print("  ❌ N_EMBD mismatch in train.py")
        return False

    if re.search(r"N_HEAD = 6", train_content):
        print("  ✅ N_HEAD = 6 (in train.py, loaded from checkpoint in harmonize.py)")
    else:
        print("  ❌ N_HEAD mismatch in train.py")
        return False

    if re.search(r"N_LAYER = 6", train_content):
        print("  ✅ N_LAYER = 6 (in train.py, loaded from checkpoint in harmonize.py)")
    else:
        print("  ❌ N_LAYER mismatch in train.py")
        return False

    return True

def check_training_data():
    """Verify training data exists."""
    print("\n🔍 Checking training data...")

    data_file = "tools/barbershop_dataset/training_sequences.txt"
    if os.path.exists(data_file):
        size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"  ✅ Found {data_file} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ❌ Not found: {data_file}")
        return False

def check_detokenizer():
    """Verify detokenizer is present."""
    print("\n🔍 Checking detokenizer...")

    if os.path.exists("tools/detokenize.py"):
        print("  ✅ tools/detokenize.py exists")
        return True
    else:
        print("  ❌ tools/detokenize.py not found")
        return False

def check_documentation():
    """Verify documentation files exist."""
    print("\n🔍 Checking documentation...")

    docs = [
        "IMPLEMENTATION_SUMMARY.md",
        "PURE_ARRANGER_QUICKSTART.md",
    ]

    all_found = True
    for doc in docs:
        if os.path.exists(doc):
            print(f"  ✅ {doc}")
        else:
            print(f"  ⚠️ {doc} not found (not critical)")

    return True  # Documentation is optional

def main():
    print("=" * 70)
    print("🎹 PURE ARRANGER IMPLEMENTATION VERIFICATION")
    print("=" * 70)
    print()

    checks = [
        check_token_order,
        check_harmonize_exists,
        check_constants,
        check_training_data,
        check_detokenizer,
        check_documentation,
    ]

    results = [check() for check in checks]

    print()
    print("=" * 70)
    if all(results):
        print("✅ ALL CHECKS PASSED!")
        print()
        print("Next steps:")
        print("  1. Delete old model:     rm tools/barbershop_dataset/arranger_model.pt")
        print("  2. Retrain model:        python tools/train.py")
        print("  3. Harmonize melody:     python tools/harmonize.py")
        print("  4. Convert to XML:       python tools/detokenize.py harmonized_output.txt out.xml")
        print("  5. Play in MuseScore:    musescore out.xml")
    else:
        print("❌ SOME CHECKS FAILED - Please review errors above")
    print("=" * 70)
    print()

if __name__ == "__main__":
    main()
