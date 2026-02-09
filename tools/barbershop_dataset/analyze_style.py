import music21
import glob
import json
import os
from collections import Counter

DATA_DIR = "tools/barbershop_dataset/processed/open_scores"

# Barbershop functional vocabulary — maps strict music21 labels to style functions.
# Raw labels stay in the JSON for reference; normalized labels are what the AI trains on.
STYLE_MAP = {
    # 1. Home Base — tonic/rest (consonant resolution targets)
    "major triad": "MAJOR_TRIAD",
    "major seventh chord": "MAJOR_TRIAD",
    "Major Third with octave doublings": "MAJOR_TRIAD",
    "enharmonic equivalent to major triad": "MAJOR_TRIAD",
    "major-second major tetrachord": "MAJOR_TRIAD",  # Add2 chord — classic barbershop tag ending
    "incomplete major-seventh chord": "MAJOR_TRIAD",

    # 2. The Engine — dominant function (tension -> resolution)
    "dominant seventh chord": "DOM7",
    "incomplete dominant-seventh chord": "DOM7",
    "enharmonic to dominant seventh chord": "DOM7",
    "German augmented sixth chord": "DOM7",
    "diminished triad": "DOM7",
    "diminished seventh chord": "DOM7",
    "French augmented sixth chord": "DOM7",
    "Italian augmented sixth chord": "DOM7",

    # 3. Color chords — secondary harmonic mechanics
    "minor seventh chord": "MINOR7",
    "incomplete minor-seventh chord": "MINOR7",
    "enharmonic equivalent to minor seventh chord": "MINOR7",

    "minor triad": "MINOR_TRIAD",
    "enharmonic equivalent to minor triad": "MINOR_TRIAD",

    "half-diminished seventh chord": "HALF_DIM",
    "enharmonic equivalent to half-diminished seventh chord": "HALF_DIM",

    "augmented triad": "AUG",
    "augmented seventh chord": "AUG",

    # 4. Texture — locking / unison moments
    "note": "UNISON",
    "Perfect Octave": "UNISON",
    "Perfect Unison": "UNISON",
    "Perfect Fifth": "OPEN_5TH",
}


def normalize_chord(raw_name):
    return STYLE_MAP.get(raw_name, "OTHER")


def analyze_dataset():
    files = glob.glob(f"{DATA_DIR}/*.xml") + \
            glob.glob(f"{DATA_DIR}/*.mxl") + \
            glob.glob(f"{DATA_DIR}/*.musicxml")

    if not files:
        print("No files found. Run separate_voices.py first!")
        return

    print(f"Analyzing {len(files)} pure quartet files...")

    stats = {
        "raw_chords": Counter(),
        "normalized_chords": Counter(),
        "total_chords": 0,
        "ranges": {
            "Tenor": {"min": 127, "max": 0},
            "Lead":  {"min": 127, "max": 0},
            "Bari":  {"min": 127, "max": 0},
            "Bass":  {"min": 127, "max": 0},
        },
        "part_crossings": 0,
        "files_parsed": 0,
        "files_failed": 0,
    }

    for i, file_path in enumerate(files):
        try:
            s = music21.converter.parse(file_path)
            stats["files_parsed"] += 1

            # 1. Harmonic Analysis
            chords = s.chordify()
            for c in chords.recurse().getElementsByClass('Chord'):
                raw = c.commonName
                stats["total_chords"] += 1
                stats["raw_chords"][raw] += 1
                stats["normalized_chords"][normalize_chord(raw)] += 1

            # 2. Part Range Analysis
            for part_name in ["Tenor", "Lead", "Bari", "Bass"]:
                part = s.getElementById(part_name)
                if part:
                    for n in part.recurse().notes:
                        if n.isNote:
                            midi_val = n.pitch.midi
                            if 22 <= midi_val <= 84:
                                stats["ranges"][part_name]["min"] = min(stats["ranges"][part_name]["min"], midi_val)
                                stats["ranges"][part_name]["max"] = max(stats["ranges"][part_name]["max"], midi_val)

            # 3. Voice Order Check
            first_chord = chords.recurse().getElementsByClass('Chord').first()
            if first_chord and len(first_chord.pitches) == 4:
                pitches = sorted([p.midi for p in first_chord.pitches])
                bass_part = s.getElementById('Bass')
                if bass_part:
                    b_note = bass_part.recurse().notes.first()
                    if b_note and b_note.isNote and b_note.pitch.midi > pitches[0]:
                        stats["part_crossings"] += 1

        except Exception as e:
            stats["files_failed"] += 1

        if i % 20 == 0:
            print(f"   Processed {i}/{len(files)}...")

    total = stats["total_chords"]
    if total == 0:
        print("No chords found!")
        return

    # --- Report ---
    print()
    print("=" * 55)
    print("      BARBERSHOP STYLE REPORT")
    print("=" * 55)

    print(f"\nFiles Analyzed: {stats['files_parsed']} ({stats['files_failed']} failed to parse)")
    print(f"Total Chords:   {total:,}")
    print(f"Part Order Issues: {stats['part_crossings']} files")

    # Normalized distribution (what the AI sees)
    print("\n--- FUNCTIONAL CHORD DISTRIBUTION (Normalized) ---")
    for label, count in stats["normalized_chords"].most_common():
        pct = (count / total) * 100
        print(f"  {label:<16} : {count:>6,}  ({pct:5.1f}%)")

    # Raw distribution (for reference)
    print("\n--- RAW CHORD QUALITY (Top 15) ---")
    for quality, count in stats["raw_chords"].most_common(15):
        pct = (count / total) * 100
        print(f"  {quality:<45} : {count:>6,}  ({pct:5.1f}%)")

    # Unmapped labels (what fell into OTHER)
    other_labels = {k: v for k, v in stats["raw_chords"].items() if normalize_chord(k) == "OTHER"}
    if other_labels:
        print(f"\n--- UNMAPPED LABELS ({len(other_labels)} types, {sum(other_labels.values()):,} chords) ---")
        for label, count in sorted(other_labels.items(), key=lambda x: -x[1])[:20]:
            pct = (count / total) * 100
            print(f"  {label:<45} : {count:>6,}  ({pct:5.1f}%)")

    # Vocal ranges
    print("\n--- VOCAL RANGES ---")
    print(f"  {'Part':<10} | {'Low':<8} | {'High':<8} | {'MIDI'}")
    print("  " + "-" * 42)
    for part, r in stats["ranges"].items():
        if r['max'] > 0:
            low_p = music21.pitch.Pitch()
            low_p.midi = r['min']
            high_p = music21.pitch.Pitch()
            high_p.midi = r['max']
            print(f"  {part:<10} | {low_p.nameWithOctave:<8} | {high_p.nameWithOctave:<8} | {r['min']}-{r['max']}")

    # Save to JSON
    profile = {
        "normalized_distribution": dict(stats["normalized_chords"].most_common()),
        "raw_distribution": dict(stats["raw_chords"].most_common(30)),
        "unmapped_labels": dict(sorted(other_labels.items(), key=lambda x: -x[1])[:30]),
        "style_map": STYLE_MAP,
        "vocal_ranges": stats["ranges"],
        "metadata": {
            "files_analyzed": stats["files_parsed"],
            "files_failed": stats["files_failed"],
            "total_chords": total,
            "part_order_warnings": stats["part_crossings"],
        }
    }
    out_path = "tools/barbershop_dataset/style_profile.json"
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nProfile saved to {out_path}")


if __name__ == "__main__":
    analyze_dataset()
