import music21
import glob
import json
import os
from collections import Counter

DATA_DIR = "tools/barbershop_dataset/raw"

def analyze_dataset():
    # Look for all XML and MIDI files
    files = glob.glob(f"{DATA_DIR}/*.xml") + \
            glob.glob(f"{DATA_DIR}/*.mxl") + \
            glob.glob(f"{DATA_DIR}/*.musicxml") + \
            glob.glob(f"{DATA_DIR}/*.mid")
            
    if not files:
        print("❌ No files found. Run fetch_tags.py first!")
        return

    print(f"🧐 Analyzing {len(files)} files for harmonic style...")

    stats = {
        "chord_counts": Counter(),
        "total_chords": 0,
        "bass_range": {"min": 100, "max": 0}, # MIDI pitch tracking
        "intervals_vertical": Counter()
    }

    for i, file_path in enumerate(files):
        try:
            # Parse file
            s = music21.converter.parse(file_path)
            
            # Reduce to block chords (vertical slices)
            # This handles polyphony automatically
            chords = s.chordify()
            
            for c in chords.recurse().getElementsByClass('Chord'):
                stats["total_chords"] += 1
                
                # 1. Chord Quality
                # music21 distinguishes 'dominant-seventh' from 'major-minor' etc.
                # We map common name to be readable.
                quality = c.commonName
                stats["chord_counts"][quality] += 1
                
                # 2. Bass Note (Lowest pitch)
                bass_note = c.bass()
                stats["bass_range"]["min"] = min(stats["bass_range"]["min"], bass_note.midi)
                stats["bass_range"]["max"] = max(stats["bass_range"]["max"], bass_note.midi)

        except Exception as e:
            # Many files will fail (bad formatting, percussion tracks, etc.)
            # Just skip them to keep the pipeline moving.
            pass
            
        if i % 10 == 0:
            print(f"   Processed {i}/{len(files)}...")

    # --- Output Report ---
    print("\n" + "="*40)
    print("      BARBERSHOP STYLE REPORT")
    print("="*40)
    
    print(f"\nAnalyzed {stats['total_chords']} vertical moments.")
    
    print("\n--- CHORD DISTRIBUTION (Top 10) ---")
    # This directly informs your solver weights!
    for quality, count in stats["chord_counts"].most_common(10):
        percent = (count / stats['total_chords']) * 100
        print(f"{quality:<20} : {percent:.1f}%")

    print("\n--- BASS RANGE (MIDI) ---")
    low_n = music21.note.Note()
    low_n.midi = stats['bass_range']['min']
    high_n = music21.note.Note()
    high_n.midi = stats['bass_range']['max']
    print(f"Lowest: {low_n.nameWithOctave} ({stats['bass_range']['min']})")
    print(f"Highest: {high_n.nameWithOctave} ({stats['bass_range']['max']})")

    # Generate JSON for potential automatic ingestion later
    with open("tools/barbershop_dataset/style_profile.json", "w") as f:
        json.dump({
            "chord_distribution": dict(stats["chord_counts"].most_common(20)),
            "bass_limits": stats["bass_range"]
        }, f, indent=2)
    print("\n✅ Saved profile to tools/barbershop_dataset/style_profile.json")

if __name__ == "__main__":
    analyze_dataset()
