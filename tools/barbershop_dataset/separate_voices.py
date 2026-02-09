import music21
import sys
import os
import glob
import copy
import subprocess
import shutil

def get_mscore_path():
    """Finds the MuseScore binary in the system path."""
    for binary in ["mscore4", "mscore3", "mscore", "musescore"]:
        path = shutil.which(binary)
        if path:
            return path
    return None

def convert_to_xml(file_path):
    """Converts .mscz to MusicXML using MuseScore."""
    mscore = get_mscore_path()
    if not mscore:
        print("   MuseScore not found. Cannot convert .mscz files.")
        return None

    temp_xml = file_path.rsplit('.', 1)[0] + "_temp.xml"
    try:
        subprocess.run(
            [mscore, "-f", "-o", temp_xml, file_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if os.path.exists(temp_xml):
            return temp_xml
    except Exception as e:
        print(f"   Conversion failed for {file_path}: {e}")
    return None

def get_max_voices_in_score(score):
    """Calculates total voices by summing the max voices found in each staff."""
    total_voices = 0
    for p in score.parts:
        measures = p.getElementsByClass(music21.stream.Measure)
        if not measures:
            if p.voices:
                total_voices += len(p.voices)
            elif p.notesAndRests:
                total_voices += 1
            continue

        max_voices_in_staff = 0
        for m in measures:
            if m.voices:
                max_voices_in_staff = max(max_voices_in_staff, len(m.voices))
            elif m.notesAndRests:
                max_voices_in_staff = max(max_voices_in_staff, 1)
        total_voices += max_voices_in_staff
    return total_voices

def explode_closed_score(file_path):
    print(f"Processing: {os.path.basename(file_path)}")

    work_file = file_path
    is_temp = False

    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.mscz':
        print(f"   Converting .mscz to XML...")
        work_file = convert_to_xml(file_path)
        if not work_file:
            return None
        is_temp = True

    try:
        s = music21.converter.parse(work_file)
    except Exception as e:
        print(f"   Failed to parse: {e}")
        if is_temp:
            os.remove(work_file)
        return None

    voice_count = get_max_voices_in_score(s)
    if voice_count != 4:
        print(f"   Skipped: Found {voice_count} voices (need exactly 4)")
        if is_temp:
            os.remove(work_file)
        return None

    score_parts = s.parts

    # CASE 1: Already Open Score (4 Parts)
    if len(score_parts) == 4:
        print("   Already open score (4 parts)")
        res = copy.deepcopy(s)
        if is_temp:
            os.remove(work_file)
        return res

    # CASE 2: Closed Score (2 Staves)
    if len(score_parts) == 2:
        print("   Exploding closed score (2 staves -> 4 parts)...")
        parts = {
            "Tenor": music21.stream.Part(id='Tenor'),
            "Lead": music21.stream.Part(id='Lead'),
            "Bari": music21.stream.Part(id='Bari'),
            "Bass": music21.stream.Part(id='Bass')
        }

        upper_staff = score_parts[0]
        lower_staff = score_parts[1]

        for m in upper_staff.getElementsByClass(music21.stream.Measure):
            parts['Tenor'].append(music21.stream.Measure(number=m.number))
            parts['Lead'].append(music21.stream.Measure(number=m.number))
            voices = m.voices
            if len(voices) >= 2:
                parts['Tenor'][-1].append(copy.deepcopy(voices[0]))
                parts['Lead'][-1].append(copy.deepcopy(voices[1]))
            else:
                items = voices[0] if voices else m.notesAndRests
                parts['Lead'][-1].append(copy.deepcopy(items))

        for m in lower_staff.getElementsByClass(music21.stream.Measure):
            parts['Bari'].append(music21.stream.Measure(number=m.number))
            parts['Bass'].append(music21.stream.Measure(number=m.number))
            voices = m.voices
            if len(voices) >= 2:
                parts['Bari'][-1].append(copy.deepcopy(voices[0]))
                parts['Bass'][-1].append(copy.deepcopy(voices[1]))
            else:
                items = voices[0] if voices else m.notesAndRests
                parts['Bass'][-1].append(copy.deepcopy(items))

        new_score = music21.stream.Score()
        for p_name in ["Tenor", "Lead", "Bari", "Bass"]:
            new_score.insert(0, parts[p_name])

        if is_temp:
            os.remove(work_file)
        return new_score

    print(f"   Unsupported layout ({len(score_parts)} parts). Skipping.")
    if is_temp:
        os.remove(work_file)
    return None


if __name__ == "__main__":
    target_dir = "tools/barbershop_dataset/raw"
    processed_dir = "tools/barbershop_dataset/processed/open_scores"
    os.makedirs(processed_dir, exist_ok=True)

    # Collect all source files, deduplicate by base name.
    # Priority: .xml > .mxl > .mscz (skip .musx — needs Finale, we have exports)
    sources_by_name = {}
    priority = {'.xml': 0, '.mxl': 1, '.mscz': 2}
    for f in glob.glob(os.path.join(target_dir, "*")):
        ext = os.path.splitext(f)[1].lower()
        if ext not in priority:
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        if name not in sources_by_name or priority[ext] < priority[sources_by_name[name][1]]:
            sources_by_name[name] = (f, ext)

    target_files = [v[0] for v in sorted(sources_by_name.values())]
    total = len(target_files)
    print(f"Found {total} unique source files")

    # Check what's already processed
    existing = set()
    for f in glob.glob(os.path.join(processed_dir, "*_pure.xml")):
        name = os.path.basename(f).replace("_pure.xml", "")
        existing.add(name)

    total_processed = 0
    skipped = 0
    failed = 0

    for i, target_file in enumerate(target_files, 1):
        name = os.path.splitext(os.path.basename(target_file))[0]

        if name in existing:
            skipped += 1
            continue

        try:
            exploded = explode_closed_score(target_file)
            if exploded:
                output_path = os.path.join(processed_dir, f"{name}_pure.xml")
                exploded.write('musicxml', output_path)
                print(f"   Saved: {output_path}")
                total_processed += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            print(f"\nInterrupted at {i}/{total}")
            break
        except Exception as e:
            print(f"   Error: {e}")
            failed += 1

        if i % 25 == 0:
            print(f"--- Progress: {i}/{total} (new: {total_processed}, skipped: {skipped}, failed: {failed}) ---")

    print()
    print("=== SUMMARY ===")
    print(f"Total sources:    {total}")
    print(f"Already existed:  {skipped}")
    print(f"Newly processed:  {total_processed}")
    print(f"Failed/skipped:   {failed}")
    print(f"Total in output:  {len(existing) + total_processed}")
