import requests
import xml.etree.ElementTree as ET
import os
import time
import re

# --- Configuration ---
API_URL = "https://www.barbershoptags.com/api.php"
OUTPUT_DIR = "tools/barbershop_dataset/raw"
MIN_RATING = 4.5       # Only high-quality tags
MAX_TAGS = 100         # How many valid files to download
BATCH_SIZE = 20        # API standard page size

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")

def fetch_barbershop_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🎵 Starting harvest. Target: {MAX_TAGS} high-quality tags...")
    
    downloaded_count = 0
    start_index = 1
    
    while downloaded_count < MAX_TAGS:
        # Query: Barbershop style, 4 parts, sorted by Downloaded
        params = {
            "Type": "bbs",
            "Parts": "4",
            "Sortby": "Downloaded",
            "start": start_index
        }
        
        try:
            print(f"   ...Querying index {start_index}...")
            response = requests.get(API_URL, params=params, timeout=10)
            response.raise_for_status()
            
            # API returns XML
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError:
                print("   ⚠️ XML Parse Error. Skipping batch.")
                start_index += BATCH_SIZE
                continue

            tags = root.findall(".//tag")
            if not tags:
                print("   🏁 End of API results.")
                break
                
            for tag in tags:
                if downloaded_count >= MAX_TAGS:
                    break
                
                # Extract Metadata
                title = tag.find("Title").text
                rating_str = tag.find("Rating").text
                rating = float(rating_str) if rating_str else 0.0
                tag_id = tag.find("id").text

                if rating < MIN_RATING:
                    continue

                # Hunt for MusicXML or MIDI links
                # The API is inconsistent. We check 'MusicXML', 'Notation', and 'Midi' fields.
                candidates = []
                
                # Priority 1: Explicit MusicXML field
                mxml = tag.find("MusicXML")
                if mxml is not None and mxml.text:
                    candidates.append(mxml.text)
                
                # Priority 2: Notation field (often contains the XML link)
                notation = tag.find("Notation")
                if notation is not None and notation.text:
                    candidates.append(notation.text)

                # Priority 3: MIDI (Good backup if XML missing)
                midi = tag.find("Midi")
                if midi is not None and midi.text:
                    candidates.append(midi.text)

                # Download the first valid candidate
                found_file = False
                valid_exts = ['.xml', '.mxl', '.musicxml', '.mid', '.midi']
                
                for url in candidates:
                    ext = os.path.splitext(url)[1].lower()
                    if ext in valid_exts:
                        try:
                            file_data = requests.get(url, timeout=10).content
                            safe_title = sanitize_filename(title)
                            filename = f"{OUTPUT_DIR}/{safe_title}_{tag_id}{ext}"
                            
                            with open(filename, 'wb') as f:
                                f.write(file_data)
                            
                            print(f"   ⬇️ [{downloaded_count+1}/{MAX_TAGS}] Saved: {safe_title}{ext}")
                            downloaded_count += 1
                            found_file = True
                            break # Move to next tag
                        except Exception as e:
                            print(f"   ❌ Failed to download {url}: {e}")
                
                if not found_file:
                    pass # Tag didn't have usable machine-readable files

            start_index += BATCH_SIZE
            time.sleep(1.0) # Be polite to their server

        except Exception as e:
            print(f"   ❌ Network/API Error: {e}")
            break

    print(f"✅ Harvest complete. {downloaded_count} files in {OUTPUT_DIR}")

if __name__ == "__main__":
    fetch_barbershop_data()
