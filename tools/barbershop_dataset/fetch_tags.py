import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
import os
import time
import re

# --- Configuration ---
API_URL = "https://www.barbershoptags.com/api.php"
OUTPUT_DIR = "tools/barbershop_dataset/raw"
MAX_TAGS = 10000        # Cover the entire database (~6800 tags)
BATCH_SIZE = 100        # Maximize batch size for efficiency
CLIENT_NAME = "MusicArrangerAI"

# Formats useful for machine analysis
VALID_EXTENSIONS = ['.xml', '.mxl', '.musicxml', '.mid', '.midi', '.mscz', '.musx']

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")

def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "MusicArrangerAI/1.0"})
    return session

def fetch_barbershop_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🎵 Harvesting up to {MAX_TAGS} machine-readable tags...")
    
    session = get_session()
    downloaded_count = 0
    start_index = 1 # Start from the beginning
    seen_ids = set()

    while downloaded_count < MAX_TAGS:
        params = {
            "start": start_index,
            "n": BATCH_SIZE,
            "client": CLIENT_NAME,
            # Added undocumented but likely fields
            "fldlist": "id,Title,Rating,Notation,SheetMusic,NotationAlt,SheetMusicAlt,MusicXML,Midi"
        }
        
        try:
            print(f"   ...Querying index {start_index}...")
            response = session.get(API_URL, params=params, timeout=25)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            tags = root.findall(".//tag")
            if not tags:
                print("   🏁 End of API results.")
                break
                
            for tag in tags:
                if downloaded_count >= MAX_TAGS:
                    break
                
                tag_id = tag.findtext("id", "0")
                if tag_id in seen_ids:
                    continue
                seen_ids.add(tag_id)

                title = tag.findtext("Title", "Unknown")
                
                # Collect all possible file URLs and types
                candidates = []
                for field_name in ["NotationAlt", "Notation", "SheetMusicAlt", "SheetMusic", "MusicXML", "Midi"]:
                    elem = tag.find(field_name)
                    if elem is not None:
                        url = elem.text
                        file_type = elem.get('type', '').lower()
                        if url:
                            candidates.append((url, file_type))

                for url, file_type in candidates:
                    if not url or not url.startswith('http'):
                        continue
                    
                    url = url.replace('&amp;', '&')
                    ext = os.path.splitext(url.split('?')[0])[1].lower()
                    
                    # Match by extension OR by the XML 'type' attribute
                    if ext in VALID_EXTENSIONS or ('.' + file_type) in VALID_EXTENSIONS:
                        try:
                            # Use type attribute if extension is missing/generic
                            final_ext = ext if ext in VALID_EXTENSIONS else ('.' + file_type)
                            
                            file_resp = session.get(url, timeout=20)
                            file_resp.raise_for_status()
                            
                            safe_title = sanitize_filename(title)
                            filename = f"{OUTPUT_DIR}/{safe_title}_{tag_id}{final_ext}"
                            
                            with open(filename, 'wb') as f:
                                f.write(file_resp.content)
                            
                            print(f"   ⬇️ [{downloaded_count+1}] Saved: {safe_title}{final_ext}")
                            downloaded_count += 1
                            break 
                        except Exception as e:
                            print(f"      ❌ Failed download {url}: {e}")
                
            start_index += BATCH_SIZE
            time.sleep(0.5)

        except Exception as e:
            print(f"   ❌ API Error: {e}")
            time.sleep(2)
            start_index += BATCH_SIZE
            if start_index > 7000: break # Safety valve

    print(f"✅ Harvest complete. {downloaded_count} files in {OUTPUT_DIR}")

if __name__ == "__main__":
    fetch_barbershop_data()
