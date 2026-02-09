import requests
import os
import glob
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

# --- CONFIG ---
RAW_DIR = "tools/barbershop_dataset/raw"
AUDIO_DIR = "tools/barbershop_dataset/audio"
API_URL = "https://www.barbershoptags.com/api.php"
CLIENT_NAME = "MusicArrangerAI"

# Mapping API fields to local filenames
STEM_MAP = {
    "dwnld_mix": "Mix",
    "dwnld_ten": "Tenor",
    "dwnld_lead": "Lead",
    "dwnld_bari": "Bari",
    "dwnld_bass": "Bass"
}

def get_existing_ids():
    """Scans RAW_DIR to find all Tag IDs we have sheet music for."""
    ids = set()
    for f in glob.glob(os.path.join(RAW_DIR, "*")):
        # filename format: Title_ID.ext
        try:
            base = os.path.splitext(os.path.basename(f))[0]
            parts = base.split('_')
            if parts[-1].isdigit():
                ids.add(parts[-1])
        except:
            pass
    return sorted(list(ids))

def download_stem(url, path):
    """Helper to download a single file."""
    if os.path.exists(path): return # Skip existing
    try:
        # Fake a browser user agent to avoid 403s
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

def process_tag(tag_id):
    """Fetches metadata for a tag and downloads its audio."""
    # 1. Get API Data
    try:
        params = {"id": tag_id, "client": CLIENT_NAME}
        r = requests.get(API_URL, params=params, timeout=10)
        root = ET.fromstring(r.content)
        tag = root.find("tag")
        if tag is None: return f"❌ {tag_id}: Not found"
    except:
        return f"❌ {tag_id}: API Error"

    title = tag.find("Title").text
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ','_')).strip().replace(" ", "_")
    
    # Create folder for this song
    song_dir = os.path.join(AUDIO_DIR, f"{safe_title}_{tag_id}")
    if not os.path.exists(song_dir):
        os.makedirs(song_dir)

    results = []
    
    # 2. Download Stems
    for xml_field, file_suffix in STEM_MAP.items():
        node = tag.find(xml_field)
        if node is not None and node.text:
            url = node.text
            ext = os.path.splitext(url)[1].lower()
            if not ext: ext = ".mp3" # Default
            
            filename = f"{file_suffix}{ext}"
            save_path = os.path.join(song_dir, filename)
            
            if download_stem(url, save_path):
                results.append(file_suffix)

    if results:
        return f"✅ {safe_title}: {', '.join(results)}"
    else:
        # Cleanup empty dir
        try: os.rmdir(song_dir) 
        except: pass
        return f"⚠️  {safe_title}: No audio found"

def main():
    if not os.path.exists(AUDIO_DIR): os.makedirs(AUDIO_DIR)
    
    ids = get_existing_ids()
    print(f"🎧 Found {len(ids)} sheet music files. Checking for matching audio...")
    
    # Parallel Download (Faster)
    with ThreadPoolExecutor(max_workers=5) as executor:
        for result in executor.map(process_tag, ids):
            print(result)

if __name__ == "__main__":
    main()
