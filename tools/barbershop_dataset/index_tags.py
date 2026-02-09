import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
import time

# --- Configuration ---
API_URL = "https://www.barbershoptags.com/api.php"
BATCH_SIZE = 100
CLIENT_NAME = "MusicArrangerAI"
VALID_EXTENSIONS = ['xml', 'mxl', 'musicxml', 'mid', 'midi']

def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "MusicArrangerAI/1.0"})
    return session

def index_tags():
    print("🔍 Deep Indexing BarbershopTags for machine-readable formats...")
    session = get_session()
    
    total_found = 0
    total_scanned = 0
    start_index = 1
    limit = 7000 # Scan the entire database
    
    try:
        while start_index < limit:
            params = {
                "start": start_index,
                "n": BATCH_SIZE,
                "client": CLIENT_NAME,
                "fldlist": "id,Title,Notation,SheetMusic,NotationAlt,SheetMusicAlt"
            }
            
            response = session.get(API_URL, params=params, timeout=20)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            tags = root.findall(".//tag")
            
            if not tags:
                break
                
            for tag in tags:
                total_scanned += 1
                found_for_tag = False
                
                # Check both URL extension AND 'type' attribute
                for field_name in ["Notation", "SheetMusic", "NotationAlt", "SheetMusicAlt"]:
                    elem = tag.find(field_name)
                    if elem is not None:
                        # Check 'type' attribute
                        file_type = elem.get('type', '').lower()
                        if file_type in VALID_EXTENSIONS:
                            found_for_tag = True
                            break
                        
                        # Check URL extension
                        url = elem.text
                        if url:
                            clean_url = url.split('?')[0].lower()
                            if any(clean_url.endswith('.' + ext) for ext in VALID_EXTENSIONS):
                                found_for_tag = True
                                break
                
                if found_for_tag:
                    total_found += 1
            
            print(f"   Scanned: {total_scanned} | Found: {total_found}")
            start_index += BATCH_SIZE
            time.sleep(0.2)

    except Exception as e:
        print(f"\n❌ Error: {e}")

    print("\n✅ Indexing complete.")
    print(f"   Total Scanned: {total_scanned}")
    if total_scanned > 0:
        print(f"   Total machine-readable: {total_found} ({ (total_found/total_scanned)*100:.1f}% hit rate)")
        print(f"   Estimated total in DB (~6800 tags): {int((total_found/total_scanned)*6800)}")

if __name__ == "__main__":
    index_tags()
