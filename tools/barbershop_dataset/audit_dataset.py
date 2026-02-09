import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
import os
import glob
import time

# --- Configuration ---
API_URL = "https://www.barbershoptags.com/api.php"
LOCAL_DIR = "tools/barbershop_dataset/raw"
BATCH_SIZE = 100
MAX_ID_CHECK = 7000 

FIELDS_TO_CHECK = [
    "Notation", "SheetMusic", 
    "NotationAlt", "SheetMusicAlt", 
    "MusicXML", "Midi"
]

USABLE_TARGETS = ['.mscz', '.xml', '.mxl', '.musicxml', '.mid', '.midi']
UNUSABLE_TARGETS = ['.musx']

def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "MusicArrangerAudit/1.0"})
    return session

def audit_data():
    print("🕵️‍♂️ Starting Precision Gap Analysis...")
    
    local_ids = set()
    files = glob.glob(os.path.join(LOCAL_DIR, "*"))
    for f in files:
        try:
            base = os.path.splitext(os.path.basename(f))[0]
            tag_id = base.split('_')[-1]
            if tag_id.isdigit():
                local_ids.add(int(tag_id))
        except:
            pass
    print(f"   ✅ Found files for {len(local_ids)} unique tags locally.")

    session = get_session()
    missing_report = []
    musx_only_count = 0
    total_scanned = 0
    usable_avail_count = 0

    start_index = 1
    while start_index < MAX_ID_CHECK:
        params = {
            "start": start_index,
            "n": BATCH_SIZE,
            "fldlist": "id,Title," + ",".join(FIELDS_TO_CHECK)
        }
        
        try:
            resp = session.get(API_URL, params=params, timeout=20)
            resp.raise_for_status()
            
            root = ET.fromstring(resp.content)
            tags = root.findall(".//tag")
            
            if not tags: break

            for tag in tags:
                total_scanned += 1
                tag_id = int(tag.find("id").text)
                title = tag.find("Title").text
                
                has_usable = False
                has_musx = False

                for field in FIELDS_TO_CHECK:
                    node = tag.find(field)
                    if node is not None:
                        url = node.text
                        type_attr = node.get('type', '').lower()
                        
                        ext = ""
                        if type_attr:
                            ext = '.' + type_attr
                        elif url:
                            ext = os.path.splitext(url.split('?')[0])[1].lower()
                        
                        if ext in USABLE_TARGETS:
                            has_usable = True
                        elif ext in UNUSABLE_TARGETS:
                            has_musx = True

                if has_usable:
                    usable_avail_count += 1
                    if tag_id not in local_ids:
                        missing_report.append(f"{tag_id}: {title}")
                elif has_musx:
                    musx_only_count += 1
                    # Technically we already have these, but they are 'ignored' in terms of target
                    if tag_id not in local_ids:
                        # We could report these as missing too if we wanted them
                        pass

            start_index += BATCH_SIZE

        except Exception as e:
            print(f"   ❌ Error batch {start_index}: {e}")
            break

    print("\n" + "="*40)
    print("      AUDIT RESULTS")
    print("="*40)
    print(f"Total Tags Scanned:      {total_scanned}")
    print(f"Usable Files Available:  {usable_avail_count}")
    print(f"Finale Only (Ignored):   {musx_only_count}")
    print(f"Total Potentially Usable:{usable_avail_count + musx_only_count}")
    print(f"Locally Downloaded:      {len(local_ids)}")
    print("-" * 40)
    print(f"MISSING USABLE FILES:    {len(missing_report)}")
    
    if missing_report:
        print(f"\nMissing {len(missing_report)} files. List saved to 'tools/barbershop_dataset/missing_tags.txt'")
        with open("tools/barbershop_dataset/missing_tags.txt", "w") as f:
            f.write("\n".join(missing_report))
    else:
        print("\n🎉 PERFECT MATCH! All machine-readable files (including hidden fields) are accounted for.")

if __name__ == "__main__":
    audit_data()