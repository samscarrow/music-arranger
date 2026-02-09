import requests
import xml.etree.ElementTree as ET
import os
import glob
import time
import re
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIG ---
SOURCE_DIR = "tools/barbershop_dataset/raw"
OUTPUT_BASE = "tools/barbershop_dataset"
API_URL = "https://www.barbershoptags.com/api.php"
RATE_LIMIT = 1.0  # seconds between API calls
API_TIMEOUT = 30  # seconds
DOWNLOAD_TIMEOUT = 60  # seconds


def get_id_from_filename(filename):
    """Extracts ID from 'Title_1234.xml' or similar."""
    match = re.search(r'_(\d+)\.[^.]+$', filename)
    return match.group(1) if match else None


def sanitize_title(title):
    """Remove filesystem-unsafe characters, replace spaces with underscores."""
    return re.sub(r'[\\/*?:"<>|]', "", title).replace(" ", "_")


def download_file(session, url, dest_path):
    """Stream-download a file. Returns True on success."""
    try:
        with session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        size = os.path.getsize(dest_path)
        print(f"      downloaded {size:,} bytes -> {dest_path}")
        return True
    except Exception as e:
        print(f"      FAILED {url}: {e}")
        # Remove partial file
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def fetch_assets():
    # 1. Build target list from raw dataset files
    print("Scanning raw dataset for targets...")
    targets = set()
    for f in glob.glob(f"{SOURCE_DIR}/*"):
        if f.endswith(('.xml', '.mxl', '.mscz', '.musx')):
            tid = get_id_from_filename(os.path.basename(f))
            if tid:
                targets.add(tid)

    print(f"Found {len(targets)} targets")

    # 2. Setup directories
    dirs = {
        "pdf": f"{OUTPUT_BASE}/pdf",
        "mix": f"{OUTPUT_BASE}/audio/mix",
        "stems": f"{OUTPUT_BASE}/audio/stems",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "MusicArrangerAI/1.0"})
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    # 3. Counters
    stats = {"ok": 0, "skip": 0, "fail": 0, "pdf": 0, "mix": 0, "stems": 0}
    total = len(targets)

    # 4. Fetch loop
    for i, tag_id in enumerate(sorted(targets, key=int), 1):
        try:
            params = {
                "id": tag_id,
                "fldlist": "id,Title,SheetMusic,dwnld_mix,dwnld_ten,dwnld_lead,dwnld_bari,dwnld_bass",
            }
            resp = session.get(API_URL, params=params, timeout=API_TIMEOUT)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            tag = root.find("tag")

            if tag is None:
                print(f"[{i}/{total}] #{tag_id}: not found in API, skipping")
                stats["fail"] += 1
                continue

            title_node = tag.find("Title")
            title = sanitize_title(title_node.text) if title_node is not None and title_node.text else f"unknown_{tag_id}"
            print(f"[{i}/{total}] #{tag_id}: {title}")

            tag_has_new = False

            # --- PDF ---
            pdf_node = tag.find("SheetMusic")
            if pdf_node is not None and pdf_node.text:
                url = pdf_node.text.strip()
                if url:
                    pdf_path = f"{dirs['pdf']}/{title}_{tag_id}.pdf"
                    if os.path.exists(pdf_path):
                        print(f"   pdf: exists, skipping")
                    else:
                        print(f"   pdf: {url}")
                        if download_file(session, url, pdf_path):
                            stats["pdf"] += 1
                            tag_has_new = True

            # --- Mix ---
            mix_node = tag.find("dwnld_mix")
            if mix_node is not None and mix_node.text:
                url = mix_node.text.strip()
                if url:
                    ext = os.path.splitext(url)[1] or ".mp3"
                    mix_path = f"{dirs['mix']}/{title}_{tag_id}{ext}"
                    if os.path.exists(mix_path):
                        print(f"   mix: exists, skipping")
                    else:
                        print(f"   mix: {url}")
                        if download_file(session, url, mix_path):
                            stats["mix"] += 1
                            tag_has_new = True

            # --- Stems ---
            stem_parts = {
                "Tenor": tag.find("dwnld_ten"),
                "Lead": tag.find("dwnld_lead"),
                "Bari": tag.find("dwnld_bari"),
                "Bass": tag.find("dwnld_bass"),
            }
            for part, node in stem_parts.items():
                if node is not None and node.text:
                    url = node.text.strip()
                    if not url:
                        continue
                    ext = os.path.splitext(url)[1] or ".mp3"
                    stem_dir = f"{dirs['stems']}/{title}_{tag_id}"
                    stem_path = f"{stem_dir}/{part}{ext}"
                    if os.path.exists(stem_path):
                        print(f"   stem {part}: exists, skipping")
                    else:
                        os.makedirs(stem_dir, exist_ok=True)
                        print(f"   stem {part}: {url}")
                        if download_file(session, url, stem_path):
                            stats["stems"] += 1
                            tag_has_new = True

            if tag_has_new:
                stats["ok"] += 1
            else:
                stats["skip"] += 1

            time.sleep(RATE_LIMIT)

        except KeyboardInterrupt:
            print(f"\nInterrupted at {i}/{total}")
            break
        except Exception as e:
            print(f"   ERROR: {e}")
            stats["fail"] += 1

    # 5. Summary
    print()
    print("=== SUMMARY ===")
    print(f"Tags processed: {i}/{total}")
    print(f"Tags with new downloads: {stats['ok']}")
    print(f"Tags already complete:   {stats['skip']}")
    print(f"Tags failed:             {stats['fail']}")
    print(f"PDFs downloaded:   {stats['pdf']}")
    print(f"Mixes downloaded:  {stats['mix']}")
    print(f"Stems downloaded:  {stats['stems']}")


if __name__ == "__main__":
    fetch_assets()
