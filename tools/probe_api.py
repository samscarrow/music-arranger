import requests
import os
import re
import xml.dom.minidom

# Path to where you just downloaded PDFs
PDF_DIR = "tools/barbershop_dataset/pdf"

def get_first_id():
    if not os.path.exists(PDF_DIR):
        return None, None
    files = os.listdir(PDF_DIR)
    for f in files:
        # Look for the ID pattern in filename (e.g., Title_1234.pdf)
        match = re.search(r'_(\d+)\.pdf$', f)
        if match:
            return match.group(1), f
    return None, None

def probe():
    tag_id, filename = get_first_id()
    if not tag_id:
        # Fallback to a known ID if no PDFs found
        tag_id = "31"
        filename = "After Today_31.pdf"
        print(f"⚠️ No PDFs found. Probing fallback Tag ID: {tag_id}")
    else:
        print(f"🕵️ Probing Tag ID: {tag_id} (from {filename})")
    
    # Request ALL fields
    url = "https://www.barbershoptags.com/api.php"
    
    # First, let's try the standard call to see what we missed
    response = requests.get(url, params={"id": tag_id})
    
    print("\n--- RAW API RESPONSE ---")
    try:
        # Pretty print the XML
        dom = xml.dom.minidom.parseString(response.text)
        print(dom.toprettyxml())
    except Exception as e:
        print(response.text)

if __name__ == "__main__":
    probe()