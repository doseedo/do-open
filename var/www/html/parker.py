import os
import requests
from pathlib import Path
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

SEARCH_API = "https://archive.org/advancedsearch.php"
META_API = "https://archive.org/metadata/"
BASE_DL_URL = "https://archive.org/download/"
DOWNLOAD_DIR = Path("/mnt/msdd/jazz_archive_downloads")
PREFERRED_FORMATS = [".mp3", ".flac", ".wav"]

JAZZ_ARTISTS = [
    "John Coltrane", "Thelonious Monk",
    "Duke Ellington", "Bill Evans", "Sonny Rollins", "Chet Baker",
    "Louis Armstrong", "Charles Mingus", "Clifford Brown", "Cannonball Adderley",
    "Dexter Gordon", "Wayne Shorter", "Wes Montgomery", "McCoy Tyner", "Count Basie"
]


DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def search_identifiers_for_artist(artist, rows=1000):
    print(f"\n🔍 Searching archive.org for: {artist}")
    params = {
        "q": f'(creator:"{artist}" OR title:"{artist}" OR description:"{artist}") AND mediatype:(audio)',
        "fl[]": "identifier",
        "rows": rows,
        "page": 1,
        "output": "json"
    }
    try:
        r = requests.get(SEARCH_API, params=params)
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
        ids = [doc["identifier"] for doc in docs]
        print(f"✅ Found {len(ids)} identifiers for {artist}")
        return ids
    except Exception as e:
        print(f"❌ Failed search for {artist}: {e}")
        return []

def get_best_files(identifier):
    try:
        r = requests.get(f"{META_API}{identifier}")
        r.raise_for_status()
        files = r.json().get("files", [])
    except Exception as e:
        print(f"❌ Failed metadata: {identifier} → {e}")
        return []

    grouped = {}
    for f in files:
        name = f.get("name", "")
        ext = os.path.splitext(name)[1].lower()
        if ext in PREFERRED_FORMATS:
            base = os.path.splitext(name)[0]
            grouped.setdefault(base, {})[ext] = name

    best_files = []
    for base, formats in grouped.items():
        for ext in PREFERRED_FORMATS:
            if ext in formats:
                fname = formats[ext]
                url = f"{BASE_DL_URL}{identifier}/{fname}"
                local = f"{identifier}__{unquote(fname)}"
                best_files.append((url, local))
                break
    return best_files

def download_file(url, fname, subdir):
    dest = subdir / fname
    if dest.exists():
        print(f"⏭ Already downloaded: {fname}")
        return
    try:
        head = requests.head(url, timeout=10, allow_redirects=True)
        if head.status_code == 401:
            print(f"🔒 Unauthorized: {fname}")
            return
        elif head.status_code == 403:
            print(f"🚫 Forbidden: {fname}")
            return

        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        print(f"✅ Downloaded: {fname}")
    except Exception as e:
        print(f"❌ Download failed: {fname} → {e}")

def process_artist(artist):
    artist_dir = DOWNLOAD_DIR / artist.replace(" ", "_")
    artist_dir.mkdir(parents=True, exist_ok=True)

    identifiers = search_identifiers_for_artist(artist)
    for ident in identifiers:
        print(f"\n🎷 [{artist}] Processing: {ident}")
        files = get_best_files(ident)
        if not files:
            print(f"⚠️ [{artist}] No audio files in {ident}")
            continue
        for url, fname in files:
            download_file(url, fname, artist_dir)

def main():
    with ThreadPoolExecutor(max_workers=len(JAZZ_ARTISTS)) as executor:
        futures = [executor.submit(process_artist, artist) for artist in JAZZ_ARTISTS]
        for future in as_completed(futures):
            future.result()

    print("\n🎉 Done. All artists processed.")
    print(f"📁 Saved in: {DOWNLOAD_DIR.resolve()}")

if __name__ == "__main__":
    main()
