import os
import shutil
from pathlib import Path
from difflib import get_close_matches

# === CONFIG ===
BASE_DIR = Path("/home/arlo/Data/sessionmetadata")
TXT_DIR = BASE_DIR / "Text"
MID_DIR = BASE_DIR / "Midi"
UNMATCHED_TXT_DIR = BASE_DIR / "Unmatched_TXT"
UNMATCHED_MID_DIR = BASE_DIR / "Unmatched_MID"
MATCH_THRESHOLD = 0.75

# === Setup ===
UNMATCHED_TXT_DIR.mkdir(exist_ok=True)
UNMATCHED_MID_DIR.mkdir(exist_ok=True)

# === Collect Files ===
txt_files = [f for f in TXT_DIR.glob("*.txt")]
mid_files = [f for f in MID_DIR.glob("*.mid")]
matched_txt = set()
matched_mid = set()

# === Build .mid name lookup ===
mid_names = {f.stem: f for f in mid_files}

# === Match and copy into folders ===
for txt in txt_files:
    matches = get_close_matches(txt.stem, mid_names.keys(), n=1, cutoff=MATCH_THRESHOLD)
    if matches:
        mid_name = matches[0]
        mid = mid_names[mid_name]

        session_folder = BASE_DIR / txt.stem
        session_folder.mkdir(exist_ok=True)

        shutil.copy2(str(txt), session_folder / txt.name)
        shutil.copy2(str(mid), session_folder / mid.name)

        matched_txt.add(txt)
        matched_mid.add(mid)
    else:
        shutil.move(str(txt), UNMATCHED_TXT_DIR / txt.name)

# === Move unmatched .mid files ===
for mid in mid_files:
    if mid not in matched_mid:
        shutil.move(str(mid), UNMATCHED_MID_DIR / mid.name)

print("✅ Matching and organizing complete.")
