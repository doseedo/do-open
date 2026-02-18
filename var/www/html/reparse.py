import os
import re
import json
import difflib
from pathlib import Path
from collections import Counter

# === CONFIG ===
SORTED_SESSIONS = Path("/home/arlo/Data/sessionmetadata/Sorted")
AUDIO_PATH_LIST = Path("/home/arlo/Data/all_audio_paths3.txt")

def normalize_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())

def build_audio_index(list_file):
    index = {}
    with open(list_file, 'r') as f:
        for line in f:
            path = line.strip()
            if not path or "/prev/" in path.lower():
                continue
            base = os.path.basename(path)
            norm = normalize_name(os.path.splitext(base)[0])
            index.setdefault(norm, []).append(path)
    return index

def parse_tracks_and_sources(folder, audio_index):
    txt_files = list(folder.glob("*.txt"))
    out_file = folder / f"{folder.name}_tracks.json"
    if not txt_files:
        print(f"⚠️ No session info for {folder.name}")
        return None

    with open(txt_files[0], 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f]

    # === Parse clip → source file mappings ===
    clip_sources = []
    in_clip_section = False
    for line in lines:
        if "CLIP NAME" in line and "Source File" in line:
            in_clip_section = True
            continue
        if in_clip_section:
            if not line or line.startswith("TRACK NAME:"):
                break
            parts = re.split(r'\s{2,}|\t+', line)
            if len(parts) == 2:
                clip, src = parts
                if src.lower().endswith(".wav"):
                    clip_sources.append(src.strip())

    if not clip_sources:
        print(f"⚠️ Skipped {folder.name}: No WAV sources found in clip map")
        return None

    most_common_source = Counter(clip_sources).most_common(1)[0][0]
    print(f"🎧 Most common source WAV: {most_common_source}")

    # === Match WAV to audio path ===
    norm = normalize_name(os.path.splitext(most_common_source)[0])
    candidates = audio_index.get(norm, [])
    if not candidates:
        matches = difflib.get_close_matches(norm, audio_index.keys(), n=1, cutoff=0.6)
        if matches:
            norm = matches[0]
            candidates = audio_index[norm]

    if not candidates:
        print(f"❌ No match for source WAV: {most_common_source}")
        return None

    source_audio_path = next((p for p in candidates if "/prev/" not in p.lower()), None)
    if not source_audio_path:
        print(f"⚠️ Found only /prev/ paths for {most_common_source}")
        return None

    # === Parse TRACK NAMEs ===
    session = {"session_id": folder.name, "tracks": {}}
    current_track = None
    for line in lines:
        if re.match(r"TRACK NAME:\s+", line):
            match = re.match(r"TRACK NAME:\s+(.*)", line)
            if match:
                current_track = match.group(1).strip()
                print(f"🎛 Found track: {current_track}")
                session["tracks"][current_track] = {
                    "is_stereo": False,
                    "track_state": "Unspecified",
                    "source_audio": source_audio_path
                }

        elif line.startswith("STATE:") and current_track:
            m = re.search(r'STATE:\s*(\w+)', line)
            if m:
                session["tracks"][current_track]["track_state"] = m.group(1)

    if not session["tracks"]:
        print(f"⚠️ Skipped {folder.name}: No TRACK NAME lines found")
        return None

    with open(out_file, "w") as f:
        json.dump(session, f, indent=2)
    print(f"✅ Parsed {folder.name} → {out_file.name}")
    return session

def run():
    audio_index = build_audio_index(AUDIO_PATH_LIST)
    print(f"📦 Indexed {sum(len(v) for v in audio_index.values())} audio files.")
    total, parsed, skipped = 0, 0, 0
    for folder in SORTED_SESSIONS.iterdir():
        if not folder.is_dir():
            continue
        total += 1
        result = parse_tracks_and_sources(folder, audio_index)
        if not result:
            skipped += 1
        else:
            parsed += 1
    print(f"\n🔎 Summary: {parsed} parsed, {skipped} skipped, {total} total")

if __name__ == "__main__":
    run()
