import os
import re
import json
import difflib
from pathlib import Path

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
            base_no_ext = os.path.splitext(base)[0]
            norm = normalize_name(base_no_ext)
            index.setdefault(norm, []).append(path)
    return index

def parse_tempo_file(tempo_file):
    events = []
    with open(tempo_file, 'r') as f:
        for line in f:
            if "Initial tempo" in line:
                m = re.search(r'Initial tempo:\s*(\d+)', line)
                if m:
                    events.append((1, 1, int(m.group(1))))
            elif "Change to" in line:
                bpm_match = re.search(r'Change to\s*(\d+)', line)
                pos_match = re.search(r'at bar\s*(\d+)\|(\d+)', line)
                if bpm_match and pos_match:
                    bpm = int(bpm_match.group(1))
                    bar = int(pos_match.group(1))
                    beat = int(pos_match.group(2))
                    events.append((bar, beat, bpm))
    events.sort(key=lambda x: (x[0] - 1) * 4 + (x[1] - 1))
    return events

def bars_beats_to_seconds(bar, beat, tempo_map):
    target_beats = (bar - 1) * 4 + (beat - 1)
    elapsed = 0
    last_beat = 0
    last_bpm = tempo_map[0][2]
    for (b_bar, b_beat, bpm) in tempo_map:
        b_total = (b_bar - 1) * 4 + (b_beat - 1)
        if b_total > target_beats:
            break
        elapsed += ((b_total - last_beat) * 60 / last_bpm)
        last_beat = b_total
        last_bpm = bpm
    elapsed += ((target_beats - last_beat) * 60 / last_bpm)
    return round(elapsed, 3)

def parse_timecode(tc):
    if tc in ("0:00", "00:00"):
        return 0.0
    match = re.match(r'(\d+):(\d+):(\d+):(\d+)', tc)
    if match:
        h, m, s, f = map(int, match.groups())
        fps = 30
        return round(h * 3600 + m * 60 + s + f / fps, 3)
    match2 = re.match(r'(\d+):(\d+)', tc)
    if match2:
        m, s = map(int, match2.groups())
        return round(m * 60 + s, 3)
    return None

def parse_session_folder(folder, audio_index):
    txt_files = list(folder.glob("*.txt"))
    tempo_file = folder / "tempoinfo.txt"
    out_file = folder / f"{folder.name}.json"

    if not txt_files or not tempo_file.exists():
        return None

    with open(txt_files[0], 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f]

    # === Parse known source audio filenames from FILES sections ===
    known_audio_sources = set()
    in_audio_file_section = False
    for line in lines:
        if "O F F L I N E  F I L E S" in line or "O N L I N E  F I L E S" in line:
            in_audio_file_section = True
            continue
        if in_audio_file_section:
            if not line or line.startswith("O N L I N E  C L I P") or line.startswith("TRACK NAME:"):
                break
            parts = re.split(r'\s{2,}|\t+', line)
            if parts:
                known_audio_sources.add(parts[0].strip())

    # === Parse clip → source mapping ===
    clip_to_source = {}
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
                clipname, srcfile = parts
                clip_to_source[clipname.strip()] = srcfile.strip()

    tempo_map = parse_tempo_file(tempo_file)
    session = {"session_id": folder.name, "tracks": {}}
    current_track = None

    for line in lines:
        if line.startswith("TRACK NAME:"):
            m = re.search(r'TRACK NAME:\s*(.*?)(\s+\(Stereo\)|\s+\(Mono\))?$', line)
            if m:
                current_track = m.group(1).strip()
                session["tracks"][current_track] = {
                    "is_stereo": "Stereo" in (m.group(2) or ""),
                    "track_state": "Unspecified",
                    "clips": []
                }
        elif line.startswith("STATE:") and current_track:
            state_match = re.search(r'STATE:\s*(\w+)', line)
            if state_match:
                session["tracks"][current_track]["track_state"] = state_match.group(1)

        elif re.match(r'^\d+\s+\d+\s+\S+', line) and current_track:
            parts = re.split(r'\s{2,}|\t+', line)
            if len(parts) >= 5:
                _, _, clip, start_bb, end_bb, *rest = parts
                if "fade" in clip.lower():
                    continue

                # === Get source name
                source_name = clip_to_source.get(clip, "")
                if not source_name and known_audio_sources:
                    base = clip.split("-")[0]
                    for known in known_audio_sources:
                        if known.startswith(base):
                            source_name = known
                            break

                norm = normalize_name(source_name if source_name else clip)
                print(f"→ Looking for: {norm}")

                candidates = audio_index.get(norm, [])
                if not candidates:
                    matches = difflib.get_close_matches(norm, audio_index.keys(), n=1, cutoff=0.6)
                    if matches:
                        print(f"  🔍 Fuzzy match: {matches[0]}")
                        candidates = audio_index[matches[0]]

                if candidates:
                    print(f"  → Candidates: {candidates}")
                else:
                    print(f"❌ No match for {clip} (normalized as {norm}) → source_name = {source_name}")
                    continue

                audio_file = next((p for p in candidates if "/prev/" not in p.lower()), None)
                if audio_file:
                    print(f"  ✅ Matched audio path: {audio_file}")
                else:
                    print(f"  ⚠️ No usable path (all were in /prev/)")
                    continue

                try:
                    if "|" in start_bb:
                        s_bar, s_beat = map(int, start_bb.split("|"))
                        e_bar, e_beat = map(int, end_bb.split("|"))
                        start_sec = bars_beats_to_seconds(s_bar, s_beat, tempo_map)
                        end_sec = bars_beats_to_seconds(e_bar, e_beat, tempo_map)
                    else:
                        start_sec = parse_timecode(start_bb)
                        end_sec = parse_timecode(end_bb)
                        if start_sec is None or end_sec is None:
                            print(f"    ⛔ Parse error: start_bb={start_bb}, end_bb={end_bb}")
                            continue
                except Exception as e:
                    print(f"    ❌ Exception parsing time: {e}")
                    continue

                clip_state = rest[-1] if rest else "Unspecified"
                session["tracks"][current_track]["clips"].append({
                    "clip_name": clip,
                    "clip_state": clip_state,
                    "audio_file": audio_file,
                    "start_bb": start_bb,
                    "end_bb": end_bb,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration": round(end_sec - start_sec, 3)
                })



    for track in session["tracks"].values():
        end_secs = [c["end_sec"] for c in track["clips"] if "end_sec" in c]
        track["total_duration"] = round(max(end_secs), 3) if end_secs else 0.0

    if not any(t["clips"] for t in session["tracks"].values()):
        print(f"⚠️ Skipped {folder.name}: No clips matched any audio files")
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
        result = parse_session_folder(folder, audio_index)
        if not result:
            skipped += 1
        else:
            parsed += 1

    print(f"\n🔎 Summary: {parsed} parsed, {skipped} skipped, {total} total")

if __name__ == "__main__":
    run()
