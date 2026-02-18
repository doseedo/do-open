import os
import re
import json
from pathlib import Path

def parse_bars_beats_to_seconds(bars_beats, bpm=120):
    try:
        bar, beat = map(int, bars_beats.split("|"))
        beats_total = (bar - 1) * 4 + (beat - 1)  # assumes 4/4
        return round((60 / bpm) * beats_total, 3)
    except:
        return None

def parse_session_txt(filepath, default_bpm=120):
    session_data = {
        "session_id": filepath.stem,
        "tempo_bpm": default_bpm,
        "tracks": {}
    }

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f]

    current_track = None

    for line in lines:
        if line.startswith("TRACK NAME:"):
            match = re.search(r'TRACK NAME:\s*(.*?)(\s+\(Stereo\)|\s+\(Mono\))?$', line)
            if match:
                current_track = match.group(1).strip()
                session_data["tracks"][current_track] = {
                    "is_stereo": "Stereo" in (match.group(2) or ""),
                    "clips": []
                }

        elif re.match(r'^\d+\s+\d+\s+\S+', line) and current_track:
            parts = re.split(r'\s{2,}|\t+', line)
            if len(parts) >= 5:
                channel, event_num, clip_name, start_bb, end_bb = parts[:5]
                if "(cross fade" in clip_name.lower() or "fade" in clip_name.lower():
                    session_data["tracks"][current_track]["clips"].append({
                        "fade": "crossfade",
                        "position_bars_beats": start_bb
                    })
                else:
                    start_sec = parse_bars_beats_to_seconds(start_bb, bpm=default_bpm)
                    end_sec = parse_bars_beats_to_seconds(end_bb, bpm=default_bpm)
                    session_data["tracks"][current_track]["clips"].append({
                        "clip_name": clip_name,
                        "file": clip_name + ".wav",
                        "start_bars_beats": start_bb,
                        "end_bars_beats": end_bb,
                        "start_sec": start_sec,
                        "end_sec": end_sec
                    })

    return session_data

def parse_all_sessions(input_folder="sessionmetadata", output_folder="parsed", default_bpm=120):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    for file in input_path.glob("*.txt"):
        parsed = parse_session_txt(file, default_bpm=default_bpm)
        out_file = output_path / f"{parsed['session_id']}.json"
        with open(out_file, 'w') as f:
            json.dump(parsed, f, indent=2)
        print(f"✅ Parsed {file.name} → {out_file.name}")

# === Run This ===
if __name__ == "__main__":
    parse_all_sessions()
