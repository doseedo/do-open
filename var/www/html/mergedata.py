import os
import json
import re
from pathlib import Path

SORTED_DIR = Path("/home/arlo/Data/sessionmetadata/Sorted")
ENCODEC_DIR = Path("/home/arlo/Data/encodec_tokens")

def extract_tempo(tempo_path):
    if not tempo_path.exists():
        return None

    tempo_data = {
        "initial_bpm": None,
        "meter": "4/4",
        "map": []
    }

    with open(tempo_path) as f:
        for line in f:
            if "Initial tempo" in line:
                match = re.search(r"(\d+)\s*BPM", line)
                if match:
                    tempo_data["initial_bpm"] = int(match.group(1))
            elif "Change to" in line:
                match = re.search(r"Change to (\d+)\s*BPM at bar (\d+)\|(\d+)", line)
                if match:
                    bpm, bar, beat = int(match[1]), int(match[2]), int(match[3])
                    tempo_data["map"].append({
                        "bar": bar,
                        "beat": beat,
                        "bpm": bpm
                    })

    return tempo_data if tempo_data["initial_bpm"] is not None else None

def merge_session(session_dir: Path):
    session_name = session_dir.name
    json_path = session_dir / f"{session_name}.json"
    yamnet_path = session_dir / "yamnet_classifications.json"
    tempo_path = session_dir / "tempoinfo.txt"
    out_path = session_dir / "final_metadata.json"

    if not json_path.exists():
        print(f"⚠️ No session JSON found: {json_path}")
        return

    with open(json_path) as f:
        session_data = json.load(f)

    # Optional YAMNet data
    yamnet_data = {}
    if yamnet_path.exists():
        with open(yamnet_path) as f:
            yamnet_data = json.load(f)

    # Tempo info
    tempo = extract_tempo(tempo_path)
    if tempo:
        session_data["tempo"] = tempo
    else:
        print(f"⚠️ No tempo found in: {tempo_path}")

    # Process clips in tracks
    for track_name, track in session_data.get("tracks", {}).items():
        for i, clip in enumerate(track.get("clips", [])):
            # YAMNet
            yam_key = f"{track_name}_clip_{i}"
            if yam_key in yamnet_data:
                clip["yamnet_labels"] = yamnet_data[yam_key].get("yamnet_labels", [])

            # MIDI
            audio_file = clip.get("audio_file", "")
            if "/protools/" in audio_file and audio_file.endswith(".wav"):
                midi_path = audio_file.replace("/protools/", "/BasicPitch/").rsplit(".", 1)[0] + ".mid"
                clip["basic_pitch_midi"] = midi_path

            # Encodec tokens
            audio_filename = Path(audio_file).name.rsplit(".", 1)[0] + ".pt"
            encodec_path = ENCODEC_DIR / session_name / audio_filename
            if encodec_path.exists():
                clip["encodec_tokens"] = str(encodec_path)
            else:
                print(f"⚠️ Missing Encodec tokens for: {encodec_path}")

    # Save final metadata
    with open(out_path, "w") as f:
        json.dump(session_data, f, indent=2)

    print(f"✅ Merged session: {session_name} → final_metadata.json")

def run_all():
    for session_dir in SORTED_DIR.iterdir():
        if session_dir.is_dir():
            merge_session(session_dir)

if __name__ == "__main__":
    run_all()
