import os
from pathlib import Path
from miditoolkit import MidiFile

# === CONFIG ===
BASE_DIR = Path("/home/arlo/Data/sessionmetadata")
SORTED_DIR = BASE_DIR / "Sorted"
MASTER_TEMPO_SHEET = BASE_DIR / "master_tempo_sheet.txt"

# === UTIL ===
def ticks_to_bars_beats(tick, ticks_per_beat):
    bar = tick // (4 * ticks_per_beat) + 1
    beat = (tick % (4 * ticks_per_beat)) // ticks_per_beat + 1
    return f"{bar}|{beat}"

# === STATS TRACKING ===
total_sessions = 0
sessions_with_tempo = 0
sessions_without_tempo = 0
sessions_with_invalid_tempos = 0

session_tempos = []

# === MAIN LOOP ===
for session_dir in sorted(SORTED_DIR.iterdir()):
    if not session_dir.is_dir():
        continue

    midi_files = list(session_dir.glob("*.mid"))
    if not midi_files:
        continue

    total_sessions += 1
    midi_path = midi_files[0]
    midi = MidiFile(midi_path)
    ticks_per_beat = midi.ticks_per_beat
    tempo_changes = midi.tempo_changes

    tempo_lines = []
    first_tempo = None
    valid_tempo_found = False

    if not tempo_changes:
        tempo_lines.append("❌ No tempo events found. Assuming default: 120 BPM")
    else:
        for event in tempo_changes:
            raw = event.tempo

            # Detect if raw is in BPM or µs/beat
            if raw < 1000:
                bpm = raw  # Logic/DAW-exported BPM
            else:
                bpm = 60000000 / raw  # Standard MIDI tempo (µs per beat)

            if bpm < 30 or bpm > 400:
                tempo_lines.append(f"⚠️ Ignored unrealistic BPM: {bpm:.2f} (raw tempo = {raw:.2f} at tick {event.time})")
                continue

            if first_tempo is None:
                tempo_lines.append(f"Initial tempo: {round(bpm)} BPM")
                first_tempo = round(bpm)
            else:
                position = ticks_to_bars_beats(event.time, ticks_per_beat)
                tempo_lines.append(f"Change to {round(bpm)} BPM at bar {position}")

            valid_tempo_found = True

    # Finalize fallback
    if not valid_tempo_found:
        first_tempo = 120
        tempo_lines.append("❌ No valid tempo events found. Assuming default: 120 BPM")
        sessions_without_tempo += 1
        if tempo_changes:
            sessions_with_invalid_tempos += 1
    else:
        sessions_with_tempo += 1

    # Check for note presence
    has_note = any(len(track.notes) > 0 for track in midi.instruments)

    # Write tempoinfo.txt
    tempo_info_path = session_dir / "tempoinfo.txt"
    with open(tempo_info_path, "w") as f:
        f.write("\n".join(tempo_lines) + "\n")
        f.write(f"\nNote present in file: {'Yes' if has_note else 'No'}\n")

    # Store for master sheet
    session_tempos.append({
        "session": session_dir.name,
        "starting_tempo": first_tempo,
        "has_note": has_note,
        "tempo_summary": "; ".join(tempo_lines)
    })

# === WRITE MASTER SHEET ===
session_tempos.sort(key=lambda x: x["starting_tempo"])
with open(MASTER_TEMPO_SHEET, "w") as f:
    f.write("Session\tStarting Tempo\tNote Present\tTempo Summary\n")
    for s in session_tempos:
        f.write(f"{s['session']}\t{s['starting_tempo']} BPM\t{'Yes' if s['has_note'] else 'No'}\t{s['tempo_summary']}\n")

# === CONSOLE SUMMARY ===
print("✅ Tempo extraction complete.")
print(f"📁 Total sessions processed: {total_sessions}")
print(f"🎼 Sessions with valid tempo events: {sessions_with_tempo}")
print(f"❌ Sessions with no valid tempo: {sessions_without_tempo}")
print(f"⚠️ Sessions with tempo events but all were unrealistic: {sessions_with_invalid_tempos}")
print(f"📄 Master sheet saved to: {MASTER_TEMPO_SHEET}")
