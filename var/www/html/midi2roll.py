#!/usr/bin/env python3
import json
import os
import sys
import traceback
import difflib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pretty_midi
from tqdm import tqdm

# ===================== CONFIG =====================
INPUT_JSON = Path("full_dataset_new.json")           # master list: entries with audio_path (+/- midi_path)
FINAL_MANIFEST_JSON = Path("final_training_manifest.json")
PIANO_ROLL_DIR = Path("/mnt/msdd/piano_rolls"); PIANO_ROLL_DIR.mkdir(exist_ok=True)

# Retry mode (reads audio paths from a log of previously failed items)
RETRY_FROM_LOG = True
RETRY_LOG_FILE = Path("log_no_piano_roll_match.txt")

# New logs we’ll write in this script
NO_MIDI_LOG = Path("log_no_midi_found.txt")
ERROR_LOG = Path("log_pianoroll_errors.txt")

# Frame params (must match the rest of your pipeline)
SAMPLE_RATE = 44100
HOP_LENGTH = 4096  # frame rate = SAMPLE_RATE / HOP_LENGTH

# Where we’ll look to detect session root (walk upward)
SESSION_HINT_DIRNAMES = {
    "IO Settings", "Audio Files", "MIDI Files", "Melodyne",
    "Bounced Files", "Session File Backups"
}
MIDI_EXTS = {".mid", ".midi"}

# How we pick the “best” MIDI when there are several
FILENAME_MATCH_THRESHOLD = 0.55  # 0..1, higher is stricter

# ==================================================


def safe_read_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r") as f:
        return json.load(f)


def safe_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def frame_rate() -> float:
    return SAMPLE_RATE / HOP_LENGTH


def detect_session_root(audio_path: Path) -> Path:
    """
    Heuristic: walk upward and if we find a directory whose name is in SESSION_HINT_DIRNAMES,
    treat its parent as the session root. Otherwise fall back to parent.parent.
    """
    p = audio_path
    for parent in p.parents:
        if parent.name in SESSION_HINT_DIRNAMES:
            return parent.parent
    # fallback: two up (like your current logic)
    return audio_path.parent.parent


def find_all_midis(session_root: Path) -> List[Path]:
    if not session_root.exists():
        return []
    midis = []
    # Common subdirs to prioritize
    prioritized = ["MIDI Files", "Midi Files", "MIDI", "Midi"]
    for sub in prioritized:
        candidate = session_root / sub
        if candidate.exists():
            for m in candidate.rglob("*"):
                if m.suffix.lower() in MIDI_EXTS:
                    midis.append(m)

    # Also scan entire session tree (guard against duplicates)
    for m in session_root.rglob("*"):
        if m.suffix.lower() in MIDI_EXTS:
            midis.append(m)

    # De-dup (preserve order)
    seen = set()
    unique = []
    for m in midis:
        if m not in seen:
            unique.append(m)
            seen.add(m)
    return unique


def best_midi_match(audio_path: Path, midi_paths: List[Path]) -> Optional[Path]:
    """
    Use fuzzy match between audio stem and midi stem.
    Return the best match if score >= threshold; else return the first MIDI as fallback (if any).
    """
    if not midi_paths:
        return None
    a_stem = audio_path.stem.lower()
    scored: List[Tuple[float, Path]] = []
    for m in midi_paths:
        score = difflib.SequenceMatcher(None, a_stem, m.stem.lower()).ratio()
        scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_score, top_path = scored[0]
    if top_score >= FILENAME_MATCH_THRESHOLD:
        return top_path
    # fallback: just take the first MIDI in sorted order (stable)
    return top_path


def ensure_subdir(output_root: Path, audio_path: Path) -> Path:
    """
    Mirror a session-level subdir scheme: /piano_rolls/<session_name>/
    We define <session_name> as the detected session root name.
    """
    session_root = detect_session_root(audio_path)
    session_name = session_root.name if session_root else audio_path.parent.parent.name
    out_dir = output_root / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def midi_to_roll_save(midi_file: Path, out_path: Path):
    midi_data = pretty_midi.PrettyMIDI(str(midi_file))
    pr = midi_data.get_piano_roll(fs=frame_rate())
    pr[pr > 0] = 1  # binarize
    np.save(out_path, pr.astype(np.uint8))


def resolve_or_autofind_midi(entry: dict) -> Optional[Path]:
    """
    Return a usable MIDI path:
      1) If entry['midi_path'] exists on disk, use it.
      2) Else, search session_root for a .mid/.midi and pick the best match.
    """
    audio_path = Path(entry["audio_path"])
    # 1) Honor existing midi_path if valid
    midi_path = Path(entry["midi_path"]) if entry.get("midi_path") else None
    if midi_path and midi_path.exists():
        return midi_path

    # 2) Autofind
    session_root = detect_session_root(audio_path)
    midi_candidates = find_all_midis(session_root)
    return best_midi_match(audio_path, midi_candidates)


def load_entries_to_process(input_json: Path,
                            final_manifest_json: Path,
                            retry_from_log: bool,
                            retry_log_file: Path) -> List[dict]:
    if not input_json.exists():
        print(f"❌ Input JSON not found: {input_json}")
        sys.exit(1)

    data = safe_read_json(input_json, [])
    data_map = {e["audio_path"]: e for e in data if "audio_path" in e}

    # Already processed
    already = set()
    if final_manifest_json.exists():
        fm = safe_read_json(final_manifest_json, [])
        already = {e["audio_path"] for e in fm if "audio_path" in e}
        print(f"↺ Loaded {len(already)} previously processed entries from {final_manifest_json.name}")

    if retry_from_log:
        if not retry_log_file.exists():
            print(f"❌ Retry log not found: {retry_log_file}")
            sys.exit(1)
        with open(retry_log_file, "r") as f:
            retry_paths = {line.strip() for line in f if line.strip()}
        entries = []
        for p in retry_paths:
            e = data_map.get(p)
            if e:
                entries.append(e)
        print(f"🔁 RETRY MODE: {len(entries)} entries from {retry_log_file.name}")
        return entries

    # Full mode: anything not already processed
    entries = [e for e in data if e["audio_path"] not in already]
    print(f"🆕 FULL MODE: {len(entries)} new entries to process")
    return entries


def merge_and_save_manifest(final_manifest_json: Path, new_entries: List[dict]):
    existing = safe_read_json(final_manifest_json, [])
    by_audio = {e["audio_path"]: e for e in existing if "audio_path" in e}
    # Update/insert new entries
    for e in new_entries:
        by_audio[e["audio_path"]] = e
    merged = list(by_audio.values())
    safe_write_json(final_manifest_json, merged)
    print(f"💾 Manifest updated: {final_manifest_json.resolve()} ({len(merged)} entries)")


def main():
    entries = load_entries_to_process(INPUT_JSON, FINAL_MANIFEST_JSON, RETRY_FROM_LOG, RETRY_LOG_FILE)

    # Starting point for final manifest (so we keep previous)
    final_manifest = safe_read_json(FINAL_MANIFEST_JSON, [])

    no_midi_log_f = NO_MIDI_LOG.open("a")
    error_log_f = ERROR_LOG.open("a")

    processed_now = 0

    for entry in tqdm(entries, desc="Converting MIDI → Piano Roll"):
        audio_path_str = entry.get("audio_path")
        if not audio_path_str:
            tqdm.write("⚠️ Entry missing 'audio_path' key; skipping.")
            continue

        audio_path = Path(audio_path_str)
        try:
            # Resolve or find MIDI
            midi_path = resolve_or_autofind_midi(entry)
            if not midi_path or not midi_path.exists():
                tqdm.write(f"🚫 No MIDI found for: {audio_path}")
                no_midi_log_f.write(audio_path_str + "\n")
                continue

            # Where we save PR
            out_dir = ensure_subdir(PIANO_ROLL_DIR, audio_path)
            # Use MIDI filename for output (stable & debuggable)
            pr_filename = midi_path.with_suffix(".pianoroll.npy").name
            pr_path = out_dir / pr_filename

            # If exists, just wire up the manifest
            if pr_path.exists():
                entry["midi_path"] = str(midi_path)
                entry["piano_roll_path"] = str(pr_path)
                final_manifest.append(entry)
                continue

            # Write PR
            midi_to_roll_save(midi_path, pr_path)

            # Record
            entry["midi_path"] = str(midi_path)
            entry["piano_roll_path"] = str(pr_path)
            final_manifest.append(entry)
            processed_now += 1

        except Exception as e:
            tqdm.write(f"❌ Error on {audio_path.name}: {e}")
            error_log_f.write(f"{audio_path_str}\n{traceback.format_exc()}\n")

    no_midi_log_f.close()
    error_log_f.close()

    # Dedup & save
    deduped = []
    seen = set()
    for e in final_manifest:
        ap = e.get("audio_path")
        if ap and ap not in seen:
            deduped.append(e)
            seen.add(ap)

    print(f"\n✅ Added {processed_now} new piano rolls.")
    print(f"📦 Total entries to save: {len(deduped)}")

    merge_and_save_manifest(FINAL_MANIFEST_JSON, deduped)


if __name__ == "__main__":
    main()
