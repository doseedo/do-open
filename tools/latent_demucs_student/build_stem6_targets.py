#!/usr/bin/env python3
"""Build 6-stem target latents for mixesV7 sessions from GT stems in Latents2.

For each mixesV7 session:
1. Read session_meta.json → active tracks + timeline
2. Classify each track into one of 6 categories (drums/bass/guitar/piano/vocals/other)
3. Load individual track latents from Latents2 (via GCS FUSE)
4. Timeline-align using start_samples from session_meta
5. Sum aligned latents within each category
6. Save as stem6_{category}.vae.pt next to full_mix.vae.pt

Usage:
    python build_stem6_targets.py [--manifest PATH] [--limit N]
"""
import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

LATENT_OUT = Path("/scratch/mixesV7_latents")
GCS_FUSE = Path("/home/arlo/gcs-bucket")
LATENTS2_FUSE = GCS_FUSE / "Latents2"
MIXESV7_FUSE = GCS_FUSE / "mixesV7"

STEMS_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]
SAMPLE_RATE = 48000
VAE_HZ = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // VAE_HZ  # 1920

# ── Track name code → 6-stem category ──────────────────────────────────
# For protoolsA sessions with structured naming like "1m01_DRUM_all_TBL"
TRACK_CODE_TO_STEM6 = {
    "DRUM": "drums", "DRM": "drums", "KIT": "drums", "PERC": "drums",
    "BASS": "bass", "BSS": "bass",
    "GTR": "guitar", "GUIT": "guitar",
    "PIANO": "piano", "PNO": "piano", "KEYS": "piano", "KEY": "piano",
    "VOX": "vocals", "VOC": "vocals", "SING": "vocals",
    "STR": "other", "SYN": "other", "FX": "other",
    "BRASS": "other", "WIND": "other", "ORG": "other",
    "MALLET": "other", "MALL": "other",
}
# Track name codes to skip entirely (not real stems)
SKIP_CODES = {"MIX", "REF", "CLICK", "CLK", "ROOM", "AMB", "TALK", "SLATE"}

# ── Manifest group → 6-stem category ──────────────────────────────────
GROUP_TO_STEM6 = {
    "drums": "drums", "percussion": "drums",
    "bass": "bass",
    "guitar": "guitar", "plucked": "guitar",
    "piano": "piano", "keys": "piano", "organ": "piano",
    "voice": "vocals", "dialogue": "vocals",
    "strings": "other", "brass": "other", "winds": "other",
    "synth": "other", "fx": "other", "mallets": "other",
    "ensemble": "other",
}
# Groups to skip
SKIP_GROUPS = {"mix", "room", "click", "silent", "junk", "undefined"}

# ── Heuristic fallback for free-form track names ─────────────────────
HEURISTIC_PATTERNS = [
    (re.compile(r"(kick|snare|hi.?hat|tom|cymbal|oh_|room_|drum|floor|rack)", re.I), "drums"),
    (re.compile(r"bass", re.I), "bass"),
    (re.compile(r"guit|gtr|strat|tele|les.paul|acou.*gtr", re.I), "guitar"),
    (re.compile(r"piano|pno|keys|wurli|rhodes|clav", re.I), "piano"),
    (re.compile(r"vox|vocal|sing|voice|lead.*v|bgv|bkv|choir|harm", re.I), "vocals"),
]


def classify_track_structured(track_name):
    """Classify from structured protoolsA names like '1m01_DRUM_all_TBL'.

    Only matches the protoolsA convention where the name starts with
    a pattern like '1m01_' (digit + 'm' + digits).
    """
    parts = track_name.split("_")
    # Must match protoolsA format: first part is NmNN (e.g. 1m01, 2m03)
    if len(parts) >= 2 and re.match(r"\d+m\d+", parts[0]):
        code = parts[1].upper()
        if code in SKIP_CODES:
            return None
        if code in TRACK_CODE_TO_STEM6:
            return TRACK_CODE_TO_STEM6[code]
    return "UNKNOWN"


def classify_track_heuristic(track_name):
    """Classify from free-form track names like 'Floor_1', 'Hihat_1'."""
    name = track_name.replace("_", " ").replace("-", " ")
    for pattern, category in HEURISTIC_PATTERNS:
        if pattern.search(name):
            return category
    return None  # truly unknown → skip or put in "other"


def classify_track(track_name, manifest_lookup=None, session_path=""):
    """Classify a track into one of 6 stem categories.

    Priority: manifest group > structured name > heuristic > other.
    Returns None to skip the track entirely.
    """
    # 1. Try manifest lookup
    if manifest_lookup:
        group = manifest_lookup.get(track_name)
        if group:
            if group in SKIP_GROUPS:
                return None
            return GROUP_TO_STEM6.get(group, "other")

    # 2. Try structured name parsing (protoolsA convention)
    result = classify_track_structured(track_name)
    if result is None:
        return None  # explicitly skip
    if result != "UNKNOWN":
        return result

    # 3. Try heuristic
    result = classify_track_heuristic(track_name)
    if result:
        return result

    # 4. Default to "other"
    return "other"


def load_latent(path):
    """Load a VAE latent from disk → [T, 64] float32."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64:
        z = z.t()  # → [T, 64]
    return z.float()


def build_manifest_lookup(manifest_path, session_path):
    """Build a {track_name: group} dict for one session from the manifest.

    Only loads the relevant entries to avoid holding the full 1.9GB in memory.
    """
    # The manifest keys are full audio paths like:
    # /home/arlo/gcs-bucket/protools/DATE/New/Session/Audio Files/track.wav
    # We need to match session_path to find the right entries.
    lookup = {}
    if not manifest_path or not os.path.exists(manifest_path):
        return lookup

    # Build the expected path prefix
    prefix = f"/home/arlo/gcs-bucket/{session_path}/Audio Files/"

    # Stream through the manifest entries (it's large, so we use a generator)
    import ijson
    try:
        with open(manifest_path, "rb") as f:
            parser = ijson.kvitems(f, "entries")
            for key, entry in parser:
                if prefix in key:
                    fname = key.split("/")[-1].replace(".wav", "")
                    lookup[fname] = entry.get("group", "undefined")
    except ImportError:
        # ijson not available — load the full manifest (slow, 1.9GB)
        pass

    return lookup


def build_full_manifest_index(manifest_path):
    """Load the full manifest and build a {session_path → {track_name: group}} index."""
    print("[manifest] Loading master manifest (this may take a minute)...")
    t0 = time.time()
    with open(manifest_path) as f:
        data = json.load(f)

    entries = data.get("entries", data)
    index = defaultdict(dict)
    for audio_path, entry in entries.items():
        group = entry.get("group", "undefined")
        # Extract session path: everything between gcs-bucket/ and /Audio Files/
        parts = audio_path.split("/Audio Files/")
        if len(parts) != 2:
            continue
        sess = parts[0]
        for prefix in ["/home/arlo/gcs-bucket/", "/mnt/data/system_home/arlo/gcs-bucket/"]:
            if sess.startswith(prefix):
                sess = sess[len(prefix):]
                break
        track_name = parts[1].replace(".wav", "").replace(".flac", "")
        index[sess][track_name] = group

    print(f"[manifest] Loaded {len(entries)} entries, {len(index)} sessions "
          f"in {time.time()-t0:.1f}s")
    return index


def find_latent_path(session_path, track_name):
    """Find the Latents2 file for a track. Returns Path or None."""
    # Map track_name → wav filename (track_regions may have different naming)
    # Standard: Latents2/{session_path}/Audio Files/{track_name}.vae.pt
    base = LATENTS2_FUSE / session_path / "Audio Files"

    # Try exact match
    exact = base / f"{track_name}.vae.pt"
    if exact.exists():
        return exact

    # Try with common suffixes stripped/added
    for candidate in base.glob(f"{track_name}*.vae.pt"):
        return candidate

    # Try matching the wav_filename from track_regions
    # (sometimes track name differs from wav filename)
    return None


def process_session(session_dir, session_meta, manifest_lookup, total_frames=None):
    """Process one session → dict of {category: aligned_latent_sum}."""
    session_path = session_meta.get("session_path", "")
    active_tracks = session_meta.get("active_tracks", [])
    track_regions = session_meta.get("track_regions", {})

    # Determine timeline length in frames
    total_samples = session_meta.get("total_length_samples", 0)
    if total_frames is None:
        if total_samples > 0:
            total_frames = total_samples // SAMPLES_PER_FRAME
        else:
            # Estimate from full_mix latent
            fm_path = session_dir / "full_mix.vae.pt"
            if fm_path.exists():
                fm = load_latent(fm_path)
                total_frames = fm.shape[0]
            else:
                return None, "no full_mix"

    # Classify and load each track
    categories = defaultdict(list)  # category → [(latent, start_frame)]
    skipped = []

    for track_name in active_tracks:
        # Classify
        cat = classify_track(track_name, manifest_lookup, session_path)
        if cat is None:
            skipped.append(track_name)
            continue

        # Find latent file
        latent_path = find_latent_path(session_path, track_name)
        if latent_path is None:
            # Try matching via wav_filename in track_regions
            regions = track_regions.get(track_name, [])
            if regions:
                wav_fname = regions[0].get("wav_filename", "")
                wav_base = wav_fname.replace(".wav", "")
                latent_path = find_latent_path(session_path, wav_base)
            if latent_path is None:
                skipped.append(f"{track_name}(no_latent)")
                continue

        try:
            z = load_latent(latent_path)  # [T_track, 64]
        except Exception as e:
            skipped.append(f"{track_name}(load_err)")
            continue

        # Get timeline offset
        start_frame = 0
        regions = track_regions.get(track_name, [])
        if regions:
            start_samples = regions[0].get("start_samples", 0)
            start_frame = start_samples // SAMPLES_PER_FRAME

        categories[cat].append((z, start_frame))

    if not categories:
        return None, f"no valid tracks (skipped: {skipped})"

    # Sum aligned latents within each category
    stem_latents = {}
    for cat, track_list in categories.items():
        canvas = torch.zeros(total_frames, 64)
        for z, sf in track_list:
            T_track = z.shape[0]
            end_frame = min(sf + T_track, total_frames)
            usable = end_frame - sf
            if usable > 0:
                canvas[sf:end_frame] += z[:usable]
        stem_latents[cat] = canvas

    active_cats = list(stem_latents.keys())
    return stem_latents, f"ok: {active_cats} (skipped: {skipped})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/arlo/gcs-bucket/DO1ckpts/master_manifest_v2.6.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    # Build manifest index
    manifest_index = {}
    if os.path.exists(args.manifest):
        manifest_index = build_full_manifest_index(args.manifest)

    # Find mixesV7 sessions that have full_mix.vae.pt
    sessions = []
    for fm in LATENT_OUT.rglob("full_mix.vae.pt"):
        session_dir = fm.parent
        # Check if already done
        if args.skip_existing:
            existing = [session_dir / f"stem6_{s}.vae.pt" for s in STEMS_6]
            if any(e.exists() for e in existing):
                continue
        sessions.append(session_dir)

    if args.limit > 0:
        sessions = sessions[:args.limit]

    print(f"[stem6] {len(sessions)} sessions to process")
    if not sessions:
        return

    ok, fail = 0, 0
    t0 = time.time()
    for i, session_dir in enumerate(sessions):
        rel = session_dir.relative_to(LATENT_OUT)

        # Load session_meta.json
        meta_path = MIXESV7_FUSE / rel / "session_meta.json"
        if not meta_path.exists():
            print(f"[{i+1}/{len(sessions)}] {rel}  SKIP (no session_meta.json)")
            fail += 1
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            fail += 1
            continue

        session_path = meta.get("session_path", str(rel))
        manifest_lookup = manifest_index.get(session_path, {})

        try:
            stem_latents, status = process_session(
                session_dir, meta, manifest_lookup
            )
        except Exception as e:
            print(f"[{i+1}/{len(sessions)}] {rel}  ERROR: {e}")
            traceback.print_exc()
            fail += 1
            continue

        if stem_latents is None:
            print(f"[{i+1}/{len(sessions)}] {rel}  SKIP ({status})")
            fail += 1
            continue

        # Save stem6_*.vae.pt
        mask = []
        for s in STEMS_6:
            if s in stem_latents:
                out_path = session_dir / f"stem6_{s}.vae.pt"
                torch.save({"latents": stem_latents[s], "stem": s}, out_path)
                mask.append(1)
            else:
                mask.append(0)

        ok += 1
        elapsed = time.time() - t0
        rate = (ok + fail) / max(elapsed, 1)
        cats_present = [s for s, m in zip(STEMS_6, mask) if m]
        print(f"[{i+1}/{len(sessions)}] {rel}  {status}  "
              f"(ok={ok} fail={fail} {rate:.2f}/s)")

    print(f"\n[done] ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
