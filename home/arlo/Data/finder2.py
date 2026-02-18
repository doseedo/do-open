#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# --- CONFIG ---
MANIFEST = Path("final_training_manifest.json")
OUT_MANIFEST = Path("final_training_manifest_repaired_pianorolls.json")
LOG_MISSING = Path("log_pianoroll_still_missing.txt")

PIANO_ROLL_ROOT = Path("/mnt/msdd/piano_rolls")

# If you want to also fix entries whose piano_roll_path is non-null but points to a missing file, set:
REPAIR_BROKEN_EXISTING = True


def build_pianoroll_index(root: Path):
    """
    Walk PIANO_ROLL_ROOT and build a mapping:
        filename (e.g., 'Beat_organ.R.pianoroll.npy') -> [full_paths...]
    Multiple matches possible (rare). We’ll prefer the shortest path depth later.
    """
    index = defaultdict(list)
    total_dirs = sum(1 for _ in os.walk(root))
    for dirpath, _, filenames in tqdm(os.walk(root), total=total_dirs, desc="Indexing piano rolls"):
        for fn in filenames:
            if fn.endswith(".pianoroll.npy"):
                full = str(Path(dirpath) / fn)
                index[fn].append(full)
    return index


def choose_best_candidate(candidates):
    """
    If multiple files share the same filename, pick the one with the shallowest path
    (heuristic: tends to be the cleaner session folder).
    """
    if not candidates:
        return None
    # Sort by path depth, then lexicographically for determinism
    candidates = sorted(candidates, key=lambda p: (p.count(os.sep), p))
    return candidates[0]


def guess_session_folders_from_audio(audio_path: Path):
    """
    Heuristics to guess the session folder name used under /mnt/msdd/piano_rolls/.
    Returns a list of possible folder names ordered by likelihood.
    """
    parts = audio_path.parts
    guesses = []

    # 1) The folder after 'Mixes/' (e.g., .../Mixes/PRO_BRAINMELT MIX/Audio Files/..)
    if "Mixes" in parts:
        i = parts.index("Mixes")
        if i + 1 < len(parts):
            guesses.append(parts[i + 1])

    # 2) The last folder before 'Audio Files'
    if "Audio Files" in parts:
        j = parts.index("Audio Files")
        if j - 1 >= 0:
            guesses.append(parts[j - 1])

    # 3) The folder right after 'New' or 'Prev'
    for key in ("New", "Prev"):
        if key in parts:
            k = parts.index(key)
            if k + 1 < len(parts):
                guesses.append(parts[k + 1])

    # 4) The top-level project-ish folder: try to find a folder name with spaces/caps close to project name
    # (Often useful, but we’ll dedup and let filename-indexing handle most cases)
    # No extra special handling here; the index will cover this.

    # Dedup while preserving order
    seen = set()
    uniq = []
    for g in guesses:
        if g not in seen and g.strip():
            uniq.append(g)
            seen.add(g)

    return uniq


def try_heuristic_paths(audio_path: Path, root: Path, filename: str):
    """
    Try a handful of deterministic paths based on audio_path heuristics.
    """
    for sess in guess_session_folders_from_audio(audio_path):
        candidate = root / sess / filename
        if candidate.exists():
            return str(candidate)
    # Also try at the root (flat)
    flat = root / filename
    if flat.exists():
        return str(flat)
    return None


def find_pianoroll_for_entry(entry, index):
    """
    Given a manifest entry, try to resolve the piano_roll_path.
    Strategy:
        - If piano_roll_path exists and file is there -> keep it.
        - Else try exact filename in index.
        - Else try heuristic folder guesses.
    """
    audio_path_str = entry.get("audio_path")
    if not audio_path_str:
        return None

    audio_path = Path(audio_path_str)
    stem = audio_path.stem  # e.g., 'Beat_organ.R'
    pr_filename = f"{stem}.pianoroll.npy"

    # 1) If non-null path present and exists (or we allow repairing broken existing)
    current = entry.get("piano_roll_path")
    if current and Path(current).exists():
        return current
    elif current and not Path(current).exists() and not REPAIR_BROKEN_EXISTING:
        # Keep the current broken path if we’re not allowed to repair
        return current

    # 2) Index lookup by filename anywhere under root
    candidates = index.get(pr_filename, [])
    best = choose_best_candidate(candidates)
    if best:
        return best

    # 3) Heuristic path guesses from audio structure
    guessed = try_heuristic_paths(audio_path, PIANO_ROLL_ROOT, pr_filename)
    if guessed:
        return guessed

    # Not found
    return None


def main():
    if not MANIFEST.exists():
        print(f"Error: Manifest not found at {MANIFEST}")
        return

    print(f"Loading manifest: {MANIFEST}")
    data = json.load(MANIFEST.open())

    # Build filename index of all piano rolls
    index = build_pianoroll_index(PIANO_ROLL_ROOT)

    updated = 0
    still_missing = []

    # We’ll process all entries, fixing null and optionally broken paths
    for entry in tqdm(data, desc="Repairing piano_roll_path"):
        pr = entry.get("piano_roll_path")
        needs_fix = (pr is None) or (REPAIR_BROKEN_EXISTING and pr and not Path(pr).exists())
        if not needs_fix:
            continue

        fixed = find_pianoroll_for_entry(entry, index)
        if fixed:
            entry["piano_roll_path"] = fixed
            updated += 1
        else:
            entry["piano_roll_path"] = None
            still_missing.append(entry.get("audio_path", "<unknown>"))

    # Save results
    OUT_MANIFEST.write_text(json.dumps(data, indent=4))
    print(f"✅ Wrote updated manifest to: {OUT_MANIFEST.resolve()}")
    print(f"   Updated entries: {updated}")

    if still_missing:
        LOG_MISSING.write_text("\n".join(still_missing))
        print(f"❌ {len(still_missing)} entries still missing a piano roll. Logged to: {LOG_MISSING.resolve()}")


if __name__ == "__main__":
    main()
