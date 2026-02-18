#!/usr/bin/env python3
import json, os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# --- CONFIGURATION ---
MANIFEST_PATH = Path("final_training_manifest.json")

# CHANGED: look in /moreconditioning now
COND_ROOT = Path("/mnt/msdd/evenmoreconditioning")   # conditioning files (.npy)
ENCODEC_ROOT = Path("/mnt/msdd/encodec_tokens")  # encodec files (.pt)

WRITE_BACKUP = True
DRY_RUN = False

COND_KEYS = ["onsets", "rframe", "rbend", "amp", "f0", "f0_masked"]
COND_SUFFIX = {
    "onsets": ".onsets.npy",
    "rframe": ".rframe.npy",
    "rbend": ".rbend.npy",
    "amp": ".amp.npy",
    "f0": ".f0.npy",
    "f0_masked": ".f0_masked.npy",
}

def walk_files(root: Path) -> List[Path]:
    out = []
    if root.exists():
        for dp, _, files in os.walk(root):
            for fn in files:
                out.append(Path(dp) / fn)
    return out

def build_cond_lookup(root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Index npy files by exact base filename (without the conditioning suffix).
    E.g. 'GTR DI.15_09.amp.npy' -> stem 'GTR DI.15_09'
    """
    by_stem = defaultdict(dict)
    for p in walk_files(root):
        if not str(p).lower().endswith(".npy"):
            continue
        for k, suf in COND_SUFFIX.items():
            if p.name.endswith(suf):
                stem = p.name[:-len(suf)]
                by_stem[stem][k] = p
    return by_stem

def build_encodec_lookup(root: Path) -> Dict[str, Path]:
    out = {}
    for p in walk_files(root):
        if p.suffix == ".pt":
            out[p.stem] = p
    return out

def load_manifest(path: Path) -> List[dict]:
    with path.open("r") as f:
        return json.load(f)

def save_manifest(path: Path, data: List[dict]):
    tmp = path.with_suffix(".tmp.json")
    with tmp.open("w") as f:
        json.dump(data, f, indent=4)
    tmp.replace(path)

def main():
    print(f"🔎 Scanning conditioning root: {COND_ROOT}")
    print(f"🔎 Scanning encodec root:      {ENCODEC_ROOT}")

    data = load_manifest(MANIFEST_PATH)
    cond_lookup = build_cond_lookup(COND_ROOT)
    encodec_lookup = build_encodec_lookup(ENCODEC_ROOT)

    total, touched = 0, 0
    fixed_by_key, encodec_fixed = Counter(), 0
    changed_examples: List[Tuple[str, Dict[str, str]]] = []

    for entry in tqdm(data, desc="Repairing conditioning"):
        audio_path = entry.get("audio_path") or ""
        if not audio_path:
            continue
        total += 1
        stem = Path(audio_path).stem  # exact base filename match

        cond = entry.get("conditioning_paths") or {k: None for k in COND_KEYS}
        repaired = False

        cand_map = cond_lookup.get(stem, {})
        for k in COND_KEYS:
            if not cond.get(k):
                p = cand_map.get(k)
                if p and p.exists():
                    cond[k] = str(p)
                    fixed_by_key[k] += 1
                    repaired = True

        prev_encodec = entry.get("encodec_path")
        if not prev_encodec:
            enc = encodec_lookup.get(stem)
            if enc and enc.exists():
                entry["encodec_path"] = str(enc)
                encodec_fixed += 1
                repaired = True

        if repaired:
            entry["conditioning_paths"] = cond
            changed_examples.append((audio_path, cond))
            touched += 1

    print("\n--- Repair Summary ---")
    print(f"Entries scanned: {total}")
    print(f"Entries updated: {touched}")
    if encodec_fixed:
        print(f"Encodec paths fixed: {encodec_fixed}")
    for k in COND_KEYS:
        print(f"  - {k}: {fixed_by_key[k]}")

    if touched and not DRY_RUN:
        if WRITE_BACKUP:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = MANIFEST_PATH.with_suffix(f".backup_{ts}.json")
            with backup_path.open("w") as bf:
                json.dump(data, bf, indent=4)
            print(f"Backup written: {backup_path.resolve()}")
        save_manifest(MANIFEST_PATH, data)
        print(f"✅ Updated manifest: {MANIFEST_PATH.resolve()}")
    elif DRY_RUN and touched:
        print("DRY_RUN active — changes not saved.")
    else:
        print("No changes needed.")

    if changed_examples:
        print("\nExamples of repaired entries:")
        for ap, cond in changed_examples[:5]:
            print(f"  * {ap}")
            for k in COND_KEYS:
                print(f"     {k}: {cond.get(k)}")

if __name__ == "__main__":
    main()
