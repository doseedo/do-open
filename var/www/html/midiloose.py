#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from tqdm import tqdm

# --- CONFIG ---
INPUT_JSON       = Path("/home/arlo/Data/midi_matchesnew.json")  # {audio_path: midi_path}
LATENT_ROOTS     = [Path("/mnt/msdd/dcae_latentsnew"),
                    Path("/mnt/msdd/dcae_latentsnew/dcae_latentsnewnew")]
ENCODEC_ROOTS    = [Path("/mnt/msdd/encodec_tokens")]
COND_ROOTS       = [Path("/mnt/msdd/moreconditioning"),
                    Path("/mnt/msdd/newconditioning")]  # <-- both roots here
OUTPUT_JSON      = Path("full_dataset_exact.json")
LOG_DIR          = Path("missing_exact_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

COND_SUFFIXES = {
    ".onsets.npy": "onsets",
    ".rframe.npy": "rframe",
    ".rbend.npy":  "rbend",
    ".amp.npy":    "amp",
    ".f0.npy":     "f0",
    ".f0_masked.npy": "f0_masked",
}

def has_dup_root(p: str) -> bool:
    return p.count("/mnt/msdd/") > 1

def walk_files(root: Path, exts: tuple) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(exts):
                out.append(Path(dp) / fn)
    return out

def index_by_basename(roots: List[Path], exts: tuple, skip_dup_root=False) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = defaultdict(list)
    for r in roots:
        for p in walk_files(r, exts):
            if skip_dup_root and has_dup_root(str(p)):
                continue
            idx[p.name].append(p)
    return idx

def best_by_path_overlap(audio_path: Path, candidates: List[Path]) -> Optional[Path]:
    if not candidates:
        return None
    a_parts = set(x.lower() for x in audio_path.parent.parts)
    best, best_score = None, -1
    for c in candidates:
        score = len(a_parts & set(x.lower() for x in c.parent.parts))
        if score > best_score:
            best, best_score = c, score
    return best

def main():
    if not INPUT_JSON.exists():
        print(f"❌ Input JSON not found: {INPUT_JSON}")
        return
    midi_matches = json.loads(INPUT_JSON.read_text())
    print(f"Loaded {len(midi_matches):,} audio→MIDI pairs")

    print("Indexing latents (.pt)…")
    latent_idx  = index_by_basename(LATENT_ROOTS, (".pt",))
    print(f"  latents indexed: {sum(len(v) for v in latent_idx.values()):,}")

    print("Indexing encodec (.pt)…")
    encodec_idx = index_by_basename(ENCODEC_ROOTS, (".pt",))
    print(f"  encodec indexed: {sum(len(v) for v in encodec_idx.values()):,}")

    print("Indexing conditioning (.npy) from moreconditioning + newconditioning…")
    cond_idx    = index_by_basename(COND_ROOTS, tuple(COND_SUFFIXES.keys()), skip_dup_root=True)
    print(f"  conditioning indexed: {sum(len(v) for v in cond_idx.values()):,}")

    miss_latent: List[str] = []
    miss_encodec: List[str] = []
    miss_by_key = {k: [] for k in COND_SUFFIXES.values()}

    out_entries = []
    for audio_path_str, midi_path in tqdm(list(midi_matches.items()), desc="Matching"):
        ap = Path(audio_path_str)
        stem = ap.stem

        # exact filenames we expect
        latent_best  = best_by_path_overlap(ap, latent_idx.get(f"{stem}.pt", []))
        encodec_best = best_by_path_overlap(ap, encodec_idx.get(f"{stem}.pt", []))

        cond_paths: Dict[str, Optional[str]] = {}
        for suf, key in COND_SUFFIXES.items():
            fname = f"{stem}{suf}"
            chosen = best_by_path_overlap(ap, cond_idx.get(fname, []))
            cond_paths[key] = str(chosen) if chosen else None

        if latent_best is None:
            miss_latent.append(audio_path_str)
        if encodec_best is None:
            miss_encodec.append(audio_path_str)
        for k in COND_SUFFIXES.values():
            if not cond_paths[k]:
                miss_by_key[k].append(audio_path_str)

        out_entries.append({
            "audio_path": audio_path_str,
            "midi_path": midi_path,
            "latent_path":  str(latent_best)  if latent_best  else None,
            "encodec_path": str(encodec_best) if encodec_best else None,
            "conditioning_paths": cond_paths,
        })

    OUTPUT_JSON.write_text(json.dumps(out_entries, indent=2))
    print(f"✅ Wrote {len(out_entries):,} entries → {OUTPUT_JSON.resolve()}")

    (LOG_DIR / "missing_latent.txt").write_text("\n".join(miss_latent) + ("\n" if miss_latent else ""))
    (LOG_DIR / "missing_encodec.txt").write_text("\n".join(miss_encodec) + ("\n" if miss_encodec else ""))
    for k in COND_SUFFIXES.values():
        (LOG_DIR / f"missing_{k}.txt").write_text("\n".join(miss_by_key[k]) + ("\n" if miss_by_key[k] else ""))

    print("\n--- Missing summary (exact filenames) ---")
    print(f"latent:  {len(miss_latent):,}")
    print(f"encodec: {len(miss_encodec):,}")
    for k in COND_SUFFIXES.values():
        print(f"{k:8s}: {len(miss_by_key[k]):,}")

if __name__ == "__main__":
    main()
