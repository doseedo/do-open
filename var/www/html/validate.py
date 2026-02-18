#!/usr/bin/env python3
import json, random, os
from pathlib import Path
from collections import defaultdict
import argparse

COND_KEYS = ["onsets","rframe","rbend","amp","f0","f0_masked"]

def exists_file(p: str) -> bool:
    if not p or not isinstance(p, str): 
        return False
    return Path(p).is_file()

def has_dup_root(p: str) -> bool:
    return isinstance(p, str) and p.count("/mnt/msdd/") > 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("final_training_manifest.json"))
    ap.add_argument("--n", type=int, default=100, help="how many random entries to check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-conditioning", action="store_true", help="only check top-level paths")
    ap.add_argument("--outdir", type=Path, default=Path("verify_logs"))
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    data = json.loads(args.manifest.read_text())
    if not data:
        raise SystemExit("Manifest is empty.")

    k = min(args.n, len(data))
    rng = random.Random(args.seed)
    idxs = rng.sample(range(len(data)), k=k)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Track failures by field -> list of audio_paths
    missing: dict[str, list[str]] = defaultdict(list)
    duproot: dict[str, list[str]] = defaultdict(list)

    # Which fields we check
    fields = ["audio_path","piano_roll_path","latent_path","encodec_path"]

    checked = 0
    for i in idxs:
        e = data[i]
        audio_id = e.get("audio_path") or f"<idx:{i}>"
        # top-level fields
        for f in fields:
            val = e.get(f)
            if has_dup_root(val): duproot[f].append(audio_id)
            if not exists_file(val): missing[f].append(audio_id)

        # conditioning fields
        if not args.skip_conditioning:
            cond = e.get("conditioning_paths") or {}
            for ck in COND_KEYS:
                key = f"cond:{ck}"
                val = cond.get(ck)
                if has_dup_root(val): duproot[key].append(audio_id)
                if not exists_file(val): missing[key].append(audio_id)

        checked += 1

    # Write logs
    def write_list(fname: str, items: list[str]):
        p = args.outdir / fname
        if items:
            p.write_text("\n".join(items) + "\n")
        return p

    print("\n--- Random Path Verification ---")
    print(f"Manifest: {args.manifest}")
    print(f"Entries checked: {checked} (out of {len(data)})")
    print("\nMissing counts:")
    for f in fields + ([] if args.skip_conditioning else [f"cond:{ck}" for ck in COND_KEYS]):
        cnt = len(missing.get(f, []))
        print(f"  {f:<12} : {cnt}")
        if cnt:
            write_list(f"missing_{f.replace(':','_')}.txt", missing[f])

    print("\nDuplicate '/mnt/msdd/' counts:")
    any_dup = False
    for f in fields + ([] if args.skip_conditioning else [f"cond:{ck}" for ck in COND_KEYS]):
        cnt = len(duproot.get(f, []))
        print(f"  {f:<12} : {cnt}")
        if cnt:
            any_dup = True
            write_list(f"duproot_{f.replace(':','_')}.txt", duproot[f])

    print(f"\nLogs written to: {args.outdir.resolve()}")
    if any(len(v)>0 for v in missing.values()):
        print("Tip: open the per-field txt to see which audio entries failed.")

if __name__ == "__main__":
    main()
