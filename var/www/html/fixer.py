#!/usr/bin/env python3
"""
Fix conditioning paths in a training manifest.

Key features:
- Full-path exact match (not just basename).
- Deterministic mirroring from /protools*/... into /mnt/msdd/moreconditioning/...
- Path normalization (repairs double-root issues like .../moreconditioning/mnt/msdd/moreconditioning/...).
- Fallback to filename-index + path-overlap heuristic only when the deterministic path is missing.
- Optional unsliced fallback (disabled by default) to avoid duplicates across slices.
- Duplicate reporting per conditioning key.

Usage:
  python fix_conditioning_paths.py \
      --input final_training_manifest.json \
      --output final_training_manifest.fixed_cond.json \
      --log-dir cond_fix_logs \
      --allow-unsliced-fallback  # (optional; off by default)

"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# ---------- Defaults (override with CLI) ----------
DEFAULT_INPUT  = "final_training_manifest.json"
DEFAULT_OUTPUT = "final_training_manifest.fixed_cond.json"
DEFAULT_LOGDIR = "cond_fix_logs"

# Pro Tools roots (compute relative from these)
DEFAULT_PROTOOLS_ROOTS = [
    "/home/arlo/gcs-bucket/protools",
    "/home/arlo/gcs-bucket/protoolsA",
]

# Conditioning roots
DEFAULT_MORECOND_ROOT = "/mnt/msdd/evenmoreconditioning"
DEFAULT_NEWCOND_ROOT  = "/mnt/msdd/newconditioning"

# Conditioning filename suffixes
COND_SUFFIX: Dict[str, str] = {
    "onsets":    ".onsets.npy",
    "rframe":    ".rframe.npy",
    "rbend":     ".rbend.npy",
    "amp":       ".amp.npy",
    "f0":        ".f0.npy",
    "f0_masked": ".f0_masked.npy",
}


# ---------- Helpers ----------
def norm_path(p: Path) -> Path:
    """
    Normalize a path and collapse common duplicate-prefix mistakes.
    e.g. /mnt/msdd/moreconditioning/mnt/msdd/moreconditioning/... -> /mnt/msdd/moreconditioning/...
    """
    s = str(p)

    # Collapse duplicate segments for both moreconditioning and newconditioning
    patterns = [
        ("/mnt/msdd/evenmoreconditioning/mnt/msdd/evenmoreconditioning/", "/mnt/msdd/evenmoreconditioning/"),
        ("/mnt/msdd/newconditioning/mnt/msdd/newconditioning/", "/mnt/msdd/newconditioning/"),
        # Guard against accidental double slashes repeated:
        ("/mnt/msdd/evenmoreconditioning//", "/mnt/msdd/evenmoreconditioning/"),
        ("/mnt/msdd/newconditioning//", "/mnt/msdd/newconditioning/"),
    ]
    changed = True
    while changed:
        changed = False
        for bad, good in patterns:
            if bad in s:
                s = s.replace(bad, good)
                changed = True

    # Normalize .. and .
    s = os.path.normpath(s)

    return Path(s)


def walk_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(exts):
                out.append(Path(dp) / fn)
    return out


def build_morecond_index(morecond_root: Path) -> Dict[str, List[Path]]:
    """
    Index /moreconditioning by filename only, e.g. "GTR Amp.02_03.onsets.npy" -> [Path, ...].
    """
    exts = tuple(COND_SUFFIX.values())
    idx: Dict[str, List[Path]] = defaultdict(list)
    all_files = walk_files(morecond_root, exts)
    for p in all_files:
        idx[p.name].append(p)
    return idx


def rel_to_any_root(ap: Path, roots: List[Path]) -> Optional[Path]:
    for r in roots:
        try:
            return ap.relative_to(r)
        except ValueError:
            continue
    return None


def expected_cond_path(audio_path: Path, suffix: str, morecond_root: Path, prot_roots: List[Path]) -> Optional[Path]:
    """
    Mirror /protools*/<rel>/Audio Files/<stem>.wav  -> /moreconditioning/<rel>/Audio Files/<stem><suffix>
    """
    rel = rel_to_any_root(audio_path, prot_roots)
    if rel is None:
        return None
    return morecond_root / rel.parent / f"{audio_path.stem}{suffix}"


def best_by_path_overlap(audio_path: Path, candidates: List[Path]) -> Optional[Path]:
    """
    Choose the candidate whose parent path shares the most components with the audio path's parent.
    """
    if not candidates:
        return None
    a = set(x.lower() for x in audio_path.parent.parts)
    best, best_score = None, -1
    for c in candidates:
        score = len(a & set(x.lower() for x in c.parent.parts))
        if score > best_score:
            best, best_score = c, score
    return best


SLICE_RE = re.compile(r"\.\d+_\d+$")  # matches ".01_02" at end of stem

def stem_base_and_sliced(stem: str) -> Tuple[str, bool]:
    """
    Return (base_name, is_sliced). For 'GTR Amp.02_03' -> ('GTR Amp', True)
    For 'Piano' -> ('Piano', False)
    """
    m = SLICE_RE.search(stem)
    if m:
        return stem[:m.start()], True
    return stem, False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix conditioning paths in a training manifest.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input manifest JSON")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output manifest JSON")
    p.add_argument("--log-dir", default=DEFAULT_LOGDIR, help="Directory for logs")
    p.add_argument("--morecond-root", default=DEFAULT_MORECOND_ROOT, help="Root of correct conditioning files")
    p.add_argument("--newcond-root", default=DEFAULT_NEWCOND_ROOT, help="Legacy/bad conditioning root (only for normalization)")
    p.add_argument("--protools-root", action="append",
                   default=DEFAULT_PROTOOLS_ROOTS,
                   help="Add a Pro Tools root. Can be specified multiple times.")
    p.add_argument("--allow-unsliced-fallback", action="store_true",
                   help="Allow mapping sliced audio to an unsliced conditioning file if exact slice is missing (may create duplicates). Default: disabled.")
    p.add_argument("--dry-run", action="store_true", help="Process but do not write output manifest.")
    return p.parse_args()


def report_dupes(entries: List[dict], log_dir: Path) -> Dict[str, int]:
    """
    Write dupes_<key>.txt with path :: count. Return a summary dict with number of unique dupes per key.
    """
    summary = {}
    for key in COND_SUFFIX.keys():
        c = Counter(
            (entry.get("conditioning_paths") or {}).get(key)
            for entry in entries
            if (entry.get("conditioning_paths") or {}).get(key)
        )
        dupes = {k: v for k, v in c.items() if k and v > 1}
        (log_dir / f"dupes_{key}.txt").write_text(
            "\n".join(f"{path}  ::  {count}" for path, count in sorted(dupes.items(), key=lambda x: -x[1]))
        )
        summary[key] = len(dupes)
    return summary


def main():
    args = parse_args()
    INPUT_MANIFEST  = Path(args.input)
    OUTPUT_MANIFEST = Path(args.output)
    LOG_DIR         = Path(args.log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    MORECOND_ROOT = norm_path(Path(args.morecond_root))
    NEWCOND_ROOT  = norm_path(Path(args.newcond_root))
    PROTOOLS_ROOTS = [norm_path(Path(p)) for p in args.protools_root]

    if not INPUT_MANIFEST.exists():
        print(f"❌ Manifest not found: {INPUT_MANIFEST}")
        return

    raw_text = INPUT_MANIFEST.read_text()
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return

    if not isinstance(data, list):
        print("❌ Manifest root must be a JSON list of entries.")
        return

    print(f"Loaded entries: {len(data):,}")
    print(f"Indexing {MORECOND_ROOT} (once)…")
    mc_index = build_morecond_index(MORECOND_ROOT)
    total_indexed = sum(len(v) for v in mc_index.values())
    print(f"  Indexed conditioning files: {total_indexed:,}")

    # Metrics
    normalized_only = 0
    replaced_exact_mirror = 0
    replaced_index_exact = 0
    replaced_index_unsliced = 0
    skipped_exact_full = 0
    unresolved = 0

    unresolved_by_key: Dict[str, List[str]] = {k: [] for k in COND_SUFFIX.keys()}
    change_log_lines: List[str] = []

    # Backup (true snapshot of original input)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = INPUT_MANIFEST.with_suffix(f".backup_{ts}.json")
    backup_path.write_text(raw_text)

    for entry in tqdm(data, desc="Fixing conditioning paths"):
        ap_str = entry.get("audio_path", "")
        if not ap_str:
            continue
        ap = norm_path(Path(ap_str))
        cond_paths: Dict[str, str] = (entry.get("conditioning_paths") or {}).copy()

        for key, suffix in COND_SUFFIX.items():
            cur = cond_paths.get(key)
            if not cur:
                continue

            # Normalize current path first (may fix double-prefix)
            try:
                cur_norm = str(norm_path(Path(cur)))
            except Exception:
                cur_norm = cur

            changed_now = (cur_norm != cur)
            if changed_now:
                cond_paths[key] = cur_norm

            # If already equals the exact expected full path, skip
            exp = expected_cond_path(ap, suffix, MORECOND_ROOT, PROTOOLS_ROOTS)
            if exp is not None and norm_path(Path(cur_norm)) == norm_path(exp):
                if changed_now:
                    normalized_only += 1
                else:
                    skipped_exact_full += 1
                continue

            # Prefer the deterministic mirrored destination if it exists
            if exp is not None and exp.exists():
                if cur_norm != str(norm_path(exp)):
                    change_log_lines.append(f"{key}: {cur_norm}  -->  {exp}")
                    cond_paths[key] = str(norm_path(exp))
                    replaced_exact_mirror += 1
                else:
                    # matched by normalization equality earlier (unlikely), treat as normalized
                    normalized_only += 1
                continue

            # Fallback 1: filename index with exact sliced name
            expected_name = f"{ap.stem}{suffix}"
            candidates = mc_index.get(expected_name, [])
            chosen = best_by_path_overlap(ap, candidates)

            if chosen:
                if cur_norm != str(norm_path(chosen)):
                    change_log_lines.append(f"{key}: {cur_norm}  -->  {chosen}")
                    cond_paths[key] = str(norm_path(chosen))
                    replaced_index_exact += 1
                else:
                    normalized_only += 1
                continue

            # Fallback 2 (optional): unsliced filename (e.g., GTR Amp.onsets.npy)
            base, is_sliced = stem_base_and_sliced(ap.stem)
            if args.allow_unsliced_fallback and is_sliced:
                unsliced_name = f"{base}{suffix}"
                candidates2 = mc_index.get(unsliced_name, [])
                chosen2 = best_by_path_overlap(ap, candidates2)
                if chosen2:
                    if cur_norm != str(norm_path(chosen2)):
                        change_log_lines.append(f"{key}: {cur_norm}  -->  {chosen2}  [UNSLICED]")
                        cond_paths[key] = str(norm_path(chosen2))
                        replaced_index_unsliced += 1
                    else:
                        normalized_only += 1
                    continue

            # If we reach here, unresolved
            unresolved += 1
            unresolved_by_key[key].append(str(ap))

        entry["conditioning_paths"] = cond_paths  # write back

    # Write output (unless dry-run)
    if not args.dry_run:
        OUTPUT_MANIFEST.write_text(json.dumps(data, indent=2))

    # Logs
    changes_path = Path(args.log_dir) / "changes.txt"
    changes_path.write_text("\n".join(change_log_lines))

    for k, lst in unresolved_by_key.items():
        (Path(args.log_dir) / f"unresolved_{k}.txt").write_text("\n".join(lst) + ("\n" if lst else ""))

    # Duplicate report
    dupe_summary = report_dupes(data, Path(args.log_dir))

    # Summary
    print("\n--- Conditioning Fix Summary ---")
    print(f"Entries:                         {len(data):,}")
    print(f"Normalized-only (fixed paths):   {normalized_only:,}")
    print(f"Replaced (exact mirror exists):  {replaced_exact_mirror:,}")
    print(f"Replaced (index exact name):     {replaced_index_exact:,}")
    print(f"Replaced (index unsliced):       {replaced_index_unsliced:,}")
    print(f"Skipped (already exact full):    {skipped_exact_full:,}")
    print(f"Unresolved (see logs):           {unresolved:,}")
    print(f"📝 Backup of original:           {backup_path.resolve()}")
    if not args.dry_run:
        print(f"✅ Output manifest:              {Path(args.output).resolve()}")
    print(f"🧾 Changes log:                  {changes_path.resolve()}")
    print(f"🗂️  Unresolved logs dir:          {Path(args.log_dir).resolve()}")
    print("\n--- Duplicate Counts (unique duplicate paths) ---")
    for key, n in dupe_summary.items():
        print(f"  {key:10s}: {n} (see {Path(args.log_dir) / f'dupes_{key}.txt'})")


if __name__ == "__main__":
    main()