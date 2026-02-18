#!/usr/bin/env python3
"""
Build NEW subcategory TXT lists from existing broad-group TXT lists,
classifying **only by filename** (ignoring folder names).

SOURCE (group TXT lists):
    /home/arlo/Data/categorized_instrument_paths/<group>.txt
      - Each line is a full path to an audio file.
      - We use ONLY Path(path).name for subcategory classification.

DEST (new TXT lists):
    /home/arlo/Data/categorized_instrument_paths_subcats_lists/<group>/<subcategory>.txt

Groups subcategorized:
  - brass  → {trumpet, trombone, french_horn, tuba, undefined}
  - winds  → {sax, flute, clarinet, oboe, bassoon, undefined}
  - strings→ {violin, viola, cello, undefined}
  - guitar → {electric_guitar, acoustic_guitar, undefined}
  - bass   → {electric_bass, electric_bass_amp, electric_bass_di,
              upright_bass, upright_bass_arco, upright_bass_pizz, undefined}
  - piano  → {acoustic_piano, keys, electric_piano, undefined}

Other groups (drums, voice, etc.) go to <group>/all.txt.

Behavior:
  - Reads ONLY the TXT lists (source of truth).
  - Does NOT move/change source files.
  - Writes TXT lists that contain original full paths.
  - Rewrites outputs each run (idempotent), de-duplicated & sorted.
"""

from pathlib import Path
import re
from collections import defaultdict, Counter
from typing import Optional
import argparse
import sys

# ====================
# CONFIG (defaults; can be overridden via CLI)
# ====================
SOURCE_LISTS_ROOT = Path("/home/arlo/Data/categorized_instrument_paths")             # existing grouped TXT lists
DEST_LISTS_ROOT   = Path("/home/arlo/Data/categorized_instrument_paths_subcats_lists")# NEW subcategory TXT lists

# Audio extensions we’ll keep (you can allow all by using --include-non-audio)
AUDIO_EXTS = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".m4a"}

# Only subcategorize these groups (per your request).
SUBCATEGORY_PATTERNS = {
    "brass": {
        "trumpet":      ["trumpet", "tpt", "trmpt", "tp", "cornet", "flugel", "flugelhorn", "flgl"],
        "trombone":     ["trombone", "bone", "tromb", "tb", "tbn", "tbone"],
        "french_horn":  ["horn", "fh", "hrn"],
        "tuba":         ["tuba"],
    },
    "winds": {
        "sax":          ["sax", "tenor", "bari", "alto", "soprano"],
        "flute":        ["flute", "flt", "piccolo", "picc", "recorder", "fl"],
        "clarinet":     ["clarinet", "clari", "clar", "cl"],
        "oboe":         ["oboe", "obo", "ob"],
        "bassoon":      ["bassoon"],
    },
    "strings": {
        "violin":       ["violin", "vln", "vlin", "fiddle", "v1", "v2"],
        "viola":        ["viola", "vla"],
        "cello":        ["cello", "violoncello", "cllo"],
    },
    "guitar": {
        "electric_guitar": ["electric", "elec", "eguitar", "egtr", "elecgtr", "egt", "amp", "di"],
        "acoustic_guitar": ["acoustic", "acgtr", "agtr", "acguitar", "aguitar", "ac", "agt", "neck", "hole", "mic", "ribbon", "sm57"],
    },
    "bass": {
        "electric_bass":      ["ebass", "elecbass", "electric", "egtrbass", "di", "bassdi", "amp", "bassamp", "bamp", "guitar"],
        "electric_bass_amp":  ["amp", "bassamp", "bamp"],
        "upright_bass":       ["arco", "trem", "pizz", "bow", "hole", "fhole", "fret", "neck", "body", "upright", "double", "dbl", "acoustic", "mic", "bridge", "SM7b", "sm57", "hollow"],
        "synth_bass":  ["synth", "synthbass"],
    },
    "piano": {
        "acoustic_piano": ["piano", "upright", "grnd", "grand", "pno"],
        "keys":           ["keys", "key", "nord", "organ", "hammond", "b3"],
        "electric_piano": ["rhodes", "wurlitzer", "wurl", "wurli", "ep", "electric", "epiano"],
    },
}

# ============== helpers ==============
def normalize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())

def tokenize(name: str):
    return re.split(r'[\s_\-.]+', name.lower())

def find_subcategory(group: str, filename: str) -> Optional[str]:
    """
    For groups with defined subcats: return subcategory name or "undefined" if no match.
    For groups without subcats: return None.
    NOTE: classification uses ONLY the filename (e.g., 'BL4_TP1.wav').
    """
    if group not in SUBCATEGORY_PATTERNS:
        return None

    lower_filename = filename.lower()
    tokens = set(tokenize(filename))
    norm = normalize_name(filename)

    for subcat, pats in SUBCATEGORY_PATTERNS[group].items():
        # multi-word substring
        for pat in pats:
            p = pat.lower()
            if ' ' in p and p in lower_filename:
                return subcat
        # token exact
        if any(p.lower() in tokens for p in pats):
            return subcat
        # normalized substring
        if any(normalize_name(p) in norm for p in pats):
            return subcat

    return "undefined"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_group_txts(source_root: Path, include_non_audio: bool = False):
    """
    Yield (group_name, list_of_paths) for each <group>.txt under source_root.
    """
    for txt in sorted(source_root.glob("*.txt")):
        group = txt.stem.lower()
        paths = []
        try:
            with open(txt, "r") as f:
                for line in f:
                    p = line.strip()
                    if not p:
                        continue
                    if include_non_audio or Path(p).suffix.lower() in AUDIO_EXTS:
                        paths.append(p)
        except Exception as e:
            print(f"⚠️ Could not read {txt}: {e}")
            continue
        yield group, paths

def write_subcat_lists(source_root: Path, dest_root: Path, include_non_audio: bool = False):
    if not source_root.exists():
        print(f"❌ Source lists root does not exist: {source_root}")
        sys.exit(1)
    ensure_dir(dest_root)

    counts = Counter()
    examined = 0
    groups_seen = set()

    # Accumulate in-memory buckets: { (group, subcat) : set(paths) }
    buckets: dict[tuple[str,str], set] = defaultdict(set)

    for group, path_list in read_group_txts(source_root, include_non_audio=include_non_audio):
        groups_seen.add(group)
        if not path_list:
            continue

        for src_str in path_list:
            examined += 1
            filename = Path(src_str).name  # filename-only classification

            subcat = find_subcategory(group, filename)
            if subcat is None:
                # Groups without subcats → <group>/all.txt
                key = (group, "all")
            else:
                # Groups with subcats → matched subcat or 'undefined'
                key = (group, subcat)

            buckets[key].add(src_str)

    # Write out per-(group/subcat) files, sorted and unique
    for (group, subcat), paths in sorted(buckets.items()):
        out_dir = dest_root / group
        ensure_dir(out_dir)
        out_file = out_dir / f"{subcat}.txt"
        sorted_unique = sorted(paths)
        with open(out_file, "w") as f:
            f.write("\n".join(sorted_unique))
        counts[(group, subcat)] = len(sorted_unique)

    # Summary
    print("\n✅ New subcategory TXT lists built.")
    print(f"   Groups found: {len(groups_seen)}")
    print(f"   Examined paths (from TXT lists): {examined}")
    total_written = sum(counts.values())
    print(f"   Total paths written across all subcat lists: {total_written}")
    if counts:
        print("   Breakdown (group/subcat → count):")
        for (grp, sub), n in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1])):
            print(f"     - {grp}/{sub}: {n}")

# ============== CLI ==============
def main():
    parser = argparse.ArgumentParser(
        description="Create NEW subcategory TXT lists from existing broad-group TXT lists (filename-only classification)."
    )
    parser.add_argument("--source", type=str, default=str(SOURCE_LISTS_ROOT),
                        help="Existing grouped TXT lists directory (source of truth).")
    parser.add_argument("--dest", type=str, default=str(DEST_LISTS_ROOT),
                        help="Destination root for NEW subcategory TXT lists.")
    parser.add_argument("--include-non-audio", action="store_true",
                        help="Also include paths with non-audio extensions.")
    args = parser.parse_args()

    write_subcat_lists(
        source_root=Path(args.source),
        dest_root=Path(args.dest),
        include_non_audio=args.include_non_audio
    )

if __name__ == "__main__":
    main()
