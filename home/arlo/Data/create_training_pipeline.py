#!/usr/bin/env python3
"""
Master Training Pipeline Script

Combines labeling + preprocessing + manifest creation into one unified workflow.
Produces a manifest in the exact format used by trainer_performerCN2.py:

{
  "audio_path": "/path/to/audio.wav",
  "piano_roll_path": "/path/to/<stem>.pianoroll.npy",
  "latent_path": "/path/to/<stem>.pt",              # DCAE latents
  "encodec_path": "/path/to/<stem>.pt",             # EnCodec tokens
  "conditioning_paths": {
    "onsets": "/path/to/<stem>.onsets.npy",
    "rframe": "/path/to/<stem>.rframe.npy",
    "rbend": "/path/to/<stem>.rbend.npy",
    "amp": "/path/to/<stem>.amp.npy",
    "f0": "/path/to/<stem>.f0.npy",
    "f0_masked": "/path/to/<stem>.f0_masked.npy"
  },
  "group": "guitar",
  "sub_group": "electric_guitar"
}

Usage:
  # Full pipeline: label + preprocess + create manifest
  python create_training_pipeline.py \
      --audio-list /home/arlo/all_audio_paths_complete.txt \
      --output-dir /mnt/msdd/training_data \
      --manifest /home/arlo/Data/training_manifest.json \
      --gpus 0,1,2,3

  # Label only (no preprocessing)
  python create_training_pipeline.py \
      --audio-list /home/arlo/all_audio_paths_complete.txt \
      --output-dir /mnt/msdd/training_data \
      --label-only

  # Create manifest from existing extracts
  python create_training_pipeline.py \
      --audio-list /home/arlo/all_audio_paths_complete.txt \
      --output-dir /mnt/msdd/training_data \
      --manifest-only \
      --manifest /home/arlo/Data/training_manifest.json

  # Use existing labels, preprocess new files only
  python create_training_pipeline.py \
      --audio-list /home/arlo/all_audio_paths_complete.txt \
      --output-dir /mnt/msdd/training_data \
      --manifest /home/arlo/Data/training_manifest.json \
      --skip-existing
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re


# ===================== CONFIGURATION =====================

@dataclass
class PipelineConfig:
    """Pipeline configuration"""

    # Output structure
    dcae_subdir: str = "dcae_latents"
    encodec_subdir: str = "encodec_tokens"
    piano_roll_subdir: str = "piano_rolls"
    conditioning_subdir: str = "conditioning"

    # Groups matching trainer_performerCN2.py
    approved_groups: List[str] = field(default_factory=lambda: [
        "piano", "guitar", "bass", "strings", "brass", "winds"
    ])

    approved_subgroups: Dict[str, List[str]] = field(default_factory=lambda: {
        "piano":   ["acoustic_piano", "keys", "electric_piano", "undefined"],
        "guitar":  ["acoustic_guitar", "electric_guitar", "plucked", "undefined"],
        "bass":    ["electric_bass", "upright_bass", "synth_bass", "undefined"],
        "strings": ["violin", "viola", "cello", "undefined"],
        "brass":   ["trumpet", "trombone", "french_horn", "tuba", "undefined"],
        "winds":   ["bassoon", "clarinet", "flute", "oboe", "sax", "undefined"],
    })

    # Exclude from training
    exclude_categories: Set[str] = field(default_factory=lambda: {
        "drums", "room", "voice", "percussion", "fx", "click", "undefined",
        "synth", "organ", "mallets", "pad"
    })

    # Conditioning types
    conditioning_types: List[str] = field(default_factory=lambda: [
        "amp", "rframe", "rbend", "f0", "f0_masked", "onsets"
    ])


# ===================== PATTERN MATCHING (from unified_labeler) =====================

INSTRUMENT_PATTERNS = {
    "drums": [
        "kick", "kik", "bd", "bdin", "bdout", "kck",
        "snare", "sn", "snr", "snrtop", "snrbottom",
        "hihat", "hh", "hat", "chh", "ohh", "closedhat", "openhat",
        "tom", "racktom", "floortom", "rtom", "ftom", "floor",
        "overhead", "oh", "ohl", "ohr", "ovh", "ovhl", "ovhr",
        "cymbal", "cym", "crash", "ride", "china", "splash", "bell", "stack",
        "drums", "drum", "drumkit", "kit", "rimshot", "djembe"
    ],
    "room": ["room", "rooml", "roomr", "rml", "rmr", "rm", "amb", "ambient"],
    "bass": ["bass", "bss", "bassamp", "bassdi", "subbass", "bsamp", "bs amp", "bamp"],
    "guitar": ["guitar", "gtr", "git", "gt", "guit", "elecgtr", "acgtr", "egtr", "agt"],
    "piano": ["piano", "key", "keys", "rhodes", "ep", "upright", "grnd", "wurlitzer", "wurl", "wurli", "nord", "pno"],
    "synth": ["synth", "moog", "juno", "prophet", "lead", "bassynth", "pad"],
    "organ": ["organ", "b3", "hammond"],
    "voice": [
        "vox", "vocal", "voice", "ldvox", "vox1", "vox2", "bgvox", "choir",
        "vo", "bgv", "double", "harm", "print", "laud", "dubs", "dbls",
        "adlibs", "libs", "chorus", "soprano", "alto", "tenor", "baritone"
    ],
    "strings": ["violin", "viola", "cello", "string", "str", "ensemble", "vln", "vla", "vc", "vlin", "fiddle"],
    "brass": [
        "trumpet", "tpt", "trombone", "bone", "horn", "flugel", "tuba",
        "trmpt", "tromb", "tb", "tpt", "fh", "hrn", "tp", "brass", "cornet"
    ],
    "winds": ["sax", "tenor", "bari", "flute", "clari", "oboe", "alto", "bassoon", "piccolo", "recorder", "flt", "cl", "ob"],
    "mallets": ["glock", "marimba", "xylo", "vibes", "vibraphone", "bells", "chimes"],
    "plucked": ["banjo", "mandolin", "ukelele", "uke", "harp", "sitar", "dulcimer"],
    "percussion": [
        "perc", "tamb", "clap", "shaker", "conga", "bongo", "cabasa",
        "cowbell", "triangle", "windchime", "timp", "timpani", "belltree", "cajon"
    ],
    "fx": ["fx", "sfx", "sweep", "impact", "boom", "whoosh", "glitch", "echo", "reverb", "riser"],
    "click": ["click", "clk", "metronome", "tempo", "count", "2pop"]
}

SUBCATEGORY_PATTERNS = {
    "brass": {
        "trumpet":     ["trumpet", "tpt", "trmpt", "tp", "cornet", "flugel", "flugelhorn", "flgl"],
        "trombone":    ["trombone", "bone", "tromb", "tb", "tbn", "tbone"],
        "french_horn": ["horn", "fh", "hrn"],
        "tuba":        ["tuba"],
    },
    "winds": {
        "sax":      ["sax", "tenor", "bari", "alto", "soprano"],
        "flute":    ["flute", "flt", "piccolo", "picc", "recorder", "fl"],
        "clarinet": ["clarinet", "clari", "clar", "cl"],
        "oboe":     ["oboe", "obo", "ob"],
        "bassoon":  ["bassoon", "bsn"],
    },
    "strings": {
        "violin": ["violin", "vln", "vlin", "fiddle", "v1", "v2"],
        "viola":  ["viola", "vla"],
        "cello":  ["cello", "violoncello", "cllo", "vc"],
    },
    "guitar": {
        "electric_guitar": ["electric", "elec", "eguitar", "egtr", "elecgtr", "egt", "amp", "di"],
        "acoustic_guitar": ["acoustic", "acgtr", "agtr", "acguitar", "aguitar", "ac", "agt", "neck", "hole", "mic", "ribbon"],
        "plucked":         ["banjo", "mandolin", "uke", "ukelele", "harp"],
    },
    "bass": {
        "electric_bass": ["electric", "elec", "ebass", "di", "bassdi", "amp", "bassamp", "bamp"],
        "upright_bass":  ["upright", "double", "dbl", "acoustic", "arco", "pizz", "bow", "fhole"],
        "synth_bass":    ["synth", "synthbass", "moog"],
    },
    "piano": {
        "acoustic_piano": ["piano", "upright", "grnd", "grand", "pno", "steinway", "yamaha"],
        "keys":           ["keys", "key", "nord", "organ", "hammond", "b3", "clav", "clavinet"],
        "electric_piano": ["rhodes", "wurlitzer", "wurl", "wurli", "ep", "electric", "epiano", "fender"],
    },
}


def normalize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def tokenize(name: str) -> List[str]:
    return re.split(r'[\s_\-.]+', name.lower())


def classify_by_filename(filename: str) -> str:
    """Classify audio file by filename patterns."""
    lower_filename = filename.lower()
    tokens = set(tokenize(filename))
    norm = normalize_name(filename)

    for category, patterns in INSTRUMENT_PATTERNS.items():
        for pat in patterns:
            if ' ' in pat and pat in lower_filename:
                return category
        if any(pat in tokens for pat in patterns):
            return category
        if any(normalize_name(pat) in norm for pat in patterns):
            return category

    return "undefined"


def get_subcategory(group: str, filename: str) -> str:
    """Get subcategory for a file within a group."""
    if group not in SUBCATEGORY_PATTERNS:
        return "undefined"

    lower_filename = filename.lower()
    tokens = set(tokenize(filename))
    norm = normalize_name(filename)

    for subcat, patterns in SUBCATEGORY_PATTERNS[group].items():
        for pat in patterns:
            if ' ' in pat and pat in lower_filename:
                return subcat
        if any(pat in tokens for pat in patterns):
            return subcat
        if any(normalize_name(pat) in norm for pat in patterns):
            return subcat

    return "undefined"


# ===================== PATH RESOLUTION =====================

def get_session_path(audio_path: Path) -> str:
    """Extract session folder structure from audio path."""
    parts = audio_path.parts

    # Try to find markers in path
    for marker in ["gcs-bucket", "protools", "protoolsA"]:
        if marker in parts:
            idx = parts.index(marker)
            # Return everything after the marker except the filename
            return "/".join(parts[idx + 1:-1])

    # Fallback: use parent folder name
    return audio_path.parent.name


def resolve_extract_paths(audio_path: Path, output_dir: Path, config: PipelineConfig) -> Dict[str, Path]:
    """
    Resolve output paths for all extract types.
    Mirrors the structure used in the original training manifest.
    """
    session = get_session_path(audio_path)
    stem = audio_path.stem

    return {
        "latent_path": output_dir / config.dcae_subdir / session / f"{stem}.pt",
        "encodec_path": output_dir / config.encodec_subdir / session / f"{stem}.pt",
        "piano_roll_path": output_dir / config.piano_roll_subdir / session / f"{stem}.pianoroll.npy",
        "conditioning": {
            cond_type: output_dir / config.conditioning_subdir / session / f"{stem}.{cond_type}.npy"
            for cond_type in config.conditioning_types
        }
    }


def check_existing_extracts(paths: Dict) -> Dict[str, bool]:
    """Check which extracts already exist."""
    exists = {
        "latent": paths["latent_path"].exists() if paths.get("latent_path") else False,
        "encodec": paths["encodec_path"].exists() if paths.get("encodec_path") else False,
        "piano_roll": paths["piano_roll_path"].exists() if paths.get("piano_roll_path") else False,
    }

    # Check conditioning
    if "conditioning" in paths:
        exists["conditioning"] = all(
            p.exists() for p in paths["conditioning"].values()
        )

    exists["all"] = all(exists.values())
    return exists


# ===================== MANIFEST CREATION =====================

def create_manifest_entry(
    audio_path: str,
    group: str,
    subgroup: str,
    extract_paths: Dict,
    verify_exists: bool = True
) -> Optional[Dict]:
    """
    Create a single manifest entry in the exact format used by trainer_performerCN2.py.
    Returns None if required files don't exist (when verify_exists=True).
    """
    latent_path = extract_paths.get("latent_path")
    encodec_path = extract_paths.get("encodec_path")
    piano_roll_path = extract_paths.get("piano_roll_path")
    conditioning = extract_paths.get("conditioning", {})

    if verify_exists:
        # Check required files exist
        if not latent_path or not latent_path.exists():
            return None
        if not piano_roll_path or not piano_roll_path.exists():
            return None

    entry = {
        "audio_path": audio_path,
        "piano_roll_path": str(piano_roll_path) if piano_roll_path else None,
        "latent_path": str(latent_path) if latent_path else None,
        "encodec_path": str(encodec_path) if (encodec_path and encodec_path.exists()) else None,
        "conditioning_paths": {},
        "group": group,
        "sub_group": subgroup
    }

    # Add conditioning paths
    for cond_type, cond_path in conditioning.items():
        if cond_path and cond_path.exists():
            entry["conditioning_paths"][cond_type] = str(cond_path)
        else:
            entry["conditioning_paths"][cond_type] = None

    return entry


def create_training_manifest(
    audio_paths: List[str],
    output_dir: Path,
    config: PipelineConfig,
    labels: Dict[str, Dict] = None,
    verify_exists: bool = True
) -> List[Dict]:
    """
    Create training manifest from audio paths.

    Args:
        audio_paths: List of audio file paths
        output_dir: Base output directory for extracts
        config: Pipeline configuration
        labels: Pre-computed labels dict (optional)
        verify_exists: Only include entries where files exist

    Returns:
        List of manifest entries
    """
    manifest = []
    stats = Counter()

    for audio_path in tqdm(audio_paths, desc="Creating manifest"):
        audio_path_obj = Path(audio_path)
        filename = audio_path_obj.name

        # Get labels
        if labels and audio_path in labels:
            group = labels[audio_path].get("group", "undefined")
            subgroup = labels[audio_path].get("subgroup", "undefined")
        else:
            group = classify_by_filename(filename)
            subgroup = get_subcategory(group, filename) if group in config.approved_groups else "undefined"

        # Skip excluded categories
        if group in config.exclude_categories:
            stats["excluded"] += 1
            continue

        # Skip if not in approved groups
        if group not in config.approved_groups:
            stats["not_approved"] += 1
            continue

        # Get extract paths
        extract_paths = resolve_extract_paths(audio_path_obj, output_dir, config)

        # Create entry
        entry = create_manifest_entry(
            audio_path, group, subgroup, extract_paths, verify_exists
        )

        if entry:
            manifest.append(entry)
            stats["included"] += 1
            stats[f"group_{group}"] += 1
        else:
            stats["missing_files"] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("MANIFEST CREATION SUMMARY")
    print("=" * 60)
    print(f"Total audio files: {len(audio_paths)}")
    print(f"Included in manifest: {stats['included']}")
    print(f"Excluded (category): {stats['excluded']}")
    print(f"Excluded (not approved): {stats['not_approved']}")
    print(f"Missing extract files: {stats['missing_files']}")

    print("\nBy group:")
    for group in config.approved_groups:
        count = stats.get(f"group_{group}", 0)
        if count > 0:
            print(f"  {group}: {count}")

    return manifest


# ===================== MAIN PIPELINE =====================

def run_labeling(audio_paths: List[str], output_dir: Path) -> Dict[str, Dict]:
    """Run labeling on audio paths."""
    print("\n" + "=" * 60)
    print("STEP 1: LABELING")
    print("=" * 60)

    labels = {}
    config = PipelineConfig()

    for path in tqdm(audio_paths, desc="Labeling"):
        filename = Path(path).name
        group = classify_by_filename(filename)
        subgroup = get_subcategory(group, filename) if group in config.approved_groups else "undefined"

        labels[path] = {
            "group": group,
            "subgroup": subgroup,
            "filename": filename
        }

    # Save labels
    labels_file = output_dir / "labels.json"
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved labels to {labels_file}")

    # Print summary
    group_counts = Counter(l["group"] for l in labels.values())
    print("\nLabel distribution:")
    for group, count in group_counts.most_common():
        marker = "*" if group in config.approved_groups else " "
        print(f"  {marker} {group}: {count}")

    return labels


def run_preprocessing(
    audio_paths: List[str],
    output_dir: Path,
    labels: Dict[str, Dict],
    gpus: List[int],
    skip_existing: bool = False,
    config: PipelineConfig = None
):
    """
    Run preprocessing using unified_preprocess.py
    """
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING")
    print("=" * 60)

    config = config or PipelineConfig()

    # Filter to only approved groups
    paths_to_process = [
        p for p in audio_paths
        if labels.get(p, {}).get("group") in config.approved_groups
    ]

    print(f"Files to preprocess: {len(paths_to_process)} (filtered from {len(audio_paths)})")

    if not paths_to_process:
        print("No files to preprocess!")
        return

    # Write temp file with paths
    temp_paths_file = output_dir / "paths_to_process.txt"
    with open(temp_paths_file, "w") as f:
        f.write("\n".join(paths_to_process))

    # Call unified_preprocess.py
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "unified_preprocess.py"),
        "--input", str(temp_paths_file),
        "--output", str(output_dir),
        "--gpus", ",".join(map(str, gpus)),
        "--formats", "dcae_latents,conditioning"  # no encodec, no midi/piano_roll
    ]

    if skip_existing:
        cmd.append("--skip-existing")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Master training pipeline: label + preprocess + create manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--audio-list", required=True,
                        help="Input audio paths file (.txt)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for extracts")
    parser.add_argument("--manifest", type=str,
                        help="Output manifest JSON path")
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated GPU IDs (default: 0)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files where extracts already exist")

    # Mode flags
    parser.add_argument("--label-only", action="store_true",
                        help="Only run labeling step")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Only create manifest from existing extracts")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Only run preprocessing (assumes labels exist)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after manifest creation")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing manifest")
    parser.add_argument("--repair", action="store_true",
                        help="Attempt to repair issues during validation")

    # Optional: use existing labels
    parser.add_argument("--labels-json", type=str,
                        help="Use existing labels JSON file")

    args = parser.parse_args()

    # Load audio paths
    audio_list = Path(args.audio_list)
    with open(audio_list) as f:
        audio_paths = [line.strip() for line in f if line.strip() and line.strip().endswith(".wav")]

    print(f"Loaded {len(audio_paths)} audio paths")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig()
    gpus = [int(g) for g in args.gpus.split(",")]

    # Load or compute labels
    if args.labels_json:
        print(f"Loading labels from {args.labels_json}")
        with open(args.labels_json) as f:
            labels = json.load(f)
    elif args.manifest_only:
        # For manifest-only mode, compute labels on the fly
        labels = None
    else:
        labels = run_labeling(audio_paths, output_dir)

    if args.label_only:
        print("\nLabel-only mode complete.")
        return

    # Run preprocessing
    if not args.manifest_only:
        run_preprocessing(
            audio_paths, output_dir, labels or {},
            gpus, args.skip_existing, config
        )

    if args.preprocess_only:
        print("\nPreprocess-only mode complete.")
        return

    # Create manifest
    if args.manifest:
        print("\n" + "=" * 60)
        print("STEP 3: CREATE MANIFEST")
        print("=" * 60)

        manifest = create_training_manifest(
            audio_paths, output_dir, config, labels, verify_exists=True
        )

        manifest_path = Path(args.manifest)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\nSaved manifest with {len(manifest)} entries to {manifest_path}")

    # Run validation if requested
    if args.validate or args.validate_only:
        print("\n" + "=" * 60)
        print("STEP 4: VALIDATION")
        print("=" * 60)

        manifest_to_validate = args.manifest
        if args.validate_only and not manifest_to_validate:
            print("Error: --validate-only requires --manifest")
            return

        run_validation(
            manifest_path=manifest_to_validate,
            output_dir=output_dir,
            repair_mode=args.repair
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


def run_validation(manifest_path: str, output_dir: Path, repair_mode: bool = False):
    """
    Run validation using unified_validator.py
    """
    from unified_validator import UnifiedValidator, write_validation_report

    # Load manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"Validating {len(manifest)} entries...")

    validator = UnifiedValidator(
        training_type="instrumental",
        conditioning_optional=True,
        repair_mode=repair_mode,
        pr_crop_dir=output_dir / "rescued_pianorolls" if repair_mode else None
    )

    results = validator.validate_manifest(manifest)

    # Print summary
    stats = results["stats"]
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total entries:    {stats['total']:,}")
    print(f"Valid entries:    {stats['valid']:,} ({100*stats['valid']/max(1,stats['checked']):.1f}%)")
    print(f"Invalid entries:  {stats['invalid']:,}")
    if repair_mode:
        print(f"Repaired entries: {stats['repaired']:,}")

    if stats["issues"]:
        print("\nTop issues:")
        for issue, count in sorted(stats["issues"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {issue}: {count}")

    # Write reports
    validation_dir = output_dir / "validation_logs"
    write_validation_report(results, validation_dir)

    # Save repaired manifest if in repair mode
    if repair_mode and results["valid_entries"]:
        repaired_path = Path(manifest_path).with_suffix(".repaired.json")
        with open(repaired_path, "w") as f:
            json.dump(results["valid_entries"], f, indent=2)
        print(f"\nRepaired manifest saved to: {repaired_path}")
        print(f"  Entries: {len(results['valid_entries']):,}")


if __name__ == "__main__":
    main()
