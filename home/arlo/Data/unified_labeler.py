#!/usr/bin/env python3
"""
Unified Audio Labeling & Categorization Script

Combines all labeling approaches into a single configurable pipeline:
  1. Filename pattern matching (fast, rule-based)
  2. YAMNet audio classification (slower, content-based)
  3. Subcategory refinement
  4. Manifest generation with filtering

Usage:
  # Full pipeline: pattern match + YAMNet verification
  python unified_labeler.py --input audio_paths.txt --output /path/to/output --mode full

  # Fast mode: pattern matching only (no audio analysis)
  python unified_labeler.py --input audio_paths.txt --output /path/to/output --mode fast

  # YAMNet only: audio content classification
  python unified_labeler.py --input audio_paths.txt --output /path/to/output --mode yamnet

  # Generate training manifest from labeled data
  python unified_labeler.py --input audio_paths.txt --output /path/to/output --manifest manifest.json

Outputs:
  <output>/categorized/<group>.txt           - Paths by group
  <output>/categorized/<group>/<subcat>.txt  - Paths by subgroup
  <output>/yamnet_labels.json                - YAMNet classification results
  <output>/instrument_groups.json            - Structured groups for training
  <output>/manifest.json                     - Training manifest (if requested)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ===================== CONFIGURATION =====================

@dataclass
class LabelConfig:
    """Configuration for labeling pipeline"""

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

    # Categories to EXCLUDE from instrumental training
    exclude_categories: Set[str] = field(default_factory=lambda: {
        "drums", "room", "voice", "percussion", "fx", "click", "undefined"
    })

    audio_extensions: Set[str] = field(default_factory=lambda: {
        ".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".m4a"
    })


# ===================== PATTERN DEFINITIONS =====================

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


# ===================== CLASSIFICATION FUNCTIONS =====================

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


def classify_with_yamnet(audio_path: str, model, class_names: List[str],
                         max_duration: float = 30.0) -> Optional[Dict]:
    """Classify audio content using YAMNet."""
    try:
        import librosa
        import numpy as np
        import tensorflow as tf

        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

        max_samples = int(max_duration * 16000)
        if len(waveform) > max_samples:
            start = (len(waveform) - max_samples) // 2
            waveform = waveform[start:start + max_samples]

        waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
        scores, embeddings, spectrogram = model(waveform_tf)

        mean_scores = np.mean(scores.numpy(), axis=0)
        top_indices = np.argsort(mean_scores)[-10:][::-1]

        predictions = [
            {"class": class_names[i], "score": float(mean_scores[i])}
            for i in top_indices
        ]

        return {
            "predictions": predictions,
            "top_class": predictions[0]["class"],
            "top_score": predictions[0]["score"],
            "num_frames": int(scores.shape[0])
        }

    except Exception as e:
        return {"error": str(e)}


def yamnet_to_group(yamnet_result: Dict, config: LabelConfig) -> Optional[str]:
    """Map YAMNet predictions to training groups."""
    if not yamnet_result or "error" in yamnet_result:
        return None

    yamnet_mapping = {
        "violin": "strings", "fiddle": "strings", "cello": "strings",
        "viola": "strings", "string": "strings",
        "trumpet": "brass", "trombone": "brass", "french horn": "brass",
        "tuba": "brass", "brass": "brass",
        "flute": "winds", "clarinet": "winds", "oboe": "winds",
        "saxophone": "winds", "bassoon": "winds",
        "piano": "piano", "keyboard": "piano", "synthesizer": "piano",
        "electric piano": "piano", "organ": "piano",
        "guitar": "guitar", "acoustic guitar": "guitar",
        "electric guitar": "guitar", "bass guitar": "bass",
        "bass": "bass", "double bass": "bass",
        "banjo": "guitar", "mandolin": "guitar", "ukulele": "guitar", "harp": "guitar",
    }

    for pred in yamnet_result.get("predictions", [])[:5]:
        pred_class = pred["class"].lower()
        for keyword, group in yamnet_mapping.items():
            if keyword in pred_class and group in config.approved_groups:
                return group

    return None


# ===================== MAIN LABELER CLASS =====================

class UnifiedLabeler:
    def __init__(self, output_dir: Path, config: LabelConfig = None):
        self.output_dir = output_dir
        self.config = config or LabelConfig()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.filename_labels: Dict[str, Dict] = {}
        self.yamnet_labels: Dict[str, Dict] = {}
        self._yamnet_model = None
        self._yamnet_classes = None

    def _load_yamnet(self):
        if self._yamnet_model is None:
            import tensorflow as tf
            import tensorflow_hub as hub

            print("Loading YAMNet model...")
            self._yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

            class_map_path = tf.keras.utils.get_file(
                'yamnet_class_map.csv',
                'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
            )
            with open(class_map_path) as f:
                self._yamnet_classes = [line.strip().split(',')[2] for line in f.readlines()[1:]]

            print(f"YAMNet loaded with {len(self._yamnet_classes)} classes")

    def label_by_filename(self, audio_paths: List[str], show_progress: bool = True) -> Dict[str, Dict]:
        results = {}
        iterator = tqdm(audio_paths, desc="Filename labeling") if show_progress else audio_paths

        for path in iterator:
            filename = Path(path).name
            group = classify_by_filename(filename)
            subgroup = get_subcategory(group, filename) if group in self.config.approved_groups else "undefined"

            results[path] = {
                "filename": filename,
                "group": group,
                "subgroup": subgroup,
                "method": "filename"
            }

        self.filename_labels.update(results)
        return results

    def label_by_yamnet(self, audio_paths: List[str], num_workers: int = 4,
                        show_progress: bool = True) -> Dict[str, Dict]:
        self._load_yamnet()
        results = {}

        def process_file(path):
            return path, classify_with_yamnet(path, self._yamnet_model, self._yamnet_classes)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_file, p): p for p in audio_paths}
            iterator = tqdm(as_completed(futures), total=len(futures),
                           desc="YAMNet labeling") if show_progress else as_completed(futures)

            for future in iterator:
                try:
                    path, result = future.result()
                    results[path] = result
                except Exception as e:
                    path = futures[future]
                    results[path] = {"error": str(e)}

        self.yamnet_labels.update(results)
        return results

    def combine_labels(self, use_yamnet_override: bool = True) -> Dict[str, Dict]:
        combined = {}

        for path in set(self.filename_labels.keys()) | set(self.yamnet_labels.keys()):
            filename_label = self.filename_labels.get(path, {})
            yamnet_label = self.yamnet_labels.get(path, {})

            result = dict(filename_label)

            if yamnet_label and "error" not in yamnet_label:
                result["yamnet"] = yamnet_label
                result["yamnet_top_class"] = yamnet_label.get("top_class")
                result["yamnet_score"] = yamnet_label.get("top_score")

                if use_yamnet_override:
                    yamnet_group = yamnet_to_group(yamnet_label, self.config)
                    if yamnet_group and result.get("group") == "undefined":
                        result["group"] = yamnet_group
                        result["subgroup"] = get_subcategory(yamnet_group, Path(path).name)
                        result["method"] = "yamnet_override"

            combined[path] = result

        return combined

    def save_categorized_lists(self, labels: Dict[str, Dict]):
        cat_dir = self.output_dir / "categorized"
        cat_dir.mkdir(exist_ok=True)

        by_group = defaultdict(list)
        by_subgroup = defaultdict(lambda: defaultdict(list))

        for path, label in labels.items():
            group = label.get("group", "undefined")
            subgroup = label.get("subgroup", "undefined")

            by_group[group].append(path)
            if group in self.config.approved_groups:
                by_subgroup[group][subgroup].append(path)

        for group, paths in by_group.items():
            out_file = cat_dir / f"{group}.txt"
            with open(out_file, "w") as f:
                f.write("\n".join(sorted(paths)))

        for group, subgroups in by_subgroup.items():
            group_dir = cat_dir / group
            group_dir.mkdir(exist_ok=True)
            for subgroup, paths in subgroups.items():
                out_file = group_dir / f"{subgroup}.txt"
                with open(out_file, "w") as f:
                    f.write("\n".join(sorted(paths)))

        print(f"Saved categorized lists to {cat_dir}")

    def save_instrument_groups_json(self, labels: Dict[str, Dict]):
        groups = {}

        for group in self.config.approved_groups:
            group_data = {
                "files": [],
                "subgroups": defaultdict(lambda: {"files": [], "count": 0}),
                "total_files": 0
            }

            for path, label in labels.items():
                if label.get("group") == group:
                    group_data["files"].append(path)
                    subgroup = label.get("subgroup", "undefined")
                    group_data["subgroups"][subgroup]["files"].append(path)

            group_data["total_files"] = len(group_data["files"])
            for sub in group_data["subgroups"].values():
                sub["count"] = len(sub["files"])

            group_data["subgroups"] = dict(group_data["subgroups"])

            if group_data["total_files"] > 0:
                groups[group] = group_data

        out_file = self.output_dir / "instrument_groups.json"
        with open(out_file, "w") as f:
            json.dump(groups, f, indent=2)

        print(f"Saved instrument_groups.json ({len(groups)} groups)")
        return groups

    def save_yamnet_labels(self):
        if self.yamnet_labels:
            out_file = self.output_dir / "yamnet_labels.json"
            with open(out_file, "w") as f:
                json.dump(self.yamnet_labels, f, indent=2)
            print(f"Saved YAMNet labels to {out_file}")

    def generate_training_manifest(self, labels: Dict[str, Dict],
                                   output_path: Optional[Path] = None,
                                   exclude_categories: Set[str] = None) -> List[Dict]:
        exclude = exclude_categories or self.config.exclude_categories
        manifest = []

        for path, label in labels.items():
            group = label.get("group")

            if group in exclude or group not in self.config.approved_groups:
                continue

            entry = {
                "audio_path": path,
                "group": group,
                "sub_group": label.get("subgroup", "undefined"),
                "labeling_method": label.get("method", "filename")
            }

            if "yamnet_score" in label:
                entry["yamnet_confidence"] = label["yamnet_score"]

            manifest.append(entry)

        if output_path:
            with open(output_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Saved manifest with {len(manifest)} entries to {output_path}")

        return manifest

    def print_summary(self, labels: Dict[str, Dict]):
        print("\n" + "=" * 60)
        print("LABELING SUMMARY")
        print("=" * 60)

        group_counts = Counter(l.get("group") for l in labels.values())
        method_counts = Counter(l.get("method") for l in labels.values())

        print(f"\nTotal files: {len(labels)}")

        print("\nBy category:")
        for group, count in group_counts.most_common():
            pct = 100 * count / len(labels)
            marker = "*" if group in self.config.approved_groups else " "
            print(f"  {marker} {group}: {count} ({pct:.1f}%)")

        print("\nBy labeling method:")
        for method, count in method_counts.most_common():
            print(f"  {method}: {count}")

        eligible = sum(1 for l in labels.values()
                      if l.get("group") in self.config.approved_groups)
        print(f"\nTraining-eligible (in approved groups): {eligible}")
        print("=" * 60)


# ===================== CLI =====================

def main():
    parser = argparse.ArgumentParser(
        description="Unified audio labeling and categorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  fast   - Filename pattern matching only (no audio loading)
  yamnet - YAMNet audio classification only
  full   - Both filename + YAMNet (recommended)

Examples:
  python unified_labeler.py --input paths.txt --output ./labels --mode fast
  python unified_labeler.py --input paths.txt --output ./labels --mode full --manifest manifest.json
"""
    )

    parser.add_argument("--input", required=True, help="Input audio paths file (.txt) or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["fast", "yamnet", "full"], default="fast",
                        help="Labeling mode (default: fast)")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for YAMNet")
    parser.add_argument("--manifest", type=str, help="Output manifest JSON path")
    parser.add_argument("--yamnet-override", action="store_true", default=True,
                        help="Allow YAMNet to override undefined filename labels")
    parser.add_argument("--skip-excluded", action="store_true", default=True,
                        help="Skip files in excluded categories for YAMNet")

    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file() and input_path.suffix == ".txt":
        with open(input_path) as f:
            audio_paths = [line.strip() for line in f if line.strip()]
    elif input_path.is_dir():
        audio_paths = [str(p) for p in input_path.rglob("*.wav")]
    else:
        raise ValueError(f"Input must be .txt file or directory: {input_path}")

    print(f"Loaded {len(audio_paths)} audio paths")

    output_dir = Path(args.output)
    labeler = UnifiedLabeler(output_dir)

    if args.mode in ["fast", "full"]:
        print("\n[1/3] Filename-based labeling...")
        labeler.label_by_filename(audio_paths)

    if args.mode in ["yamnet", "full"]:
        if args.skip_excluded and args.mode == "full":
            paths_to_analyze = [
                p for p in audio_paths
                if labeler.filename_labels.get(p, {}).get("group") not in labeler.config.exclude_categories
            ]
            print(f"\n[2/3] YAMNet labeling ({len(paths_to_analyze)} files after filtering)...")
        else:
            paths_to_analyze = audio_paths
            print(f"\n[2/3] YAMNet labeling ({len(paths_to_analyze)} files)...")

        if paths_to_analyze:
            labeler.label_by_yamnet(paths_to_analyze, num_workers=args.workers)

    print("\n[3/3] Combining results and saving...")

    if args.mode == "full":
        labels = labeler.combine_labels(use_yamnet_override=args.yamnet_override)
    elif args.mode == "yamnet":
        labeler.label_by_filename(audio_paths, show_progress=False)
        labels = labeler.combine_labels(use_yamnet_override=True)
    else:
        labels = labeler.filename_labels

    labeler.save_categorized_lists(labels)
    labeler.save_instrument_groups_json(labels)

    if labeler.yamnet_labels:
        labeler.save_yamnet_labels()

    if args.manifest:
        labeler.generate_training_manifest(labels, Path(args.manifest))

    labeler.print_summary(labels)


if __name__ == "__main__":
    main()
