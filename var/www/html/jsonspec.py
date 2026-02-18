#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
FINAL_MANIFEST_JSON = Path("final_training_manifest_final.json")
OUTPUT_GRAPH_PATH   = Path("dataset_specs_by_group.png")

# Where to write path lists for missing items
LOG_DIR = Path(".")  # change if you want them in a different folder

# Conditioning keys we care about
COND_KEYS = ["onsets", "rframe", "rbend", "amp", "f0", "f0_masked"]

# --- SCRIPT LOGIC ---

def analyze_manifest(manifest_path: Path):
    """
    Reads the final manifest, calculates statistics per instrument group and subgroup,
    prints a detailed report, and writes path logs for each missing type:
      - missing_encodec_paths.txt
      - missing_pianoroll_paths.txt
      - missing_<condkey>_paths.txt  (for each conditioning key)
    """
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}")
        return None

    print(f"📊 Analyzing dataset from: {manifest_path}")
    with open(manifest_path, 'r') as f:
        data = json.load(f)

    if not data:
        print("Manifest is empty. No data to analyze.")
        return None

    # stats[group] structure
    def subgroup_bucket():
        return {
            'total': 0,
            'complete': 0,
            'incomplete': 0,
            'missing_by_key': defaultdict(int),  # conditioning keys
            'missing_encodec': 0,
            'missing_pianoroll': 0,
        }

    def group_bucket():
        return {
            'total': 0,
            'complete': 0,
            'incomplete': 0,
            'missing_by_key': defaultdict(int),  # conditioning keys
            'missing_encodec': 0,
            'missing_pianoroll': 0,
            'subgroups': defaultdict(subgroup_bucket)
        }

    stats = defaultdict(group_bucket)

    # NEW: path collectors for missing items
    missing_encodec_paths   = []
    missing_pianoroll_paths = []
    missing_cond_paths      = {k: [] for k in COND_KEYS}

    # 1) Aggregate
    for entry in tqdm(data, desc="Analyzing entries"):
        group = entry.get("group")
        sub_group = entry.get("sub_group")
        audio_path = entry.get("audio_path") or "<unknown>"

        if not group:
            # still collect missing paths even if group is absent
            cond = entry.get("conditioning_paths") or {}
            encodec_ok = bool(entry.get("encodec_path"))
            pr_ok = bool(entry.get("piano_roll_path"))
            if not encodec_ok:
                missing_encodec_paths.append(audio_path)
            if not pr_ok:
                missing_pianoroll_paths.append(audio_path)
            for k in COND_KEYS:
                if not cond.get(k):
                    missing_cond_paths[k].append(audio_path)
            continue

        cond = entry.get("conditioning_paths") or {}
        encodec_ok = bool(entry.get("encodec_path"))
        pr_ok = bool(entry.get("piano_roll_path"))
        cond_ok = cond and all(cond.get(k) for k in COND_KEYS)

        is_complete = bool(encodec_ok and pr_ok and cond_ok)

        # Update group totals
        g = stats[group]
        g['total'] += 1
        if is_complete:
            g['complete'] += 1
        else:
            g['incomplete'] += 1
            # Which pieces are missing?
            if not encodec_ok:
                g['missing_encodec'] += 1
                missing_encodec_paths.append(audio_path)
            if not pr_ok:
                g['missing_pianoroll'] += 1
                missing_pianoroll_paths.append(audio_path)
            for k in COND_KEYS:
                if not cond.get(k):
                    g['missing_by_key'][k] += 1
                    missing_cond_paths[k].append(audio_path)

        # Subgroup
        if sub_group:
            s = g['subgroups'][sub_group]
            s['total'] += 1
            if is_complete:
                s['complete'] += 1
            else:
                s['incomplete'] += 1
                if not encodec_ok:
                    s['missing_encodec'] += 1
                if not pr_ok:
                    s['missing_pianoroll'] += 1
                for k in COND_KEYS:
                    if not cond.get(k):
                        s['missing_by_key'][k] += 1

    # 2) Report
    print("\n--- Dataset Specification Report ---")
    print(f"{'Category':<25} | {'Total Files':>12} | {'Complete':>12} | {'Incomplete':>12}")
    print("-" * 70)

    total_files = 0
    total_complete = 0
    total_incomplete = 0
    total_missing_by_key = defaultdict(int)
    total_missing_encodec = 0
    total_missing_pianoroll = 0

    sorted_groups = sorted(stats.items(), key=lambda item: item[1]['total'], reverse=True)

    for group, g in sorted_groups:
        print(f"{group.upper():<25} | {g['total']:>12,} | {g['complete']:>12,} | {g['incomplete']:>12,}")
        total_files += g['total']
        total_complete += g['complete']
        total_incomplete += g['incomplete']

        # Aggregate totals
        total_missing_encodec += g['missing_encodec']
        total_missing_pianoroll += g['missing_pianoroll']
        for k in COND_KEYS:
            total_missing_by_key[k] += g['missing_by_key'].get(k, 0)

        # Group-level missing details
        if g['incomplete'] > 0:
            print(f"    Missing encodec: {g['missing_encodec']:,} | Missing piano_roll: {g['missing_pianoroll']:,}")
            miss_line = "    Missing conditioning -> " + ", ".join(
                f"{k}: {g['missing_by_key'].get(k, 0):,}" for k in COND_KEYS
            )
            print(miss_line)

        # Subgroups
        sorted_subgroups = sorted(g['subgroups'].items(), key=lambda item: item[1]['total'], reverse=True)
        for sname, s in sorted_subgroups:
            print(f"  - {sname:<22} | {s['total']:>12,} | {s['complete']:>12,} | {s['incomplete']:>12,}")
            if s['incomplete'] > 0:
                print(f"      Missing encodec: {s['missing_encodec']:,} | Missing piano_roll: {s['missing_pianoroll']:,}")
                s_miss_line = "      Missing conditioning -> " + ", ".join(
                    f"{k}: {s['missing_by_key'].get(k, 0):,}" for k in COND_KEYS
                )
                print(s_miss_line)

    print("-" * 70)
    print(f"{'GRAND TOTAL':<25} | {total_files:>12,} | {total_complete:>12,} | {total_incomplete:>12,}")
    if total_incomplete > 0:
        print("Overall missing:")
        print(f"  encodec: {total_missing_encodec:,}, piano_roll: {total_missing_pianoroll:,}")
        print("  conditioning -> " + ", ".join(f"{k}: {total_missing_by_key.get(k,0):,}" for k in COND_KEYS))
    print("----------------------------------------\n")

    # 3) Write per-type path logs
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _write_list(paths, filename):
        # de-dupe while preserving order
        seen = set()
        ordered = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        out_path = LOG_DIR / filename
        with out_path.open("w") as f:
            f.write("\n".join(ordered))
        print(f"📝 Wrote {len(ordered):,} paths → {out_path.resolve()}")

    _write_list(missing_encodec_paths,   "missing_encodec_paths.txt")
    _write_list(missing_pianoroll_paths, "missing_pianoroll_paths.txt")
    for k in COND_KEYS:
        _write_list(missing_cond_paths[k], f"missing_{k}_paths.txt")

    return stats

def plot_stats(stats: dict, output_path: Path):
    """
    Generates and saves a bar chart of the total files per instrument group
    (complete vs incomplete).
    """
    if not stats:
        print("No stats to plot.")
        return

    sorted_groups = sorted(stats.items(), key=lambda item: item[1]['total'], reverse=True)
    labels = [item[0] for item in sorted_groups]
    complete_counts = [item[1]['complete'] for item in sorted_groups]
    incomplete_counts = [item[1]['incomplete'] for item in sorted_groups]

    plt.figure(figsize=(14, 8))
    plt.bar(labels, complete_counts, label='Complete')
    plt.bar(labels, incomplete_counts, bottom=complete_counts, label='Incomplete')
    plt.title("Dataset Composition by Instrument Group", fontsize=16)
    plt.ylabel("Number of Files", fontsize=12)
    plt.xlabel("Instrument Group", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        print(f"✅ Graph saved successfully to: {output_path.resolve()}")
    except Exception as e:
        print(f"❌ Error saving graph: {e}")

if __name__ == "__main__":
    stats = analyze_manifest(FINAL_MANIFEST_JSON)
    if stats:
        plot_stats(stats, OUTPUT_GRAPH_PATH)
