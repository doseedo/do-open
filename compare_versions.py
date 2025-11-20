#!/usr/bin/env python3
"""
Compare versions of common files between harmonymodule and standalone midi_generator
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
HARMONY_BASE = "home/arlo/harmonymodule/midi_generator"
STANDALONE_BASE = "midi_generator"

# Common files
COMMON_FILES = []
with open("/tmp/common_files.txt", "r") as f:
    COMMON_FILES = [line.strip() for line in f if line.strip()]

print("=" * 100)
print("FILE VERSION COMPARISON REPORT")
print("=" * 100)
print(f"\nComparing {len(COMMON_FILES)} common files\n")

results = []

for rel_path in COMMON_FILES:
    harmony_path = Path(HARMONY_BASE) / rel_path
    standalone_path = Path(STANDALONE_BASE) / rel_path

    # Get file sizes
    harmony_size = harmony_path.stat().st_size if harmony_path.exists() else 0
    standalone_size = standalone_path.stat().st_size if standalone_path.exists() else 0

    # Get modification times
    harmony_mtime = datetime.fromtimestamp(harmony_path.stat().st_mtime) if harmony_path.exists() else None
    standalone_mtime = datetime.fromtimestamp(standalone_path.stat().st_mtime) if standalone_path.exists() else None

    # Get line counts
    try:
        with open(harmony_path, 'r', encoding='utf-8', errors='ignore') as f:
            harmony_lines = len(f.readlines())
    except:
        harmony_lines = 0

    try:
        with open(standalone_path, 'r', encoding='utf-8', errors='ignore') as f:
            standalone_lines = len(f.readlines())
    except:
        standalone_lines = 0

    # Determine difference
    size_diff = harmony_size - standalone_size
    line_diff = harmony_lines - standalone_lines

    # Recommendation
    if abs(size_diff) < 100 and abs(line_diff) < 5:
        recommendation = "IDENTICAL (or nearly)"
        keep = "Either"
    elif harmony_size > standalone_size:
        recommendation = "Harmony is LARGER"
        keep = "Harmony"
    elif standalone_size > harmony_size:
        recommendation = "Standalone is LARGER"
        keep = "Standalone"
    else:
        recommendation = "EQUAL size, check content"
        keep = "Need manual review"

    results.append({
        'file': rel_path,
        'harmony_size': harmony_size,
        'standalone_size': standalone_size,
        'harmony_lines': harmony_lines,
        'standalone_lines': standalone_lines,
        'size_diff': size_diff,
        'line_diff': line_diff,
        'harmony_mtime': harmony_mtime,
        'standalone_mtime': standalone_mtime,
        'recommendation': recommendation,
        'keep': keep
    })

# Print summary
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

identical = sum(1 for r in results if "IDENTICAL" in r['recommendation'])
harmony_larger = sum(1 for r in results if r['keep'] == "Harmony")
standalone_larger = sum(1 for r in results if r['keep'] == "Standalone")
manual_review = sum(1 for r in results if r['keep'] == "Need manual review")

print(f"\nIdentical files: {identical}")
print(f"Harmony version larger: {harmony_larger}")
print(f"Standalone version larger: {standalone_larger}")
print(f"Need manual review: {manual_review}")

# Print detailed comparison for significant differences
print("\n" + "=" * 100)
print("FILES WITH SIGNIFICANT DIFFERENCES (>100 bytes or >5 lines)")
print("=" * 100)

significant_diffs = [r for r in results if abs(r['size_diff']) > 100 or abs(r['line_diff']) > 5]

for result in sorted(significant_diffs, key=lambda x: abs(x['size_diff']), reverse=True):
    print(f"\n{result['file']}")
    print(f"  Harmony:    {result['harmony_size']:>8} bytes, {result['harmony_lines']:>5} lines")
    print(f"  Standalone: {result['standalone_size']:>8} bytes, {result['standalone_lines']:>5} lines")
    print(f"  Difference: {result['size_diff']:>+8} bytes, {result['line_diff']:>+5} lines")
    print(f"  → KEEP: {result['keep']}")

# Save full report
print("\n" + "=" * 100)
print("Saving detailed report to /tmp/version_comparison.csv")

with open("/tmp/version_comparison.csv", "w") as f:
    f.write("File,Harmony Size,Standalone Size,Size Diff,Harmony Lines,Standalone Lines,Line Diff,Keep\n")
    for r in sorted(results, key=lambda x: abs(x['size_diff']), reverse=True):
        f.write(f"{r['file']},{r['harmony_size']},{r['standalone_size']},{r['size_diff']},")
        f.write(f"{r['harmony_lines']},{r['standalone_lines']},{r['line_diff']},{r['keep']}\n")

print("Done!")
print("=" * 100)
