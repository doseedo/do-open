#!/usr/bin/env python3
"""
Extract only the 5000 patterns needed by the model from v55.
Uses streaming to avoid loading 2GB into memory.
"""
import json
import sys
import re

# Load pattern names we need
print("Loading pattern ID mapping...")
with open('pattern_id_to_name.json') as f:
    id_to_name = json.load(f)

needed_names = set(id_to_name.values())
print(f"Need {len(needed_names)} patterns")
print(f"Sample names: {list(needed_names)[:5]}")

# Stream through v55 file
pattern_file = 'checkpoint_v55_pure_contour_1000files_patterns.json'
print(f"\nStreaming through {pattern_file}...")

found = {}
buffer = ""
current_key = None
brace_depth = 0
in_pattern = False

with open(pattern_file, 'r') as f:
    # Skip first line (opening brace)
    f.readline()

    for line_num, line in enumerate(f, 2):
        if line_num % 500000 == 0:
            print(f"  Line {line_num:,}, found {len(found)}/{len(needed_names)}")
            if len(found) >= len(needed_names):
                break

        # Look for pattern start: "GM128_129": {
        if not in_pattern:
            match = re.match(r'\s*"(GM\d+_\d+)":\s*\{', line)
            if match:
                key = match.group(1)
                if key in needed_names and key not in found:
                    current_key = key
                    buffer = "{"
                    brace_depth = 1
                    in_pattern = True
                    continue

        if in_pattern:
            buffer += line
            brace_depth += line.count('{') - line.count('}')

            if brace_depth == 0:
                # Complete pattern - strip trailing comma
                buffer = buffer.rstrip().rstrip(',')
                try:
                    pattern = json.loads(buffer)
                    # Keep only essential data (skip occurrences to save space)
                    found[current_key] = {
                        'canonical_pitches': pattern.get('canonical_pitches', []),
                        'pitch_intervals': pattern.get('pitch_intervals', []),
                        'rhythm_ratios': pattern.get('rhythm_ratios', []),
                        'duration_ratios': pattern.get('duration_ratios', []),
                        'velocity_ratios': pattern.get('velocity_ratios', []),
                        'gm_program': int(current_key.split('_')[0].replace('GM', ''))
                    }
                    if len(found) % 500 == 0:
                        print(f"  Extracted {len(found)} patterns")
                except json.JSONDecodeError as e:
                    print(f"  Failed to parse {current_key}: {e}")

                in_pattern = False
                current_key = None
                buffer = ""

print(f"\nExtracted {len(found)} of {len(needed_names)} needed patterns")

# Save
output_file = 'needed_patterns.json'
print(f"Saving to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(found, f)

import os
file_size = os.path.getsize(output_file)
print(f"Done! File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

# Show sample
if found:
    key = list(found.keys())[0]
    p = found[key]
    print(f"\nSample pattern {key}:")
    print(f"  canonical_pitches: {p['canonical_pitches']}")
    print(f"  rhythm_ratios: {p['rhythm_ratios']}")
    print(f"  duration_ratios: {p['duration_ratios']}")
    print(f"  velocity_ratios: {p['velocity_ratios']}")
