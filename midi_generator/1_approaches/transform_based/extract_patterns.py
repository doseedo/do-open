#!/usr/bin/env python3
"""
Extract only the 5000 patterns needed by the model.
Streams through the file to avoid memory issues.
"""
import json
import sys

# Load pattern mappings first
print("Loading pattern ID mapping...")
with open('pattern_id_to_name.json') as f:
    id_to_name = json.load(f)

# Get the pattern names we need
needed_names = set(id_to_name.values())
print(f"Need {len(needed_names)} unique patterns")

# Stream through the large file and extract only needed patterns
pattern_file = sys.argv[1] if len(sys.argv) > 1 else 'checkpoint_v55_pure_contour_1000files_patterns.json'
print(f"Streaming through {pattern_file}...")

found = {}
current_key = None
current_content = []
brace_depth = 0
in_pattern = False

with open(pattern_file, 'r') as f:
    # Skip opening brace
    line = f.readline()

    for line_num, line in enumerate(f, 2):
        if line_num % 100000 == 0:
            print(f"  Line {line_num:,}, found {len(found)}/{len(needed_names)} patterns...")
            if len(found) >= len(needed_names):
                print("  Found all needed patterns!")
                break

        # Look for pattern key starts
        stripped = line.strip()
        if stripped.startswith('"GM') and '": {' in stripped:
            # Extract key name
            key = stripped.split('"')[1]
            if key in needed_names and key not in found:
                current_key = key
                current_content = ['{']
                brace_depth = 1
                in_pattern = True
                continue

        if in_pattern:
            current_content.append(line.rstrip().rstrip(','))
            brace_depth += line.count('{') - line.count('}')

            if brace_depth == 0:
                # Complete pattern
                try:
                    pattern_json = '\n'.join(current_content)
                    found[current_key] = json.loads(pattern_json)
                    if len(found) % 500 == 0:
                        print(f"  Extracted {len(found)} patterns...")
                except json.JSONDecodeError as e:
                    print(f"  Failed to parse {current_key}: {e}")

                in_pattern = False
                current_key = None
                current_content = []

print(f"\nExtracted {len(found)} of {len(needed_names)} needed patterns")

# Save to smaller file
output_file = 'needed_patterns.json'
print(f"Saving to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(found, f)

print(f"Done! File size: {len(json.dumps(found)):,} bytes")

# Show some stats
if found:
    sample_key = list(found.keys())[0]
    sample = found[sample_key]
    print(f"\nSample pattern {sample_key}:")
    print(f"  canonical_pitches: {sample.get('canonical_pitches', [])[:5]}...")
    print(f"  rhythm_ratios: {sample.get('rhythm_ratios', [])[:5]}...")
    print(f"  duration_ratios: {sample.get('duration_ratios', [])[:5]}...")
    print(f"  velocity_ratios: {sample.get('velocity_ratios', [])[:5]}...")
