#!/usr/bin/env python3
"""Verify factored checkpoint contents."""
import numpy as np
import json

# Load the factored checkpoint
data = np.load('checkpoint_factored.npz', allow_pickle=True)

print('=== FACTORED CHECKPOINT CONTENTS ===')
print()
print('Keys:', list(data.keys()))
print()
print('Version:', data['version'][0])
print('Is Factored:', data['is_factored'][0])
print('Factors:', data['factors'][0])
print()
print('Stats:')
print('  Files:', data["n_files"][0])
print('  Tracks:', data["n_tracks"][0])
print('  Notes:', format(data["n_notes"][0], ','))
print('  Grammar rules:', data["n_grammar_rules"][0])
print('  Canonical patterns:', data["n_canonicals"][0])
print('  Transform vocab:', data["n_transform_vocab"][0])
print()

# Check a sample pattern
patterns = json.loads(data['canonical_patterns_json'][0])
print('Sample pattern (ID 0):')
p = patterns[0]
print('  pitch_classes:', p["pitch_classes"][:10], '...')
print('  octaves:', p["octaves"][:10], '...')
print('  velocities:', p["velocities"][:10], '...')
print('  durations:', p["durations"][:10], '...')
print()

# Show transforms
transforms = json.loads(data['transform_vocabulary_json'][0])
n_transforms = len(transforms)
print('Transforms discovered (' + str(n_transforms) + '):')
print('  ', transforms[:20], '...')
