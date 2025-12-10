#!/usr/bin/env python3
"""
Diagnose harmonic relationships in pattern corpus.

Checks if:
1. Pitch class distribution shows tonal bias (concentrated on diatonic notes)
2. Simultaneous patterns have consistent pitch intervals
3. There's enough harmonic signal to build pitch-aware co-occurrence
"""

import sys
import os
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def diagnose_harmonic_relationships(patterns: dict):
    """Check if patterns show harmonic structure."""

    print("=" * 60)
    print("HARMONIC STRUCTURE DIAGNOSTIC")
    print("=" * 60)

    # 1. Pitch class distribution - do patterns concentrate on diatonic notes?
    pc_distribution = Counter()
    total_occs = 0
    occs_with_pitch = 0

    for pid, p in patterns.items():
        for occ in p.get('occurrences', []):
            total_occs += 1
            fp = occ.get('first_pitch')
            if fp is not None:
                occs_with_pitch += 1
                pc_distribution[fp % 12] += 1

    print(f"\n1. PITCH CLASS DISTRIBUTION")
    print(f"   Total occurrences: {total_occs}")
    print(f"   Occurrences with first_pitch: {occs_with_pitch} ({100*occs_with_pitch/max(1,total_occs):.1f}%)")

    if occs_with_pitch == 0:
        print("   ⚠️  NO PITCH DATA FOUND - Cannot build harmonic co-occurrence!")
        return None

    total = sum(pc_distribution.values())
    print(f"\n   Distribution (sorted by frequency):")
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for pc in sorted(range(12), key=lambda x: -pc_distribution[x]):
        count = pc_distribution[pc]
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"   {note_names[pc]:2s} ({pc:2d}): {pct:5.1f}% {bar}")

    # Check diatonic bias (C major = 0, 2, 4, 5, 7, 9, 11)
    c_major_pcs = {0, 2, 4, 5, 7, 9, 11}
    in_c_major = sum(pc_distribution[pc] for pc in c_major_pcs)
    print(f"\n   In C major scale (diatonic bias): {100*in_c_major/total:.1f}%")
    if in_c_major / total > 0.75:
        print("   ✓ Strong diatonic bias - corpus has tonal structure")
    elif in_c_major / total > 0.60:
        print("   ~ Moderate diatonic bias")
    else:
        print("   ⚠️  Weak diatonic bias - corpus may be atonal or use other modes")

    # 2. Simultaneous patterns - what intervals are common?
    print(f"\n2. SIMULTANEOUS PATTERN INTERVALS")

    # Group occurrences by (piece, time_bucket)
    TIME_BUCKET = 480  # Quarter note
    time_buckets = defaultdict(list)

    for pid, p in patterns.items():
        gm = p.get('gm_program', 0)
        for occ in p.get('occurrences', []):
            piece = occ.get('piece_id', occ.get('piece_idx', 'unknown'))
            onset = occ.get('onset_time', occ.get('onset', 0))
            first_pitch = occ.get('first_pitch')
            if first_pitch is not None:
                bucket = onset // TIME_BUCKET
                time_buckets[(piece, bucket)].append((gm, pid, first_pitch))

    # Count intervals between simultaneous patterns
    interval_counts = Counter()
    gm_pair_intervals = defaultdict(Counter)  # (gm1, gm2) -> interval -> count
    total_pairs = 0

    for (piece, bucket), items in time_buckets.items():
        for i, (gm1, p1, pitch1) in enumerate(items):
            for gm2, p2, pitch2 in items[i+1:]:
                if gm1 != gm2:  # Only cross-instrument
                    interval = (pitch2 - pitch1) % 12
                    interval_counts[interval] += 1
                    total_pairs += 1

                    key = (min(gm1, gm2), max(gm1, gm2))
                    if gm1 < gm2:
                        gm_pair_intervals[key][interval] += 1
                    else:
                        gm_pair_intervals[key][(12 - interval) % 12] += 1

    if total_pairs == 0:
        print("   ⚠️  No simultaneous cross-instrument patterns found!")
        return pc_distribution

    print(f"   Total cross-instrument pairs: {total_pairs}")
    print(f"\n   Interval distribution:")
    interval_names = ["P1", "m2", "M2", "m3", "M3", "P4", "TT", "P5", "m6", "M6", "m7", "M7"]
    for iv in sorted(range(12), key=lambda x: -interval_counts[x]):
        count = interval_counts[iv]
        pct = 100 * count / total_pairs
        bar = "█" * int(pct / 2)
        print(f"   {interval_names[iv]:3s} ({iv:2d}): {pct:5.1f}% {bar}")

    # Check if consonant intervals dominate
    consonant = {0, 3, 4, 5, 7, 8, 9}  # P1, m3, M3, P4, P5, m6, M6
    consonant_count = sum(interval_counts[iv] for iv in consonant)
    print(f"\n   Consonant intervals (P1,m3,M3,P4,P5,m6,M6): {100*consonant_count/total_pairs:.1f}%")
    if consonant_count / total_pairs > 0.70:
        print("   ✓ Strong consonance preference - corpus has harmonic structure")
    elif consonant_count / total_pairs > 0.55:
        print("   ~ Moderate consonance preference")
    else:
        print("   ⚠️  Weak consonance preference - may need different approach")

    # 3. Show top GM pair interval preferences
    print(f"\n3. TOP GM PAIR INTERVAL PREFERENCES")
    gm_names = {
        0: "Piano", 32: "A.Bass", 33: "E.Bass",
        56: "Trumpet", 57: "Trombone", 58: "Tuba",
        64: "Sop.Sax", 65: "Alto.Sax", 66: "Ten.Sax", 67: "Bari.Sax",
        40: "Violin", 42: "Cello",
    }

    # Sort by total count
    sorted_pairs = sorted(gm_pair_intervals.items(),
                         key=lambda x: sum(x[1].values()),
                         reverse=True)

    for (gm1, gm2), iv_counts in sorted_pairs[:10]:
        total = sum(iv_counts.values())
        top_3 = sorted(iv_counts.items(), key=lambda x: -x[1])[:3]
        name1 = gm_names.get(gm1, f"GM{gm1}")
        name2 = gm_names.get(gm2, f"GM{gm2}")
        top_str = ", ".join(f"{interval_names[iv]}={100*c/total:.0f}%" for iv, c in top_3)
        print(f"   {name1:8s} + {name2:8s} (n={total:5d}): {top_str}")

    # 4. Summary and recommendation
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    has_pitch_data = occs_with_pitch > 0
    has_diatonic_bias = (in_c_major / total > 0.60) if total > 0 else False
    has_consonance_pref = (consonant_count / total_pairs > 0.55) if total_pairs > 0 else False

    if has_pitch_data and (has_diatonic_bias or has_consonance_pref):
        print("✓ HARMONIC STRUCTURE DETECTED")
        print("  → Building harmonic co-occurrence should improve generation")
        print("  → Pitch-interval-aware sampling is recommended")
    else:
        if not has_pitch_data:
            print("⚠️  NO PITCH DATA IN OCCURRENCES")
            print("  → Need to extract first_pitch during grammar learning")
        else:
            print("⚠️  WEAK HARMONIC STRUCTURE")
            print("  → Corpus may be atonal or have different tonal system")
            print("  → Generic dissonance penalties may not help")

    return {
        'pc_distribution': dict(pc_distribution),
        'interval_distribution': dict(interval_counts),
        'gm_pair_intervals': {str(k): dict(v) for k, v in gm_pair_intervals.items()},
        'has_pitch_data': has_pitch_data,
        'has_diatonic_bias': has_diatonic_bias,
        'has_consonance_preference': has_consonance_pref,
    }


def build_harmonic_cooccurrence(patterns: dict, time_tolerance: int = 480):
    """Build co-occurrence that tracks pitch relationships.

    Returns:
        harmonic_cooc: {(gm1, gm2, interval): {(p1, p2): count}}
    """
    from collections import defaultdict

    print("\n" + "=" * 60)
    print("BUILDING HARMONIC CO-OCCURRENCE")
    print("=" * 60)

    # Group occurrences by (piece, time_bucket)
    time_buckets = defaultdict(list)

    for pid, p in patterns.items():
        gm = p.get('gm_program', 0)
        for occ in p.get('occurrences', []):
            piece = occ.get('piece_id', occ.get('piece_idx', 'unknown'))
            onset = occ.get('onset_time', occ.get('onset', 0))
            first_pitch = occ.get('first_pitch')
            if first_pitch is not None:
                bucket = onset // time_tolerance
                time_buckets[(piece, bucket)].append((gm, pid, first_pitch))

    # Count co-occurrences WITH pitch interval
    harmonic_cooc = defaultdict(lambda: defaultdict(int))

    for (piece, bucket), items in time_buckets.items():
        for i, (gm1, p1, pitch1) in enumerate(items):
            for gm2, p2, pitch2 in items[i+1:]:
                if gm1 != gm2:
                    # Key includes pitch interval!
                    interval = (pitch2 - pitch1) % 12
                    key = (min(gm1, gm2), max(gm1, gm2), interval)
                    pair = (p1, p2) if gm1 < gm2 else (p2, p1)
                    harmonic_cooc[key][pair] += 1

    print(f"Harmonic co-occurrence keys: {len(harmonic_cooc)}")

    # Analyze what intervals are common per GM pair
    gm_pair_totals = defaultdict(int)
    for (gm1, gm2, interval), pairs in harmonic_cooc.items():
        total = sum(pairs.values())
        gm_pair_totals[(gm1, gm2)] += total

    print(f"Unique GM pairs: {len(gm_pair_totals)}")
    print(f"Total co-occurrence entries: {sum(len(v) for v in harmonic_cooc.values())}")

    return dict(harmonic_cooc)


def load_patterns_streaming(json_path: str, max_patterns: int = None, sample_occs: int = 100):
    """Load patterns with sampling for large files.

    For very large files, samples occurrences to keep memory reasonable.
    """
    import json

    print(f"Loading patterns from: {json_path}")

    patterns = {}
    n_loaded = 0

    with open(json_path, 'r') as f:
        data = json.load(f)

    for pid, pdata in data.items():
        # Sample occurrences if too many
        occs = pdata.get('occurrences', [])
        if len(occs) > sample_occs:
            import random
            occs = random.sample(occs, sample_occs)
            pdata = {**pdata, 'occurrences': occs}

        patterns[pid] = pdata
        n_loaded += 1

        if max_patterns and n_loaded >= max_patterns:
            break

        if n_loaded % 10000 == 0:
            print(f"  Loaded {n_loaded} patterns...")

    print(f"Loaded {len(patterns)} patterns total")
    return patterns


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Diagnose harmonic structure in corpus')
    parser.add_argument('checkpoint', help='Path to checkpoint .npz file or patterns .json')
    parser.add_argument('--max-patterns', type=int, default=None,
                       help='Max patterns to load (for faster testing)')
    parser.add_argument('--sample-occs', type=int, default=100,
                       help='Sample this many occurrences per pattern')
    args = parser.parse_args()

    if args.checkpoint.endswith('.json'):
        patterns = load_patterns_streaming(args.checkpoint, args.max_patterns, args.sample_occs)
    else:
        # Load from npz - get patterns JSON path
        print(f"Loading checkpoint: {args.checkpoint}")
        data = np.load(args.checkpoint, allow_pickle=True)
        patterns_file = str(data['patterns_json_file'][0]) if isinstance(data['patterns_json_file'], np.ndarray) else str(data['patterns_json_file'])

        # Build full path
        import os
        base_dir = os.path.dirname(args.checkpoint)
        json_path = os.path.join(base_dir, patterns_file)
        patterns = load_patterns_streaming(json_path, args.max_patterns, args.sample_occs)

    # Run diagnostic
    results = diagnose_harmonic_relationships(patterns)

    if results and results.get('has_pitch_data'):
        # Build harmonic co-occurrence
        harmonic_cooc = build_harmonic_cooccurrence(patterns)
