#!/usr/bin/env python3
"""
Check the current status of the vocal training pipeline.
"""

import json
from pathlib import Path
from collections import Counter

def check_file(path, description):
    """Check if file exists and show info."""
    p = Path(path)
    if p.exists():
        with open(p, 'r') as f:
            data = json.load(f)
        print(f"✅ {description}")
        print(f"   Path: {path}")
        print(f"   Entries: {len(data):,}")
        return len(data)
    else:
        print(f"❌ {description}")
        print(f"   Path: {path}")
        print(f"   Status: NOT FOUND")
        return 0

def analyze_yamnet(path):
    """Analyze YAMNet labeled manifest."""
    if not Path(path).exists():
        return

    with open(path, 'r') as f:
        data = json.load(f)

    total = len(data)
    with_labels = sum(1 for e in data if 'yamnet_labels' in e and e['yamnet_labels'].get('status') == 'success')
    with_warnings = sum(1 for e in data if 'yamnet_labels' in e and e['yamnet_labels'].get('warnings'))

    print(f"\n   YAMNet Status:")
    print(f"   - Successfully labeled: {with_labels:,} ({100*with_labels/total:.1f}%)")
    print(f"   - With warnings: {with_warnings:,} ({100*with_warnings/total:.1f}%)")

def analyze_flagged(path):
    """Analyze flagged manifest."""
    if not Path(path).exists():
        return

    with open(path, 'r') as f:
        data = json.load(f)

    print(f"\n   Flagged Analysis:")
    decisions = Counter(e['review']['decision'] for e in data)
    print(f"   - Flagged: {len(data):,}")

    reasons = [e['review']['reason'].split(';')[0].strip() for e in data]
    print(f"\n   Top reasons:")
    for reason, count in Counter(reasons).most_common(5):
        print(f"   - {reason}: {count}")

def main():
    print("=" * 70)
    print("VOCAL TRAINING PIPELINE STATUS")
    print("=" * 70)
    print()

    # Check alternates manifest
    print("📋 STEP 1: Alternate Takes Manifest")
    print("-" * 70)
    alt_count = check_file(
        "./vocal_training_manifest_with_alternates.json",
        "Alternate takes manifest"
    )
    print()

    # Check YAMNet labeled
    print("🔊 STEP 2: YAMNet Labeling")
    print("-" * 70)

    # Check test version (1000 entries)
    test_count = check_file(
        "./vocal_training_manifest_yamnet_labeled.json",
        "YAMNet labeled (TEST)"
    )
    if test_count:
        analyze_yamnet("./vocal_training_manifest_yamnet_labeled.json")

    print()

    # Check full version (32K entries)
    full_count = check_file(
        "./vocal_training_manifest_yamnet_labeled_FULL.json",
        "YAMNet labeled (FULL)"
    )
    if full_count:
        analyze_yamnet("./vocal_training_manifest_yamnet_labeled_FULL.json")

    print()

    # Check flagged manifests
    print("🚩 STEP 3: Flagged/Clean Manifests")
    print("-" * 70)

    # Ollama version
    ollama_flag_count = check_file(
        "./vocal_training_manifest_ollama_flagged.json",
        "Ollama flagged"
    )
    if ollama_flag_count:
        analyze_flagged("./vocal_training_manifest_ollama_flagged.json")

    print()

    ollama_clean_count = check_file(
        "./vocal_training_manifest_ollama_clean.json",
        "Ollama clean"
    )

    print()

    # Keyword version
    keyword_flag_count = check_file(
        "./vocal_flagged_FULL.json",
        "Keyword flagged (FULL)"
    )
    if keyword_flag_count:
        analyze_flagged("./vocal_flagged_FULL.json")

    print()

    keyword_clean_count = check_file(
        "./vocal_clean_FULL.json",
        "Keyword clean (FULL)"
    )

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY & NEXT STEPS")
    print("=" * 70)
    print()

    if alt_count == 0:
        print("⚠️  No alternate takes manifest found!")
        print("   Run: python find_alternate_takes.py")

    elif full_count == 0:
        print("📋 You have the alternate takes manifest")
        print(f"   Entries: {alt_count:,}")
        print()
        if test_count > 0:
            print(f"✅ YAMNet test complete ({test_count:,} entries)")
            print(f"   Keep rate: ~{100 * ollama_clean_count / test_count:.1f}%" if ollama_clean_count else "")
            print()
        print("⏭️  NEXT STEP: Run YAMNet on FULL dataset")
        print()
        print("   ./run_full_yamnet.sh")
        print()
        print(f"   This will process {alt_count:,} entries (~10-11 hours)")

    elif keyword_clean_count == 0 and ollama_clean_count == 0:
        print(f"✅ YAMNet labeling complete ({full_count:,} entries)")
        print()
        print("⏭️  NEXT STEP: Flag non-vocal entries")
        print()
        print("   Option 1 (Fast - Recommended):")
        print("   python flag_non_vocals.py --input ./vocal_training_manifest_yamnet_labeled_FULL.json")
        print()
        print("   Option 2 (Accurate):")
        print("   python ollama_review_yamnet.py --input ./vocal_training_manifest_yamnet_labeled_FULL.json")

    else:
        print("✅ Pipeline complete!")
        print()
        if keyword_clean_count:
            print(f"📊 Keyword method: {keyword_clean_count:,} clean entries")
        if ollama_clean_count:
            print(f"📊 Ollama method: {ollama_clean_count:,} clean entries")
        print()
        print("⏭️  NEXT STEP: Start training")
        print()
        manifest = "./vocal_clean_FULL.json" if keyword_clean_count else "./vocal_ollama_clean.json"
        print(f"   python trainer_performervox.py \\")
        print(f"       --manifest_json {manifest} \\")
        print(f"       --checkpoint_dir /path/to/checkpoint \\")
        print(f"       --batch_size 4")

if __name__ == "__main__":
    main()
