#!/usr/bin/env python3
"""
Fast keyword-based approach to flag non-vocal entries.
Use this if Ollama is slow or unavailable.
"""

import json
from pathlib import Path
from typing import List, Set, Dict
from collections import Counter

# Vocal-related keywords (KEEP these)
VOCAL_KEYWORDS = {
    'singing', 'speech', 'voice', 'vocal', 'choir', 'chant',
    'male singing', 'female singing', 'child singing',
    'male speech', 'female speech', 'narration', 'conversation',
    'yodel', 'hum', 'humming', 'whispering', 'spoken'
}

# Non-vocal keywords (FLAG these if primary)
NON_VOCAL_KEYWORDS = {
    'silence', 'music', 'instrumental', 'guitar', 'piano', 'drum',
    'bass', 'synthesizer', 'organ', 'strings', 'brass', 'woodwind',
    'percussion', 'electronic', 'ambient', 'noise', 'static',
    'white noise', 'pink noise', 'hum', 'buzz', 'distortion',
    'wind', 'rain', 'water', 'bird', 'animal', 'dog', 'cat',
    'traffic', 'car', 'engine', 'door', 'footsteps', 'clicking'
}

def analyze_entry(entry: Dict, strict: bool = False) -> Dict:
    """
    Analyze entry and determine if it should be flagged.

    Args:
        entry: Manifest entry with yamnet_labels
        strict: If True, require strong vocal signal; if False, allow some music

    Returns:
        Dict with decision, reason, confidence
    """
    yamnet_labels = entry.get("yamnet_labels", {})

    if yamnet_labels.get("status") != "success":
        return {
            "decision": "FLAG",
            "reason": "YAMNet processing failed",
            "confidence": "high"
        }

    top_preds = yamnet_labels.get("top_predictions", [])
    warnings = yamnet_labels.get("warnings", [])
    top_class = yamnet_labels.get("top_class", "").lower()
    top_score = yamnet_labels.get("top_score", 0)

    # Check top prediction
    has_vocal_top = any(vk in top_class for vk in VOCAL_KEYWORDS)
    has_nonvocal_top = any(nvk in top_class for nvk in NON_VOCAL_KEYWORDS)

    # Check top 3 predictions
    top3_classes = [p['class'].lower() for p in top_preds[:3]]
    has_vocal_top3 = any(any(vk in cls for vk in VOCAL_KEYWORDS) for cls in top3_classes)

    # Check for specific bad patterns
    is_silence = 'silence' in top_class and top_score > 0.7
    is_pure_music = 'music' in top_class and top_score > 0.8 and not has_vocal_top3
    is_noise = any(w in top_class for w in ['noise', 'static', 'distortion'])

    # Decision logic
    decision = "KEEP"
    reason = []
    confidence = "medium"

    # Clear bad cases
    if is_silence:
        decision = "FLAG"
        reason.append("Mostly silence")
        confidence = "high"
    elif is_pure_music:
        decision = "FLAG"
        reason.append("Pure instrumental music, no vocals detected")
        confidence = "high"
    elif is_noise:
        decision = "FLAG"
        reason.append("Heavy noise/distortion")
        confidence = "high"

    # No vocal content in top 5
    elif not has_vocal_top3:
        if strict:
            decision = "FLAG"
            reason.append("No vocal content in top 3 predictions")
            confidence = "medium"
        else:
            # More lenient - check top 5
            top5_classes = [p['class'].lower() for p in top_preds[:5]]
            has_vocal_top5 = any(any(vk in cls for vk in VOCAL_KEYWORDS) for cls in top5_classes)
            if not has_vocal_top5:
                decision = "FLAG"
                reason.append("No vocal content in top 5 predictions")
                confidence = "medium"

    # Check warnings
    critical_warnings = [
        w for w in warnings
        if 'no clear vocal' in w.lower() or 'silence' in w.lower()
    ]
    if critical_warnings and decision == "KEEP":
        if strict:
            decision = "FLAG"
            reason.append(f"Critical warnings: {', '.join(critical_warnings[:2])}")
            confidence = "low"

    # Vocal is present
    if decision == "KEEP":
        if has_vocal_top:
            reason.append(f"Strong vocal signal: {top_class}")
            confidence = "high"
        elif has_vocal_top3:
            reason.append("Vocal content detected in top 3")
            confidence = "medium"
        else:
            reason.append("Passed basic checks")
            confidence = "low"

    return {
        "decision": decision,
        "reason": "; ".join(reason) if reason else top_class,
        "confidence": confidence,
        "top_class": top_class,
        "top_score": top_score
    }

def flag_non_vocals(
    input_manifest: str,
    output_flagged: str,
    output_clean: str,
    strict: bool = False
):
    """Flag non-vocal entries and create clean manifest."""

    print(f"Loading manifest: {input_manifest}")
    with open(input_manifest, 'r') as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}")
    print(f"Mode: {'STRICT' if strict else 'LENIENT'}")
    print()

    flagged_entries = []
    keep_count = 0
    flag_count = 0

    # Analyze each entry
    for idx, entry in enumerate(manifest):
        review = analyze_entry(entry, strict=strict)

        if review["decision"] == "FLAG":
            flagged_entry = {
                "original_index": idx,
                "audio_path": entry.get("audio_path", ""),
                "yamnet_labels": entry.get("yamnet_labels", {}),
                "review": review,
                "full_entry": entry
            }
            flagged_entries.append(flagged_entry)
            flag_count += 1
        else:
            keep_count += 1

    # Save flagged manifest
    with open(output_flagged, 'w') as f:
        json.dump(flagged_entries, f, indent=2)

    # Create clean manifest
    flagged_indices = {item["original_index"] for item in flagged_entries}
    clean = [entry for idx, entry in enumerate(manifest) if idx not in flagged_indices]

    with open(output_clean, 'w') as f:
        json.dump(clean, f, indent=2)

    # Statistics
    print("=" * 70)
    print("FLAGGING COMPLETE")
    print("=" * 70)
    print(f"Total entries: {len(manifest)}")
    print(f"Keep: {keep_count} ({100*keep_count/len(manifest):.1f}%)")
    print(f"Flagged: {flag_count} ({100*flag_count/len(manifest):.1f}%)")
    print()
    print(f"Flagged manifest: {output_flagged}")
    print(f"Clean manifest: {output_clean}")

    # Show reason distribution
    if flagged_entries:
        print("\nTop flagging reasons:")
        reasons = [item["review"]["reason"].split(";")[0] for item in flagged_entries]
        reason_counts = Counter(reasons)
        for reason, count in reason_counts.most_common(10):
            print(f"  - {reason}: {count}")

    # Show examples
    if flagged_entries:
        print("\nExample flagged entries:")
        for item in flagged_entries[:5]:
            print(f"\n[{item['original_index']}] {Path(item['audio_path']).name}")
            print(f"  Reason: {item['review']['reason']}")
            print(f"  Top class: {item['review']['top_class']} ({item['review']['top_score']*100:.1f}%)")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Flag non-vocal entries based on YAMNet labels")
    ap.add_argument("--input", type=str,
                    default="./vocal_training_manifest_yamnet_labeled.json",
                    help="Input manifest with YAMNet labels")
    ap.add_argument("--output_flagged", type=str,
                    default="./vocal_training_manifest_flagged.json",
                    help="Output manifest with flagged entries")
    ap.add_argument("--output_clean", type=str,
                    default="./vocal_training_manifest_clean.json",
                    help="Output clean manifest")
    ap.add_argument("--strict", action="store_true",
                    help="Use strict mode (flag more aggressively)")

    args = ap.parse_args()

    flag_non_vocals(
        input_manifest=args.input,
        output_flagged=args.output_flagged,
        output_clean=args.output_clean,
        strict=args.strict
    )
