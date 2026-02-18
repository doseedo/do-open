#!/usr/bin/env python3
"""
Filter YAMNet labeled manifest to exclude instrumental content.
Keeps vocal entries (opera, lullaby, singing, etc.) but removes entries with:
- Guitar, Drum, Percussion in top predictions or warnings
- High Music score (>80%) with no vocal indicators
"""

import json
import sys

def has_instrumental_content(entry):
    """Check if entry contains instrumental content to exclude."""

    # Instrumental keywords to exclude
    instrumental_keywords = [
        'guitar', 'drum', 'percussion', 'bass', 'piano',
        'plucked string instrument', 'musical instrument',
        'keyboard', 'synthesizer', 'electric guitar'
    ]

    # Vocal keywords to keep (even with high music score)
    vocal_keywords = [
        'singing', 'opera', 'lullaby', 'mantra', 'choir',
        'female singing', 'male singing', 'child singing',
        'humming', 'vocal', 'speech', 'narration'
    ]

    yamnet = entry.get('yamnet_labels', {})
    if yamnet.get('status') != 'success':
        return False  # Keep entries that weren't analyzed

    # Check warnings
    warnings = yamnet.get('warnings', [])
    for warning in warnings:
        warning_lower = warning.lower()
        for keyword in instrumental_keywords:
            if keyword in warning_lower:
                return True  # Exclude

    # Check top predictions
    top_preds = yamnet.get('top_predictions', [])

    # Check if any instrumental content in top 5 predictions
    for i, pred in enumerate(top_preds[:5]):
        class_name = pred.get('class', '').lower()

        # Direct instrumental match
        for keyword in instrumental_keywords:
            if keyword in class_name:
                return True  # Exclude

    # Check for high Music score without vocal indicators
    if top_preds:
        top_class = top_preds[0].get('class', '').lower()
        top_score = top_preds[0].get('percentage', 0)

        # If top prediction is "Music" with >80% and no vocal keywords
        if top_class == 'music' and top_score > 80:
            has_vocal_indicator = False

            # Check top 5 predictions for vocal content
            for pred in top_preds[:5]:
                class_name = pred.get('class', '').lower()
                for vocal_kw in vocal_keywords:
                    if vocal_kw in class_name:
                        has_vocal_indicator = True
                        break
                if has_vocal_indicator:
                    break

            # Also check warnings for vocal content
            for warning in warnings:
                warning_lower = warning.lower()
                for vocal_kw in vocal_keywords:
                    if vocal_kw in warning_lower:
                        has_vocal_indicator = True
                        break

            if not has_vocal_indicator:
                return True  # Exclude high music score without vocals

    return False  # Keep this entry


def main():
    input_file = './vocal_training_manifest_yamnet_labeled_FULL.json'
    output_file = './vocal_training_manifest_yamnet_filtered.json'

    print(f"Loading {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")

    # Filter out instrumental content
    filtered_data = []
    excluded_count = 0
    excluded_reasons = {}

    for entry in data:
        if has_instrumental_content(entry):
            excluded_count += 1

            # Track exclusion reasons
            yamnet = entry.get('yamnet_labels', {})
            warnings = yamnet.get('warnings', [])
            top_pred = ''
            if yamnet.get('top_predictions'):
                top_pred = yamnet['top_predictions'][0].get('class', '')

            reason = f"{top_pred} | {', '.join(warnings[:3])}"
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
        else:
            filtered_data.append(entry)

    print(f"\nFiltered results:")
    print(f"  Kept: {len(filtered_data)}")
    print(f"  Excluded: {excluded_count}")
    print(f"  Retention rate: {len(filtered_data)/len(data)*100:.1f}%")

    print(f"\nTop exclusion reasons:")
    for reason, count in sorted(excluded_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {count:5d} - {reason[:100]}")

    # Save filtered manifest
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"✅ Done! Filtered manifest saved to {output_file}")


if __name__ == '__main__':
    main()
