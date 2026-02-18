#!/usr/bin/env python3
"""
Use Ollama LLM to review YAMNet labels and flag non-vocal entries.
More intelligent than simple keyword matching.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import time

def call_ollama(prompt: str, model: str = "llama3.2:3b") -> str:
    """Call Ollama API via subprocess."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        print(f"Ollama error: {e}")
        return "ERROR"

def create_review_prompt(top_predictions: List[Dict], warnings: List[str]) -> str:
    """Create a focused prompt for Ollama to review YAMNet classifications."""

    # Format predictions
    preds_text = "\n".join([
        f"- {p['class']}: {p['percentage']:.1f}%"
        for p in top_predictions[:5]
    ])

    warnings_text = "\n".join([f"- {w}" for w in warnings]) if warnings else "None"

    prompt = f"""You are reviewing audio classifications for a VOCAL training dataset.

YAMNet Top Predictions:
{preds_text}

Warnings:
{warnings_text}

Task: Determine if this audio is suitable for VOCAL training.

KEEP if:
- Primary content is singing, speech, or vocals
- Minor music/instrumental background is OK
- Some reverb/processing is OK

FLAG if:
- No vocal content detected
- Primarily instrumental/music without vocals
- Heavy noise/distortion/corruption
- Silence or environmental sounds only
- Wrong content (animal sounds, machinery, etc.)

Respond with ONLY one word:
- KEEP (if suitable for vocal training)
- FLAG (if not suitable or problematic)

Your response:"""

    return prompt

def review_entry_with_ollama(entry: Dict, model: str = "llama3.2:3b") -> Dict:
    """Review a single entry using Ollama."""

    yamnet_labels = entry.get("yamnet_labels", {})

    if yamnet_labels.get("status") != "success":
        return {
            "index": None,
            "decision": "FLAG",
            "reason": "YAMNet processing failed",
            "confidence": "high"
        }

    top_preds = yamnet_labels.get("top_predictions", [])
    warnings = yamnet_labels.get("warnings", [])
    top_class = yamnet_labels.get("top_class", "Unknown")

    # Create prompt
    prompt = create_review_prompt(top_preds, warnings)

    # Call Ollama
    response = call_ollama(prompt, model)

    # Parse response
    decision = "FLAG"  # Default to FLAG if unclear
    if "KEEP" in response.upper():
        decision = "KEEP"
    elif "FLAG" in response.upper():
        decision = "FLAG"

    # Determine confidence based on response clarity
    confidence = "medium"
    if response in ["KEEP", "FLAG"]:
        confidence = "high"

    return {
        "decision": decision,
        "reason": f"Top class: {top_class}, Warnings: {len(warnings)}",
        "llm_response": response,
        "confidence": confidence
    }

def batch_review_with_ollama(
    input_manifest: str,
    output_flagged: str,
    model: str = "llama3.2:3b",
    max_entries: Optional[int] = None,
    skip_existing: bool = True
):
    """Review manifest entries with Ollama and create flagged manifest."""

    print(f"Loading manifest: {input_manifest}")
    with open(input_manifest, 'r') as f:
        manifest = json.load(f)

    if max_entries:
        manifest = manifest[:max_entries]

    print(f"Total entries to review: {len(manifest)}")

    # Check if we should resume
    reviewed_indices = set()
    if skip_existing and Path(output_flagged).exists():
        print("Loading existing flagged manifest...")
        with open(output_flagged, 'r') as f:
            existing = json.load(f)

        for item in existing:
            if "original_index" in item:
                reviewed_indices.add(item["original_index"])

        print(f"Found {len(reviewed_indices)} already reviewed entries")

    # Review entries
    flagged_entries = []
    keep_count = 0
    flag_count = 0

    print(f"\nUsing Ollama model: {model}")
    print("Reviewing entries...\n")

    for idx, entry in enumerate(tqdm(manifest, desc="Ollama Review")):
        # Skip if already reviewed
        if idx in reviewed_indices:
            # Add from existing if flagged
            if skip_existing and Path(output_flagged).exists():
                with open(output_flagged, 'r') as f:
                    existing = json.load(f)
                for item in existing:
                    if item.get("original_index") == idx:
                        flagged_entries.append(item)
                        flag_count += 1
                        break
            continue

        # Review with Ollama
        review = review_entry_with_ollama(entry, model)

        if review["decision"] == "FLAG":
            flagged_entry = {
                "original_index": idx,
                "audio_path": entry.get("audio_path", ""),
                "yamnet_labels": entry.get("yamnet_labels", {}),
                "review": review,
                "full_entry": entry  # Include full entry for reference
            }
            flagged_entries.append(flagged_entry)
            flag_count += 1
        else:
            keep_count += 1

        # Save checkpoint every 50 entries
        if (idx + 1) % 50 == 0:
            with open(output_flagged, 'w') as f:
                json.dump(flagged_entries, f, indent=2)
            print(f"\nCheckpoint saved at {idx + 1} entries")
            print(f"  Keep: {keep_count}, Flag: {flag_count}")

    # Final save
    with open(output_flagged, 'w') as f:
        json.dump(flagged_entries, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("OLLAMA REVIEW COMPLETE")
    print("=" * 70)
    print(f"Total reviewed: {len(manifest)}")
    print(f"Keep: {keep_count} ({100*keep_count/len(manifest):.1f}%)")
    print(f"Flagged: {flag_count} ({100*flag_count/len(manifest):.1f}%)")
    print(f"\nFlagged manifest saved to: {output_flagged}")

    # Show some examples
    if flagged_entries:
        print("\nExample flagged entries:")
        for i, item in enumerate(flagged_entries[:3]):
            print(f"\n[{item['original_index']}] {Path(item['audio_path']).name}")
            print(f"  Decision: {item['review']['decision']}")
            print(f"  Reason: {item['review']['reason']}")
            print(f"  LLM: {item['review']['llm_response']}")

def create_clean_manifest(original_manifest: str, flagged_manifest: str, output_clean: str):
    """Create clean manifest by removing flagged entries."""

    print(f"Loading original manifest: {original_manifest}")
    with open(original_manifest, 'r') as f:
        original = json.load(f)

    print(f"Loading flagged entries: {flagged_manifest}")
    with open(flagged_manifest, 'r') as f:
        flagged = json.load(f)

    # Get flagged indices
    flagged_indices = {item["original_index"] for item in flagged}

    # Create clean manifest
    clean = [entry for idx, entry in enumerate(original) if idx not in flagged_indices]

    # Save
    with open(output_clean, 'w') as f:
        json.dump(clean, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("CLEAN MANIFEST CREATED")
    print("=" * 70)
    print(f"Original entries: {len(original)}")
    print(f"Flagged entries: {len(flagged_indices)}")
    print(f"Clean entries: {len(clean)}")
    print(f"Removed: {100*len(flagged_indices)/len(original):.1f}%")
    print(f"\nClean manifest saved to: {output_clean}")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Use Ollama to review YAMNet labels")
    ap.add_argument("--input", type=str,
                    default="./vocal_training_manifest_yamnet_labeled.json",
                    help="Input manifest with YAMNet labels")
    ap.add_argument("--output_flagged", type=str,
                    default="./vocal_training_manifest_ollama_flagged.json",
                    help="Output manifest with flagged entries")
    ap.add_argument("--output_clean", type=str,
                    default="./vocal_training_manifest_ollama_clean.json",
                    help="Output clean manifest (without flagged)")
    ap.add_argument("--model", type=str, default="llama3.2:3b",
                    help="Ollama model to use")
    ap.add_argument("--max_entries", type=int, default=None,
                    help="Max entries to review (for testing)")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip already reviewed entries")
    ap.add_argument("--create_clean", action="store_true", default=True,
                    help="Create clean manifest after review")

    args = ap.parse_args()

    # Check if Ollama is available
    try:
        subprocess.run(["ollama", "--version"],
                      capture_output=True, check=True)
        print("✅ Ollama is available")
    except:
        print("❌ Ollama not found. Please install: https://ollama.ai/")
        exit(1)

    # Check if model is available
    try:
        result = subprocess.run(["ollama", "list"],
                               capture_output=True, text=True)
        if args.model.split(":")[0] not in result.stdout:
            print(f"📥 Pulling model: {args.model}")
            subprocess.run(["ollama", "pull", args.model], check=True)
    except Exception as e:
        print(f"⚠️  Could not check/pull model: {e}")

    # Run review
    batch_review_with_ollama(
        input_manifest=args.input,
        output_flagged=args.output_flagged,
        model=args.model,
        max_entries=args.max_entries,
        skip_existing=args.skip_existing
    )

    # Create clean manifest
    if args.create_clean:
        create_clean_manifest(
            original_manifest=args.input,
            flagged_manifest=args.output_flagged,
            output_clean=args.output_clean
        )
