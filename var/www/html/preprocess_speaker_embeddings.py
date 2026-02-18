#!/usr/bin/env python3
"""
Preprocess speaker embeddings for all alternate takes in the manifest.
Saves embeddings to disk for fast loading during training.

Output structure:
/mnt/msdd/speaker_embeddings/
    ├── <hash>_<filename>_spk.pt
    └── ...
"""

import json
import torch
import torchaudio
from pathlib import Path
from resemblyzer import VoiceEncoder
from tqdm import tqdm
import hashlib
import argparse
from typing import Optional, List, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_file_hash(filepath: str) -> str:
    """Generate short hash from filepath for unique identification."""
    return hashlib.md5(filepath.encode()).hexdigest()[:8]


def load_and_resample_audio(audio_path: str, target_sr: int = 16000) -> Optional[torch.Tensor]:
    """
    Load audio and resample to 16kHz for Resemblyzer.

    Returns:
        audio_np: numpy array [T] or None if failed
    """
    try:
        if not Path(audio_path).exists():
            return None

        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample to 16kHz
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)

        # Convert to numpy
        audio_np = audio.squeeze().numpy()

        return audio_np

    except Exception as e:
        print(f"⚠ Error loading {audio_path}: {e}")
        return None


def extract_speaker_embedding(
    audio_path: str,
    encoder: VoiceEncoder,
    output_dir: Path
) -> Optional[str]:
    """
    Extract speaker embedding and save to disk.

    Returns:
        Path to saved embedding or None if failed
    """
    # Load and resample audio
    audio_np = load_and_resample_audio(audio_path)
    if audio_np is None:
        return None

    try:
        # Extract speaker embedding
        speaker_emb = encoder.embed_utterance(audio_np)  # [256]

        # Generate output path
        file_hash = get_file_hash(audio_path)
        filename = Path(audio_path).stem
        output_path = output_dir / f"{file_hash}_{filename}_spk.pt"

        # Save embedding
        torch.save(torch.from_numpy(speaker_emb).float(), output_path)

        return str(output_path)

    except Exception as e:
        print(f"⚠ Error extracting embedding for {audio_path}: {e}")
        return None


def collect_all_audio_files(manifest: List[Dict]) -> set:
    """
    Collect all unique audio files that need speaker embeddings.
    Includes main files + alternate takes.
    """
    audio_files = set()

    for entry in manifest:
        # Add main audio file
        audio_path = entry.get("audio_path")
        if audio_path:
            audio_files.add(audio_path)

        # Add alternate takes
        alternates = entry.get("alternate_takes", [])
        for alt in alternates:
            alt_audio = alt.get("audio_path")
            if alt_audio:
                audio_files.add(alt_audio)

    return audio_files


def process_batch_gpu(
    audio_paths: List[str],
    output_dir: Path,
    device: str = "cuda"
) -> List[Dict[str, str]]:
    """
    Process a batch of audio files on GPU.

    Returns:
        List of dicts with audio_path and embedding_path
    """
    # Initialize encoder on GPU
    encoder = VoiceEncoder(device=device)
    encoder.eval()

    results = []

    for audio_path in tqdm(audio_paths, desc="Processing batch", leave=False):
        embedding_path = extract_speaker_embedding(audio_path, encoder, output_dir)

        results.append({
            "audio_path": audio_path,
            "embedding_path": embedding_path,
            "success": embedding_path is not None
        })

    return results


def update_manifest_with_embeddings(
    manifest: List[Dict],
    embedding_map: Dict[str, str],
    output_manifest_path: str
):
    """
    Update manifest with speaker_embedding_path field.
    """
    updated_manifest = []

    for entry in manifest:
        audio_path = entry.get("audio_path")

        # Add speaker embedding path if available
        if audio_path in embedding_map:
            entry["speaker_embedding_path"] = embedding_map[audio_path]

        # Update alternate takes with embedding paths
        alternates = entry.get("alternate_takes", [])
        for alt in alternates:
            alt_audio = alt.get("audio_path")
            if alt_audio in embedding_map:
                alt["speaker_embedding_path"] = embedding_map[alt_audio]

        updated_manifest.append(entry)

    # Save updated manifest
    with open(output_manifest_path, 'w') as f:
        json.dump(updated_manifest, f, indent=2)

    print(f"✅ Updated manifest saved to: {output_manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess speaker embeddings for training")
    parser.add_argument("--manifest", type=str,
                       default="./vocal_training_manifest_yamnet_filtered.json",
                       help="Input manifest path")
    parser.add_argument("--output_dir", type=str,
                       default="/mnt/msdd/speaker_embeddings",
                       help="Output directory for embeddings")
    parser.add_argument("--output_manifest", type=str,
                       default="./vocal_training_manifest_with_speaker_embs.json",
                       help="Output manifest with embedding paths")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Files per batch (GPU stays warm)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda or cpu")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of parallel GPU workers (if multiple GPUs)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip files that already have embeddings")

    args = parser.parse_args()

    print("=" * 80)
    print("Speaker Embedding Preprocessing")
    print("=" * 80)
    print(f"Manifest: {args.manifest}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print("Loading manifest...")
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    print(f"Loaded {len(manifest)} entries\n")

    # Collect all unique audio files
    print("Collecting audio files...")
    audio_files = collect_all_audio_files(manifest)
    print(f"Found {len(audio_files)} unique audio files\n")

    # Filter out existing embeddings if skip_existing
    if args.skip_existing:
        print("Checking for existing embeddings...")
        to_process = []
        for audio_path in audio_files:
            file_hash = get_file_hash(audio_path)
            filename = Path(audio_path).stem
            expected_path = output_dir / f"{file_hash}_{filename}_spk.pt"

            if not expected_path.exists():
                to_process.append(audio_path)

        print(f"Skipping {len(audio_files) - len(to_process)} existing embeddings")
        print(f"Processing {len(to_process)} new files\n")
        audio_files = to_process
    else:
        audio_files = list(audio_files)

    if len(audio_files) == 0:
        print("✅ All embeddings already exist!")
        return

    # Split into batches
    batches = [audio_files[i:i+args.batch_size]
               for i in range(0, len(audio_files), args.batch_size)]

    print(f"Processing {len(batches)} batches...")
    print()

    # Process batches
    all_results = []

    if args.device == "cuda" and torch.cuda.is_available():
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print()

        # Process each batch on GPU
        for i, batch in enumerate(batches):
            print(f"Batch {i+1}/{len(batches)} ({len(batch)} files)")

            results = process_batch_gpu(batch, output_dir, args.device)
            all_results.extend(results)

            # Stats
            success_count = sum(1 for r in results if r["success"])
            print(f"  ✅ Success: {success_count}/{len(batch)}")
            print()

    else:
        print(f"⚠ Using CPU (slower)")
        print()

        # Initialize encoder on CPU
        encoder = VoiceEncoder(device="cpu")
        encoder.eval()

        # Process all files
        for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
            embedding_path = extract_speaker_embedding(audio_path, encoder, output_dir)

            all_results.append({
                "audio_path": audio_path,
                "embedding_path": embedding_path,
                "success": embedding_path is not None
            })

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in all_results if r["success"])
    failed_count = len(all_results) - success_count

    print(f"Total processed: {len(all_results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print()

    if failed_count > 0:
        print(f"First 10 failures:")
        failures = [r for r in all_results if not r["success"]]
        for r in failures[:10]:
            print(f"  - {Path(r['audio_path']).name}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        print()

    # Create embedding map
    embedding_map = {
        r["audio_path"]: r["embedding_path"]
        for r in all_results
        if r["success"]
    }

    # Update manifest
    print("Updating manifest with embedding paths...")
    update_manifest_with_embeddings(manifest, embedding_map, args.output_manifest)

    print()
    print(f"✅ Done! Embeddings saved to: {args.output_dir}")
    print(f"✅ Updated manifest: {args.output_manifest}")


if __name__ == "__main__":
    main()
