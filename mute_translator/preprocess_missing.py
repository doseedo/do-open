#!/usr/bin/env python3
"""
Preprocess Missing Entries

Extracts latents and conditioning for manifest entries that are missing them.
Updates the manifest in-place with the new paths.

Usage:
    python preprocess_missing.py --manifest ./mute_manifest_deduped.json --output-dir /mnt/msdd2

    # Specific formats only
    python preprocess_missing.py --manifest ./manifest.json --output-dir /mnt/msdd2 --formats dcae_latents,onsets

    # Multi-GPU
    python preprocess_missing.py --manifest ./manifest.json --output-dir /mnt/msdd2 --gpus 0,1,2,3
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import gc
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from collections import defaultdict
import re

import numpy as np
import torch
import torchaudio


# ===================== PATH FIXING =====================
def fix_mount_path(path: str) -> str:
    """Fix mount path from msdd to msdd2 if needed."""
    if path and '/mnt/msdd/' in path:
        return path.replace('/mnt/msdd/', '/mnt/msdd2/')
    return path


# ===================== CONFIGURATION =====================
@dataclass
class Config:
    sample_rate: int = 44100
    hop_length: int = 4096
    n_fft: int = 8192
    fmin: float = 65.41
    fmax: float = 2093.00
    dcae_sr: int = 44100
    dcae_hop: int = 4096
    latent_shape: Tuple[int, int] = (8, 16)
    silence_threshold: float = 1e-6
    min_duration_sec: float = 0.5
    max_duration_sec: float = 480.0

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length


# Available formats
ALL_FORMATS = {"dcae_latents", "amp", "rframe", "rbend", "f0", "f0_masked", "onsets"}
CONDITIONING_FORMATS = {"amp", "rframe", "rbend", "f0", "f0_masked", "onsets"}


# ===================== LAZY MODEL LOADING =====================
class ModelCache:
    def __init__(self, checkpoint_dir: str, device: torch.device):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self._dcae_model = None

    @property
    def dcae(self):
        if self._dcae_model is None:
            print(f"Loading DCAE model on {self.device}...")
            from acestep.pipeline_ace_step import ACEStepPipeline
            pipeline = ACEStepPipeline(checkpoint_dir=self.checkpoint_dir)
            pipeline.load_checkpoint(self.checkpoint_dir)
            self._dcae_model = pipeline.music_dcae.eval().to(self.device)
            print(f"DCAE model loaded on {self.device}")
        return self._dcae_model

    def clear(self):
        self._dcae_model = None
        torch.cuda.empty_cache()
        gc.collect()


# ===================== EXTRACTION FUNCTIONS =====================

def load_audio(audio_path: Path, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


def extract_dcae_latents(
    waveform: torch.Tensor,
    sr: int,
    model_cache: ModelCache,
    config: Config
) -> torch.Tensor:
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    waveform = waveform / (waveform.abs().max() + 1e-8)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        waveform = waveform.to(model_cache.device)
        audio_batch = waveform.unsqueeze(0).float()
        audio_lengths = torch.tensor([waveform.shape[-1]], device=model_cache.device)
        latents, _ = model_cache.dcae.encode(
            audios=audio_batch,
            audio_lengths=audio_lengths,
            sr=sr
        )

    latents = latents.float().squeeze(0).cpu()
    if not torch.isfinite(latents).all():
        raise ValueError("DCAE latents contain NaN/inf")
    return latents


def extract_conditioning_features(
    waveform: torch.Tensor,
    sr: int,
    config: Config
) -> Dict[str, np.ndarray]:
    import torchcrepe

    device = "cuda"
    y = waveform[0]

    win = config.n_fft
    hop = config.hop_length
    if y.numel() < win:
        y = torch.nn.functional.pad(y, (0, win - y.numel()))

    crepe_sr = 16000
    crepe_hop = int(crepe_sr / config.frame_rate)
    y_16k = torchaudio.functional.resample(y.unsqueeze(0), sr, crepe_sr).squeeze(0)

    # RMS amplitude
    frames = y.unfold(0, win, hop)
    amp = torch.sqrt((frames ** 2).mean(dim=1) + 1e-12)
    if amp.max() > 0:
        amp = amp / amp.max()
    amp = amp.numpy().astype(np.float32)

    # F0 with torchcrepe
    y_crepe = y_16k.unsqueeze(0).to(device)
    with torch.inference_mode():
        f0, periodicity = torchcrepe.predict(
            y_crepe,
            sample_rate=crepe_sr,
            hop_length=crepe_hop,
            fmin=config.fmin,
            fmax=config.fmax,
            pad=True,
            model="tiny",
            batch_size=256,
            device=device,
            return_periodicity=True
        )
        f0 = f0[0].cpu()
        periodicity = periodicity[0].cpu()

    T = min(len(amp), len(f0))
    amp = amp[:T]
    f0 = f0[:T]
    periodicity = periodicity[:T]

    vmask = ((periodicity > 0.5) & (torch.from_numpy(amp) > 0.02)).float().numpy()
    f0_safe = np.where(f0.numpy() > 0, f0.numpy(), 1.0)
    rbend = 12.0 * np.log2(f0_safe / 440.0)
    rbend = np.where(np.isfinite(rbend), rbend, 0.0).astype(np.float32)
    rbend = rbend * vmask

    f0_np = f0.numpy().astype(np.float32)
    f0_masked = f0_np * vmask

    amp_diff = np.diff(amp, prepend=amp[0])
    onsets = (amp_diff > 0.05).astype(np.float32)

    return {
        "amp": amp.astype(np.float32),
        "rframe": vmask.astype(np.float32),
        "rbend": rbend.astype(np.float32),
        "f0": f0_np,
        "f0_masked": f0_masked.astype(np.float32),
        "onsets": onsets.astype(np.float32)
    }


# ===================== MAIN PROCESSOR =====================

def get_output_paths(audio_path: Path, output_base: Path) -> Dict[str, Path]:
    """Generate output paths based on audio path structure"""
    parts = audio_path.parts

    # Find session structure from path
    for marker in ["gcs-bucket", "protools", "protoolsA"]:
        if marker in parts:
            idx = parts.index(marker)
            rel_parts = parts[idx + 1:-1]
            break
    else:
        rel_parts = [audio_path.parent.name]

    # Create output directory structure
    latent_dir = output_base / "dcae_latentsnew" / "/".join(rel_parts)
    cond_dir = output_base / "evenmoreconditioning" / "/".join(rel_parts)

    stem = audio_path.stem

    return {
        "dcae_latents": latent_dir / f"{stem}.pt",
        "amp": cond_dir / f"{stem}.amp.npy",
        "rframe": cond_dir / f"{stem}.rframe.npy",
        "rbend": cond_dir / f"{stem}.rbend.npy",
        "f0": cond_dir / f"{stem}.f0.npy",
        "f0_masked": cond_dir / f"{stem}.f0_masked.npy",
        "onsets": cond_dir / f"{stem}.onsets.npy",
    }


def process_entry(
    entry: Dict,
    output_base: Path,
    model_cache: ModelCache,
    config: Config,
    formats: Set[str]
) -> Tuple[str, Dict[str, Any]]:
    """Process a single manifest entry"""

    audio_path = Path(entry.get("audio_path", ""))
    if not audio_path.exists():
        return "failed", {"error": f"Audio not found: {audio_path}"}

    out_paths = get_output_paths(audio_path, output_base)
    results = {}

    try:
        # Load audio
        waveform, sr = load_audio(audio_path, config.sample_rate)

        duration = waveform.shape[-1] / sr
        if duration < config.min_duration_sec:
            raise ValueError(f"Too short: {duration:.2f}s")
        if duration > config.max_duration_sec:
            raise ValueError(f"Too long: {duration:.2f}s")
        if waveform.abs().max() < config.silence_threshold:
            raise ValueError("Silent audio")

        # Extract DCAE latents
        if "dcae_latents" in formats:
            out_paths["dcae_latents"].parent.mkdir(parents=True, exist_ok=True)
            latents = extract_dcae_latents(waveform, sr, model_cache, config)
            torch.save({
                "latents": latents,
                "length": latents.shape[-1],
                "original_path": str(audio_path),
                "original_duration": duration
            }, out_paths["dcae_latents"])
            results["latent_path"] = str(out_paths["dcae_latents"])

        # Extract conditioning
        cond_formats = formats & CONDITIONING_FORMATS
        if cond_formats:
            out_paths["amp"].parent.mkdir(parents=True, exist_ok=True)
            cond_features = extract_conditioning_features(waveform, sr, config)
            results["conditioning_paths"] = {}
            for fmt in cond_formats:
                np.save(out_paths[fmt], cond_features[fmt])
                results["conditioning_paths"][fmt] = str(out_paths[fmt])

        return "success", results

    except Exception as e:
        return "failed", {"error": str(e), "trace": traceback.format_exc()}

    finally:
        torch.cuda.empty_cache()
        gc.collect()


def find_missing_entries(manifest: List[Dict], formats: Set[str]) -> List[int]:
    """Find indices of entries missing required paths"""
    missing_indices = []

    for i, entry in enumerate(manifest):
        needs_processing = False

        # Check latent_path
        if "dcae_latents" in formats:
            latent_path = entry.get("latent_path", "")
            if not latent_path or not Path(latent_path).exists():
                needs_processing = True

        # Check conditioning paths
        cond_formats = formats & CONDITIONING_FORMATS
        if cond_formats:
            cond_paths = entry.get("conditioning_paths", {})
            for fmt in cond_formats:
                path = cond_paths.get(fmt, "")
                if not path or not Path(path).exists():
                    needs_processing = True
                    break

        if needs_processing:
            missing_indices.append(i)

    return missing_indices


def gpu_worker(
    gpu_id: int,
    entries_with_indices: List[Tuple[int, Dict]],
    output_base: Path,
    checkpoint_dir: str,
    formats: Set[str],
    config: Config,
    progress_queue: Queue,
    result_queue: Queue
):
    """Worker process for a single GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    model_cache = ModelCache(checkpoint_dir, device)

    results = []
    for i, (idx, entry) in enumerate(entries_with_indices):
        status, result = process_entry(entry, output_base, model_cache, config, formats)
        results.append((idx, status, result))
        progress_queue.put((gpu_id, i + 1, len(entries_with_indices), status))

    result_queue.put((gpu_id, results))


def main():
    parser = argparse.ArgumentParser(description="Preprocess missing manifest entries")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--output-dir", required=True, help="Base output directory (e.g., /mnt/msdd2)")
    parser.add_argument("--checkpoint-dir", default="/home/arlo/Data/ACE-Step/checkpoints",
                        help="DCAE checkpoint directory")
    parser.add_argument("--formats", default="dcae_latents,onsets,amp,f0,f0_masked,rframe,rbend",
                        help="Comma-separated formats to extract")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--filter-group", type=str, help="Only process entries with this sub_group")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be processed")

    args = parser.parse_args()

    # Parse formats
    formats = set(f.strip() for f in args.formats.split(","))
    print(f"Extracting formats: {', '.join(sorted(formats))}")

    # Parse GPUs
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    print(f"Using GPUs: {gpu_ids}")

    # Load manifest
    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Loaded manifest: {len(manifest)} entries")

    # Filter by group if specified
    if args.filter_group:
        filtered_indices = [i for i, e in enumerate(manifest)
                          if e.get("sub_group") == args.filter_group]
        print(f"Filtered to {len(filtered_indices)} entries with sub_group='{args.filter_group}'")
    else:
        filtered_indices = list(range(len(manifest)))

    # Find missing entries (apply path fix when checking existence)
    missing_indices = []
    for i in filtered_indices:
        entry = manifest[i]
        needs_processing = False

        if "dcae_latents" in formats:
            latent_path = fix_mount_path(entry.get("latent_path", ""))
            if not latent_path or not Path(latent_path).exists():
                needs_processing = True

        cond_formats = formats & CONDITIONING_FORMATS
        if cond_formats:
            cond_paths = entry.get("conditioning_paths", {})
            for fmt in cond_formats:
                path = fix_mount_path(cond_paths.get(fmt, ""))
                if not path or not Path(path).exists():
                    needs_processing = True
                    break

        if needs_processing:
            missing_indices.append(i)

    print(f"Found {len(missing_indices)} entries missing required paths")

    if args.dry_run:
        # Count totals across ALL missing entries
        total_missing_latent = 0
        total_missing_cond = 0
        for i in missing_indices:
            entry = manifest[i]
            latent_path = fix_mount_path(entry.get("latent_path", ""))
            if not latent_path or not Path(latent_path).exists():
                total_missing_latent += 1

            cond_paths = entry.get("conditioning_paths", {})
            for fmt in CONDITIONING_FORMATS:
                path = fix_mount_path(cond_paths.get(fmt, ""))
                if not path or not Path(path).exists():
                    total_missing_cond += 1
                    break

        print(f"\nSummary: {total_missing_latent} missing latents, {total_missing_cond} missing conditioning")
        print("\nDry run - sample entries to process:")

        # Show sample of entries
        for i in missing_indices[:20]:
            entry = manifest[i]
            latent_path = fix_mount_path(entry.get("latent_path", ""))
            has_latent = latent_path and Path(latent_path).exists()

            cond_paths = entry.get("conditioning_paths", {})
            missing_conds = []
            for fmt in CONDITIONING_FORMATS:
                path = fix_mount_path(cond_paths.get(fmt, ""))
                if not path or not Path(path).exists():
                    missing_conds.append(fmt)

            status = []
            if not has_latent:
                status.append("no_latent")
            if missing_conds:
                status.append(f"no_cond:{','.join(missing_conds[:3])}")

            print(f"  [{i}] {Path(entry.get('audio_path', 'N/A')).name} - {' | '.join(status)}")

        if len(missing_indices) > 20:
            print(f"  ... and {len(missing_indices) - 20} more")
        return

    if not missing_indices:
        print("Nothing to process!")
        return

    output_base = Path(args.output_dir)
    config = Config()

    # Prepare work items
    work_items = [(i, manifest[i]) for i in missing_indices]

    # Split across GPUs
    chunks = [work_items[i::len(gpu_ids)] for i in range(len(gpu_ids))]

    # Create queues
    progress_queue = Queue()
    result_queue = Queue()

    # Start workers
    workers = []
    for gpu_id, chunk in zip(gpu_ids, chunks):
        if not chunk:
            continue
        p = Process(target=gpu_worker, args=(
            gpu_id, chunk, output_base, args.checkpoint_dir,
            formats, config, progress_queue, result_queue
        ))
        p.start()
        workers.append(p)

    # Monitor progress
    from tqdm import tqdm
    pbar = tqdm(total=len(missing_indices), desc="Processing")

    completed = 0
    stats = defaultdict(int)
    while completed < len(missing_indices):
        try:
            gpu_id, current, total, status = progress_queue.get(timeout=1)
            pbar.update(1)
            pbar.set_postfix({"gpu": gpu_id, "status": status})
            stats[status] += 1
            completed += 1
        except:
            if not any(p.is_alive() for p in workers):
                break

    pbar.close()

    # Collect results and update manifest
    all_results = []
    for _ in workers:
        try:
            gpu_id, results = result_queue.get(timeout=5)
            all_results.extend(results)
        except:
            pass

    for p in workers:
        p.join()

    # Update manifest with new paths
    updated = 0
    for idx, status, result in all_results:
        if status == "success":
            if "latent_path" in result:
                manifest[idx]["latent_path"] = result["latent_path"]
            if "conditioning_paths" in result:
                # Merge conditioning paths
                existing_cond = manifest[idx].get("conditioning_paths", {})
                existing_cond.update(result["conditioning_paths"])
                manifest[idx]["conditioning_paths"] = existing_cond
            updated += 1

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Total processed: {len(missing_indices)}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  Manifest updated: {updated} entries")
    print(f"  Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
