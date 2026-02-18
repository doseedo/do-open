#!/usr/bin/env python3
"""
Combined script to extract missing piano rolls and DCAE latents.
Processes entries from the vocal manifest that are missing either PR or DCAE.
"""

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import json
import os
import sys
import traceback
import difflib
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict
import re

import numpy as np
import pretty_midi
import torch
import torchaudio
from tqdm import tqdm
from acestep.pipeline_ace_step import ACEStepTrainComponents

# ===================== CONFIG =====================
INPUT_MANIFEST = Path("/home/arlo/Data/vocal_training_manifest_yamnet_filtered_COMPLETE.json")
OUTPUT_MANIFEST = Path("/home/arlo/Data/vocal_training_manifest_yamnet_filtered_FINAL.json")

# Output directories
PIANO_ROLL_DIR = Path("/mnt/msdd/piano_rolls")
DCAE_DIR = Path("/mnt/msdd/dcae_latentsnew")
PIANO_ROLL_DIR.mkdir(exist_ok=True, parents=True)
DCAE_DIR.mkdir(exist_ok=True, parents=True)

# Checkpoints
CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints"

# Logs
LOG_DIR = Path("/home/arlo/Data/extraction_logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)
PR_ERROR_LOG = LOG_DIR / "pr_errors.txt"
DCAE_ERROR_LOG = LOG_DIR / "dcae_errors.jsonl"
NO_MIDI_LOG = LOG_DIR / "no_midi_found.txt"

# MIDI/Piano Roll params
SAMPLE_RATE = 44100
HOP_LENGTH = 4096
SESSION_HINT_DIRNAMES = {
    "IO Settings", "Audio Files", "MIDI Files", "Melodyne",
    "Bounced Files", "Session File Backups"
}
MIDI_EXTS = {".mid", ".midi"}
FILENAME_MATCH_THRESHOLD = 0.55

# DCAE params
LATENT_SHAPE = (8, 16)
DOWNSAMPLE_FACTOR = 4096
MIN_SAMPLES = int(3 * 48000)
MAX_SAMPLES = int(12 * 60 * 48000)
SILENCE_THRESHOLD = 1e-6
STD_RANGE = (0.4, 2.2)
MEAN_RANGE = (-1.3, 1.3)

# GPU setup
NUM_GPUS = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(NUM_GPUS)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==================================================
# PIANO ROLL FUNCTIONS
# ==================================================

def frame_rate() -> float:
    return SAMPLE_RATE / HOP_LENGTH

def detect_session_root(audio_path: Path) -> Path:
    """Walk upward to find session root."""
    p = audio_path
    for parent in p.parents:
        if parent.name in SESSION_HINT_DIRNAMES:
            return parent.parent
    return audio_path.parent.parent

def find_all_midis(session_root: Path) -> List[Path]:
    """Find all MIDI files in session."""
    if not session_root.exists():
        return []
    midis = []
    prioritized = ["MIDI Files", "Midi Files", "MIDI", "Midi"]
    for sub in prioritized:
        candidate = session_root / sub
        if candidate.exists():
            for m in candidate.rglob("*"):
                if m.suffix.lower() in MIDI_EXTS:
                    midis.append(m)

    for m in session_root.rglob("*"):
        if m.suffix.lower() in MIDI_EXTS:
            midis.append(m)

    # De-dup
    seen = set()
    unique = []
    for m in midis:
        if m not in seen:
            unique.append(m)
            seen.add(m)
    return unique

def pick_best_midi(midis: List[Path], audio_stem: str) -> Optional[Path]:
    """Pick MIDI with best filename match."""
    if not midis:
        return None
    if len(midis) == 1:
        return midis[0]

    scores = []
    for m in midis:
        ratio = difflib.SequenceMatcher(None, audio_stem.lower(), m.stem.lower()).ratio()
        scores.append(ratio)

    best_idx = scores.index(max(scores))
    if scores[best_idx] >= FILENAME_MATCH_THRESHOLD:
        return midis[best_idx]
    return midis[0]

def midi_to_piano_roll(midi_path: Path, duration_sec: float) -> np.ndarray:
    """Convert MIDI to piano roll."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    fr = frame_rate()
    num_frames = int(np.ceil(duration_sec * fr))

    roll = np.zeros((128, num_frames), dtype=np.float32)

    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            start_frame = int(note.start * fr)
            end_frame = int(note.end * fr)
            if 0 <= note.pitch < 128:
                roll[note.pitch, start_frame:end_frame] = note.velocity / 127.0

    return roll

def extract_piano_roll(audio_path: str) -> Optional[str]:
    """Extract piano roll for audio file. Returns saved path or None."""
    try:
        audio_file = Path(audio_path)

        # Get duration
        info = torchaudio.info(str(audio_file))
        duration_sec = info.num_frames / info.sample_rate

        # Find session and MIDI
        session_root = detect_session_root(audio_file)
        midis = find_all_midis(session_root)

        if not midis:
            with open(NO_MIDI_LOG, "a") as f:
                f.write(f"{audio_path}\n")
            return None

        midi_file = pick_best_midi(midis, audio_file.stem)
        if not midi_file:
            return None

        # Convert to piano roll
        roll = midi_to_piano_roll(midi_file, duration_sec)

        # Save
        session_name = session_root.name
        out_dir = PIANO_ROLL_DIR / session_name
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{audio_file.stem}.pianoroll.npy"
        np.save(out_path, roll)

        return str(out_path)

    except Exception as e:
        with open(PR_ERROR_LOG, "a") as f:
            f.write(f"{audio_path}: {str(e)}\n")
        return None

# ==================================================
# DCAE FUNCTIONS
# ==================================================

class LatentProcessor:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.model = self._load_model()

    def _load_model(self):
        """Load DCAE model."""
        torch.cuda.set_device(self.gpu_id)
        components = ACEStepTrainComponents(checkpoint_dir=CHECKPOINT_DIR, device_id=self.gpu_id)
        model = components.load_dcae()  # This loads and returns the DCAE model
        return model

    def _validate_audio(self, waveform):
        """Validate audio tensor."""
        if not torch.isfinite(waveform).all():
            return False, "Non-finite values"
        if torch.abs(waveform).max() < SILENCE_THRESHOLD:
            return False, "Silent audio"
        return True, None

    def _validate_latent(self, latent):
        """Validate latent tensor."""
        if not torch.isfinite(latent).all():
            return False, "Non-finite latent"
        std_val = latent.std().item()
        mean_val = latent.mean().item()
        if not (STD_RANGE[0] <= std_val <= STD_RANGE[1]):
            return False, f"STD out of range: {std_val:.3f}"
        if not (MEAN_RANGE[0] <= mean_val <= MEAN_RANGE[1]):
            return False, f"Mean out of range: {mean_val:.3f}"
        return True, None

    def process_file(self, audio_path: str) -> Optional[str]:
        """Process audio file and save DCAE latent. Returns saved path or None."""
        try:
            audio_file = Path(audio_path)

            # Load audio
            waveform, sr = torchaudio.load(str(audio_file))

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 48kHz
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                waveform = resampler(waveform)

            # Check duration
            num_samples = waveform.shape[1]
            if num_samples < MIN_SAMPLES or num_samples > MAX_SAMPLES:
                return None

            # Validate audio
            valid, msg = self._validate_audio(waveform)
            if not valid:
                return None

            # Encode
            waveform = waveform.to(self.gpu_id)
            with torch.no_grad():
                latent = self.model.encode(waveform.unsqueeze(0))
                latent = latent.squeeze(0).cpu()

            # Validate latent
            valid, msg = self._validate_latent(latent)
            if not valid:
                return None

            # Determine save path based on audio path structure
            parts = audio_file.parts

            # Find protools/protoolsA and date
            protools_root = None
            date = None
            new_prev = None
            session = None

            if 'gcs-bucket' in parts:
                gcs_idx = parts.index('gcs-bucket')
                if gcs_idx + 1 < len(parts):
                    next_part = parts[gcs_idx + 1]
                    if next_part in ['protools', 'protoolsA']:
                        protools_root = next_part

            for part in parts:
                if part.startswith('2025-') and len(part) == 10:
                    date = part
                    break

            if 'New' in parts:
                idx = parts.index('New')
                new_prev = 'New'
                session = parts[idx + 1]
            elif 'Prev' in parts:
                idx = parts.index('Prev')
                new_prev = 'Prev'
                session = parts[idx + 1]

            # Build save path
            if protools_root and date and new_prev and session:
                out_dir = DCAE_DIR / protools_root / date / new_prev / session / "Audio Files"
            elif date and new_prev and session:
                out_dir = DCAE_DIR / date / new_prev / session / "Audio Files"
            else:
                return None

            out_dir.mkdir(exist_ok=True, parents=True)
            out_path = out_dir / f"{audio_file.stem}.pt"
            torch.save(latent, out_path)

            return str(out_path)

        except Exception as e:
            error_entry = {
                "audio_path": audio_path,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            with open(DCAE_ERROR_LOG, "a") as f:
                f.write(json.dumps(error_entry) + "\n")
            return None

# ==================================================
# MAIN PROCESSING
# ==================================================

def process_entries_gpu(gpu_id, entries):
    """Process entries on a specific GPU."""
    processor = LatentProcessor(gpu_id=gpu_id)

    results = []
    for entry in tqdm(entries, desc=f"GPU {gpu_id}", position=gpu_id):
        audio_path = entry['audio_path']

        pr_path = entry.get('piano_roll_path', '')
        dcae_path = entry.get('dcae_path', '')

        needs_pr = not (pr_path and Path(pr_path).exists())
        needs_dcae = not (dcae_path and Path(dcae_path).exists())

        result = entry.copy()

        # Extract piano roll if needed
        if needs_pr:
            pr_path = extract_piano_roll(audio_path)
            if pr_path:
                result['piano_roll_path'] = pr_path

        # Extract DCAE if needed
        if needs_dcae:
            dcae_path = processor.process_file(audio_path)
            if dcae_path:
                result['dcae_path'] = dcae_path

        results.append(result)

    return results

def main():
    print("="*80)
    print("Extract Missing Piano Rolls and DCAE Latents")
    print("="*80)
    print(f"Input manifest: {INPUT_MANIFEST}")
    print(f"Output manifest: {OUTPUT_MANIFEST}\n")

    # Load manifest
    with open(INPUT_MANIFEST) as f:
        manifest = json.load(f)

    print(f"Total entries: {len(manifest)}")

    # Find entries missing PR or DCAE (check if path exists)
    to_process = []
    for entry in manifest:
        pr_path = entry.get('piano_roll_path', '')
        dcae_path = entry.get('dcae_path', '')

        needs_pr = not (pr_path and Path(pr_path).exists())
        needs_dcae = not (dcae_path and Path(dcae_path).exists())

        if needs_pr or needs_dcae:
            to_process.append(entry)

    print(f"Entries needing extraction: {len(to_process)}\n")

    if not to_process:
        print("No entries need processing!")
        return

    # Split work across GPUs
    if NUM_GPUS > 1:
        chunks = np.array_split(to_process, NUM_GPUS)
        processes = []

        for gpu_id in range(NUM_GPUS):
            p = mp.Process(target=process_entries_gpu, args=(gpu_id, chunks[gpu_id].tolist()))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # Single GPU
        results = process_entries_gpu(0, to_process)

        # Update manifest
        result_dict = {r['audio_path']: r for r in results}
        for i, entry in enumerate(manifest):
            if entry['audio_path'] in result_dict:
                manifest[i] = result_dict[entry['audio_path']]

        # Save
        print(f"\nSaving to {OUTPUT_MANIFEST}...")
        with open(OUTPUT_MANIFEST, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"✅ Done! Saved: {OUTPUT_MANIFEST}")

if __name__ == "__main__":
    main()
