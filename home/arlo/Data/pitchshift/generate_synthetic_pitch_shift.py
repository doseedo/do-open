#!/usr/bin/env python3
"""
Generate Synthetic Pitch-Shifted Data

Step 3 of the pipeline: After validating the translator, generate synthetic
pitch-shifted audio from the corpus for training the student model.

Creates paired data:
- (original_audio, shift_N) -> synthetic_shifted_audio

This creates training data for the lightweight student model.

Usage:
    python generate_synthetic_pitch_shift.py \
        --checkpoint ./checkpoints/best.pt \
        --output_dir /path/to/synthetic_data \
        --shifts -12 -7 -5 -3 3 5 7 12
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/do')

from models_pitch_shift import (
    RegisterAwareTranslator,
    RegisterAwareTranslatorDirect,
    RegisterAwareTranslatorLarge,
)


def pitch_shift_audio(audio: torch.Tensor, sr: int, shift: int) -> torch.Tensor:
    """Apply pitch shift to audio using librosa."""
    import librosa

    audio_np = audio.numpy()
    shifted_channels = []

    for ch in range(audio_np.shape[0]):
        shifted = librosa.effects.pitch_shift(
            audio_np[ch],
            sr=sr,
            n_steps=shift,
        )
        shifted_channels.append(shifted)

    return torch.from_numpy(np.stack(shifted_channels))


def estimate_pitch(audio: torch.Tensor, sr: int) -> int:
    """Estimate dominant pitch from audio."""
    try:
        import librosa
        mono = audio.mean(dim=0).numpy()
        pitches, magnitudes = librosa.piptrack(y=mono, sr=sr)
        pitch_idx = magnitudes.argmax()
        pitch_hz = pitches.flatten()[pitch_idx]
        if pitch_hz > 0:
            midi_pitch = int(12 * np.log2(pitch_hz / 440) + 69)
            return max(24, min(96, midi_pitch))
    except Exception:
        pass
    return 60


class SyntheticDataGenerator:
    """Generate synthetic pitch-shifted data for student training."""

    def __init__(
        self,
        translator_checkpoint: str,
        dcae_checkpoint: str,
        manifest_path: str,
        output_dir: str,
        instrument: str = 'trumpet',
        shifts: list = None,
        device: str = "cuda",
        max_files: int = None,
        skip_existing: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.shifts = shifts or [-12, -7, -5, -3, 3, 5, 7, 12]
        self.skip_existing = skip_existing

        # Load DCAE
        print("Loading DCAE...")
        from do.pipeline_do import DoTrainComponents
        comps = DoTrainComponents(checkpoint_dir=dcae_checkpoint, dtype="float32")
        self.dcae = comps.load_dcae()
        self.dcae = self.dcae.to(self.device).eval()

        # Load translator
        print("Loading translator...")
        checkpoint = torch.load(translator_checkpoint, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})

        model_type = config.get('model_type', 'residual')
        if model_type == 'direct':
            self.translator = RegisterAwareTranslatorDirect()
        elif model_type == 'large':
            self.translator = RegisterAwareTranslatorLarge()
        else:
            self.translator = RegisterAwareTranslator()

        self.translator.load_state_dict(state_dict)
        self.translator = self.translator.to(self.device).eval()

        print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")

        # Load manifest
        print("Loading manifest...")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Filter entries
        self.entries = []
        for entry in manifest:
            if instrument and entry.get('sub_group') != instrument:
                continue
            audio_path = entry.get('audio_path', '')
            if os.path.exists(audio_path):
                self.entries.append(entry)

        print(f"Found {len(self.entries)} {instrument} entries")

        if max_files:
            self.entries = self.entries[:max_files]
            print(f"Limited to: {len(self.entries)} files")

    @torch.no_grad()
    def process_single_file(self, entry: dict, shift: int) -> dict:
        """Process a single file with a specific shift."""
        audio_path = entry['audio_path']
        result = {
            'input': audio_path,
            'shift': shift,
            'success': False,
        }

        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)

            # Ensure stereo
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]

            # Estimate source pitch
            source_pitch = estimate_pitch(audio, sr)
            target_pitch = max(24, min(96, source_pitch + shift))

            # Apply pitch shift to audio
            shifted_audio = pitch_shift_audio(audio, sr, shift)

            # Encode to latent
            shifted_audio_batch = shifted_audio.unsqueeze(0).to(self.device)
            latent = self.dcae.encode(shifted_audio_batch, sr=sr)

            # Apply register-aware correction
            target_pitch_tensor = torch.tensor([target_pitch], device=self.device)
            shift_tensor = torch.tensor([float(shift)], device=self.device)

            corrected_latent = self.translator(latent, target_pitch_tensor, shift_tensor)

            # Decode to audio
            corrected_audio = self.dcae.decode(corrected_latent)

            # Prepare output paths
            stem = Path(audio_path).stem
            parent_name = Path(audio_path).parent.name

            # Create output directories
            audio_output_dir = self.output_dir / "audio" / parent_name
            audio_output_dir.mkdir(parents=True, exist_ok=True)

            # Save corrected audio
            corrected_audio = corrected_audio.squeeze(0).cpu()
            corrected_audio = corrected_audio / (corrected_audio.abs().max() + 1e-8) * 0.9

            output_path = audio_output_dir / f"{stem}_shift{shift:+d}.wav"
            torchaudio.save(str(output_path), corrected_audio, 44100)

            result['output_audio'] = str(output_path)
            result['source_pitch'] = source_pitch
            result['target_pitch'] = target_pitch
            result['success'] = True
            result['duration_sec'] = audio.shape[-1] / sr

        except Exception as e:
            result['error'] = str(e)

        return result

    def generate(self):
        """Generate synthetic pitch-shifted data for all files and shifts."""
        print("\n" + "=" * 60)
        print("SYNTHETIC PITCH-SHIFTED DATA GENERATION")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Files to process: {len(self.entries)}")
        print(f"Shifts: {self.shifts}")
        print(f"Total pairs: {len(self.entries) * len(self.shifts)}")
        print("=" * 60)

        start_time = datetime.now()

        all_results = []
        total_duration = 0.0

        for shift in self.shifts:
            print(f"\nProcessing shift = {shift:+d} semitones...")

            pbar = tqdm(self.entries, desc=f"Shift {shift:+d}")
            for entry in pbar:
                # Check if output exists
                if self.skip_existing:
                    stem = Path(entry['audio_path']).stem
                    parent = Path(entry['audio_path']).parent.name
                    expected_output = self.output_dir / "audio" / parent / f"{stem}_shift{shift:+d}.wav"
                    if expected_output.exists():
                        continue

                result = self.process_single_file(entry, shift)
                all_results.append(result)

                if result['success']:
                    total_duration += result.get('duration_sec', 0)

                pbar.set_postfix({
                    'success': sum(1 for r in all_results if r['success']),
                    'duration': f"{total_duration/3600:.1f}h"
                })

        # Summary
        successful = [r for r in all_results if r['success']]
        failed = [r for r in all_results if not r['success']]

        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total audio duration: {total_duration/3600:.2f} hours")
        print(f"Processing time: {datetime.now() - start_time}")
        print(f"Output: {self.output_dir}")

        # Save manifest
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'translator_checkpoint': str(self.output_dir),
            'shifts': self.shifts,
            'num_files': len(successful),
            'total_duration_hours': total_duration / 3600,
            'files': [
                {
                    'original_audio': r['input'],
                    'shifted_audio': r['output_audio'],
                    'shift_semitones': r['shift'],
                    'source_pitch': r.get('source_pitch'),
                    'target_pitch': r.get('target_pitch'),
                }
                for r in successful
            ]
        }

        manifest_path = self.output_dir / "synthetic_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved to: {manifest_path}")

        # Save failed list
        if failed:
            failed_path = self.output_dir / "failed_files.json"
            with open(failed_path, 'w') as f:
                json.dump(failed, f, indent=2)
            print(f"Failed files list: {failed_path}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Pitch-Shifted Data")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to translator checkpoint')
    parser.add_argument('--dcae_checkpoint', type=str,
                        default='/home/arlo/Data/ACE-Step/checkpoints',
                        help='Path to DCAE checkpoint directory')
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/Data/final_training_manifest_final.json',
                        help='Path to manifest JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for synthetic data')
    parser.add_argument('--instrument', type=str, default='trumpet',
                        help='Instrument to process')
    parser.add_argument('--shifts', type=int, nargs='+',
                        default=[-12, -7, -5, -3, 3, 5, 7, 12],
                        help='Pitch shifts to generate (in semitones)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Limit number of files to process')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip files that already have output')

    args = parser.parse_args()

    generator = SyntheticDataGenerator(
        translator_checkpoint=args.checkpoint,
        dcae_checkpoint=args.dcae_checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        instrument=args.instrument,
        shifts=args.shifts,
        device=args.device,
        max_files=args.max_files,
        skip_existing=args.skip_existing,
    )

    generator.generate()


if __name__ == "__main__":
    main()
