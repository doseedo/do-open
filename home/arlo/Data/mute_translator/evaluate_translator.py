#!/usr/bin/env python3
"""
Evaluate Mute Translator

Step 2: Test the trained translator before generating synthetic data.

This script:
1. Loads trained translator checkpoint
2. Converts sample dry latents to muted
3. Decodes both to audio using DCAE
4. Saves comparison audio files for listening tests
5. Computes quantitative metrics

Usage:
    python evaluate_translator.py --checkpoint ./checkpoints/best.pt --output_dir ./eval_output
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/mute_translator')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/dø')

from models import MuteTranslator, MuteTranslatorLarge
from dataset import load_manifest, VERIFIED_MUTED_PATTERNS


def load_dcae(checkpoint_dir: str, device: str = "cuda"):
    """Load DCAE for encoding/decoding audio."""
    from do.pipeline_do import DoTrainComponents

    comps = DoTrainComponents(checkpoint_dir=checkpoint_dir, dtype="float32")
    dcae = comps.load_dcae()
    dcae = dcae.to(device).eval()
    return dcae


def load_translator(checkpoint_path: str, device: str = "cuda"):
    """Load trained translator from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Detect model type from state dict
    state_dict = checkpoint['model_state_dict']
    if 'input_proj.weight' in state_dict:
        # Check input proj channels to determine type
        in_channels = state_dict['input_proj.weight'].shape[1]
        if in_channels == 8:
            model = MuteTranslator()
        else:
            model = MuteTranslatorLarge()
    else:
        model = MuteTranslator()

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"Loaded translator from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Training loss: {checkpoint.get('metrics', {}).get('loss', '?')}")

    return model, checkpoint


def encode_audio(dcae, audio_path: str, device: str = "cuda") -> torch.Tensor:
    """Encode audio file to latent using DCAE."""
    # Load audio
    audio, sr = torchaudio.load(audio_path)

    # Ensure stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    # Add batch dim
    audio = audio.unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        latent = dcae.encode(audio, sr=sr)

    return latent


def decode_latent(dcae, latent: torch.Tensor, output_path: str):
    """Decode latent to audio and save."""
    with torch.no_grad():
        audio = dcae.decode(latent)

    # audio shape: [B, 2, T]
    audio = audio.squeeze(0).cpu()

    # Normalize
    audio = audio / (audio.abs().max() + 1e-8) * 0.9

    torchaudio.save(output_path, audio, 44100)
    return output_path


def compute_spectral_distance(audio1: torch.Tensor, audio2: torch.Tensor) -> float:
    """Compute spectral distance between two audio tensors."""
    # Simple mel-spectrogram comparison
    n_fft = 2048
    hop = 512
    n_mels = 128

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )

    mel1 = mel_transform(audio1.mean(dim=0, keepdim=True))
    mel2 = mel_transform(audio2.mean(dim=0, keepdim=True))

    # Log mel
    mel1 = torch.log(mel1 + 1e-8)
    mel2 = torch.log(mel2 + 1e-8)

    # Truncate to same length
    min_len = min(mel1.shape[-1], mel2.shape[-1])
    mel1 = mel1[..., :min_len]
    mel2 = mel2[..., :min_len]

    return F.mse_loss(mel1, mel2).item()


def compute_centroid_shift(audio1: torch.Tensor, audio2: torch.Tensor) -> float:
    """Compute spectral centroid shift (muted should have higher centroid)."""
    from scipy import signal

    def spectral_centroid(audio_np):
        f, t, Sxx = signal.spectrogram(audio_np, fs=44100, nperseg=2048, noverlap=1024)
        centroid = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8)
        return np.mean(centroid)

    audio1_np = audio1.mean(dim=0).numpy()
    audio2_np = audio2.mean(dim=0).numpy()

    c1 = spectral_centroid(audio1_np)
    c2 = spectral_centroid(audio2_np)

    return c2 - c1  # Positive = muted has higher centroid (expected)


class TranslatorEvaluator:
    def __init__(
        self,
        checkpoint_path: str,
        dcae_checkpoint_dir: str,
        manifest_path: str,
        output_dir: str,
        device: str = "cuda",
        num_samples: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_samples = num_samples

        # Load models
        print("Loading DCAE...")
        self.dcae = load_dcae(dcae_checkpoint_dir, device)

        print("Loading translator...")
        self.translator, self.checkpoint = load_translator(checkpoint_path, device)

        # Load manifest
        print("Loading manifest...")
        self.dry_entries, self.muted_entries = load_manifest(manifest_path)
        print(f"  Dry: {len(self.dry_entries)}, Muted: {len(self.muted_entries)}")

    def evaluate_sample(
        self,
        dry_audio_path: str,
        sample_name: str,
    ) -> dict:
        """Evaluate translator on a single dry sample."""
        sample_dir = self.output_dir / sample_name
        sample_dir.mkdir(exist_ok=True)

        results = {'name': sample_name}

        try:
            # 1. Encode dry audio
            print(f"  Encoding {sample_name}...")
            dry_latent = encode_audio(self.dcae, dry_audio_path, self.device)
            results['latent_shape'] = list(dry_latent.shape)

            # 2. Translate to muted
            print(f"  Translating...")
            with torch.no_grad():
                muted_latent = self.translator(dry_latent)

            # 3. Decode both
            print(f"  Decoding dry...")
            dry_output = sample_dir / "dry_original.wav"
            decode_latent(self.dcae, dry_latent, str(dry_output))

            print(f"  Decoding muted...")
            muted_output = sample_dir / "muted_translated.wav"
            decode_latent(self.dcae, muted_latent, str(muted_output))

            results['dry_audio'] = str(dry_output)
            results['muted_audio'] = str(muted_output)

            # 4. Compute metrics
            dry_audio, _ = torchaudio.load(str(dry_output))
            muted_audio, _ = torchaudio.load(str(muted_output))

            results['spectral_distance'] = compute_spectral_distance(dry_audio, muted_audio)
            results['centroid_shift'] = compute_centroid_shift(dry_audio, muted_audio)

            # 5. Latent statistics
            results['dry_latent_mean'] = dry_latent.mean().item()
            results['dry_latent_std'] = dry_latent.std().item()
            results['muted_latent_mean'] = muted_latent.mean().item()
            results['muted_latent_std'] = muted_latent.std().item()
            results['latent_diff_norm'] = (muted_latent - dry_latent).norm().item()

            results['success'] = True

        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            print(f"  Error: {e}")

        return results

    def evaluate_against_real_muted(self, sample_idx: int = 0) -> dict:
        """
        Compare translated muted vs real muted.

        This is a sanity check to see if the translation
        produces latents similar to real muted recordings.
        """
        if sample_idx >= len(self.muted_entries):
            return {}

        muted_entry = self.muted_entries[sample_idx]
        muted_audio_path = muted_entry['audio_path']
        muted_name = Path(muted_audio_path).stem

        sample_dir = self.output_dir / f"comparison_{muted_name}"
        sample_dir.mkdir(exist_ok=True)

        results = {'name': muted_name, 'type': 'comparison'}

        try:
            # Encode real muted
            print(f"  Encoding real muted: {muted_name}")
            real_muted_latent = encode_audio(self.dcae, muted_audio_path, self.device)

            # Decode real muted (roundtrip)
            real_output = sample_dir / "real_muted.wav"
            decode_latent(self.dcae, real_muted_latent, str(real_output))

            # Get a random dry sample and translate
            import random
            dry_entry = random.choice(self.dry_entries)
            dry_audio_path = dry_entry['audio_path']

            if os.path.exists(dry_audio_path):
                dry_latent = encode_audio(self.dcae, dry_audio_path, self.device)

                with torch.no_grad():
                    translated_latent = self.translator(dry_latent)

                translated_output = sample_dir / "translated_muted.wav"
                decode_latent(self.dcae, translated_latent, str(translated_output))

                # Compare distributions
                results['real_muted_mean'] = real_muted_latent.mean().item()
                results['real_muted_std'] = real_muted_latent.std().item()
                results['translated_mean'] = translated_latent.mean().item()
                results['translated_std'] = translated_latent.std().item()

                results['real_audio'] = str(real_output)
                results['translated_audio'] = str(translated_output)
                results['success'] = True

        except Exception as e:
            results['success'] = False
            results['error'] = str(e)

        return results

    def run_evaluation(self):
        """Run full evaluation."""
        print("\n" + "=" * 60)
        print("MUTE TRANSLATOR EVALUATION")
        print("=" * 60)

        all_results = {
            'checkpoint': str(self.checkpoint.get('epoch', 'unknown')),
            'timestamp': datetime.now().isoformat(),
            'samples': [],
            'comparisons': [],
        }

        # Evaluate on dry samples
        print("\n--- Evaluating dry → muted translation ---")
        import random
        sample_entries = random.sample(
            self.dry_entries,
            min(self.num_samples, len(self.dry_entries))
        )

        for i, entry in enumerate(sample_entries):
            audio_path = entry['audio_path']
            if not os.path.exists(audio_path):
                continue

            name = f"sample_{i:02d}_{Path(audio_path).stem[:30]}"
            print(f"\n[{i+1}/{len(sample_entries)}] {name}")

            results = self.evaluate_sample(audio_path, name)
            all_results['samples'].append(results)

        # Compare with real muted
        print("\n--- Comparing with real muted ---")
        for i in range(min(3, len(self.muted_entries))):
            print(f"\n[Comparison {i+1}]")
            results = self.evaluate_against_real_muted(i)
            if results:
                all_results['comparisons'].append(results)

        # Summary statistics
        successful = [r for r in all_results['samples'] if r.get('success')]
        if successful:
            avg_centroid_shift = np.mean([r['centroid_shift'] for r in successful])
            avg_spectral_dist = np.mean([r['spectral_distance'] for r in successful])
            avg_latent_diff = np.mean([r['latent_diff_norm'] for r in successful])

            all_results['summary'] = {
                'num_samples': len(successful),
                'avg_centroid_shift_hz': avg_centroid_shift,
                'avg_spectral_distance': avg_spectral_dist,
                'avg_latent_diff_norm': avg_latent_diff,
            }

            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Successful samples: {len(successful)}/{len(all_results['samples'])}")
            print(f"Avg centroid shift: {avg_centroid_shift:.1f} Hz")
            print(f"Avg spectral distance: {avg_spectral_dist:.4f}")
            print(f"Avg latent diff norm: {avg_latent_diff:.4f}")

        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {results_path}")
        print(f"Audio samples saved to: {self.output_dir}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mute Translator")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to translator checkpoint')
    parser.add_argument('--dcae_checkpoint', type=str,
                        default='/home/arlo/Data/ACE-Step/checkpoints',
                        help='Path to DCAE checkpoint directory')
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/Data.backup/final_training_manifest_brass_only.json',
                        help='Path to manifest JSON')
    parser.add_argument('--output_dir', type=str,
                        default='/home/arlo/Data/mute_translator/eval_output',
                        help='Output directory for evaluation results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    evaluator = TranslatorEvaluator(
        checkpoint_path=args.checkpoint,
        dcae_checkpoint_dir=args.dcae_checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
    )

    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
