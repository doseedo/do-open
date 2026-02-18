#!/usr/bin/env python3
"""
Evaluation script for Inverse Audio Effects System.
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inverse_afx.data.datasets import InverseAFxDataset
from inverse_afx.training.train_system import InverseAFxSystem
from inverse_afx.evaluation.metrics import evaluate_system
from inverse_afx.export.daw_export import chain_to_daw_preset, generate_processing_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Inverse AFx System")

    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_dir", "-t",
        type=str,
        required=True,
        help="Directory containing test audio files",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--effect_types",
        type=str,
        nargs="+",
        default=["eq", "compressor", "reverb", "distortion", "chorus", "delay"],
        help="Effect types",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        help="Save recovered audio files",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = InverseAFxSystem.load_from_checkpoint(
        args.checkpoint,
        map_location=args.device,
    )
    model.eval()
    model.to(args.device)

    # Create test dataset
    print(f"Loading test data from {args.test_dir}...")
    test_dataset = InverseAFxDataset(
        audio_dir=args.test_dir,
        sample_rate=44100,
        segment_length=131072,
        max_chain_length=6,
        effect_types=args.effect_types,
        mode='online',
        augment=False,
    )

    # Run evaluation
    print("Running evaluation...")
    results = evaluate_system(
        model=model,
        test_dataset=test_dataset,
        effect_types=args.effect_types,
        device=args.device,
        max_samples=args.max_samples,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\nDry Signal Recovery:")
    for metric, value in results['dry_recovery'].items():
        print(f"  {metric}: {value:.4f}")

    print("\nChain Estimation:")
    for metric, value in results['chain_estimation'].items():
        print(f"  {metric}: {value:.4f}")

    print("\nReconstruction:")
    for metric, value in results['reconstruction'].items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Optionally save audio samples
    if args.save_audio:
        audio_dir = output_dir / "audio_samples"
        audio_dir.mkdir(exist_ok=True)

        print("\nSaving audio samples...")
        num_samples = min(10, len(test_dataset))

        with torch.no_grad():
            for i in range(num_samples):
                sample = test_dataset[i]
                wet = sample['wet_audio'].unsqueeze(0).to(args.device)
                dry_gt = sample['dry_audio']

                # Process
                dry_est, chain = model(wet)
                dry_est = dry_est.squeeze(0).cpu()

                # Save
                torchaudio.save(
                    str(audio_dir / f"sample_{i:03d}_wet.wav"),
                    wet.squeeze(0).cpu(), 44100
                )
                torchaudio.save(
                    str(audio_dir / f"sample_{i:03d}_dry_gt.wav"),
                    dry_gt, 44100
                )
                torchaudio.save(
                    str(audio_dir / f"sample_{i:03d}_dry_est.wav"),
                    dry_est, 44100
                )

                # Save chain
                chain_spec = [(fx, p) for fx, p, _ in chain]
                chain_json = chain_to_daw_preset(chain_spec, format='json')
                with open(audio_dir / f"sample_{i:03d}_chain.json", 'w') as f:
                    json.dump(chain_json, f, indent=2)

        print(f"Audio samples saved to {audio_dir}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
