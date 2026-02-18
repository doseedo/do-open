#!/usr/bin/env python3
"""
Analyze what operations emerged from training.

After training the LearnedOperationCodec, run this to discover:
- What does each operation produce?
- Are they harmonic series? Formants? Noise? Something else?
- Which operations are most used?
- Do operations specialize by instrument type?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train_learned_ops import LearnedOperationCodec, SlotOperationCodec, SMSDataset


def load_model(checkpoint_path: str, model_type: str = 'codec'):
    """Load trained model."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    if model_type == 'codec':
        model = LearnedOperationCodec()
    else:
        model = SlotOperationCodec()

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, ckpt.get('op_embeddings')


def analyze_channel_weights(model):
    """Analyze what the network learned about z structure."""
    print("="*70)
    print("CHANNEL WEIGHT ANALYSIS - Did it discover SAMI structure?")
    print("="*70)

    if hasattr(model, 'channel_weights'):
        cw = F.softplus(model.channel_weights).detach().cpu()
        print("\nLearned channel weights (8 channels of z):")
        for i, w in enumerate(cw):
            bar = '█' * int(w * 20)
            label = "coarse" if i < 4 else "fine"
            print(f"  Ch {i} ({label}): {w:.3f} {bar}")

        coarse_avg = cw[:4].mean().item()
        fine_avg = cw[4:].mean().item()
        print(f"\n  Coarse (0-3) avg: {coarse_avg:.3f}")
        print(f"  Fine (4-7) avg:   {fine_avg:.3f}")

        if abs(coarse_avg - fine_avg) > 0.2:
            print("  >>> Network discovered coarse/fine distinction!")
        else:
            print("  >>> Coarse/fine weights similar - SAMI split may not be optimal")

    if hasattr(model, 'dim_weights'):
        dw = F.softplus(model.dim_weights).detach().cpu()
        print("\nLearned dimension weights (16 dims per channel):")
        for i, w in enumerate(dw):
            bar = '█' * int(w * 20)
            print(f"  Dim {i:2d}: {w:.3f} {bar}")

        print(f"\n  Most important dims: {dw.argsort(descending=True)[:5].tolist()}")
        print(f"  Least important dims: {dw.argsort()[:5].tolist()}")

    print()


def analyze_op_embeddings(op_embeddings: torch.Tensor):
    """Analyze the learned operation embeddings."""
    print("="*70)
    print("OPERATION EMBEDDING ANALYSIS")
    print("="*70)

    n_ops, dim = op_embeddings.shape
    print(f"\n{n_ops} operations, {dim} dimensions each")

    # 1. Norms (how "strong" is each operation)
    norms = op_embeddings.norm(dim=1)
    print(f"\nOperation norms (strength):")
    for i, n in enumerate(norms):
        bar = '█' * int(n * 10)
        print(f"  Op {i:2d}: {n:.3f} {bar}")

    # 2. Pairwise similarities
    normalized = F.normalize(op_embeddings, dim=1)
    sims = normalized @ normalized.T

    print(f"\nMost similar operation pairs:")
    pairs = []
    for i in range(n_ops):
        for j in range(i+1, n_ops):
            pairs.append((sims[i,j].item(), i, j))
    pairs.sort(reverse=True)
    for sim, i, j in pairs[:5]:
        print(f"  Op {i} <-> Op {j}: {sim:.3f}")

    print(f"\nMost different operation pairs:")
    for sim, i, j in pairs[-5:]:
        print(f"  Op {i} <-> Op {j}: {sim:.3f}")

    # 3. PCA to visualize
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(op_embeddings.numpy())

        print(f"\nOperation positions (PCA 2D):")
        for i, (x, y) in enumerate(coords):
            print(f"  Op {i:2d}: ({x:+.2f}, {y:+.2f})")

        print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    except ImportError:
        pass

    return sims


def analyze_op_outputs(model, device='cpu'):
    """
    For each operation, see what frequencies/amplitudes it produces.
    """
    print("\n" + "="*70)
    print("OPERATION OUTPUT ANALYSIS - What does each op produce?")
    print("="*70)

    model = model.to(device)
    model.eval()

    n_ops = model.n_ops

    for op_idx in range(n_ops):
        print(f"\n--- Operation {op_idx} ---")

        # Create synthetic input that activates ONLY this operation
        B, T = 1, 22
        z = torch.randn(B, 8, 16, T, device=device) * 0.1

        with torch.no_grad():
            pred = model(z)
            op_weights = pred['op_weights']  # [B, T, n_ops]
            freqs = pred['freqs']  # [B, T, n_sines]
            amps = pred['amps']

        # Average contribution of this op
        avg_weight = op_weights[0, :, op_idx].mean().item()
        print(f"  Average weight: {avg_weight:.3f}")

        # When this op is dominant, what freqs does it produce?
        # Find frames where this op has highest weight
        dominant_frames = op_weights[0, :, op_idx] > op_weights[0].mean(dim=-1)
        if dominant_frames.sum() > 0:
            dom_freqs = freqs[0, dominant_frames]
            dom_amps = amps[0, dominant_frames]

            # Get top-K sines by amplitude
            mean_amps = dom_amps.mean(dim=0)
            top_k = 8
            top_indices = mean_amps.argsort(descending=True)[:top_k]

            top_freqs = dom_freqs[:, top_indices].mean(dim=0)
            top_amp_vals = mean_amps[top_indices]

            print(f"  Top {top_k} frequencies when dominant:")
            for freq, amp in zip(top_freqs, top_amp_vals):
                print(f"    {freq:.1f} Hz (amp={amp:.3f})")

            # Check if harmonic
            sorted_freqs = top_freqs.sort()[0]
            if len(sorted_freqs) >= 2:
                f0 = sorted_freqs[0].item()
                if f0 > 20:
                    ratios = sorted_freqs / f0
                    print(f"  Harmonic ratios (if f0={f0:.1f}): {ratios.tolist()}")

                    # Count how many are near-integer ratios
                    near_int = ((ratios - ratios.round()).abs() < 0.1).sum().item()
                    print(f"  Near-integer ratios: {near_int}/{len(ratios)}")
                    if near_int >= 3:
                        print(f"  >>> LIKELY HARMONIC SERIES <<<")


def analyze_op_usage(model, dataset, device='cpu', max_samples=500):
    """
    Analyze which operations are used for different types of audio.
    """
    print("\n" + "="*70)
    print("OPERATION USAGE ANALYSIS - Which ops for which audio?")
    print("="*70)

    model = model.to(device)
    model.eval()

    # Collect op usage per sample
    op_usage_by_path = defaultdict(list)
    all_op_usages = []

    for i in range(min(len(dataset), max_samples)):
        sample = dataset[i]
        latent = sample['latent'].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(latent)
            op_weights = pred['op_weights'][0]  # [T, n_ops]

        # Average op weights for this sample
        avg_weights = op_weights.mean(dim=0)  # [n_ops]
        all_op_usages.append(avg_weights.cpu())

        # Track by audio path (for instrument analysis)
        if 'audio_path' in sample:
            path = sample['audio_path'].lower()
            for keyword in ['guitar', 'bass', 'vocal', 'synth', 'piano', 'string']:
                if keyword in path:
                    op_usage_by_path[keyword].append(avg_weights.cpu())
                    break

    all_op_usages = torch.stack(all_op_usages)  # [N, n_ops]

    # Overall op usage distribution
    print("\nOverall operation usage:")
    mean_usage = all_op_usages.mean(dim=0)
    for i, usage in enumerate(mean_usage):
        bar = '█' * int(usage * 50)
        print(f"  Op {i:2d}: {usage:.3f} {bar}")

    # Find specialized vs generalist ops
    usage_std = all_op_usages.std(dim=0)
    print("\nOperation specialization (high std = specialized):")
    for i in torch.argsort(usage_std, descending=True)[:5]:
        print(f"  Op {i}: std={usage_std[i]:.3f}, mean={mean_usage[i]:.3f}")

    # Usage by instrument type (if we have path info)
    if op_usage_by_path:
        print("\nOperation usage by instrument type:")
        for instrument, usages in op_usage_by_path.items():
            if len(usages) < 5:
                continue
            usages = torch.stack(usages)
            mean = usages.mean(dim=0)
            dominant_op = mean.argmax().item()
            print(f"\n  {instrument} ({len(usages)} samples):")
            print(f"    Most used op: {dominant_op} ({mean[dominant_op]:.3f})")
            top3 = mean.argsort(descending=True)[:3]
            print(f"    Top 3 ops: {top3.tolist()}")


def synthesize_from_op(model, op_idx: int, device='cpu', duration_s=2.0):
    """
    Synthesize audio using ONLY one operation.
    Listen to understand what the operation does.
    """
    print(f"\nSynthesizing audio for Operation {op_idx}...")

    model = model.to(device)
    model.eval()

    sr = 44100
    n_frames = 22
    n_samples = int(sr * duration_s)

    # Create input with moderate activation
    z = torch.randn(1, 8, 16, n_frames, device=device) * 0.5

    with torch.no_grad():
        pred = model(z)
        freqs = pred['freqs'][0].cpu().numpy()  # [T, n_sines]
        amps = pred['amps'][0].cpu().numpy()
        phases = pred['phases'][0].cpu().numpy()
        op_weights = pred['op_weights'][0].cpu()

    # Report op weights
    print(f"  Op weights: {op_weights.mean(dim=0).tolist()}")

    # Synthesize
    audio = np.zeros(n_samples)
    hop = n_samples // n_frames

    for t in range(n_frames):
        for s in range(freqs.shape[1]):
            if amps[t, s] < 0.01:
                continue
            freq = freqs[t, s]
            amp = amps[t, s]

            start = t * hop
            end = min((t + 1) * hop, n_samples)
            t_samples = np.arange(end - start) / sr

            audio[start:end] += amp * np.sin(2 * np.pi * freq * t_samples + phases[t, s])

    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.8

    return audio, sr


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/learned_ops_codec/best_model.pt')
    parser.add_argument('--model_type', type=str, default='codec')
    parser.add_argument('--sms_manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_audio', action='store_true')
    args = parser.parse_args()

    print("="*70)
    print("Analyzing Learned Operations - What Emerged?")
    print("="*70)

    # Load model
    model, op_embeddings = load_model(args.checkpoint, args.model_type)
    print(f"Loaded model from {args.checkpoint}")

    # Analyze what the network learned about z structure
    analyze_channel_weights(model)

    # Analyze embeddings
    if op_embeddings is not None:
        analyze_op_embeddings(op_embeddings)

    # Analyze what each op produces
    analyze_op_outputs(model, args.device)

    # Analyze usage patterns (requires dataset)
    try:
        dataset = SMSDataset(args.sms_manifest, max_samples=1000, skip_drums=True)
        analyze_op_usage(model, dataset, args.device)
    except Exception as e:
        print(f"Skipping usage analysis: {e}")

    # Optionally synthesize audio for each op
    if args.save_audio:
        import torchaudio
        out_dir = Path(args.checkpoint).parent / "op_audio"
        out_dir.mkdir(exist_ok=True)

        for op_idx in range(model.n_ops):
            audio, sr = synthesize_from_op(model, op_idx, args.device)
            path = out_dir / f"op_{op_idx:02d}.wav"
            torchaudio.save(str(path), torch.tensor(audio).unsqueeze(0), sr)
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
