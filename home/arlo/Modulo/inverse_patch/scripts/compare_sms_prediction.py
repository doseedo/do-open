#!/usr/bin/env python3
"""
Compare predicted SMS vs original extracted SMS.

This shows how well the mapper predicts the actual sine parameters,
independent of neural vocoder comparison.
"""

import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')

from mel_to_sines_mapper import MelMapperV2
from train_mel_to_sms import MelToSMSMapper
import orjson


def additive_synth(freqs, amps, sample_rate=44100, hop_length=512):
    """Synthesize audio from sine parameters."""
    B, T, n_sines = freqs.shape
    device = freqs.device
    n_samples = T * hop_length
    audio = torch.zeros(B, n_samples, device=device)
    phase = torch.zeros(B, n_sines, device=device)

    for t in range(T):
        frame_freqs = freqs[:, t, :]
        frame_amps = amps[:, t, :]
        phase_inc = 2 * np.pi * frame_freqs / sample_rate
        t_samples = torch.arange(hop_length, device=device).float()
        sample_phase = phase.unsqueeze(-1) + t_samples * phase_inc.unsqueeze(-1)
        frame_audio = (torch.sin(sample_phase) * frame_amps.unsqueeze(-1)).sum(dim=1)
        audio[:, t * hop_length:(t + 1) * hop_length] = frame_audio
        phase = (phase + hop_length * phase_inc) % (2 * np.pi)

    return audio


def main():
    device = 'cuda'
    output_dir = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/audio_comparison'
    os.makedirs(output_dir, exist_ok=True)

    # Load models
    print("Loading mel_mapper...")
    mel_ckpt = torch.load(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_mapper/best_model.pt',
        weights_only=True
    )
    mel_mapper = MelMapperV2(hidden_dim=mel_ckpt.get('hidden_dim', 256)).to(device)
    mel_mapper.load_state_dict(mel_ckpt['model_state_dict'])
    mel_mapper.eval()

    print("Loading sms_mapper...")
    sms_ckpt = torch.load(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_to_sms/best_model_v1.pt',
        weights_only=True
    )
    sms_mapper = MelToSMSMapper(n_sines=64, hidden_dim=256).to(device)
    sms_mapper.load_state_dict(sms_ckpt['model_state_dict'])
    sms_mapper.eval()

    # Load test sample
    print("\nLoading test sample...")
    with open('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json', 'rb') as f:
        manifest = orjson.loads(f.read())

    # Find a good sample (not drums)
    sample_path = None
    for entry in manifest['entries'][:50]:
        path = entry['path']
        if not any(kw in path.lower() for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
            sample_path = path
            break

    sms_data = torch.load(sample_path, weights_only=True, map_location='cpu')
    print(f"  Sample: {sample_path}")

    # Get original SMS params (ground truth)
    gt_freqs = sms_data['freqs']  # [T, n_sines]
    gt_amps = sms_data['amps']
    print(f"  GT SMS shape: {gt_freqs.shape}")
    print(f"  GT amp range: [{gt_amps.min():.4f}, {gt_amps.max():.4f}]")

    # Limit to 32 frames for comparison
    max_T = 32
    if gt_freqs.shape[0] > max_T:
        gt_freqs = gt_freqs[:max_T]
        gt_amps = gt_amps[:max_T]

    # Get latent
    lat_data = torch.load(sms_data['latent_path'], weights_only=True, map_location='cpu')
    if isinstance(lat_data, dict):
        z = lat_data.get('latents', lat_data)
    else:
        z = lat_data
    if z.dim() == 4:
        z = z.squeeze(0)
    z = z[..., :max_T].unsqueeze(0).to(device)
    print(f"  z shape: {z.shape}")

    # Predict SMS from z via mel_mapper -> sms_mapper
    print("\nPredicting SMS...")
    with torch.no_grad():
        # z -> mel
        mel = mel_mapper(z)  # [1, T*8, 128]
        mel = mel.permute(0, 2, 1)  # [1, 128, T*8]

        # Downsample mel to SMS rate
        mel_for_sms = F.avg_pool1d(mel, kernel_size=8, stride=8)  # [1, 128, T]

        # mel -> (freqs, amps)
        pred_freqs, pred_amps = sms_mapper(mel_for_sms)  # [1, T, 64]

        # Scale amps
        AMP_SCALE = 0.4
        pred_amps = pred_amps * AMP_SCALE

    print(f"  Pred freqs shape: {pred_freqs.shape}")
    print(f"  Pred amp range: [{pred_amps.min():.4f}, {pred_amps.max():.4f}]")

    # Compare statistics
    print("\n" + "=" * 60)
    print("COMPARISON: Ground Truth vs Predicted")
    print("=" * 60)

    # Take top 64 sines from GT by amplitude
    gt_amp_order = gt_amps.mean(dim=0).argsort(descending=True)[:64]
    gt_freqs_top = gt_freqs[:, gt_amp_order]
    gt_amps_top = gt_amps[:, gt_amp_order]

    # Sort both by amplitude for fair comparison
    gt_order = gt_amps_top.mean(dim=0).argsort(descending=True)
    pred_order = pred_amps.squeeze().mean(dim=0).argsort(descending=True)

    gt_f_sorted = gt_freqs_top[:, gt_order]
    gt_a_sorted = gt_amps_top[:, gt_order]
    pred_f_sorted = pred_freqs.squeeze()[:, pred_order]
    pred_a_sorted = pred_amps.squeeze()[:, pred_order]

    print("\nTop 10 sines (by amplitude):")
    print(f"{'Rank':<6} {'GT Freq':>10} {'Pred Freq':>10} {'GT Amp':>10} {'Pred Amp':>10} {'Freq Err':>10}")
    print("-" * 60)

    for i in range(10):
        gt_f = gt_f_sorted[:, i].mean().item()
        pred_f = pred_f_sorted[:, i].mean().item()
        gt_a = gt_a_sorted[:, i].mean().item()
        pred_a = pred_a_sorted[:, i].mean().item()

        # Frequency error in semitones
        freq_err_st = 12 * np.log2(pred_f / (gt_f + 1e-8) + 1e-8)

        print(f"{i:<6} {gt_f:>10.1f} {pred_f:>10.1f} {gt_a:>10.4f} {pred_a:>10.4f} {freq_err_st:>+10.1f} st")

    # Render ground truth SMS
    print("\n" + "=" * 60)
    print("RENDERING AUDIO")
    print("=" * 60)

    # GT audio (upsample to mel rate for synthesis)
    gt_freqs_up = F.interpolate(
        gt_freqs_top.T.unsqueeze(0).float(),
        scale_factor=8, mode='nearest'
    ).permute(0, 2, 1).to(device)
    gt_amps_up = F.interpolate(
        gt_amps_top.T.unsqueeze(0).float(),
        scale_factor=8, mode='linear', align_corners=False
    ).permute(0, 2, 1).to(device)

    audio_gt = additive_synth(gt_freqs_up, gt_amps_up)
    audio_gt = audio_gt.squeeze() / (audio_gt.abs().max() + 1e-8) * 0.9

    # Predicted audio
    pred_freqs_up = F.interpolate(
        pred_freqs.permute(0, 2, 1), scale_factor=8, mode='nearest'
    ).permute(0, 2, 1)
    pred_amps_up = F.interpolate(
        pred_amps.permute(0, 2, 1), scale_factor=8, mode='linear', align_corners=False
    ).permute(0, 2, 1)

    audio_pred = additive_synth(pred_freqs_up, pred_amps_up)
    audio_pred = audio_pred.squeeze() / (audio_pred.abs().max() + 1e-8) * 0.9

    # Match lengths
    min_len = min(audio_gt.shape[-1], audio_pred.shape[-1])
    audio_gt = audio_gt[:min_len]
    audio_pred = audio_pred[:min_len]

    # Save
    print("\nSaving audio files...")
    torchaudio.save(f'{output_dir}/sms_ground_truth.wav', audio_gt.unsqueeze(0).cpu(), 44100)
    torchaudio.save(f'{output_dir}/sms_predicted.wav', audio_pred.unsqueeze(0).cpu(), 44100)

    # A/B
    silence = torch.zeros(22050)
    combined = torch.cat([audio_gt.cpu(), silence, audio_pred.cpu()])
    torchaudio.save(f'{output_dir}/sms_AB_gt_then_pred.wav', combined.unsqueeze(0), 44100)

    print(f"\n  {output_dir}/sms_ground_truth.wav - Original extracted SMS")
    print(f"  {output_dir}/sms_predicted.wav - Our prediction")
    print(f"  {output_dir}/sms_AB_gt_then_pred.wav - A/B comparison")

    # Compute metrics
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    # Spectral convergence
    window = torch.hann_window(2048, device=device)
    gt_spec = torch.stft(audio_gt, 2048, 512, window=window, return_complex=True).abs()
    pred_spec = torch.stft(audio_pred, 2048, 512, window=window, return_complex=True).abs()
    sc = (torch.norm(gt_spec - pred_spec, p='fro') / (torch.norm(gt_spec, p='fro') + 1e-8)).item()

    # Log spectral distance
    lsd = torch.sqrt(torch.mean((20 * torch.log10((gt_spec + 1e-8) / (pred_spec + 1e-8))) ** 2)).item()

    print(f"  Spectral Convergence: {sc:.3f}")
    print(f"  Log Spectral Distance: {lsd:.1f} dB")

    if sc < 0.3:
        print("  → EXCELLENT match to ground truth SMS")
    elif sc < 0.5:
        print("  → GOOD match")
    elif sc < 0.8:
        print("  → FAIR match")
    else:
        print("  → POOR match")


if __name__ == "__main__":
    main()
