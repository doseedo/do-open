"""
Evaluation metrics for the Inverse Audio Effects System.
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def compute_si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio.

    Args:
        estimate: Estimated signal [B, T] or [B, 1, T]
        target: Target signal [B, T] or [B, 1, T]

    Returns:
        SI-SDR in dB [B]
    """
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    # Zero mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Compute SI-SDR
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    s_target = (target ** 2).sum(dim=-1, keepdim=True)

    proj = dot * target / (s_target + 1e-8)
    noise = estimate - proj

    si_sdr = 10 * torch.log10(
        (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + 1e-8)
    )

    return si_sdr


def compute_stft_distance(
    estimate: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Compute STFT magnitude distance.

    Args:
        estimate: Estimated signal
        target: Target signal
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        STFT distance
    """
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    window = torch.hann_window(n_fft, device=estimate.device)

    est_stft = torch.stft(
        estimate, n_fft, hop_length, window=window, return_complex=True
    )
    tgt_stft = torch.stft(
        target, n_fft, hop_length, window=window, return_complex=True
    )

    est_mag = est_stft.abs()
    tgt_mag = tgt_stft.abs()

    # Log magnitude distance
    log_est = torch.log(est_mag + 1e-8)
    log_tgt = torch.log(tgt_mag + 1e-8)

    distance = F.l1_loss(log_est, log_tgt, reduction='none')
    return distance.mean(dim=(1, 2))


def compute_mel_distance(
    estimate: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 44100,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Compute mel spectrogram distance.

    Args:
        estimate: Estimated signal
        target: Target signal
        sample_rate: Sample rate
        n_mels: Number of mel bins
        n_fft: FFT size
        hop_length: Hop length

    Returns:
        Mel distance
    """
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    ).to(estimate.device)

    est_mel = mel_transform(estimate)
    tgt_mel = mel_transform(target)

    # Log mel distance
    log_est = torch.log(est_mel + 1e-8)
    log_tgt = torch.log(tgt_mel + 1e-8)

    distance = F.l1_loss(log_est, log_tgt, reduction='none')
    return distance.mean(dim=(1, 2))


def compute_pesq(
    estimate: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
) -> float:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality).
    Requires pesq package.

    Args:
        estimate: Estimated signal
        target: Target signal
        sample_rate: Sample rate (will resample if needed)

    Returns:
        PESQ score
    """
    try:
        from pesq import pesq
    except ImportError:
        return 0.0

    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    # Convert to numpy
    est_np = estimate[0].cpu().numpy()
    tgt_np = target[0].cpu().numpy()

    # Resample if needed
    if sample_rate != 16000:
        import scipy.signal
        est_np = scipy.signal.resample(est_np, int(len(est_np) * 16000 / sample_rate))
        tgt_np = scipy.signal.resample(tgt_np, int(len(tgt_np) * 16000 / sample_rate))

    try:
        score = pesq(16000, tgt_np, est_np, 'wb')
    except Exception:
        score = 0.0

    return score


def evaluate_dry_recovery(
    estimate: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 44100,
) -> Dict[str, float]:
    """
    Evaluate dry signal recovery quality.

    Args:
        estimate: Estimated dry signal
        target: Target dry signal
        sample_rate: Sample rate

    Returns:
        Dictionary of metrics
    """
    si_sdr = compute_si_sdr(estimate, target).mean().item()
    stft_dist = compute_stft_distance(estimate, target).mean().item()
    mel_dist = compute_mel_distance(estimate, target, sample_rate).mean().item()

    # MSE
    mse = F.mse_loss(estimate, target).item()

    # Correlation
    est_flat = estimate.flatten()
    tgt_flat = target.flatten()
    correlation = torch.corrcoef(torch.stack([est_flat, tgt_flat]))[0, 1].item()

    return {
        'si_sdr': si_sdr,
        'stft_distance': stft_dist,
        'mel_distance': mel_dist,
        'mse': mse,
        'correlation': correlation,
    }


def evaluate_chain_estimation(
    estimated_chain: List[Tuple[str, torch.Tensor]],
    target_chain: List[Tuple[str, torch.Tensor]],
    effect_types: List[str],
) -> Dict[str, float]:
    """
    Evaluate effect chain estimation accuracy.

    Args:
        estimated_chain: Estimated chain [(effect_type, params), ...]
        target_chain: Target chain [(effect_type, params), ...]
        effect_types: List of all effect types

    Returns:
        Dictionary of metrics
    """
    # Extract effect types
    est_types = [fx_type for fx_type, _ in estimated_chain]
    tgt_types = [fx_type for fx_type, _ in target_chain]

    # Effect presence accuracy (multi-label)
    est_set = set(est_types)
    tgt_set = set(tgt_types)

    if len(tgt_set) > 0:
        precision = len(est_set & tgt_set) / (len(est_set) + 1e-8)
        recall = len(est_set & tgt_set) / len(tgt_set)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        precision = recall = f1 = 0.0

    # Effect order accuracy
    min_len = min(len(est_types), len(tgt_types))
    order_matches = sum(
        1 for i in range(min_len) if est_types[i] == tgt_types[i]
    )
    order_accuracy = order_matches / (max(len(est_types), len(tgt_types)) + 1e-8)

    # Parameter MAE (for matching effects)
    param_mae = {}
    for est_fx, est_params in estimated_chain:
        for tgt_fx, tgt_params in target_chain:
            if est_fx == tgt_fx:
                mae = F.l1_loss(est_params, tgt_params).item()
                param_mae[est_fx] = mae
                break

    avg_param_mae = np.mean(list(param_mae.values())) if param_mae else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'order_accuracy': order_accuracy,
        'param_mae': avg_param_mae,
        'chain_length_diff': abs(len(est_types) - len(tgt_types)),
    }


def evaluate_system(
    model,
    test_dataset,
    effect_types: List[str],
    device: str = 'cuda',
    max_samples: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of the inverse AFx system.

    Args:
        model: InverseAFxSystem model
        test_dataset: Test dataset
        effect_types: List of effect types
        device: Device to run on
        max_samples: Maximum number of samples to evaluate

    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    model.to(device)

    results = {
        'dry_recovery': defaultdict(list),
        'chain_estimation': defaultdict(list),
        'reconstruction': defaultdict(list),
    }

    num_samples = len(test_dataset)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    with torch.no_grad():
        for i in range(num_samples):
            sample = test_dataset[i]

            dry = sample['dry_audio'].unsqueeze(0).to(device)
            wet = sample['wet_audio'].unsqueeze(0).to(device)

            # Get ground truth chain
            gt_chain_types = sample['effect_types']
            gt_chain_length = sample['chain_length']

            # Forward pass
            dry_est, estimated_chain = model(wet)

            # Evaluate dry recovery
            dry_metrics = evaluate_dry_recovery(dry_est, dry)
            for k, v in dry_metrics.items():
                results['dry_recovery'][k].append(v)

            # Evaluate chain estimation
            # Convert ground truth to comparable format
            gt_chain = []
            for j in range(gt_chain_length):
                if j < len(gt_chain_types):
                    fx_idx = gt_chain_types[j].item()
                    if fx_idx < len(effect_types):
                        gt_chain.append((
                            effect_types[fx_idx],
                            sample['effect_params'][j]
                        ))

            est_chain_simple = [
                (fx_type, params)
                for fx_type, params, _ in estimated_chain
            ]

            chain_metrics = evaluate_chain_estimation(
                est_chain_simple, gt_chain, effect_types
            )
            for k, v in chain_metrics.items():
                results['chain_estimation'][k].append(v)

            # Evaluate reconstruction
            if len(estimated_chain) > 0:
                wet_recon = model.fx_chain(dry_est, est_chain_simple)
                recon_metrics = evaluate_dry_recovery(wet_recon, wet)
                for k, v in recon_metrics.items():
                    results['reconstruction'][k].append(v)

    # Compute means
    final_results = {}
    for category, metrics in results.items():
        final_results[category] = {
            k: np.mean(v) for k, v in metrics.items()
        }

    return final_results
