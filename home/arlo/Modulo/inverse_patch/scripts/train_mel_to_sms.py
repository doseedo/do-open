#!/usr/bin/env python3
"""
Train Mel → SMS Params mapper.

The full chain becomes:
  z → mel_mapper → mel → sms_mapper → (freqs, amps) → additive synth → audio

This is more tractable than z → sms directly because:
1. mel is already in audio domain (128 frequency bands over time)
2. SMS params are (freqs, amps) - which mel directly represents
3. The mapping should be more linear/explicit

The key insight: mel spectrogram IS a kind of frequency representation.
Converting mel → sines is essentially "which bins have energy → which frequencies"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import sys
import os
import orjson

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
from mel_to_sines_mapper import MelMapperV2


# ============================================================================
# MEL BIN TO HZ MAPPING
# ============================================================================

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def get_mel_frequencies(n_mels=128, f_min=40, f_max=16000):
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mels = np.linspace(mel_min, mel_max, n_mels)
    return mel_to_hz(mels)

MEL_FREQS = torch.tensor(get_mel_frequencies(128, 40, 16000), dtype=torch.float32)


# ============================================================================
# DATASET
# ============================================================================

class MelToSMSDataset(Dataset):
    """
    Dataset providing:
    - mel: mel_mapper output mel spectrogram (downsampled to match SMS rate)
    - freqs, amps: ground truth SMS params from audio

    IMPORTANT: SMS is at z's frame rate, mel is 8x upsampled.
    We downsample mel to match SMS rate.

    Uses mel_mapper outputs (not decoder mel) to avoid distribution shift
    between training and inference.
    """

    def __init__(self, mel_mapper, sms_manifest_path, max_samples=1000, device='cuda'):
        self.mel_mapper = mel_mapper
        self.device = device
        self.data = []

        with open(sms_manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        print(f"Loading dataset from {sms_manifest_path}...")
        count = 0

        for entry in manifest['entries']:
            if count >= max_samples:
                break

            path = entry['path']

            # Skip drums
            if any(kw in path.lower() for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
                continue

            if not os.path.exists(path):
                continue

            try:
                sms_data = torch.load(path, weights_only=True, map_location='cpu')

                # Get latent
                latent_path = sms_data.get('latent_path')
                if not latent_path or not os.path.exists(latent_path):
                    continue

                lat_data = torch.load(latent_path, weights_only=True, map_location='cpu')
                if isinstance(lat_data, dict):
                    z = lat_data.get('latents', lat_data.get('latent'))
                else:
                    z = lat_data

                if z is None:
                    continue

                if z.dim() == 4:
                    z = z.squeeze(0)

                # Get SMS params - these are at z's frame rate
                freqs = sms_data['freqs']  # [T_sms, n_sines]
                amps = sms_data['amps']
                T_sms = freqs.shape[0]

                # Limit length (at z/SMS frame rate)
                max_T = 32
                T_z = z.shape[-1]
                if T_z > max_T:
                    z = z[..., :max_T]
                    # Also truncate SMS to match
                    freqs = freqs[:max_T]
                    amps = amps[:max_T]
                    T_sms = max_T
                    T_z = max_T

                # Get mel from mel_mapper (will be 8x upsampled)
                with torch.no_grad():
                    z_4d = z.unsqueeze(0).to(device)
                    mel = mel_mapper(z_4d)  # [1, T*8, 128]
                    mel = mel.permute(0, 2, 1).squeeze(0).cpu()  # [128, T*8]

                # Downsample mel to match SMS frame rate
                # mel is [128, T_z * 8], we want [128, T_z]
                T_mel = mel.shape[-1]
                mel_downsampled = F.avg_pool1d(
                    mel.unsqueeze(0),  # [1, 128, T_mel]
                    kernel_size=8,
                    stride=8
                ).squeeze(0)  # [128, T_z]

                # Ensure same length
                min_T = min(mel_downsampled.shape[-1], T_sms)
                mel_downsampled = mel_downsampled[..., :min_T]
                freqs = freqs[:min_T]
                amps = amps[:min_T]

                self.data.append({
                    'mel': mel_downsampled,  # [128, T] at z's frame rate
                    'freqs': freqs,          # [T, n_sines] at z's frame rate
                    'amps': amps,            # [T, n_sines] at z's frame rate
                })

                count += 1
                if count % 50 == 0:
                    print(f"  Loaded {count} samples...")

            except Exception as e:
                continue

        print(f"  Total: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate with padding to max length in batch.

    Now mel and SMS are at same frame rate (no interpolation needed).
    """
    max_T = max(b['mel'].shape[-1] for b in batch)

    mels = []
    freqs_list = []
    amps_list = []

    for b in batch:
        mel = b['mel']  # [128, T]
        freqs = b['freqs']  # [T, n_sines]
        amps = b['amps']  # [T, n_sines]

        T = mel.shape[-1]

        # Pad to max_T
        if T < max_T:
            mel = F.pad(mel, (0, max_T - T))
            freqs = F.pad(freqs, (0, 0, 0, max_T - T))  # Pad time dim
            amps = F.pad(amps, (0, 0, 0, max_T - T))

        mels.append(mel)
        freqs_list.append(freqs)
        amps_list.append(amps)

    return {
        'mel': torch.stack(mels),           # [B, 128, T]
        'freqs': torch.stack(freqs_list),   # [B, T, n_sines]
        'amps': torch.stack(amps_list),     # [B, T, n_sines]
    }


# ============================================================================
# MEL TO SMS MAPPER
# ============================================================================

class MelToSMSMapper(nn.Module):
    """
    Map mel spectrogram to SMS sine parameters.

    Input: mel [B, 128, T] - 128 mel bins over T frames
    Output: freqs [B, T, n_sines], amps [B, T, n_sines]

    Key insight: mel bins already represent frequency bands.
    The mapping is roughly "which bins have energy" → "which frequencies"
    """

    def __init__(self, n_sines=64, hidden_dim=256):
        super().__init__()
        self.n_sines = n_sines

        # Register mel bin center frequencies
        self.register_buffer('mel_freqs', MEL_FREQS.clone())

        # Encoder: mel → hidden
        self.encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Temporal context
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Step 1: Predict amplitudes first
        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_sines),
        )

        # Step 2: Predict frequencies CONDITIONED ON amplitudes
        # This way freq[i] knows about amp[i]
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_sines, hidden_dim),  # h + amps
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_sines),
        )

    def forward(self, mel):
        """
        mel: [B, 128, T]
        Returns: freqs [B, T, n_sines] in Hz, amps [B, T, n_sines] in [0, 1]
        """
        B, n_mels, T = mel.shape

        # Transpose to [B, T, 128]
        mel = mel.permute(0, 2, 1)

        # Encode
        h = self.encoder(mel)  # [B, T, hidden]

        # Temporal context
        h, _ = self.temporal(h)  # [B, T, hidden*2]

        # Step 1: Predict amplitudes first
        amps = torch.sigmoid(self.amp_head(h))  # [B, T, n_sines]

        # Step 2: Predict frequencies conditioned on amplitudes
        # Concatenate h with amps so freq_head knows amp[i] when predicting freq[i]
        h_with_amps = torch.cat([h, amps], dim=-1)  # [B, T, hidden*2 + n_sines]
        freq_logits = self.freq_head(h_with_amps)  # [B, T, n_sines]

        # Map to frequency range [20, 16000] in log space
        log_freq_min = np.log(20)
        log_freq_max = np.log(16000)
        log_freqs = log_freq_min + torch.sigmoid(freq_logits) * (log_freq_max - log_freq_min)
        freqs = torch.exp(log_freqs)

        return freqs, amps


class MelToSMSMapperV2(nn.Module):
    """
    V2: Explicit mel bin selection approach.

    Instead of predicting arbitrary frequencies, select from mel bin centers
    and predict amplitude for each bin. Then take top-k as the "sines".

    This is more aligned with what mel actually represents.
    """

    def __init__(self, n_sines=64, hidden_dim=256):
        super().__init__()
        self.n_sines = n_sines

        self.register_buffer('mel_freqs', MEL_FREQS.clone())

        # Predict amplitude for each mel bin (refinement of mel energy)
        self.amp_refiner = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),  # Back to 128 bins
        )

        # Predict frequency offset within each bin (fine-tuning)
        self.freq_offset = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
        )

    def forward(self, mel):
        """
        mel: [B, 128, T]
        Returns: freqs [B, T, n_sines], amps [B, T, n_sines]
        """
        B, n_mels, T = mel.shape
        device = mel.device

        mel = mel.permute(0, 2, 1)  # [B, T, 128]

        # Refine amplitudes
        amp_refined = torch.sigmoid(self.amp_refiner(mel))  # [B, T, 128]

        # Get frequency offsets (small corrections to bin centers)
        # Output in [-0.5, 0.5] bin widths
        freq_offset = 0.5 * torch.tanh(self.freq_offset(mel))  # [B, T, 128]

        # For each frame, select top-k bins by amplitude
        topk_amps, topk_idx = amp_refined.topk(self.n_sines, dim=-1)  # [B, T, n_sines]

        # Get frequencies: bin center + offset
        mel_freqs = self.mel_freqs.to(device)  # [128]

        # Compute bin widths for offset scaling
        bin_widths = torch.zeros(128, device=device)
        bin_widths[:-1] = mel_freqs[1:] - mel_freqs[:-1]
        bin_widths[-1] = bin_widths[-2]

        # Get base frequencies for selected bins
        base_freqs = mel_freqs[topk_idx]  # [B, T, n_sines]

        # Get offsets for selected bins
        selected_offsets = freq_offset.gather(-1, topk_idx)  # [B, T, n_sines]
        selected_widths = bin_widths[topk_idx]  # [B, T, n_sines]

        # Final frequencies
        freqs = base_freqs + selected_offsets * selected_widths

        return freqs, topk_amps


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def sinkhorn_loss(pred_freqs, pred_amps, target_freqs, target_amps, n_iter=10, reg=0.1):
    """
    Permutation-invariant loss using Sinkhorn for soft assignment.
    Handles different number of sines in pred vs target.
    """
    B, T, N_pred = pred_freqs.shape
    N_target = target_freqs.shape[-1]
    device = pred_freqs.device

    # Take top N_pred from target by amplitude
    target_order = target_amps.argsort(dim=-1, descending=True)
    target_freqs_top = target_freqs.gather(-1, target_order)[..., :N_pred]
    target_amps_top = target_amps.gather(-1, target_order)[..., :N_pred]

    # Log frequencies for perceptual weighting
    log_pred = torch.log(pred_freqs.clamp(min=20))
    log_target = torch.log(target_freqs_top.clamp(min=20))

    # Cost matrix: |log(f_pred) - log(f_target)|²
    # Shape: [B, T, N_pred, N_pred]
    freq_cost = (log_pred.unsqueeze(-1) - log_target.unsqueeze(-2)).pow(2)

    # Add amplitude difference to cost
    amp_cost = (pred_amps.unsqueeze(-1) - target_amps_top.unsqueeze(-2)).pow(2)
    cost = freq_cost + amp_cost

    # Sinkhorn iterations (in log space for stability)
    log_P = -cost / reg
    for _ in range(n_iter):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)

    P = torch.exp(log_P)  # [B, T, N_pred, N_pred] soft assignment

    # Mask for active sines
    mask = (target_amps_top > 0.001).float()  # [B, T, N_pred]

    # Matched target frequencies and amps
    matched_target_freqs = (P * target_freqs_top.unsqueeze(-2)).sum(dim=-1)  # [B, T, N_pred]
    matched_target_amps = (P * target_amps_top.unsqueeze(-2)).sum(dim=-1)

    # Frequency loss (log scale)
    log_matched = torch.log(matched_target_freqs.clamp(min=20))
    freq_loss = (mask * (log_pred - log_matched).pow(2)).sum() / (mask.sum() + 1e-8)

    # Amplitude loss
    amp_loss = (mask * (pred_amps - matched_target_amps).pow(2)).sum() / (mask.sum() + 1e-8)

    return freq_loss, amp_loss


def amplitude_matched_loss(pred_freqs, pred_amps, target_freqs, target_amps):
    """
    Match sines by amplitude rank (strongest pred → strongest target).
    This prevents frequency collapse since freqs must match their amplitude-matched partners.
    """
    B, T, N_pred = pred_freqs.shape
    N_target = target_freqs.shape[-1]

    # Sort both by amplitude (descending) - strongest first
    pred_order = pred_amps.argsort(dim=-1, descending=True)
    target_order = target_amps.argsort(dim=-1, descending=True)

    pred_f = pred_freqs.gather(-1, pred_order)
    pred_a = pred_amps.gather(-1, pred_order)
    target_f = target_freqs.gather(-1, target_order)
    target_a = target_amps.gather(-1, target_order)

    # Take top N_pred from target
    target_f = target_f[..., :N_pred]
    target_a = target_a[..., :N_pred]

    # Now compute loss WITHOUT resorting by frequency
    # The i-th strongest pred should match the i-th strongest target

    # Mask inactive sines (low amplitude in target)
    mask = (target_a > 0.001).float()

    # Log frequency loss
    log_pred = torch.log(pred_f.clamp(min=20))
    log_target = torch.log(target_f.clamp(min=20))
    freq_loss = (mask * (log_pred - log_target).pow(2)).sum() / (mask.sum() + 1e-8)

    # Amplitude loss
    amp_loss = (mask * (pred_a - target_a).pow(2)).sum() / (mask.sum() + 1e-8)

    return freq_loss, amp_loss


def hungarian_loss(pred_freqs, pred_amps, target_freqs, target_amps):
    """
    Optimal assignment using cost matrix (slower but more accurate).
    """
    from scipy.optimize import linear_sum_assignment

    B, T, N_pred = pred_freqs.shape
    N_target = target_freqs.shape[-1]
    device = pred_freqs.device

    # Take top N_pred from target by amplitude
    target_order = target_amps.argsort(dim=-1, descending=True)
    target_f = target_freqs.gather(-1, target_order)[..., :N_pred]
    target_a = target_amps.gather(-1, target_order)[..., :N_pred]

    total_freq_loss = 0
    total_amp_loss = 0
    count = 0

    for b in range(B):
        for t in range(T):
            pf = pred_freqs[b, t]  # [N_pred]
            pa = pred_amps[b, t]
            tf = target_f[b, t]
            ta = target_a[b, t]

            # Cost matrix: log freq distance + amp distance
            log_pf = torch.log(pf.clamp(min=20))
            log_tf = torch.log(tf.clamp(min=20))

            freq_cost = (log_pf.unsqueeze(1) - log_tf.unsqueeze(0)).pow(2)  # [N, N]
            amp_cost = (pa.unsqueeze(1) - ta.unsqueeze(0)).pow(2)
            cost = freq_cost + amp_cost

            # Hungarian assignment
            cost_np = cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            # Compute matched loss
            mask = (ta[col_ind] > 0.001).float()
            freq_loss = (mask * (log_pf[row_ind] - log_tf[col_ind]).pow(2)).sum()
            amp_loss = (mask * (pa[row_ind] - ta[col_ind]).pow(2)).sum()

            total_freq_loss += freq_loss
            total_amp_loss += amp_loss
            count += mask.sum()

    return total_freq_loss / (count + 1e-8), total_amp_loss / (count + 1e-8)


# ============================================================================
# TRAINING
# ============================================================================

def train_mel_to_sms(model, dataloader, n_epochs=100, lr=1e-3, device='cuda', save_path=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_loss = float('inf')

    for epoch in range(n_epochs):
        total_freq_loss = 0
        total_amp_loss = 0
        n_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            mel = batch['mel'].to(device)
            target_freqs = batch['freqs'].to(device)
            target_amps = batch['amps'].to(device)

            pred_freqs, pred_amps = model(mel)

            # Match temporal dimension
            min_T = min(pred_freqs.shape[1], target_freqs.shape[1])
            pred_freqs = pred_freqs[:, :min_T]
            pred_amps = pred_amps[:, :min_T]
            target_freqs = target_freqs[:, :min_T]
            target_amps = target_amps[:, :min_T]

            # Scale target amps to [0,1] range (SMS amps are typically [0, 0.1])
            # Use fixed scale factor so model learns consistent mapping
            AMP_SCALE = 0.4  # Match actual max amp in data (~0.39)  # targets are ~[0, 0.1], scale to ~[0, 1]
            target_amps_scaled = target_amps / AMP_SCALE

            freq_loss, amp_loss = hungarian_loss(pred_freqs, pred_amps, target_freqs, target_amps_scaled)
            loss = freq_loss + amp_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_freq_loss += freq_loss.item()
            total_amp_loss += amp_loss.item()
            n_batches += 1

        scheduler.step()

        avg_freq = total_freq_loss / n_batches
        avg_amp = total_amp_loss / n_batches
        avg_loss = avg_freq + avg_amp

        if epoch % 10 == 0:
            # Convert freq_loss to semitones (rough estimate)
            semitones = np.sqrt(avg_freq) * 12 / np.log(2)
            print(f"  Epoch {epoch}: freq_loss={avg_freq:.4f} (~{semitones:.1f} st), amp_loss={avg_amp:.4f}")

        if save_path and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'freq_loss': avg_freq,
                'amp_loss': avg_amp,
            }, save_path)

    print(f"\nBest loss: {best_loss:.4f}")
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_sines', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--model', type=str, default='v2', choices=['v1', 'v2'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load mel_mapper (instead of DCAE decoder)
    print("\nLoading mel_mapper...")
    mel_mapper_path = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_mapper/best_model.pt'
    mel_ckpt = torch.load(mel_mapper_path, weights_only=True)
    mel_mapper = MelMapperV2(hidden_dim=mel_ckpt.get('hidden_dim', 256)).to(device)
    mel_mapper.load_state_dict(mel_ckpt['model_state_dict'])
    mel_mapper.eval()
    print(f"  Mel mapper: {sum(p.numel() for p in mel_mapper.parameters()):,} params")

    # Load dataset (preloads all mel using mel_mapper)
    print("\nLoading dataset...")
    dataset = MelToSMSDataset(
        mel_mapper=mel_mapper,
        sms_manifest_path='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        max_samples=args.max_samples,
        device=device,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Create model
    print(f"\nCreating MelToSMSMapper {args.model}...")
    if args.model == 'v1':
        model = MelToSMSMapper(n_sines=args.n_sines, hidden_dim=args.hidden_dim)
    else:
        model = MelToSMSMapperV2(n_sines=args.n_sines, hidden_dim=args.hidden_dim)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    # Train
    print("\n" + "=" * 60)
    print("TRAINING MEL → SMS MAPPER")
    print("=" * 60)

    save_dir = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_to_sms'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/best_model_{args.model}.pt'

    model = train_mel_to_sms(
        model, dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=save_path,
    )

    print(f"\nSaved to {save_path}")

    # Test: full chain z → mel_mapper → sms_mapper → sines
    print("\n" + "=" * 60)
    print("TESTING FULL CHAIN")
    print("=" * 60)
    print("  mel_mapper already loaded and used for training data")
    print("  Run test_full_chain.py for audio comparison")
    print("\nDone!")


if __name__ == "__main__":
    main()
