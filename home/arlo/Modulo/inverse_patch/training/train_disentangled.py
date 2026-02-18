#!/usr/bin/env python3
"""
Disentangled Mapper: Learn transform that makes z → sines interpretable.

Key insight from z_analysis:
- Raw z dims don't map cleanly to frequencies
- But after learned transform (ICA, dictionary), z becomes sparse
- Disentangling transform + sparse mapping = interpretable operations

Architecture:
    z → disentangle → sparse_z → simple_map → sines
        (128→128)      (few dims     (linear,
         rotation)      per sine)    sparse)

After training:
- disentangle.weight: how to transform z to interpretable space
- freq_map.weight: which transformed dims affect which sines (sparse)
- Both are explicit matrices, not black boxes
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import orjson

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# DISENTANGLED MAPPER
# ============================================================

class DisentangledMapper(nn.Module):
    """
    Learn z → interpretable_z → sines mapping.

    The disentangle layer learns a basis change where the
    mapping to sines becomes sparse and interpretable.
    """

    def __init__(
        self,
        n_sines: int = 64,
        hidden_dim: int = 128,  # Disentangled space dimension
        n_disentangle_layers: int = 1,
        use_temporal_attention: bool = False,
        n_attention_layers: int = 1,
        freq_min: float = 20.0,
        freq_max: float = 8000.0,
    ):
        super().__init__()
        self.n_sines = n_sines
        self.hidden_dim = hidden_dim
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.use_temporal_attention = use_temporal_attention

        # Temporal context (like harmonic ops)
        if use_temporal_attention:
            self.temporal_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
                for _ in range(n_attention_layers)
            ])
            self.temporal_norms = nn.ModuleList([
                nn.LayerNorm(128) for _ in range(n_attention_layers)
            ])
        else:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=5, padding=2, groups=8),
                nn.GELU(),
            )

        # Step 1: Disentangling transform
        # Learn rotation/basis change that makes downstream mapping sparse
        if n_disentangle_layers == 1:
            self.disentangle = nn.Linear(128, hidden_dim)
        else:
            layers = [nn.Linear(128, hidden_dim), nn.GELU()]
            for _ in range(n_disentangle_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            self.disentangle = nn.Sequential(*layers[:-1])  # Remove last GELU

        # Step 2: Sparse mapping to sines
        # These weights should become sparse after training with L1 regularization
        self.freq_map = nn.Linear(hidden_dim, n_sines)
        self.amp_map = nn.Linear(hidden_dim, n_sines)
        self.phase_map = nn.Linear(hidden_dim, n_sines)

        # Noise bands
        self.noise_map = nn.Linear(hidden_dim, 8)

        # Log frequency range
        self.log_freq_min = np.log(freq_min)
        self.log_freq_max = np.log(freq_max)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, 8, 16, T] DCAE latent

        Returns:
            freqs: [B, T, n_sines]
            amps: [B, T, n_sines]
            phases: [B, T, n_sines]
            z_disentangled: [B, T, hidden_dim] for analysis
        """
        B, C, H, T = z.shape

        # Temporal context
        z_flat = z.reshape(B, C * H, T)  # [B, 128, T]

        if self.use_temporal_attention:
            z_seq = z_flat.permute(0, 2, 1)  # [B, T, 128]
            for attn, norm in zip(self.temporal_attention_layers, self.temporal_norms):
                z_attended, _ = attn(z_seq, z_seq, z_seq)
                z_seq = norm(z_seq + z_attended)
            z_temporal = z_seq  # [B, T, 128]
        else:
            z_temporal = self.temporal_conv(z_flat).permute(0, 2, 1)  # [B, T, 128]

        # Step 1: Disentangle
        z_dis = self.disentangle(z_temporal)  # [B, T, hidden_dim]

        # Step 2: Sparse mapping to sines
        freq_logits = self.freq_map(z_dis)
        freq_norm = torch.sigmoid(freq_logits)
        log_freq = self.log_freq_min + freq_norm * (self.log_freq_max - self.log_freq_min)
        freqs = torch.exp(log_freq)  # [B, T, n_sines]

        amps = torch.sigmoid(self.amp_map(z_dis))  # [B, T, n_sines]

        phases = torch.tanh(self.phase_map(z_dis)) * np.pi  # [B, T, n_sines]

        noise_amps = torch.sigmoid(self.noise_map(z_dis)) * 0.2  # [B, T, 8]

        return {
            'freqs': freqs,
            'amps': amps,
            'phases': phases,
            'noise_amps': noise_amps,
            'z_disentangled': z_dis,  # For analysis
        }

    def get_sparsity_stats(self) -> Dict[str, float]:
        """Analyze sparsity of learned mappings."""
        with torch.no_grad():
            freq_w = self.freq_map.weight.abs()  # [n_sines, hidden_dim]
            amp_w = self.amp_map.weight.abs()

            # Sparsity = fraction of weights below threshold
            threshold = 0.1
            freq_sparsity = (freq_w < threshold).float().mean().item()
            amp_sparsity = (amp_w < threshold).float().mean().item()

            # Per-sine sparsity (how many dims each sine uses)
            freq_dims_per_sine = (freq_w > threshold).float().sum(dim=1).mean().item()
            amp_dims_per_sine = (amp_w > threshold).float().sum(dim=1).mean().item()

            return {
                'freq_sparsity': freq_sparsity,
                'amp_sparsity': amp_sparsity,
                'freq_dims_per_sine': freq_dims_per_sine,
                'amp_dims_per_sine': amp_dims_per_sine,
            }


# ============================================================
# DATASET (copy from train_harmonic_ops)
# ============================================================

class DisentangledDataset(Dataset):
    """Dataset for disentangled mapper training."""

    DRUM_KEYWORDS = ['drum', 'kick', 'snare', 'hat', 'tom', 'perc', 'cymbal',
                     'overhead', ' oh ', '_oh_', 'hihat', 'hh_', '_hh']

    def __init__(
        self,
        sms_manifest_path: str,
        max_samples: Optional[int] = None,
        target_frames: int = 22,
        amp_scale: float = 10.0,
        skip_drums: bool = True,
        n_sines: int = 64,
    ):
        self.target_frames = target_frames
        self.amp_scale = amp_scale
        self.n_sines = n_sines

        print(f"Loading SMS manifest from {sms_manifest_path}...")
        with open(sms_manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        entries = manifest['entries']
        if max_samples:
            entries = entries[:max_samples]

        print(f"  Found {len(entries)} entries, using {n_sines} sines")

        self.data = []
        skipped = 0

        for entry in entries:
            try:
                sms_data = torch.load(entry['path'], weights_only=True, map_location='cpu')

                if skip_drums:
                    audio_path = sms_data.get('audio_path', '').lower()
                    if any(kw in audio_path for kw in self.DRUM_KEYWORDS):
                        skipped += 1
                        continue

                lat_data = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
                if 'latents' in lat_data:
                    latent = lat_data['latents']
                elif 'latent' in lat_data:
                    latent = lat_data['latent']
                else:
                    continue

                C, H, T = latent.shape
                freqs = sms_data['freqs']
                amps = sms_data['amps'] * self.amp_scale
                phases = sms_data['phases']
                noise_amps = sms_data.get('noise_amps', torch.zeros(freqs.shape[0], 8))

                # Slice to requested sines
                if freqs.shape[1] > self.n_sines:
                    freqs = freqs[:, :self.n_sines]
                    amps = amps[:, :self.n_sines]
                    phases = phases[:, :self.n_sines]

                # Crop/pad to target frames
                if T < target_frames:
                    latent = F.pad(latent, (0, target_frames - T))
                    freqs = F.pad(freqs, (0, 0, 0, target_frames - freqs.shape[0]))
                    amps = F.pad(amps, (0, 0, 0, target_frames - amps.shape[0]))
                    phases = F.pad(phases, (0, 0, 0, target_frames - phases.shape[0]))
                    noise_amps = F.pad(noise_amps, (0, 0, 0, target_frames - noise_amps.shape[0]))
                elif T > target_frames:
                    activity = amps.sum(dim=1)
                    cumsum = torch.cumsum(activity, dim=0)
                    padded = F.pad(cumsum, (1, 0))
                    window_sums = padded[target_frames:] - padded[:-target_frames]
                    start = min(window_sums.argmax().item(), T - target_frames)
                    latent = latent[:, :, start:start + target_frames]
                    freqs = freqs[start:start + target_frames]
                    amps = amps[start:start + target_frames]
                    phases = phases[start:start + target_frames]
                    noise_amps = noise_amps[start:start + target_frames]

                self.data.append({
                    'latent': latent,
                    'freqs': freqs,
                    'amps': amps,
                    'phases': phases,
                    'noise_amps': noise_amps,
                })

            except Exception:
                continue

        print(f"  Loaded {len(self.data)} samples (skipped {skipped} drums)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return {
        'latent': torch.stack([b['latent'] for b in batch]),
        'freqs': torch.stack([b['freqs'] for b in batch]),
        'amps': torch.stack([b['amps'] for b in batch]),
        'phases': torch.stack([b['phases'] for b in batch]),
        'noise_amps': torch.stack([b['noise_amps'] for b in batch]),
    }


# ============================================================
# LOSS FUNCTIONS (copy from train_harmonic_ops)
# ============================================================

def sinkhorn_matching_loss(
    pred_freqs: torch.Tensor,
    pred_amps: torch.Tensor,
    target_freqs: torch.Tensor,
    target_amps: torch.Tensor,
    pred_phases: Optional[torch.Tensor] = None,
    target_phases: Optional[torch.Tensor] = None,
    n_iters: int = 50,
    tau: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sinkhorn-based optimal transport matching."""
    B, T, N_pred = pred_freqs.shape
    N_target = target_freqs.shape[-1]
    device = pred_freqs.device

    log_pred = torch.log(pred_freqs.clamp(min=20))
    log_target = torch.log(target_freqs.clamp(min=20))

    freq_cost = (log_pred.unsqueeze(-1) - log_target.unsqueeze(-2)).pow(2)

    active_mask = (target_amps > 0.01).float()
    cost = freq_cost + (1 - active_mask.unsqueeze(-2)) * 100.0

    log_P = -cost / tau
    for _ in range(n_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)

    P = torch.exp(log_P)

    freq_loss = (P * freq_cost * active_mask.unsqueeze(-2)).sum() / active_mask.sum().clamp(min=1)

    matched_target_amp = (P * target_amps.unsqueeze(-2)).sum(dim=-1)
    amp_loss = F.mse_loss(pred_amps, matched_target_amp)

    phase_loss = torch.tensor(0.0, device=device)
    if pred_phases is not None and target_phases is not None:
        phase_diff = (pred_phases.unsqueeze(-1) - target_phases.unsqueeze(-2)).abs()
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
        phase_cost = phase_diff.pow(2)
        phase_loss = (P * phase_cost * active_mask.unsqueeze(-2)).sum() / active_mask.sum().clamp(min=1)

    pred_active = (pred_amps > 0.01).float().sum(dim=-1)
    target_active = active_mask.sum(dim=-1)
    count_loss = F.mse_loss(pred_active, target_active)

    total_loss = freq_loss + amp_loss + 0.1 * phase_loss + 0.1 * count_loss

    metrics = {
        'freq_loss': freq_loss.item(),
        'amp_loss': amp_loss.item(),
        'phase_loss': phase_loss.item(),
        'count_loss': count_loss.item(),
        'n_active_pred': pred_amps.gt(0.01).float().sum(dim=-1).mean().item(),
        'n_active_target': target_active.mean().item(),
    }

    return total_loss, metrics


# ============================================================
# TRAINER
# ============================================================

class DisentangledTrainer:
    """Train disentangled mapper with sparsity regularization."""

    def __init__(
        self,
        n_sines: int = 64,
        hidden_dim: int = 128,
        n_disentangle_layers: int = 1,
        use_temporal_attention: bool = False,
        n_attention_layers: int = 1,
        sparsity_weight: float = 0.01,  # L1 weight on mapping layers
        device: str = 'cuda',
    ):
        self.device = device
        self.sparsity_weight = sparsity_weight

        self.mapper = DisentangledMapper(
            n_sines=n_sines,
            hidden_dim=hidden_dim,
            n_disentangle_layers=n_disentangle_layers,
            use_temporal_attention=use_temporal_attention,
            n_attention_layers=n_attention_layers,
        ).to(device)

        self.scaler = torch.amp.GradScaler('cuda')

        params = sum(p.numel() for p in self.mapper.parameters())
        print(f"\nDisentangledTrainer:")
        print(f"  Sines: {n_sines}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Disentangle layers: {n_disentangle_layers}")
        attn_str = f'attention x{n_attention_layers}' if use_temporal_attention else 'conv'
        print(f"  Temporal: {attn_str}")
        print(f"  Sparsity weight: {sparsity_weight}")
        print(f"  Params: {params:,}")

    def get_sparsity_loss(self):
        """L1 regularization on mapping weights."""
        freq_l1 = self.mapper.freq_map.weight.abs().mean()
        amp_l1 = self.mapper.amp_map.weight.abs().mean()
        return freq_l1 + amp_l1

    def train_step(self, batch, optimizer):
        optimizer.zero_grad()

        latent = batch['latent'].to(self.device)
        target_freqs = batch['freqs'].to(self.device)
        target_amps = batch['amps'].to(self.device)
        target_phases = batch['phases'].to(self.device)
        target_noise = batch['noise_amps'].to(self.device)

        with torch.amp.autocast('cuda'):
            pred = self.mapper(latent)

            loss, metrics = sinkhorn_matching_loss(
                pred['freqs'], pred['amps'],
                target_freqs, target_amps,
                pred['phases'], target_phases,
            )

            # Noise loss
            noise_loss = F.mse_loss(pred['noise_amps'], target_noise)
            loss = loss + 0.5 * noise_loss
            metrics['noise_loss'] = noise_loss.item()

            # Sparsity loss (L1 on mapping weights)
            sparsity_loss = self.get_sparsity_loss()
            loss = loss + self.sparsity_weight * sparsity_loss
            metrics['sparsity_loss'] = sparsity_loss.item()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        metrics['loss'] = loss.item()

        # Sparsity stats
        sparsity_stats = self.mapper.get_sparsity_stats()
        metrics.update(sparsity_stats)

        return metrics

    def train(self, dataloader, n_epochs: int = 100, lr: float = 1e-3,
              save_dir: Optional[str] = None):
        optimizer = torch.optim.AdamW(self.mapper.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        print("\n" + "=" * 70)
        print("Disentangled Mapper Training")
        print("=" * 70)
        print("Goal: Learn transform where z → sines mapping is SPARSE")
        print("After training, mapping weights show which z dims affect which sines")
        print("=" * 70)

        for epoch in range(n_epochs):
            self.mapper.train()
            metrics_sum = {}
            n_batches = 0

            for batch in dataloader:
                m = self.train_step(batch, optimizer)
                for k, v in m.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v
                n_batches += 1

            scheduler.step()

            metrics_avg = {k: v / n_batches for k, v in metrics_sum.items()}

            print(f"Epoch {epoch:4d}: loss={metrics_avg['loss']:.4f} "
                  f"freq={metrics_avg['freq_loss']:.4f} "
                  f"amp={metrics_avg['amp_loss']:.4f} "
                  f"sparse={metrics_avg['freq_sparsity']:.2%} "
                  f"dims/sine={metrics_avg['freq_dims_per_sine']:.1f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e}")

            if save_dir and metrics_avg['loss'] < best_loss:
                best_loss = metrics_avg['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.mapper.state_dict(),
                    'loss': best_loss,
                }, str(save_path / "best_model.pt"))

            if epoch % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Final sparsity analysis
        print("\n" + "=" * 70)
        print("FINAL SPARSITY ANALYSIS")
        print("=" * 70)
        self.analyze_learned_structure(save_dir)

        if save_dir:
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': self.mapper.state_dict(),
                'loss': metrics_avg['loss'],
            }, str(save_path / "final_model.pt"))
            print(f"\nSaved to {save_dir}")

        print(f"Training complete! Best loss: {best_loss:.4f}")
        return self.mapper

    def analyze_learned_structure(self, save_dir: Optional[str] = None):
        """Analyze the learned disentangling transform and sparse mappings."""
        self.mapper.eval()

        with torch.no_grad():
            freq_w = self.mapper.freq_map.weight.cpu()  # [n_sines, hidden_dim]
            amp_w = self.mapper.amp_map.weight.cpu()

            print(f"\nFrequency mapping weights: {freq_w.shape}")
            print(f"  Sparsity (|w| < 0.1): {(freq_w.abs() < 0.1).float().mean():.2%}")
            print(f"  Dims per sine (|w| > 0.1): {(freq_w.abs() > 0.1).float().sum(dim=1).mean():.1f}")

            # Find which dims are most important overall
            freq_importance = freq_w.abs().sum(dim=0)  # [hidden_dim]
            top_dims = freq_importance.argsort(descending=True)[:20]
            print(f"  Top 20 most important dims: {top_dims.tolist()}")

            # Check if different sines use different dims (specialization)
            freq_w_binary = (freq_w.abs() > 0.1).float()
            # Overlap between sines: do they share dims?
            overlap = freq_w_binary @ freq_w_binary.T  # [n_sines, n_sines]
            avg_overlap = overlap.mean().item()
            print(f"  Avg dim overlap between sines: {avg_overlap:.2f}")

            print(f"\nAmplitude mapping weights: {amp_w.shape}")
            print(f"  Sparsity (|w| < 0.1): {(amp_w.abs() < 0.1).float().mean():.2%}")
            print(f"  Dims per sine (|w| > 0.1): {(amp_w.abs() > 0.1).float().sum(dim=1).mean():.1f}")

            # Disentangle layer
            if isinstance(self.mapper.disentangle, nn.Linear):
                dis_w = self.mapper.disentangle.weight.cpu()  # [hidden_dim, 128]
                print(f"\nDisentangle weights: {dis_w.shape}")
                print(f"  This matrix transforms raw z to interpretable space")

                # Check if it's close to identity or learned something
                if dis_w.shape[0] == dis_w.shape[1]:
                    identity_dist = (dis_w - torch.eye(dis_w.shape[0])).norm().item()
                    print(f"  Distance from identity: {identity_dist:.4f}")

        if save_dir:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            save_path = Path(save_dir)

            # Plot frequency mapping weights
            plt.figure(figsize=(12, 8))
            plt.imshow(freq_w.numpy(), aspect='auto', cmap='RdBu', vmin=-0.5, vmax=0.5)
            plt.colorbar(label='Weight')
            plt.xlabel('Disentangled z dimension')
            plt.ylabel('Sine index')
            plt.title('Frequency Mapping Weights (sparse = interpretable)')
            plt.savefig(save_path / 'freq_mapping_weights.png', dpi=150)
            plt.close()

            print(f"\nSaved weight visualization to {save_path / 'freq_mapping_weights.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sms_manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_sines', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_disentangle_layers', type=int, default=1)
    parser.add_argument('--temporal_attention', action='store_true')
    parser.add_argument('--n_attention_layers', type=int, default=1)
    parser.add_argument('--sparsity_weight', type=float, default=0.01,
                        help='L1 regularization on mapping weights')
    parser.add_argument('--skip_drums', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("Disentangled Mapper")
    print("=" * 70)
    print(f"\nArchitecture:")
    print(f"  z (128) → disentangle ({args.hidden_dim}) → sparse_map → {args.n_sines} sines")
    print(f"\nKey insight: Learn transform that makes z→sines SPARSE")
    print(f"After training, weights reveal which z dims control which sines")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    dataset = DisentangledDataset(
        sms_manifest_path=args.sms_manifest,
        max_samples=args.max_samples,
        skip_drums=args.skip_drums,
        n_sines=args.n_sines,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Dataloader: {len(dataloader)} batches")

    trainer = DisentangledTrainer(
        n_sines=args.n_sines,
        hidden_dim=args.hidden_dim,
        n_disentangle_layers=args.n_disentangle_layers,
        use_temporal_attention=args.temporal_attention,
        n_attention_layers=args.n_attention_layers,
        sparsity_weight=args.sparsity_weight,
        device=device,
    )

    save_dir = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/disentangled"

    trainer.train(
        dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
