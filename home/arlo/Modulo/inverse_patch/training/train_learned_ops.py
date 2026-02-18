#!/usr/bin/env python3
"""
Learned Operations Training: Discover what operations DCAE learned.

Philosophy: No predefined bias. Let operations emerge from data.

Architecture:
  z_dcae → op_selector → which ops active + parameters
                ↓
        op_embeddings → learned operation "recipes"
                ↓
        op_expander → how each op produces sines
                ↓
        combine → weighted sum of all ops' sine contributions
                ↓
        (freqs, amps) → compare to SMS extraction

After training, examine op_embeddings to discover what emerged:
  - Did it learn harmonic series?
  - Did it learn formants?
  - Did it learn something unexpected?
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
# LEARNED OPERATION CODEC
# ============================================================

class LearnedOperationCodec(nn.Module):
    """
    Discovers operations from data - no hardcoded assumptions.

    Components:
    - op_embeddings: What each operation "is" - learned from data
    - z_to_ops: Which operations are active for this z
    - op_expander: How each operation produces sines

    NO PRESCRIPTIONS:
    - No SAMI coarse/fine split - network learns what parts of z matter
    - No hardcoded harmonic formulas - network discovers patterns
    - After training, analyze to see what structure emerged

    After training, analyze op_embeddings to see what emerged.
    """

    def __init__(
        self,
        n_ops: int = 16,           # Number of operation types to discover
        n_params_per_op: int = 8,  # Parameters per operation
        op_embed_dim: int = 64,    # Dimension of operation embeddings
        n_sines: int = 64,         # Output sines
        hidden_dim: int = 256,
        freq_min: float = 20.0,
        freq_max: float = 16000.0,
    ):
        super().__init__()
        self.n_ops = n_ops
        self.n_params_per_op = n_params_per_op
        self.n_sines = n_sines
        self.freq_min = freq_min
        self.freq_max = freq_max

        # ===== LEARNED OPERATION EMBEDDINGS =====
        # These define WHAT each operation IS
        # Initialized random, learned from data
        # After training, examine these to see what emerged
        self.op_embeddings = nn.Parameter(torch.randn(n_ops, op_embed_dim) * 0.1)

        # ===== LEARNABLE CHANNEL WEIGHTING =====
        # Network learns which dimensions of z matter
        # Initialized uniform - no SAMI prescription
        # After training, check: did it discover coarse/fine split?
        self.channel_weights = nn.Parameter(torch.ones(8))  # Per-channel
        self.dim_weights = nn.Parameter(torch.ones(16))     # Per height dim

        # ===== Z → OPERATION SELECTION =====
        # Full z projection - network decides what to use
        self.z_proj = nn.Linear(128, hidden_dim)

        self.z_encoder = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Operation weights (which ops are active)
        self.op_weight_head = nn.Linear(hidden_dim, n_ops)

        # Operation parameters (how each op is configured)
        self.op_param_head = nn.Linear(hidden_dim, n_ops * n_params_per_op)

        # ===== OPERATION → SINES =====
        # Each operation produces its OWN subset of sines
        # 16 ops × 4 sines each = 64 total
        self.sines_per_op = n_sines // n_ops

        self.op_expander = nn.Sequential(
            nn.Linear(op_embed_dim + n_params_per_op, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.sines_per_op * 2),  # freq + amp for THIS op's sines
        )

        # Phase head (separate, simpler)
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, n_sines),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z: [B, 8, 16, T] DCAE latent

        Returns:
            freqs: [B, T, n_sines]
            amps: [B, T, n_sines]
            phases: [B, T, n_sines]
            op_weights: [B, T, n_ops] - for analysis/sparsity
        """
        B, C, H, T = z.shape

        # Apply learnable channel and dimension weighting
        # Network learns what parts of z matter - no hardcoded SAMI split
        channel_w = F.softplus(self.channel_weights).view(1, C, 1, 1)
        dim_w = F.softplus(self.dim_weights).view(1, 1, H, 1)
        z_weighted = z * channel_w * dim_w

        # Flatten to [B, T, 128]
        z_flat = z_weighted.permute(0, 3, 1, 2).reshape(B, T, C * H)

        # Project and encode - network decides what to use
        z_proj = self.z_proj(z_flat)
        h = self.z_encoder(z_proj)  # [B, T, hidden]

        # Get operation parameters - each op gets its own params from z
        op_params = self.op_param_head(h)  # [B, T, n_ops * n_params]
        op_params = op_params.reshape(B, T, self.n_ops, self.n_params_per_op)

        # Each operation owns a subset of sines
        # 16 ops × 4 sines each = 64 total sines
        # NO GATING - all ops always active, sparsity emerges from amplitude learning
        sines_per_op = self.sines_per_op

        all_freqs = []
        all_amps = []

        for op_idx in range(self.n_ops):
            # Get this operation's embedding (what this op IS)
            op_emb = self.op_embeddings[op_idx]  # [op_embed_dim]
            op_emb_expanded = op_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

            # Get this operation's parameters (how it's configured for this z)
            params = op_params[:, :, op_idx, :]  # [B, T, n_params]

            # Combine embedding + params
            op_input = torch.cat([op_emb_expanded, params], dim=-1)  # [B, T, embed + params]

            # This op produces ITS OWN sines - no gating
            sines = self.op_expander(op_input)  # [B, T, sines_per_op * 2]
            sines = sines.reshape(B, T, sines_per_op, 2)

            freq_logits = sines[:, :, :, 0]  # [B, T, sines_per_op]
            amp_logits = sines[:, :, :, 1]   # No gating - amp learned directly

            all_freqs.append(freq_logits)
            all_amps.append(amp_logits)

        # Concatenate all ops' sines
        combined_freq_logits = torch.cat(all_freqs, dim=-1)  # [B, T, n_sines]
        combined_amp_logits = torch.cat(all_amps, dim=-1)

        # For analysis: compute which ops are "active" (have high amplitude sines)
        # This is just for logging, not used in forward pass
        op_activity = torch.stack([a.abs().mean(dim=-1) for a in all_amps], dim=-1)  # [B, T, n_ops]
        op_weights = op_activity / (op_activity.sum(dim=-1, keepdim=True) + 1e-8)  # Normalized

        # Apply output activations
        log_freq_min = np.log(self.freq_min)
        log_freq_max = np.log(self.freq_max)
        log_freqs = log_freq_min + torch.sigmoid(combined_freq_logits) * (log_freq_max - log_freq_min)
        freqs = torch.exp(log_freqs)

        amps = torch.sigmoid(combined_amp_logits)

        # Phases
        phases = torch.tanh(self.phase_head(h)) * np.pi

        return {
            'freqs': freqs,
            'amps': amps,
            'phases': phases,
            'op_weights': op_weights,
        }


# ============================================================
# ALTERNATIVE: SLOT-BASED OPERATIONS
# ============================================================

class SlotOperationCodec(nn.Module):
    """
    Alternative architecture: Operations produce SLOTS of sines.

    Each operation "owns" some sine slots and decides their freqs/amps.
    More interpretable - you can see which op controls which sines.
    """

    def __init__(
        self,
        n_ops: int = 8,
        sines_per_op: int = 8,  # Each op controls 8 sines
        n_params_per_op: int = 8,
        op_embed_dim: int = 64,
        hidden_dim: int = 256,
        freq_min: float = 20.0,
        freq_max: float = 16000.0,
    ):
        super().__init__()
        self.n_ops = n_ops
        self.sines_per_op = sines_per_op
        self.n_sines = n_ops * sines_per_op
        self.n_params_per_op = n_params_per_op
        self.freq_min = freq_min
        self.freq_max = freq_max

        # Operation embeddings - what each op IS
        self.op_embeddings = nn.Parameter(torch.randn(n_ops, op_embed_dim) * 0.1)

        # Z encoder
        self.z_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Per-operation activation (is this op active?)
        self.op_gate = nn.Linear(hidden_dim, n_ops)

        # Per-operation parameters
        self.op_param_head = nn.Linear(hidden_dim, n_ops * n_params_per_op)

        # Each operation's sine generator
        self.op_to_sines = nn.Sequential(
            nn.Linear(op_embed_dim + n_params_per_op, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, sines_per_op * 3),  # freq, amp, phase per sine
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, T = z.shape
        z_flat = z.permute(0, 3, 1, 2).reshape(B, T, C * H)

        h = self.z_encoder(z_flat)

        # Operation gates (soft activation)
        op_gates = torch.sigmoid(self.op_gate(h))  # [B, T, n_ops]

        # Operation parameters
        op_params = self.op_param_head(h).reshape(B, T, self.n_ops, self.n_params_per_op)

        all_freqs = []
        all_amps = []
        all_phases = []

        for op_idx in range(self.n_ops):
            op_emb = self.op_embeddings[op_idx].unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            params = op_params[:, :, op_idx, :]

            op_input = torch.cat([op_emb, params], dim=-1)
            sines = self.op_to_sines(op_input)  # [B, T, sines_per_op * 3]
            sines = sines.reshape(B, T, self.sines_per_op, 3)

            # Apply gate
            gate = op_gates[:, :, op_idx:op_idx+1, None]  # [B, T, 1, 1]

            freqs = sines[:, :, :, 0]  # [B, T, sines_per_op]
            amps = sines[:, :, :, 1] * gate.squeeze(-1)  # Gated amplitude
            phases = sines[:, :, :, 2]

            all_freqs.append(freqs)
            all_amps.append(amps)
            all_phases.append(phases)

        # Concatenate all slots
        freqs = torch.cat(all_freqs, dim=-1)  # [B, T, n_sines]
        amps = torch.cat(all_amps, dim=-1)
        phases = torch.cat(all_phases, dim=-1)

        # Output activations
        log_freq_min = np.log(self.freq_min)
        log_freq_max = np.log(self.freq_max)
        freqs = torch.exp(log_freq_min + torch.sigmoid(freqs) * (log_freq_max - log_freq_min))
        amps = torch.sigmoid(amps)
        phases = torch.tanh(phases) * np.pi

        return {
            'freqs': freqs,
            'amps': amps,
            'phases': phases,
            'op_weights': op_gates,
        }


# ============================================================
# DATASET (reuse from train_sms_hybrid.py)
# ============================================================

class SMSDataset(Dataset):
    """Load SMS extractions with latents."""

    DRUM_KEYWORDS = ['drum', 'kick', 'snare', 'hat', 'tom', 'perc', 'cymbal', 'overhead', ' oh ', '_oh_', 'hihat', 'hh_', '_hh']

    def __init__(
        self,
        sms_manifest_path: str,
        max_samples: Optional[int] = None,
        target_frames: int = 22,
        skip_drums: bool = True,
        n_sines: int = 64,
    ):
        self.target_frames = target_frames
        self.n_sines = n_sines

        print(f"Loading SMS manifest from {sms_manifest_path}...")
        with open(sms_manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        entries = manifest['entries']
        if max_samples:
            entries = entries[:max_samples]

        print(f"  Found {len(entries)} entries, skip_drums={skip_drums}")

        self.data = []
        skipped_drums = 0

        for entry in entries:
            try:
                sms_data = torch.load(entry['path'], weights_only=True, map_location='cpu')

                if skip_drums:
                    audio_path = sms_data.get('audio_path', '').lower()
                    if any(kw in audio_path for kw in self.DRUM_KEYWORDS):
                        skipped_drums += 1
                        continue

                lat_data = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
                latent = lat_data.get('latents', lat_data.get('latent'))
                if latent is None:
                    continue

                C, H, T = latent.shape
                freqs = sms_data['freqs'][:, :n_sines]
                amps = sms_data['amps'][:, :n_sines]
                phases = sms_data['phases'][:, :n_sines]

                # Crop/pad to target frames
                if T < target_frames:
                    latent = F.pad(latent, (0, target_frames - T))
                    freqs = F.pad(freqs, (0, 0, 0, target_frames - freqs.shape[0]))
                    amps = F.pad(amps, (0, 0, 0, target_frames - amps.shape[0]))
                    phases = F.pad(phases, (0, 0, 0, target_frames - phases.shape[0]))
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

                self.data.append({
                    'latent': latent,
                    'freqs': freqs,
                    'amps': amps,
                    'phases': phases,
                })

                if len(self.data) % 500 == 0:
                    print(f"\r  Loaded {len(self.data)}...", end="", flush=True)

            except Exception:
                continue

        print(f"\r  Loaded {len(self.data)} samples (skipped {skipped_drums} drums)          ")

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
    }


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def sine_matching_loss(pred_freqs, pred_amps, target_freqs, target_amps,
                       pred_phases=None, target_phases=None):
    """Amplitude-sorted matching loss."""
    B, T, N = pred_freqs.shape

    # Sort by amplitude (descending)
    pred_order = pred_amps.argsort(dim=-1, descending=True)
    target_order = target_amps.argsort(dim=-1, descending=True)

    pred_f = pred_freqs.gather(-1, pred_order)
    pred_a = pred_amps.gather(-1, pred_order)
    target_f = target_freqs.gather(-1, target_order)
    target_a = target_amps.gather(-1, target_order)

    # Active mask
    active_mask = (target_a > 0.01).float()

    # Frequency loss (log scale)
    log_pred_f = torch.log(pred_f.clamp(min=20))
    log_target_f = torch.log(target_f.clamp(min=20))
    freq_loss = (active_mask * (log_pred_f - log_target_f).pow(2)).sum() / (active_mask.sum() + 1e-8)

    # Amplitude loss (all slots)
    amp_loss = (pred_a - target_a).pow(2).mean()

    # Phase loss
    phase_loss = torch.tensor(0.0, device=pred_freqs.device)
    if pred_phases is not None and target_phases is not None:
        pred_ph = pred_phases.gather(-1, pred_order)
        target_ph = target_phases.gather(-1, target_order)
        phase_diff = torch.abs(pred_ph - target_ph)
        phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)
        phase_loss = (active_mask * phase_diff.pow(2)).sum() / (active_mask.sum() + 1e-8)

    return freq_loss, amp_loss, phase_loss


def operation_sparsity_loss(op_weights: torch.Tensor, target_active: int = 3) -> torch.Tensor:
    """
    Encourage sparse operation usage.

    We want each frame to use only a few operations, not all of them.
    This forces specialization - each op must become good at something specific.
    """
    # Entropy-based: encourage peaked distribution
    entropy = -(op_weights * (op_weights + 1e-8).log()).sum(dim=-1)  # [B, T]

    # We want low entropy (peaked at few ops)
    # Max entropy is log(n_ops), we want much less
    max_entropy = np.log(op_weights.shape[-1])
    target_entropy = np.log(target_active)  # Allow ~target_active ops

    excess_entropy = F.relu(entropy - target_entropy)

    return excess_entropy.mean()


def operation_diversity_loss(op_weights: torch.Tensor) -> torch.Tensor:
    """
    Encourage different frames to use different operations.
    Prevents all ops from collapsing to the same thing.
    """
    # Average op usage across batch [n_ops]
    avg_usage = op_weights.mean(dim=(0, 1))

    # Penalty 1: Deviation from uniform (all ops should be used equally on average)
    uniform = torch.ones_like(avg_usage) / len(avg_usage)
    deviation = ((avg_usage - uniform) ** 2).sum()

    # Penalty 2: Dead ops (any op with very low usage)
    min_usage = 0.02  # Each op should be used at least 2% of the time
    dead_penalty = F.relu(min_usage - avg_usage).sum() * 10.0

    # Penalty 3: Dominant op (any op used more than 50%)
    max_usage = 0.5
    dominant_penalty = F.relu(avg_usage - max_usage).sum() * 10.0

    return deviation + dead_penalty + dominant_penalty


# ============================================================
# TRAINER
# ============================================================

class LearnedOpsTrainer:
    def __init__(
        self,
        model_type: str = 'codec',  # 'codec' or 'slot'
        n_ops: int = 16,
        n_sines: int = 64,
        hidden_dim: int = 256,
        sparsity_weight: float = 0.1,
        diversity_weight: float = 0.1,
        phase_weight: float = 0.1,
        device: str = 'cuda',
    ):
        self.device = device
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.phase_weight = phase_weight

        if model_type == 'codec':
            self.model = LearnedOperationCodec(
                n_ops=n_ops,
                n_sines=n_sines,
                hidden_dim=hidden_dim,
            ).to(device)
        else:
            sines_per_op = n_sines // n_ops
            self.model = SlotOperationCodec(
                n_ops=n_ops,
                sines_per_op=sines_per_op,
                hidden_dim=hidden_dim,
            ).to(device)

        self.scaler = torch.amp.GradScaler('cuda')

        params = sum(p.numel() for p in self.model.parameters())
        print(f"\nLearnedOpsTrainer ({model_type}):")
        print(f"  N operations: {n_ops}")
        print(f"  N sines: {n_sines}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Params: {params:,}")
        print(f"  Sparsity weight: {sparsity_weight}")
        print(f"  Diversity weight: {diversity_weight}")

    def train_step(self, batch, optimizer):
        optimizer.zero_grad()

        latent = batch['latent'].to(self.device)
        target_freqs = batch['freqs'].to(self.device)
        target_amps = batch['amps'].to(self.device)
        target_phases = batch['phases'].to(self.device)

        with torch.amp.autocast('cuda'):
            pred = self.model(latent)

            # Reconstruction losses
            freq_loss, amp_loss, phase_loss = sine_matching_loss(
                pred['freqs'], pred['amps'],
                target_freqs, target_amps,
                pred['phases'], target_phases,
            )

            # Operation regularization
            sparsity_loss = operation_sparsity_loss(pred['op_weights'])
            diversity_loss = operation_diversity_loss(pred['op_weights'])

            # Combined loss
            loss = (freq_loss +
                    amp_loss +
                    self.phase_weight * phase_loss +
                    self.sparsity_weight * sparsity_loss +
                    self.diversity_weight * diversity_loss)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()

        # Stats
        with torch.no_grad():
            op_usage = pred['op_weights'].mean(dim=(0, 1))  # [n_ops]
            active_ops = (op_usage > 0.1).sum().item()
            dominant_op = op_usage.argmax().item()

        return {
            'loss': loss.item(),
            'freq_loss': freq_loss.item(),
            'amp_loss': amp_loss.item(),
            'phase_loss': phase_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'active_ops': active_ops,
            'dominant_op': dominant_op,
        }

    def train(self, dataloader, n_epochs: int = 100, lr: float = 1e-3,
              save_dir: Optional[str] = None):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        best_loss = float('inf')

        print("\n" + "="*70)
        print("Learned Operations Training - Discovering Operations from Data")
        print("="*70)

        for epoch in range(n_epochs):
            self.model.train()
            metrics_sum = {}
            n_batches = 0

            for batch in dataloader:
                m = self.train_step(batch, optimizer)
                for k, v in m.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v
                n_batches += 1

            scheduler.step()

            metrics = {k: v / n_batches for k, v in metrics_sum.items()}

            print(f"Epoch {epoch:4d}: loss={metrics['loss']:.4f} "
                  f"freq={metrics['freq_loss']:.4f} "
                  f"amp={metrics['amp_loss']:.4f} "
                  f"sparse={metrics['sparsity_loss']:.4f} "
                  f"active_ops={metrics['active_ops']:.0f} "
                  f"| lr={scheduler.get_last_lr()[0]:.2e}")

            if save_dir and metrics['loss'] < best_loss:
                best_loss = metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'op_embeddings': self.model.op_embeddings.detach().cpu(),
                    'loss': best_loss,
                }, str(save_path / "best_model.pt"))

            # Analyze operations every 50 epochs
            if epoch % 50 == 0 and epoch > 0:
                self.analyze_operations()

        if save_dir:
            torch.save({
                'epoch': n_epochs,
                'model_state_dict': self.model.state_dict(),
                'op_embeddings': self.model.op_embeddings.detach().cpu(),
                'loss': metrics['loss'],
            }, str(save_path / "final_model.pt"))
            print(f"\nSaved to {save_dir}")

        # Final analysis
        self.analyze_operations()

        return self.model

    def analyze_operations(self):
        """Examine what operations emerged."""
        print("\n" + "-"*50)
        print("OPERATION ANALYSIS - What Emerged?")
        print("-"*50)

        op_emb = self.model.op_embeddings.detach().cpu()

        # Similarity between operations
        norms = op_emb.norm(dim=1, keepdim=True)
        normalized = op_emb / (norms + 1e-8)
        similarity = normalized @ normalized.T

        print("\nOperation embedding similarities:")
        for i in range(min(8, len(op_emb))):
            sims = [f"{similarity[i,j]:.2f}" for j in range(min(8, len(op_emb)))]
            print(f"  Op {i}: [{', '.join(sims)}]")

        # Cluster operations by embedding similarity
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist

        try:
            distances = pdist(op_emb.numpy())
            Z = linkage(distances, method='ward')
            clusters = fcluster(Z, t=3, criterion='maxclust')

            print("\nOperation clusters (by embedding similarity):")
            for c in range(1, 4):
                ops_in_cluster = [i for i, cl in enumerate(clusters) if cl == c]
                print(f"  Cluster {c}: Ops {ops_in_cluster}")
        except:
            pass

        print("-"*50 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sms_manifest', type=str,
                        default='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json')
    parser.add_argument('--model_type', type=str, default='codec', choices=['codec', 'slot'])
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_ops', type=int, default=16)
    parser.add_argument('--n_sines', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--sparsity', type=float, default=0.0,
                        help='Sparsity weight (0 = off)')
    parser.add_argument('--diversity', type=float, default=0.0,
                        help='Diversity weight (0 = off, test raw reconstruction first)')
    parser.add_argument('--skip_drums', action='store_true')
    args = parser.parse_args()

    print("="*70)
    print("Learned Operations Training")
    print("="*70)
    print("\nPhilosophy: Discover operations from data, no hardcoded assumptions.")
    print("After training, analyze op_embeddings to see what emerged.")
    sys.stdout.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    dataset = SMSDataset(
        sms_manifest_path=args.sms_manifest,
        max_samples=args.max_samples,
        skip_drums=args.skip_drums,
        n_sines=args.n_sines,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    trainer = LearnedOpsTrainer(
        model_type=args.model_type,
        n_ops=args.n_ops,
        n_sines=args.n_sines,
        hidden_dim=args.hidden_dim,
        sparsity_weight=args.sparsity,
        diversity_weight=args.diversity,
        device=device,
    )

    save_dir = f"/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/learned_ops_{args.model_type}"

    trainer.train(
        dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
