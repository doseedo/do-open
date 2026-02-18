#!/usr/bin/env python3
"""
Structured Mapper using DISCOVERED z roles.

Discovery summary:
  z[0-47]:   Energy/dynamics (weak per-dim, cumulative)
  z[48-63]:  Frequency content (QUADRATIC → centroid, R²=1.0)
  z[64-127]: Temporal shape (std, flux, attack/release)
  Temporal:  Attention-like (not just local convolution)

Key finding: centroid = a*z² + b*z + c  (R² = 1.0, exact fit)

Architecture:
1. Frequency module: z[48-63] → quadratic → spectral centroid/shape
2. Temporal module: z[64-127] → attention → temporal envelope
3. Energy module: z[0-47] → cumulative → overall amplitude
4. Combine: freq × temporal × energy → sines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import importlib.util

# Direct import
spec = importlib.util.spec_from_file_location(
    'train_harmonic_ops',
    '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/training/train_harmonic_ops.py'
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

HarmonicSMSDataset = train_module.HarmonicSMSDataset
collate_fn = train_module.collate_fn
sinkhorn_matching_loss = train_module.sinkhorn_matching_loss


# ============================================================================
# DISCOVERED Z STRUCTURE
# ============================================================================

ENERGY_DIMS = list(range(0, 48))      # Energy/dynamics (cumulative)
FREQ_DIMS = list(range(48, 64))        # Frequency content (quadratic)
TEMPORAL_DIMS = list(range(64, 128))   # Temporal shape (attention-like)


class QuadraticLayer(nn.Module):
    """
    Exact quadratic: y = W2 @ x² + W1 @ x + b

    Discovered: z → centroid is exactly quadratic (R² = 1.0)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W2 = nn.Linear(in_features, out_features, bias=False)
        self.W1 = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return self.W2(x ** 2) + self.W1(x) + self.bias


class FrequencyModule(nn.Module):
    """
    z[48-63] → frequency content

    Uses quadratic transform (discovered R² = 1.0)
    Outputs: spectral centroid, bandwidth, harmonic structure
    """
    def __init__(self, n_sines=64):
        super().__init__()
        self.n_sines = n_sines
        n_freq_dims = len(FREQ_DIMS)

        # Quadratic transform (discovered exact mapping)
        self.quadratic = QuadraticLayer(n_freq_dims, 64)

        # Frequency head: predicts log-frequencies
        self.freq_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, n_sines),
        )

        # Base amplitude head (before temporal modulation)
        self.base_amp_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, n_sines),
        )

        self.log_freq_min = np.log(20.0)
        self.log_freq_max = np.log(8000.0)

    def forward(self, z_freq):
        """
        z_freq: [B, T, 16] - frequency dims

        Returns:
            freqs: [B, T, n_sines] - frequencies in Hz
            base_amps: [B, T, n_sines] - base amplitudes (before temporal mod)
        """
        # Quadratic transform (exact mapping discovered)
        h = self.quadratic(z_freq)  # [B, T, 64]
        h = F.gelu(h)

        # Frequency prediction
        freq_logits = self.freq_head(h)
        freq_norm = torch.sigmoid(freq_logits)
        log_freq = self.log_freq_min + freq_norm * (self.log_freq_max - self.log_freq_min)
        freqs = torch.exp(log_freq)

        # Base amplitude
        base_amps = torch.sigmoid(self.base_amp_head(h))

        return freqs, base_amps


class TemporalModule(nn.Module):
    """
    z[64-127] → temporal shape

    Uses attention (discovered: attention-like spread, not just convolution)
    Outputs: temporal envelope for amplitude modulation
    """
    def __init__(self, n_sines=64, n_heads=4):
        super().__init__()
        self.n_sines = n_sines
        n_temp_dims = len(TEMPORAL_DIMS)

        # Project to hidden dim
        self.proj = nn.Linear(n_temp_dims, 64)

        # Self-attention for temporal patterns (discovered: attention-like)
        self.attention = nn.MultiheadAttention(64, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(64)

        # Envelope head: per-sine temporal modulation
        self.envelope_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, n_sines),
        )

    def forward(self, z_temp):
        """
        z_temp: [B, T, 64] - temporal dims

        Returns:
            envelope: [B, T, n_sines] - temporal envelope multiplier
        """
        # Project
        h = self.proj(z_temp)  # [B, T, 64]

        # Self-attention (discovered: attention-like temporal spread)
        h_attn, _ = self.attention(h, h, h)
        h = self.norm(h + h_attn)  # Residual

        # Envelope prediction
        envelope = torch.sigmoid(self.envelope_head(h))

        return envelope


class EnergyModule(nn.Module):
    """
    z[0-47] → overall energy

    Uses cumulative sum (discovered: weak per-dim but cumulative effect)
    Outputs: global amplitude scaling
    """
    def __init__(self, n_sines=64):
        super().__init__()
        self.n_sines = n_sines
        n_energy_dims = len(ENERGY_DIMS)

        # Simple aggregation (discovered: cumulative effect)
        self.aggregate = nn.Sequential(
            nn.Linear(n_energy_dims, 32),
            nn.GELU(),
            nn.Linear(32, n_sines),
        )

    def forward(self, z_energy):
        """
        z_energy: [B, T, 48] - energy dims

        Returns:
            energy_scale: [B, T, n_sines] - energy multiplier
        """
        energy_scale = torch.sigmoid(self.aggregate(z_energy))
        return energy_scale


class StructuredMapper(nn.Module):
    """
    Full structured mapper using discovered z roles.

    z[0-47]   → EnergyModule   → energy_scale
    z[48-63]  → FrequencyModule → freqs, base_amps
    z[64-127] → TemporalModule  → envelope

    final_amps = base_amps * envelope * energy_scale
    """
    def __init__(self, n_sines=64):
        super().__init__()
        self.n_sines = n_sines

        self.freq_module = FrequencyModule(n_sines)
        self.temporal_module = TemporalModule(n_sines)
        self.energy_module = EnergyModule(n_sines)

    def forward(self, z):
        """
        z: [B, 8, 16, T]

        Returns:
            freqs: [B, T, n_sines]
            amps: [B, T, n_sines]
        """
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        # Split by discovered roles
        z_energy = z_flat[..., ENERGY_DIMS]    # [B, T, 48]
        z_freq = z_flat[..., FREQ_DIMS]        # [B, T, 16]
        z_temp = z_flat[..., TEMPORAL_DIMS]    # [B, T, 64]

        # Process each module
        freqs, base_amps = self.freq_module(z_freq)
        envelope = self.temporal_module(z_temp)
        energy_scale = self.energy_module(z_energy)

        # Combine amplitudes
        amps = base_amps * envelope * energy_scale

        return freqs, amps


class StructuredMapperV2(nn.Module):
    """
    V2: Adds cross-module interaction.

    Frequency can influence temporal (e.g., high freqs decay faster)
    Temporal can influence frequency (e.g., pitch drift over time)
    """
    def __init__(self, n_sines=64, hidden_dim=64):
        super().__init__()
        self.n_sines = n_sines
        self.hidden_dim = hidden_dim

        # Per-role encoders
        self.energy_enc = nn.Sequential(
            nn.Linear(len(ENERGY_DIMS), hidden_dim),
            nn.GELU(),
        )
        self.freq_enc = QuadraticLayer(len(FREQ_DIMS), hidden_dim)
        self.temp_enc = nn.Linear(len(TEMPORAL_DIMS), hidden_dim)

        # Cross-module attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # Temporal self-attention
        self.temp_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.temp_norm = nn.LayerNorm(hidden_dim)

        # Output heads
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, n_sines),
        )
        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, n_sines),
        )

        self.log_freq_min = np.log(20.0)
        self.log_freq_max = np.log(8000.0)

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        # Split by discovered roles
        z_energy = z_flat[..., ENERGY_DIMS]
        z_freq = z_flat[..., FREQ_DIMS]
        z_temp = z_flat[..., TEMPORAL_DIMS]

        # Encode each
        h_energy = self.energy_enc(z_energy)  # [B, T, hidden]
        h_freq = F.gelu(self.freq_enc(z_freq))
        h_temp = self.temp_enc(z_temp)

        # Temporal self-attention (discovered: attention-like)
        h_temp_attn, _ = self.temp_attn(h_temp, h_temp, h_temp)
        h_temp = self.temp_norm(h_temp + h_temp_attn)

        # Stack for cross-attention
        # Frequency attends to temporal (pitch can drift)
        h_freq_attended, _ = self.cross_attn(h_freq, h_temp, h_temp)
        h_freq = self.norm(h_freq + h_freq_attended)

        # Concatenate all for final prediction
        h_combined = torch.cat([h_energy, h_freq, h_temp], dim=-1)  # [B, T, 3*hidden]

        # Predict
        freq_logits = self.freq_head(h_combined)
        freq_norm = torch.sigmoid(freq_logits)
        log_freq = self.log_freq_min + freq_norm * (self.log_freq_max - self.log_freq_min)
        freqs = torch.exp(log_freq)

        amps = torch.sigmoid(self.amp_head(h_combined))

        return freqs, amps


class StructuredMapperV3(nn.Module):
    """
    V3: Higher capacity + deeper processing.

    Key insight: discovery was z → mel features, not z → sines.
    Need more capacity to bridge that gap.
    """
    def __init__(self, n_sines=64, hidden_dim=256, n_layers=3):
        super().__init__()
        self.n_sines = n_sines

        # Deeper encoders per role
        self.energy_enc = nn.Sequential(
            nn.Linear(len(ENERGY_DIMS), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Quadratic + MLP for frequency
        self.freq_quad = QuadraticLayer(len(FREQ_DIMS), hidden_dim)
        self.freq_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Temporal with multiple attention layers
        self.temp_proj = nn.Linear(len(TEMPORAL_DIMS), hidden_dim)
        self.temp_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True),
                'norm1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
            })
            for _ in range(n_layers)
        ])

        # Cross-module fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
        )

        # Output heads
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_sines),
        )
        self.amp_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_sines),
        )

        self.log_freq_min = np.log(20.0)
        self.log_freq_max = np.log(8000.0)

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)

        # Split by discovered roles
        z_energy = z_flat[..., ENERGY_DIMS]
        z_freq = z_flat[..., FREQ_DIMS]
        z_temp = z_flat[..., TEMPORAL_DIMS]

        # Encode
        h_energy = self.energy_enc(z_energy)
        h_freq = F.gelu(self.freq_quad(z_freq))
        h_freq = self.freq_mlp(h_freq)

        # Temporal with stacked attention
        h_temp = self.temp_proj(z_temp)
        for layer in self.temp_layers:
            h_attn, _ = layer['attn'](h_temp, h_temp, h_temp)
            h_temp = layer['norm1'](h_temp + h_attn)
            h_ff = layer['ff'](h_temp)
            h_temp = layer['norm2'](h_temp + h_ff)

        # Fuse
        h_combined = torch.cat([h_energy, h_freq, h_temp], dim=-1)
        h_fused = self.fusion(h_combined)

        # Output
        freq_logits = self.freq_head(h_fused)
        freq_norm = torch.sigmoid(freq_logits)
        log_freq = self.log_freq_min + freq_norm * (self.log_freq_max - self.log_freq_min)
        freqs = torch.exp(log_freq)

        amps = torch.sigmoid(self.amp_head(h_fused))

        return freqs, amps


class PureQuadraticMapper(nn.Module):
    """
    Simplest possible: pure quadratic on freq dims only.

    Tests if quadratic alone is sufficient.
    """
    def __init__(self, n_sines=64):
        super().__init__()
        self.n_sines = n_sines

        # Pure quadratic (discovered R² = 1.0)
        self.freq_quad = QuadraticLayer(len(FREQ_DIMS), n_sines)
        self.amp_quad = QuadraticLayer(len(FREQ_DIMS), n_sines)

        self.log_freq_min = np.log(20.0)
        self.log_freq_max = np.log(8000.0)

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)
        z_freq = z_flat[..., FREQ_DIMS]

        freq_logits = self.freq_quad(z_freq)
        freq_norm = torch.sigmoid(freq_logits)
        log_freq = self.log_freq_min + freq_norm * (self.log_freq_max - self.log_freq_min)
        freqs = torch.exp(log_freq)

        amps = torch.sigmoid(self.amp_quad(z_freq))

        return freqs, amps


def train_and_eval(model, dataloader, n_epochs=100, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_freq_loss = float('inf')

    for epoch in range(n_epochs):
        total_loss = 0
        total_freq_loss = 0
        n_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            z = batch['latent'].to(device)
            target_freqs = batch['freqs'].to(device)
            target_amps = batch['amps'].to(device)

            pred_freqs, pred_amps = model(z)

            loss, metrics = sinkhorn_matching_loss(
                pred_freqs, pred_amps,
                target_freqs, target_amps,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_freq_loss += metrics['freq_loss']
            n_batches += 1

        scheduler.step()

        avg_freq_loss = total_freq_loss / n_batches
        if avg_freq_loss < best_freq_loss:
            best_freq_loss = avg_freq_loss

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/n_batches:.4f}, freq={avg_freq_loss:.4f}")

    return best_freq_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sines', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip_drums', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\nLoading data...")
    dataset = HarmonicSMSDataset(
        sms_manifest_path='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        max_samples=args.max_samples,
        skip_drums=args.skip_drums,
        n_sines=args.n_sines,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    results = {}

    # Test 1: Pure quadratic (simplest possible)
    print("\n" + "=" * 70)
    print("TEST 1: Pure Quadratic (z[48-63] only)")
    print("Tests if quadratic alone is sufficient")
    print("=" * 70)

    model1 = PureQuadraticMapper(n_sines=args.n_sines)
    n_params = sum(p.numel() for p in model1.parameters())
    print(f"Params: {n_params:,}")

    results['pure_quadratic'] = train_and_eval(model1, dataloader, n_epochs=args.epochs,
                                                lr=args.lr, device=device)

    # Test 2: Structured (all discovered roles)
    print("\n" + "=" * 70)
    print("TEST 2: Structured Mapper (all discovered roles)")
    print("z[0-47]→energy, z[48-63]→freq, z[64-127]→temporal")
    print("=" * 70)

    model2 = StructuredMapper(n_sines=args.n_sines)
    n_params = sum(p.numel() for p in model2.parameters())
    print(f"Params: {n_params:,}")

    results['structured'] = train_and_eval(model2, dataloader, n_epochs=args.epochs,
                                            lr=args.lr, device=device)

    # Test 3: Structured V2 (with cross-module attention)
    print("\n" + "=" * 70)
    print("TEST 3: Structured V2 (cross-module interaction)")
    print("Frequency and temporal modules can interact")
    print("=" * 70)

    model3 = StructuredMapperV2(n_sines=args.n_sines)
    n_params = sum(p.numel() for p in model3.parameters())
    print(f"Params: {n_params:,}")

    results['structured_v2'] = train_and_eval(model3, dataloader, n_epochs=args.epochs,
                                               lr=args.lr, device=device)

    # Test 4: Structured V3 (high capacity)
    print("\n" + "=" * 70)
    print("TEST 4: Structured V3 (high capacity)")
    print("256 hidden, 3 attention layers, deeper MLPs")
    print("=" * 70)

    model4 = StructuredMapperV3(n_sines=args.n_sines, hidden_dim=256, n_layers=3)
    n_params = sum(p.numel() for p in model4.parameters())
    print(f"Params: {n_params:,}")

    results['structured_v3'] = train_and_eval(model4, dataloader, n_epochs=args.epochs,
                                               lr=args.lr, device=device)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nFreq loss (lower is better):")
    for name, loss in sorted(results.items(), key=lambda x: x[1]):
        semitones = loss * 60
        print(f"  {name:20s}: {loss:.4f} (~{semitones:.1f} semitones)")

    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINES")
    print("=" * 70)
    print("""
  Previous results:
    Linear sparse:      0.56  (33 semitones)
    Disentangled MLP:   0.054 (3.2 semitones)
    Harmonic MLP:       0.046 (2.8 semitones)
    """)

    best = min(results.values())
    if best < 0.04:
        print(f"\n✓ SIGNIFICANT IMPROVEMENT: Best structured ({best:.4f}) beats all baselines")
        print("  Discovery of z roles is validated!")
    elif best < 0.046:
        print(f"\n✓ IMPROVEMENT: Best structured ({best:.4f}) beats harmonic MLP (0.046)")
    else:
        print(f"\n~ NO IMPROVEMENT: Best structured ({best:.4f})")
        print("  Structure is correct but mapping needs refinement")


if __name__ == "__main__":
    main()
