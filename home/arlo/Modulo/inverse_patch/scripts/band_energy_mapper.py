#!/usr/bin/env python3
"""
Band Energy Mapper - matches what z ACTUALLY encodes.

Discovery:
  z[48-51] → LOW band (40-300 Hz)     +68.6%
  z[52-55] → MID band (300-1000 Hz)   +87.2%
  z[56-59] → HIGH band (1000-4000 Hz) +56.0%
  z[60-63] → VERY_HIGH (4000-16000 Hz) +94.1%

z encodes BAND ENERGIES, not individual frequencies.
Individual frequencies within bands are NOT controlled by z.

Approach:
1. Predict band energies from z (quadratic, matches discovery)
2. Distribute sines within each band
3. Use temporal dims (64-127) for envelope
4. Use energy dims (0-47) for overall amplitude

This should achieve better accuracy because we're predicting
what z actually encodes, not fighting the structure.
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
# DISCOVERED STRUCTURE
# ============================================================================

# Band control dims (discovered clean separation)
BAND_DIMS = {
    'low': (list(range(48, 52)), (40, 300)),        # z[48-51] → 40-300 Hz
    'mid': (list(range(52, 56)), (300, 1000)),      # z[52-55] → 300-1000 Hz
    'high': (list(range(56, 60)), (1000, 4000)),    # z[56-59] → 1000-4000 Hz
    'very_high': (list(range(60, 64)), (4000, 8000)),  # z[60-63] → 4000-8000 Hz
}

TEMPORAL_DIMS = list(range(64, 128))  # Temporal envelope
ENERGY_DIMS = list(range(0, 48))      # Overall energy


class QuadraticLayer(nn.Module):
    """Exact quadratic: y = W2 @ x² + W1 @ x + b"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W2 = nn.Linear(in_features, out_features, bias=False)
        self.W1 = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return self.W2(x ** 2) + self.W1(x) + self.bias


class BandEnergyMapper(nn.Module):
    """
    Predict band energies, then distribute sines within bands.

    This matches what z actually encodes:
    - z[48-51] controls low band energy
    - z[52-55] controls mid band energy
    - etc.
    """

    def __init__(self, n_sines=64, sines_per_band=None):
        super().__init__()
        self.n_sines = n_sines

        # Distribute sines across bands
        if sines_per_band is None:
            sines_per_band = {
                'low': 12,       # Bass fundamentals
                'mid': 20,       # Most musical content
                'high': 20,      # Harmonics, brightness
                'very_high': 12, # Air, presence
            }
        self.sines_per_band = sines_per_band
        self.band_names = list(BAND_DIMS.keys())

        # Per-band quadratic energy predictor (matches discovery)
        self.band_energy_predictors = nn.ModuleDict()
        for band_name, (dims, freq_range) in BAND_DIMS.items():
            self.band_energy_predictors[band_name] = QuadraticLayer(len(dims), 1)

        # Per-band frequency distribution (where within band)
        self.band_freq_predictors = nn.ModuleDict()
        for band_name, (dims, freq_range) in BAND_DIMS.items():
            n_sines_band = sines_per_band[band_name]
            self.band_freq_predictors[band_name] = nn.Sequential(
                QuadraticLayer(len(dims), 32),
                nn.GELU(),
                nn.Linear(32, n_sines_band),
            )

        # Per-band amplitude distribution (relative amps within band)
        self.band_amp_predictors = nn.ModuleDict()
        for band_name in self.band_names:
            n_sines_band = sines_per_band[band_name]
            dims = BAND_DIMS[band_name][0]
            self.band_amp_predictors[band_name] = nn.Sequential(
                QuadraticLayer(len(dims), 32),
                nn.GELU(),
                nn.Linear(32, n_sines_band),
            )

        # Temporal envelope (attention-based, discovered)
        self.temporal_proj = nn.Linear(len(TEMPORAL_DIMS), 64)
        self.temporal_attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.temporal_norm = nn.LayerNorm(64)
        self.envelope_head = nn.Linear(64, n_sines)

        # Global energy (cumulative effect of dims 0-47)
        self.energy_predictor = nn.Sequential(
            nn.Linear(len(ENERGY_DIMS), 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        """
        z: [B, 8, 16, T]
        Returns: freqs [B, T, n_sines], amps [B, T, n_sines]
        """
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        all_freqs = []
        all_amps = []

        # Process each band
        for band_name in self.band_names:
            dims, (freq_min, freq_max) = BAND_DIMS[band_name]
            n_sines_band = self.sines_per_band[band_name]

            # Extract band dims
            z_band = z_flat[..., dims]  # [B, T, 4]

            # Predict band energy (quadratic, matches discovery)
            band_energy = torch.sigmoid(self.band_energy_predictors[band_name](z_band))  # [B, T, 1]

            # Predict frequency positions within band
            freq_logits = self.band_freq_predictors[band_name](z_band)  # [B, T, n_sines_band]
            freq_positions = torch.sigmoid(freq_logits)  # [0, 1] within band

            # Map to actual frequencies
            log_freq_min = np.log(freq_min)
            log_freq_max = np.log(freq_max)
            log_freqs = log_freq_min + freq_positions * (log_freq_max - log_freq_min)
            freqs_band = torch.exp(log_freqs)  # [B, T, n_sines_band]

            # Predict relative amplitudes within band
            amp_logits = self.band_amp_predictors[band_name](z_band)
            relative_amps = F.softmax(amp_logits, dim=-1)  # Sums to 1 within band

            # Scale by band energy
            amps_band = relative_amps * band_energy  # [B, T, n_sines_band]

            all_freqs.append(freqs_band)
            all_amps.append(amps_band)

        # Concatenate all bands
        freqs = torch.cat(all_freqs, dim=-1)  # [B, T, n_sines]
        amps = torch.cat(all_amps, dim=-1)    # [B, T, n_sines]

        # Apply temporal envelope
        z_temp = z_flat[..., TEMPORAL_DIMS]
        h_temp = self.temporal_proj(z_temp)
        h_attn, _ = self.temporal_attn(h_temp, h_temp, h_temp)
        h_temp = self.temporal_norm(h_temp + h_attn)
        envelope = torch.sigmoid(self.envelope_head(h_temp))  # [B, T, n_sines]

        # Apply global energy
        z_energy = z_flat[..., ENERGY_DIMS]
        global_energy = torch.sigmoid(self.energy_predictor(z_energy))  # [B, T, 1]

        # Final amplitude
        amps = amps * envelope * global_energy

        return freqs, amps


class BandEnergyMapperV2(nn.Module):
    """
    V2: Adds cross-band interaction and deeper temporal processing.
    """

    def __init__(self, n_sines=64, hidden_dim=64, n_temporal_layers=2):
        super().__init__()
        self.n_sines = n_sines
        self.hidden_dim = hidden_dim

        sines_per_band = {'low': 12, 'mid': 20, 'high': 20, 'very_high': 12}
        self.sines_per_band = sines_per_band
        self.band_names = list(BAND_DIMS.keys())

        # Per-band encoders (quadratic)
        self.band_encoders = nn.ModuleDict()
        for band_name, (dims, _) in BAND_DIMS.items():
            self.band_encoders[band_name] = QuadraticLayer(len(dims), hidden_dim)

        # Cross-band attention (bands can influence each other)
        self.cross_band_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.cross_band_norm = nn.LayerNorm(hidden_dim)

        # Per-band output heads
        self.freq_heads = nn.ModuleDict()
        self.amp_heads = nn.ModuleDict()
        for band_name in self.band_names:
            n_sines_band = sines_per_band[band_name]
            self.freq_heads[band_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_sines_band),
            )
            self.amp_heads[band_name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_sines_band),
            )

        # Temporal processing (multi-layer attention)
        self.temporal_proj = nn.Linear(len(TEMPORAL_DIMS), hidden_dim)
        self.temporal_layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True),
                'norm1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
            })
            for _ in range(n_temporal_layers)
        ])
        self.envelope_head = nn.Linear(hidden_dim, n_sines)

        # Energy
        self.energy_head = nn.Sequential(
            nn.Linear(len(ENERGY_DIMS), 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        # Encode each band
        band_features = []
        for band_name, (dims, _) in BAND_DIMS.items():
            z_band = z_flat[..., dims]
            h_band = F.gelu(self.band_encoders[band_name](z_band))  # [B, T, hidden]
            band_features.append(h_band)

        # Stack for cross-band attention: [B, T, 4, hidden]
        band_stack = torch.stack(band_features, dim=2)
        BT_shape = B * T
        band_flat = band_stack.reshape(BT_shape, 4, self.hidden_dim)

        # Cross-band attention
        band_attn, _ = self.cross_band_attn(band_flat, band_flat, band_flat)
        band_flat = self.cross_band_norm(band_flat + band_attn)
        band_stack = band_flat.reshape(B, T, 4, self.hidden_dim)

        # Generate frequencies and amplitudes per band
        all_freqs = []
        all_amps = []

        for i, band_name in enumerate(self.band_names):
            _, (freq_min, freq_max) = BAND_DIMS[band_name]
            h_band = band_stack[:, :, i, :]  # [B, T, hidden]

            # Frequencies
            freq_logits = self.freq_heads[band_name](h_band)
            freq_pos = torch.sigmoid(freq_logits)
            log_freq_min = np.log(freq_min)
            log_freq_max = np.log(freq_max)
            log_freqs = log_freq_min + freq_pos * (log_freq_max - log_freq_min)
            freqs_band = torch.exp(log_freqs)

            # Amplitudes
            amp_logits = self.amp_heads[band_name](h_band)
            amps_band = torch.sigmoid(amp_logits)

            all_freqs.append(freqs_band)
            all_amps.append(amps_band)

        freqs = torch.cat(all_freqs, dim=-1)
        amps = torch.cat(all_amps, dim=-1)

        # Temporal envelope
        z_temp = z_flat[..., TEMPORAL_DIMS]
        h_temp = self.temporal_proj(z_temp)
        for layer in self.temporal_layers:
            h_attn, _ = layer['attn'](h_temp, h_temp, h_temp)
            h_temp = layer['norm1'](h_temp + h_attn)
            h_ff = layer['ff'](h_temp)
            h_temp = layer['norm2'](h_temp + h_ff)

        envelope = torch.sigmoid(self.envelope_head(h_temp))

        # Global energy
        z_energy = z_flat[..., ENERGY_DIMS]
        global_energy = torch.sigmoid(self.energy_head(z_energy))

        amps = amps * envelope * global_energy

        return freqs, amps


class SimpleBandMapper(nn.Module):
    """
    Simplest version: just quadratic per band, no cross-band.
    Tests if pure band structure is sufficient.
    """

    def __init__(self, n_sines=64):
        super().__init__()
        sines_per_band = {'low': 12, 'mid': 20, 'high': 20, 'very_high': 12}
        self.sines_per_band = sines_per_band
        self.band_names = list(BAND_DIMS.keys())

        self.freq_predictors = nn.ModuleDict()
        self.amp_predictors = nn.ModuleDict()

        for band_name, (dims, _) in BAND_DIMS.items():
            n_sines_band = sines_per_band[band_name]
            self.freq_predictors[band_name] = QuadraticLayer(len(dims), n_sines_band)
            self.amp_predictors[band_name] = QuadraticLayer(len(dims), n_sines_band)

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)

        all_freqs = []
        all_amps = []

        for band_name in self.band_names:
            dims, (freq_min, freq_max) = BAND_DIMS[band_name]
            z_band = z_flat[..., dims]

            freq_logits = self.freq_predictors[band_name](z_band)
            freq_pos = torch.sigmoid(freq_logits)
            log_freq_min = np.log(freq_min)
            log_freq_max = np.log(freq_max)
            log_freqs = log_freq_min + freq_pos * (log_freq_max - log_freq_min)
            freqs_band = torch.exp(log_freqs)

            amps_band = torch.sigmoid(self.amp_predictors[band_name](z_band))

            all_freqs.append(freqs_band)
            all_amps.append(amps_band)

        return torch.cat(all_freqs, dim=-1), torch.cat(all_amps, dim=-1)


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

    # Test 1: Simple band mapper (pure quadratic per band)
    print("\n" + "=" * 70)
    print("TEST 1: Simple Band Mapper")
    print("Pure quadratic: z[48-51]→low, z[52-55]→mid, etc.")
    print("=" * 70)

    model1 = SimpleBandMapper(n_sines=args.n_sines)
    n_params = sum(p.numel() for p in model1.parameters())
    print(f"Params: {n_params:,}")

    results['simple_band'] = train_and_eval(model1, dataloader, n_epochs=args.epochs,
                                             lr=args.lr, device=device)

    # Test 2: Band Energy Mapper (full structure)
    print("\n" + "=" * 70)
    print("TEST 2: Band Energy Mapper")
    print("Band energy + temporal envelope + global energy")
    print("=" * 70)

    model2 = BandEnergyMapper(n_sines=args.n_sines)
    n_params = sum(p.numel() for p in model2.parameters())
    print(f"Params: {n_params:,}")

    results['band_energy'] = train_and_eval(model2, dataloader, n_epochs=args.epochs,
                                             lr=args.lr, device=device)

    # Test 3: V2 with cross-band attention
    print("\n" + "=" * 70)
    print("TEST 3: Band Energy Mapper V2")
    print("Cross-band attention + multi-layer temporal")
    print("=" * 70)

    model3 = BandEnergyMapperV2(n_sines=args.n_sines, hidden_dim=64, n_temporal_layers=2)
    n_params = sum(p.numel() for p in model3.parameters())
    print(f"Params: {n_params:,}")

    results['band_energy_v2'] = train_and_eval(model3, dataloader, n_epochs=args.epochs,
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
    print("COMPARISON")
    print("=" * 70)
    print("""
  Previous best results:
    Harmonic MLP:       0.046 (2.8 semitones)
    Disentangled MLP:   0.054 (3.2 semitones)

  Theoretical limit (if z only encodes bands):
    ~1-2 semitones within each band (band width)

  If band mapper approaches 0.046, the structure is correct.
  If worse, individual sine info is missing from z.
    """)

    best = min(results.values())
    if best < 0.046:
        print(f"✓ BETTER: Band mapper ({best:.4f}) beats harmonic MLP!")
    elif best < 0.06:
        print(f"~ COMPARABLE: Band mapper ({best:.4f}) is close to baselines")
    else:
        print(f"✗ WORSE: Band mapper ({best:.4f}) - structure incomplete")


if __name__ == "__main__":
    main()
