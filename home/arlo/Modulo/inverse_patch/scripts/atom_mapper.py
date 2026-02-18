#!/usr/bin/env python3
"""
Atom-based mapper using DISCOVERED structure.

Findings from discover_atoms.py:
1. Dims 48-63 are most influential (frequency-band organized)
2. Responses are NONLINEAR (quadratic fits 100x better)
3. Dims are relatively independent
4. Structure is frequency-band based, not harmonic

Discovered atoms:
  - dims 48-51: LOW frequencies
  - dims 52-54: MID frequencies
  - dims 55-59: MID-HIGH frequencies
  - dims 60-62: HIGH frequencies
  - dim 63: BRIGHTNESS control

Approach:
1. Extract discovered atoms from z
2. Apply nonlinear (quadratic) transforms per atom
3. Each atom controls sines in its frequency band
4. Combine bands for full spectrum
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
# DISCOVERED ATOM STRUCTURE
# ============================================================================

DISCOVERED_ATOMS = {
    'low': {
        'dims': [48, 49, 50, 51],
        'freq_range': (20, 200),      # Hz
        'effect': 'low_energy',
    },
    'mid': {
        'dims': [52, 53, 54],
        'freq_range': (200, 800),
        'effect': 'mid_energy',
    },
    'mid_high': {
        'dims': [55, 56, 57, 58, 59],
        'freq_range': (800, 3000),
        'effect': 'mid_energy',
    },
    'high': {
        'dims': [60, 61, 62],
        'freq_range': (3000, 8000),
        'effect': 'high_energy',
    },
    'brightness': {
        'dims': [63],
        'freq_range': (20, 8000),  # Global modifier
        'effect': 'peak_bin',
    },
}

# Additional dims that showed some influence (for refinement)
SECONDARY_DIMS = list(range(0, 48)) + list(range(64, 128))


class QuadraticTransform(nn.Module):
    """
    Learnable quadratic transform: y = a*x^2 + b*x + c

    Discovered: quadratic fits 100x better than linear.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.quadratic = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return self.quadratic(x ** 2) + self.linear(x) + self.bias


class AtomMapper(nn.Module):
    """
    Map z to sines using discovered atom structure.

    Each atom (frequency band) has:
    - Its specific z dimensions
    - Quadratic nonlinear transform
    - Output sines in its frequency range
    """

    def __init__(self, n_sines=64, sines_per_band=None, use_secondary=False):
        super().__init__()
        self.n_sines = n_sines
        self.use_secondary = use_secondary

        # Distribute sines across bands
        if sines_per_band is None:
            # More sines for wider/more important bands
            sines_per_band = {
                'low': 8,
                'mid': 12,
                'mid_high': 20,
                'high': 16,
                'brightness': 8,  # These modify other bands
            }
        self.sines_per_band = sines_per_band

        # Build per-atom transforms
        self.atom_transforms = nn.ModuleDict()
        self.freq_heads = nn.ModuleDict()
        self.amp_heads = nn.ModuleDict()

        for atom_name, atom_info in DISCOVERED_ATOMS.items():
            n_dims = len(atom_info['dims'])
            n_sines_atom = sines_per_band[atom_name]

            # Quadratic transform (discovered nonlinearity)
            self.atom_transforms[atom_name] = QuadraticTransform(n_dims, 32)

            # Frequency and amplitude heads
            self.freq_heads[atom_name] = nn.Sequential(
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Linear(32, n_sines_atom),
            )
            self.amp_heads[atom_name] = nn.Sequential(
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Linear(32, n_sines_atom),
            )

        # Optional: use secondary dims for refinement
        if use_secondary:
            self.secondary_transform = nn.Sequential(
                nn.Linear(len(SECONDARY_DIMS), 64),
                nn.GELU(),
                nn.Linear(64, 32),
            )
            self.secondary_freq_mod = nn.Linear(32, n_sines)
            self.secondary_amp_mod = nn.Linear(32, n_sines)

        # Brightness modulation (global)
        self.brightness_mod = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, n_sines),
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

        brightness_features = None

        for atom_name, atom_info in DISCOVERED_ATOMS.items():
            dims = atom_info['dims']
            freq_min, freq_max = atom_info['freq_range']
            n_sines_atom = self.sines_per_band[atom_name]

            # Extract atom dimensions
            z_atom = z_flat[..., dims]  # [B, T, n_dims]

            # Quadratic transform
            h = self.atom_transforms[atom_name](z_atom)  # [B, T, 32]

            if atom_name == 'brightness':
                # Store for global modulation
                brightness_features = h
                # Brightness atom also produces its own sines

            # Frequency prediction (within band range)
            freq_logits = self.freq_heads[atom_name](h)  # [B, T, n_sines_atom]
            freq_norm = torch.sigmoid(freq_logits)

            # Map to frequency range
            log_freq_min = np.log(freq_min)
            log_freq_max = np.log(freq_max)
            log_freq = log_freq_min + freq_norm * (log_freq_max - log_freq_min)
            freqs = torch.exp(log_freq)  # [B, T, n_sines_atom]

            # Amplitude prediction
            amp_logits = self.amp_heads[atom_name](h)
            amps = torch.sigmoid(amp_logits)  # [B, T, n_sines_atom]

            all_freqs.append(freqs)
            all_amps.append(amps)

        # Concatenate all bands
        freqs = torch.cat(all_freqs, dim=-1)  # [B, T, n_sines]
        amps = torch.cat(all_amps, dim=-1)    # [B, T, n_sines]

        # Apply brightness modulation (global spectral tilt)
        if brightness_features is not None:
            brightness_mod = torch.sigmoid(self.brightness_mod(brightness_features))
            amps = amps * brightness_mod

        # Optional: secondary dimension refinement
        if self.use_secondary:
            z_secondary = z_flat[..., SECONDARY_DIMS]
            h_sec = self.secondary_transform(z_secondary)

            freq_mod = torch.sigmoid(self.secondary_freq_mod(h_sec)) * 0.2 + 0.9  # [0.9, 1.1]
            amp_mod = torch.sigmoid(self.secondary_amp_mod(h_sec))

            freqs = freqs * freq_mod
            amps = amps * amp_mod

        return freqs, amps


class AtomMapperV2(nn.Module):
    """
    Version 2: Atoms control harmonic structure within bands.

    Instead of predicting raw frequencies, predict:
    - f0 per band (fundamental for that band)
    - Harmonic weights within band
    """

    def __init__(self, n_sines=64, harmonics_per_band=8):
        super().__init__()
        self.n_sines = n_sines
        self.harmonics_per_band = harmonics_per_band

        n_bands = len(DISCOVERED_ATOMS)
        self.n_bands = n_bands

        # Per-band transforms
        self.band_transforms = nn.ModuleDict()
        self.f0_heads = nn.ModuleDict()
        self.harmonic_heads = nn.ModuleDict()

        for atom_name, atom_info in DISCOVERED_ATOMS.items():
            n_dims = len(atom_info['dims'])

            self.band_transforms[atom_name] = QuadraticTransform(n_dims, 32)
            self.f0_heads[atom_name] = nn.Linear(32, 1)  # One f0 per band
            self.harmonic_heads[atom_name] = nn.Sequential(
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Linear(32, harmonics_per_band),  # Amplitude per harmonic
            )

        # Cross-band interaction (discovered: some interactions exist)
        self.cross_band = nn.MultiheadAttention(32, num_heads=4, batch_first=True)

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        band_features = []
        band_names = list(DISCOVERED_ATOMS.keys())

        # Extract features per band
        for atom_name, atom_info in DISCOVERED_ATOMS.items():
            dims = atom_info['dims']
            z_atom = z_flat[..., dims]
            h = self.band_transforms[atom_name](z_atom)  # [B, T, 32]
            band_features.append(h)

        # Stack for cross-band attention
        band_stack = torch.stack(band_features, dim=2)  # [B, T, n_bands, 32]
        BT = B * T
        band_stack_flat = band_stack.reshape(BT, self.n_bands, 32)

        # Cross-band interaction
        band_attended, _ = self.cross_band(band_stack_flat, band_stack_flat, band_stack_flat)
        band_attended = band_attended.reshape(B, T, self.n_bands, 32)

        all_freqs = []
        all_amps = []

        for i, (atom_name, atom_info) in enumerate(DISCOVERED_ATOMS.items()):
            freq_min, freq_max = atom_info['freq_range']

            h = band_attended[:, :, i, :]  # [B, T, 32]

            # Predict f0 for this band
            f0_logit = self.f0_heads[atom_name](h).squeeze(-1)  # [B, T]
            f0_norm = torch.sigmoid(f0_logit)
            log_f0_min = np.log(freq_min)
            log_f0_max = np.log(freq_max / self.harmonics_per_band)  # Room for harmonics
            log_f0 = log_f0_min + f0_norm * (log_f0_max - log_f0_min)
            f0 = torch.exp(log_f0)  # [B, T]

            # Harmonic frequencies
            harmonics = torch.arange(1, self.harmonics_per_band + 1, device=z.device).float()
            freqs_band = f0.unsqueeze(-1) * harmonics  # [B, T, n_harmonics]

            # Clip to band range
            freqs_band = freqs_band.clamp(freq_min, freq_max)

            # Harmonic amplitudes
            amp_logits = self.harmonic_heads[atom_name](h)  # [B, T, n_harmonics]
            amps_band = torch.sigmoid(amp_logits)

            all_freqs.append(freqs_band)
            all_amps.append(amps_band)

        freqs = torch.cat(all_freqs, dim=-1)  # [B, T, n_bands * harmonics_per_band]
        amps = torch.cat(all_amps, dim=-1)

        return freqs, amps


class AtomMapperV3(nn.Module):
    """
    Version 3: Focus only on discovered atoms (48-63), ignore rest.

    Simplest model that uses the discovery.
    """

    def __init__(self, n_sines=64):
        super().__init__()
        self.n_sines = n_sines

        # Only use dims 48-63 (16 dims)
        self.atom_dims = list(range(48, 64))
        n_atom_dims = len(self.atom_dims)

        # Quadratic transform
        self.transform = QuadraticTransform(n_atom_dims, 64)

        # Simple heads
        self.freq_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, n_sines),
        )
        self.amp_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, n_sines),
        )

        self.log_freq_min = np.log(20.0)
        self.log_freq_max = np.log(8000.0)

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        # Extract only discovered atoms
        z_atoms = z_flat[..., self.atom_dims]  # [B, T, 16]

        # Quadratic transform
        h = self.transform(z_atoms)  # [B, T, 64]

        # Predict frequencies
        freq_logits = self.freq_head(h)
        freq_norm = torch.sigmoid(freq_logits)
        log_freq = self.log_freq_min + freq_norm * (self.log_freq_max - self.log_freq_min)
        freqs = torch.exp(log_freq)

        # Predict amplitudes
        amps = torch.sigmoid(self.amp_head(h))

        return freqs, amps


def train_and_eval(model, dataloader, n_epochs=100, lr=1e-3, device='cuda'):
    """Train model and return final loss."""
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

    # Load data
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

    # Test 1: V3 - Simplest (only dims 48-63, quadratic)
    print("\n" + "=" * 70)
    print("TEST 1: AtomMapperV3 (only dims 48-63, quadratic)")
    print("Uses ONLY the discovered influential dimensions")
    print("=" * 70)

    model1 = AtomMapperV3(n_sines=args.n_sines)
    n_params = sum(p.numel() for p in model1.parameters())
    print(f"Params: {n_params:,}")

    results['v3_atoms_only'] = train_and_eval(model1, dataloader, n_epochs=args.epochs,
                                               lr=args.lr, device=device)

    # Test 2: V1 - Band-structured
    print("\n" + "=" * 70)
    print("TEST 2: AtomMapper V1 (band-structured)")
    print("Each discovered atom → frequency band")
    print("=" * 70)

    model2 = AtomMapper(n_sines=args.n_sines, use_secondary=False)
    n_params = sum(p.numel() for p in model2.parameters())
    print(f"Params: {n_params:,}")

    results['v1_bands'] = train_and_eval(model2, dataloader, n_epochs=args.epochs,
                                          lr=args.lr, device=device)

    # Test 3: V1 with secondary dims
    print("\n" + "=" * 70)
    print("TEST 3: AtomMapper V1 + secondary dims")
    print("Primary atoms + refinement from other dims")
    print("=" * 70)

    model3 = AtomMapper(n_sines=args.n_sines, use_secondary=True)
    n_params = sum(p.numel() for p in model3.parameters())
    print(f"Params: {n_params:,}")

    results['v1_with_secondary'] = train_and_eval(model3, dataloader, n_epochs=args.epochs,
                                                   lr=args.lr, device=device)

    # Test 4: V2 - Harmonic within bands
    print("\n" + "=" * 70)
    print("TEST 4: AtomMapper V2 (harmonic structure per band)")
    print("Each band has f0 + harmonics")
    print("=" * 70)

    model4 = AtomMapperV2(n_sines=args.n_sines, harmonics_per_band=8)
    n_params = sum(p.numel() for p in model4.parameters())
    print(f"Params: {n_params:,}")

    # Adjust n_sines to match output
    actual_sines = len(DISCOVERED_ATOMS) * 8
    print(f"Note: V2 outputs {actual_sines} sines")

    results['v2_harmonic_bands'] = train_and_eval(model4, dataloader, n_epochs=args.epochs,
                                                   lr=args.lr, device=device)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nFreq loss (lower is better):")
    for name, loss in sorted(results.items(), key=lambda x: x[1]):
        semitones = loss * 60  # Rough conversion
        print(f"  {name:25s}: {loss:.4f} (~{semitones:.1f} semitones)")

    print("\n" + "=" * 70)
    print("COMPARISON TO PREVIOUS APPROACHES")
    print("=" * 70)
    print("""
  Previous results:
    Linear sparse:      0.56  (33 semitones) - FAILED
    Disentangled MLP:   0.054 (3.2 semitones)
    Harmonic MLP:       0.046 (2.8 semitones)

  If atom-based approaches beat these, the discovery is useful.
  If not, we need to reconsider the approach.
    """)

    best = min(results.values())
    if best < 0.046:
        print(f"\n✓ DISCOVERY HELPS: Best atom-based ({best:.4f}) beats harmonic MLP (0.046)")
    elif best < 0.054:
        print(f"\n~ PARTIAL SUCCESS: Best atom-based ({best:.4f}) beats disentangled (0.054)")
    else:
        print(f"\n✗ NO IMPROVEMENT: Best atom-based ({best:.4f}) doesn't beat previous approaches")
        print("  The discovered structure may need different utilization")


if __name__ == "__main__":
    main()
