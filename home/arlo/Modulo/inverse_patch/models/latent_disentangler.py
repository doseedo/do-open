"""
Latent Disentangler: Post-hoc direction finding for DCAE latent space.

Learns a nonlinear transform from mixing-linear space (where interpolation = mixing)
to parameter-linear space (where axis movement = parameter change).

No DCAE retraining required - just learns the transform on top.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from scipy import signal


class LatentDisentangler(nn.Module):
    """
    Transform from mixing-linear space to parameter-linear space.

    In the disentangled space:
    - Moving along axis 0 = filter cutoff change
    - Moving along axis 1 = resonance change
    - Moving along axis 2 = attack change
    - etc.

    The transform should be (approximately) invertible.
    """

    # Parameter axis assignments
    PARAM_AXES = {
        'filter_cutoff': 0,
        'filter_resonance': 1,
        'amp_attack': 2,
        'amp_decay': 3,
        'amp_sustain': 4,
        'amp_release': 5,
        'osc_detune': 6,
        'osc_mix': 7,
    }

    def __init__(
        self,
        latent_dim: int = 128,  # Flattened latent dimension (8 * 16 = 128 for DCAE)
        hidden_dim: int = 256,
        n_layers: int = 3,
        use_residual: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        # Forward transform: DCAE space -> disentangled space
        self.to_disentangled = self._build_mlp(latent_dim, hidden_dim, latent_dim, n_layers)

        # Inverse transform: disentangled space -> DCAE space
        self.from_disentangled = self._build_mlp(latent_dim, hidden_dim, latent_dim, n_layers)

        # Learnable scaling per axis (to normalize parameter ranges)
        self.axis_scales = nn.Parameter(torch.ones(latent_dim))

    def _build_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Module:
        """Build an MLP with residual connections."""
        layers = []

        # Input projection
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(ResidualBlock(hidden_dim))

        # Output projection
        layers.append(nn.Linear(hidden_dim, out_dim))

        return nn.Sequential(*layers)

    def disentangle(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform from DCAE latent space to disentangled space.

        Args:
            z: [B, latent_dim] or [B, latent_dim, T] DCAE latent

        Returns:
            z_prime: [B, latent_dim] or [B, latent_dim, T] disentangled latent
        """
        # Handle temporal dimension
        if z.dim() == 3:
            B, D, T = z.shape
            z_flat = z.permute(0, 2, 1).reshape(B * T, D)  # [B*T, D]
            z_prime_flat = self.to_disentangled(z_flat)
            z_prime = z_prime_flat.reshape(B, T, D).permute(0, 2, 1)  # [B, D, T]
        else:
            z_prime = self.to_disentangled(z)

        # Apply learned scaling
        z_prime = z_prime * self.axis_scales.view(1, -1, *([1] * (z_prime.dim() - 2)))

        return z_prime

    def entangle(self, z_prime: torch.Tensor) -> torch.Tensor:
        """
        Transform from disentangled space back to DCAE latent space.

        Args:
            z_prime: [B, latent_dim] or [B, latent_dim, T] disentangled latent

        Returns:
            z: [B, latent_dim] or [B, latent_dim, T] DCAE latent
        """
        # Remove scaling
        z_prime = z_prime / (self.axis_scales.view(1, -1, *([1] * (z_prime.dim() - 2))) + 1e-8)

        # Handle temporal dimension
        if z_prime.dim() == 3:
            B, D, T = z_prime.shape
            z_prime_flat = z_prime.permute(0, 2, 1).reshape(B * T, D)
            z_flat = self.from_disentangled(z_prime_flat)
            z = z_flat.reshape(B, T, D).permute(0, 2, 1)
        else:
            z = self.from_disentangled(z_prime)

        return z

    def apply_param_change(
        self,
        z: torch.Tensor,
        param_name: str,
        delta: float,
    ) -> torch.Tensor:
        """
        Apply a parameter change by moving along the corresponding axis.

        Args:
            z: DCAE latent
            param_name: Name of parameter (e.g., 'filter_cutoff')
            delta: Amount to change (in normalized units)

        Returns:
            z_modified: Modified DCAE latent
        """
        axis = self.PARAM_AXES.get(param_name)
        if axis is None:
            raise ValueError(f"Unknown parameter: {param_name}. Valid: {list(self.PARAM_AXES.keys())}")

        z_prime = self.disentangle(z)

        # Move along the axis
        if z_prime.dim() == 3:
            z_prime[:, axis, :] += delta
        else:
            z_prime[:, axis] += delta

        return self.entangle(z_prime)

    def get_param_direction(self, param_name: str) -> torch.Tensor:
        """Get the direction vector for a parameter in disentangled space."""
        axis = self.PARAM_AXES[param_name]
        direction = torch.zeros(self.latent_dim)
        direction[axis] = 1.0
        return direction

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: disentangle and re-entangle.

        Returns both the disentangled representation and reconstruction.
        """
        z_prime = self.disentangle(z)
        z_recon = self.entangle(z_prime)
        return z_prime, z_recon


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DisentanglementLoss(nn.Module):
    """
    Loss function for training the disentangler.

    Components:
    1. Reconstruction: z ≈ entangle(disentangle(z))
    2. Cycle consistency: z' ≈ disentangle(entangle(z'))
    3. Direction alignment: parameter changes should align with axes
    4. Orthogonality: different parameters should be orthogonal
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        cycle_weight: float = 1.0,
        direction_weight: float = 10.0,
        orthogonality_weight: float = 1.0,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.cycle_weight = cycle_weight
        self.direction_weight = direction_weight
        self.orthogonality_weight = orthogonality_weight

    def forward(
        self,
        disentangler: LatentDisentangler,
        z1: torch.Tensor,
        z2: torch.Tensor,
        param_name: str,
        param_delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a pair of latents that differ by a known parameter change.

        Args:
            disentangler: The disentangler module
            z1: First DCAE latent [B, D] or [B, D, T]
            z2: Second DCAE latent (same sound with different param value)
            param_name: Name of parameter that changed
            param_delta: Amount of change [B] (normalized)

        Returns:
            Dict of loss components
        """
        losses = {}

        # Ensure param_delta is on the same device
        param_delta = param_delta.to(z1.device)

        # 1. Reconstruction loss
        z1_prime, z1_recon = disentangler(z1)
        z2_prime, z2_recon = disentangler(z2)

        losses['recon'] = self.recon_weight * (
            F.mse_loss(z1_recon, z1) + F.mse_loss(z2_recon, z2)
        ) / 2

        # 2. Cycle consistency (in disentangled space)
        z1_prime_recon = disentangler.disentangle(z1_recon)
        losses['cycle'] = self.cycle_weight * F.mse_loss(z1_prime_recon, z1_prime)

        # 3. Direction alignment loss
        # The difference z2' - z1' should be aligned with the parameter axis
        axis = disentangler.PARAM_AXES[param_name]

        # Flatten temporal dimension if present
        if z1_prime.dim() == 3:
            z1_prime_flat = z1_prime.mean(dim=-1)  # [B, D]
            z2_prime_flat = z2_prime.mean(dim=-1)
        else:
            z1_prime_flat = z1_prime
            z2_prime_flat = z2_prime

        delta_z_prime = z2_prime_flat - z1_prime_flat  # [B, D]

        # Expected: delta should be along the axis, proportional to param_delta
        expected_delta = torch.zeros_like(delta_z_prime)
        expected_delta[:, axis] = param_delta

        # Loss: actual delta should match expected delta direction
        # But we don't know the scale, so use cosine similarity for direction
        # and separate loss for magnitude on the correct axis

        # Direction loss: other axes should be ~0
        other_axes_mask = torch.ones(delta_z_prime.shape[-1], device=delta_z_prime.device, dtype=torch.bool)
        other_axes_mask[axis] = False

        losses['direction'] = self.direction_weight * (
            # Other axes should be zero
            delta_z_prime[:, other_axes_mask].pow(2).mean() +
            # Target axis should correlate with param_delta (sign matters)
            F.mse_loss(
                delta_z_prime[:, axis].sign(),
                param_delta.sign()
            )
        )

        # 4. Orthogonality loss (optional regularization)
        # Encourage the transform to preserve orthogonality
        if self.orthogonality_weight > 0:
            # Sample random directions and check they stay orthogonal
            # This is expensive, so we do it probabilistically
            if torch.rand(1).item() < 0.1:  # 10% of batches
                losses['orthogonality'] = self.orthogonality_weight * self._orthogonality_loss(
                    disentangler, z1.device
                )
            else:
                losses['orthogonality'] = torch.tensor(0.0, device=z1.device)

        # Total loss
        losses['total'] = sum(losses.values())

        return losses

    def _orthogonality_loss(self, disentangler: LatentDisentangler, device) -> torch.Tensor:
        """Regularization to encourage near-orthogonal transforms."""
        # Create orthogonal vectors in disentangled space
        n_samples = 16
        z_prime = torch.randn(n_samples, disentangler.latent_dim, device=device)
        z_prime = F.normalize(z_prime, dim=-1)

        # Transform to DCAE space and back
        z = disentangler.entangle(z_prime)
        z_prime_recon = disentangler.disentangle(z)

        # Check if dot products are preserved (orthogonality preserved)
        orig_dots = torch.mm(z_prime, z_prime.t())
        recon_dots = torch.mm(z_prime_recon, z_prime_recon.t())

        return F.mse_loss(recon_dots, orig_dots)


class SynthParameterDataset(torch.utils.data.Dataset):
    """
    Dataset that generates synth sounds with known parameters.

    Each sample is a pair of sounds that differ by exactly one parameter.
    """

    def __init__(
        self,
        codec,  # DCAE codec for encoding
        n_samples: int = 10000,
        sample_rate: int = 44100,
        duration: float = 2.0,
        device: str = 'cuda',
    ):
        self.codec = codec
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.device = device
        self.n_audio_samples = int(sample_rate * duration)

        # Parameters and their ranges
        self.param_ranges = {
            'filter_cutoff': (200, 8000),  # Hz
            'filter_resonance': (0.1, 10.0),  # Q
            'amp_attack': (0.001, 0.5),  # seconds
            'amp_decay': (0.01, 1.0),
            'amp_sustain': (0.0, 1.0),
            'amp_release': (0.01, 2.0),
            'osc_detune': (0.0, 50.0),  # cents
            'osc_mix': (0.0, 1.0),  # saw vs square
        }

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Generate a pair of sounds differing by one parameter.

        Returns:
            z1: DCAE latent of first sound
            z2: DCAE latent of second sound
            param_name: Which parameter changed
            param_delta: Normalized change amount
        """
        import numpy as np
        from scipy import signal

        # Pick a random parameter to vary
        param_name = np.random.choice(list(self.param_ranges.keys()))
        param_min, param_max = self.param_ranges[param_name]

        # Generate two random values for this parameter
        val1 = np.random.uniform(param_min, param_max)
        val2 = np.random.uniform(param_min, param_max)

        # Normalized delta
        param_delta = (val2 - val1) / (param_max - param_min)

        # Generate base parameters (random)
        base_params = {
            name: np.random.uniform(lo, hi)
            for name, (lo, hi) in self.param_ranges.items()
        }

        # Create two parameter sets
        params1 = base_params.copy()
        params1[param_name] = val1

        params2 = base_params.copy()
        params2[param_name] = val2

        # Synthesize audio
        audio1 = self._synthesize(params1)
        audio2 = self._synthesize(params2)

        # Encode to DCAE latent
        z1 = self._encode(audio1)
        z2 = self._encode(audio2)

        return {
            'z1': z1,
            'z2': z2,
            'param_name': param_name,
            'param_delta': torch.tensor(param_delta, dtype=torch.float32),
        }

    def _synthesize(self, params: Dict[str, float]) -> np.ndarray:
        """Synthesize audio from parameters."""
        t = np.linspace(0, self.duration, self.n_audio_samples, endpoint=False)
        freq = 220.0  # Base frequency

        # Generate oscillators
        saw = signal.sawtooth(2 * np.pi * freq * t)
        square = signal.square(2 * np.pi * freq * t)

        # Mix oscillators
        osc_mix = params['osc_mix']
        osc = (1 - osc_mix) * saw + osc_mix * square

        # Apply detune (add slightly detuned copy)
        detune_cents = params['osc_detune']
        detune_ratio = 2 ** (detune_cents / 1200)
        detuned = signal.sawtooth(2 * np.pi * freq * detune_ratio * t)
        osc = 0.5 * osc + 0.5 * detuned

        # Apply filter
        cutoff = params['filter_cutoff']
        resonance = params['filter_resonance']
        nyq = self.sample_rate / 2
        normalized_cutoff = min(cutoff / nyq, 0.99)

        # Resonant lowpass (using bandpass + lowpass trick)
        b, a = signal.butter(2, normalized_cutoff, btype='low')
        filtered = signal.filtfilt(b, a, osc)

        # Apply ADSR envelope
        attack = int(params['amp_attack'] * self.sample_rate)
        decay = int(params['amp_decay'] * self.sample_rate)
        sustain = params['amp_sustain']
        release = int(params['amp_release'] * self.sample_rate)

        envelope = np.ones(self.n_audio_samples)

        # Attack
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)

        # Decay
        decay_end = attack + decay
        if decay > 0 and decay_end < self.n_audio_samples:
            envelope[attack:decay_end] = np.linspace(1, sustain, decay)

        # Sustain (already 1s or sustain level)
        sustain_end = self.n_audio_samples - release
        if sustain_end > decay_end:
            envelope[decay_end:sustain_end] = sustain

        # Release
        if release > 0:
            envelope[sustain_end:] = np.linspace(sustain, 0, self.n_audio_samples - sustain_end)

        audio = filtered * envelope

        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-8) * 0.8

        return audio.astype(np.float32)

    def _encode(self, audio: np.ndarray) -> torch.Tensor:
        """Encode audio to DCAE latent."""
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        audio_stereo = audio_tensor.unsqueeze(0).unsqueeze(0).expand(-1, 2, -1)  # [1, 2, T]

        T = audio_stereo.shape[-1]
        audio_lengths = torch.tensor([T], device=self.device)

        with torch.no_grad():
            latent, _ = self.codec.encode(audio_stereo, audio_lengths=audio_lengths, sr=self.sample_rate)

        # Flatten spatial dims: [1, C, H, T'] -> [C*H, T']
        B, C, H, T_latent = latent.shape
        latent_flat = latent.reshape(B, C * H, T_latent).squeeze(0)  # [C*H, T']

        return latent_flat


def train_disentangler(
    codec,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_path: str = None,
):
    """
    Train the latent disentangler.

    Args:
        codec: DCAE codec (for generating training data)
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        save_path: Where to save the trained model
    """
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = SynthParameterDataset(
        codec=codec,
        n_samples=10000,
        device=device,
    )

    # Note: Custom collate since we generate on-the-fly
    # For efficiency, we'll generate batches directly

    # Get latent dim from a sample
    sample = dataset[0]
    latent_dim = sample['z1'].shape[0]  # C*H
    print(f"Latent dimension: {latent_dim}")

    # Create model
    disentangler = LatentDisentangler(
        latent_dim=latent_dim,
        hidden_dim=256,
        n_layers=4,
    ).to(device)

    # Loss and optimizer
    loss_fn = DisentanglementLoss(
        recon_weight=1.0,
        cycle_weight=0.5,
        direction_weight=10.0,
        orthogonality_weight=0.1,
    )

    optimizer = torch.optim.AdamW(disentangler.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Training loop
    print(f"Training disentangler for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        disentangler.train()
        epoch_losses = {k: 0.0 for k in ['total', 'recon', 'cycle', 'direction', 'orthogonality']}
        n_batches = len(dataset) // batch_size

        for batch_idx in range(n_batches):
            # Generate batch
            batch_z1 = []
            batch_z2 = []
            batch_param_names = []
            batch_param_deltas = []

            for _ in range(batch_size):
                idx = torch.randint(0, len(dataset), (1,)).item()
                sample = dataset[idx]
                batch_z1.append(sample['z1'])
                batch_z2.append(sample['z2'])
                batch_param_names.append(sample['param_name'])
                batch_param_deltas.append(sample['param_delta'])

            # Stack tensors
            z1 = torch.stack(batch_z1, dim=0).to(device)  # [B, D, T]
            z2 = torch.stack(batch_z2, dim=0).to(device)
            param_deltas = torch.stack(batch_param_deltas, dim=0).to(device)

            # Use first param name for this batch (simplification)
            # In practice, you'd want to handle mixed param batches
            param_name = batch_param_names[0]

            # Forward pass
            optimizer.zero_grad()
            losses = loss_fn(disentangler, z1, z2, param_name, param_deltas)

            # Backward pass
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(disentangler.parameters(), 1.0)
            optimizer.step()

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] += v.item()

        scheduler.step()

        # Print epoch summary
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Total: {avg_losses['total']:.4f} | "
                  f"Recon: {avg_losses['recon']:.4f} | "
                  f"Dir: {avg_losses['direction']:.4f}")

    # Save model
    if save_path:
        torch.save({
            'model_state_dict': disentangler.state_dict(),
            'latent_dim': latent_dim,
        }, save_path)
        print(f"Saved model to {save_path}")

    return disentangler


if __name__ == "__main__":
    # Quick test
    import sys
    DCAE_PATH = "/home/arlo/Data/ACE-Step"
    if DCAE_PATH not in sys.path:
        sys.path.insert(0, DCAE_PATH)

    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

    DEFAULT_DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    DEFAULT_VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

    print("Loading DCAE...")
    codec = MusicDCAE(
        source_sample_rate=44100,
        dcae_checkpoint_path=DEFAULT_DCAE_PATH,
        vocoder_checkpoint_path=DEFAULT_VOCODER_PATH,
    )
    codec = codec.to('cuda')
    codec.eval()

    # Train
    disentangler = train_disentangler(
        codec=codec,
        n_epochs=50,
        batch_size=16,
        lr=1e-4,
        device='cuda',
        save_path='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/disentangler.pt',
    )
