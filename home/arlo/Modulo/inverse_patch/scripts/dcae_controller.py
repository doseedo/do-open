#!/usr/bin/env python3
"""
DCAE Controller - Unified interface for generation, editing, and analysis.

Based on test findings:
- Forward (params → z): R² = 0.31 (nonlinear, use neural oscillator)
- Inverse (z → params): R² = 1.00 (linear, use simple extractor!)

Three capabilities:
1. GENERATE: oscillator(params) → z → audio
2. ANALYZE: audio → z → extractor → params
3. EDIT: audio → z → extractor → current → transform(z, current, target) → z' → audio'
"""

import os
import sys
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy import signal

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False

SAMPLE_RATE = 44100
DURATION = 2.0

DCAE_PATH = "/home/arlo/Data/ACE-Step"
if DCAE_PATH not in sys.path:
    sys.path.insert(0, DCAE_PATH)

DEFAULT_DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
DEFAULT_VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Parameter Extractor (Linear - R²=1.0!)
# ============================================================

class ParamExtractor(nn.Module):
    """
    Extract synth params from z using linear regression.
    Works because z → params has R²=1.0
    """
    def __init__(self, n_channels: int = 8, latent_dim: int = 16, n_frames: int = 22, n_params: int = 3):
        super().__init__()
        self.flat_size = n_channels * latent_dim * n_frames
        self.linear = nn.Linear(self.flat_size, n_params)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, n_channels, latent_dim, n_frames]
        Returns:
            params: [batch, n_params] normalized to [0, 1]
        """
        z_flat = z.view(z.shape[0], -1)
        params = self.linear(z_flat)
        return torch.sigmoid(params)  # Constrain to [0, 1]


# ============================================================
# Latent Oscillator (Nonlinear - for generation)
# ============================================================

class LatentOscillator(nn.Module):
    """Generate z from params using learned nonlinear mapping."""

    def __init__(self, n_params: int, n_channels: int, latent_dim: int, n_frames: int):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.total_size = n_channels * latent_dim * n_frames

        self.net = nn.Sequential(
            nn.Linear(n_params, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, self.total_size)
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        z_flat = self.net(params)
        return z_flat.view(-1, self.n_channels, self.latent_dim, self.n_frames)


# ============================================================
# Conditional Transform (Nonlinear - for editing)
# ============================================================

class ConditionalTransform(nn.Module):
    """Transform z based on current and target params."""

    def __init__(self, n_channels: int, latent_dim: int, n_frames: int, n_params: int = 3):
        super().__init__()
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.flat_size = n_channels * latent_dim

        # Z encoder
        self.z_encoder = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # Param encoder (current + target)
        self.param_encoder = nn.Sequential(
            nn.Linear(n_params * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Combined predictor
        self.combined = nn.Sequential(
            nn.Linear(512 + 256, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, self.flat_size * n_frames),
        )

    def forward(self, z: torch.Tensor, current_params: torch.Tensor,
                target_params: torch.Tensor) -> torch.Tensor:
        B, C, H, T = z.shape

        z_pooled = z.mean(dim=-1).view(B, -1)
        z_enc = self.z_encoder(z_pooled)

        params_combined = torch.cat([current_params, target_params], dim=-1)
        param_enc = self.param_encoder(params_combined)

        combined = torch.cat([z_enc, param_enc], dim=-1)
        z_new_flat = self.combined(combined)
        z_new = z_new_flat.view(B, C, H, T)

        # Residual blending
        alpha = 0.8
        return alpha * z_new + (1 - alpha) * z


# ============================================================
# DCAE Controller (Unified Interface)
# ============================================================

class DCAEController:
    """
    Unified interface for DCAE with generation, editing, and analysis.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

        # Load DCAE
        print("Loading DCAE...")
        self.dcae = MusicDCAE(
            source_sample_rate=SAMPLE_RATE,
            dcae_checkpoint_path=DEFAULT_DCAE_PATH,
            vocoder_checkpoint_path=DEFAULT_VOCODER_PATH,
        ).to(device)
        self.dcae.eval()

        # Latent shape (discovered from encoding)
        self.n_channels = 8
        self.latent_dim = 16
        self.n_frames = 22
        self.n_params = 3  # pitch, cutoff, attack

        # Components (will be trained)
        self.extractor = ParamExtractor(
            self.n_channels, self.latent_dim, self.n_frames, self.n_params
        ).to(device)

        self.oscillator = LatentOscillator(
            self.n_params, self.n_channels, self.latent_dim, self.n_frames
        ).to(device)

        self.transform = ConditionalTransform(
            self.n_channels, self.latent_dim, self.n_frames, self.n_params
        ).to(device)

        # Param normalization ranges
        self.param_ranges = {
            'pitch': (110, 440),
            'cutoff': (500, 4000),
            'attack': (0.01, 0.1),
        }

    def encode(self, audio: np.ndarray) -> torch.Tensor:
        """Encode audio to latent."""
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        audio_stereo = audio_tensor.expand(-1, 2, -1).to(self.device)
        audio_lengths = torch.tensor([audio_stereo.shape[-1]], device=self.device)
        with torch.no_grad():
            z, _ = self.dcae.encode(audio_stereo, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
        return z

    def decode(self, z: torch.Tensor) -> np.ndarray:
        """Decode latent to audio."""
        if z.dim() == 3:
            z = z.unsqueeze(0)
        audio_lengths = torch.tensor([int(DURATION * SAMPLE_RATE)], device=self.device)
        with torch.no_grad():
            sr, wavs = self.dcae.decode(z, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
        return wavs[0].mean(dim=0).cpu().numpy()

    def normalize_params(self, pitch: float, cutoff: float, attack: float) -> torch.Tensor:
        """Normalize params to [0, 1]."""
        pitch_norm = (pitch - self.param_ranges['pitch'][0]) / (self.param_ranges['pitch'][1] - self.param_ranges['pitch'][0])
        cutoff_norm = (cutoff - self.param_ranges['cutoff'][0]) / (self.param_ranges['cutoff'][1] - self.param_ranges['cutoff'][0])
        attack_norm = (attack - self.param_ranges['attack'][0]) / (self.param_ranges['attack'][1] - self.param_ranges['attack'][0])
        return torch.tensor([[pitch_norm, cutoff_norm, attack_norm]], dtype=torch.float32, device=self.device)

    def denormalize_params(self, params: torch.Tensor) -> Dict[str, float]:
        """Denormalize params from [0, 1]."""
        p = params.cpu().numpy().flatten()
        return {
            'pitch': p[0] * (self.param_ranges['pitch'][1] - self.param_ranges['pitch'][0]) + self.param_ranges['pitch'][0],
            'cutoff': p[1] * (self.param_ranges['cutoff'][1] - self.param_ranges['cutoff'][0]) + self.param_ranges['cutoff'][0],
            'attack': p[2] * (self.param_ranges['attack'][1] - self.param_ranges['attack'][0]) + self.param_ranges['attack'][0],
        }

    # ==================== MAIN CAPABILITIES ====================

    def generate(self, pitch: float, cutoff: float, attack: float) -> np.ndarray:
        """Generate audio from params."""
        params = self.normalize_params(pitch, cutoff, attack)
        self.oscillator.eval()
        with torch.no_grad():
            z = self.oscillator(params)
        return self.decode(z)

    def analyze(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract params from audio."""
        z = self.encode(audio)
        self.extractor.eval()
        with torch.no_grad():
            params = self.extractor(z)
        return self.denormalize_params(params)

    def edit(self, audio: np.ndarray,
             target_pitch: Optional[float] = None,
             target_cutoff: Optional[float] = None,
             target_attack: Optional[float] = None) -> np.ndarray:
        """Edit audio by changing params."""
        z = self.encode(audio)

        # Extract current params
        self.extractor.eval()
        with torch.no_grad():
            current_params = self.extractor(z)

        # Build target (keep current if not specified)
        current = self.denormalize_params(current_params)
        target = self.normalize_params(
            target_pitch if target_pitch is not None else current['pitch'],
            target_cutoff if target_cutoff is not None else current['cutoff'],
            target_attack if target_attack is not None else current['attack'],
        )

        # Transform
        self.transform.eval()
        with torch.no_grad():
            z_edited = self.transform(z, current_params, target)

        return self.decode(z_edited)

    # ==================== TRAINING ====================

    def train_all(self, training_data: Dict, n_epochs: int = 500):
        """Train all components on provided data."""

        z_all = training_data['z'].to(self.device)
        params_all = training_data['params'].to(self.device)
        N = len(z_all)

        print("\n" + "="*60)
        print("Training DCAE Controller Components")
        print("="*60)

        # 1. Train Extractor (linear, should be fast)
        print("\n[1/3] Training Param Extractor...")
        self._train_extractor(z_all, params_all, n_epochs=200)

        # 2. Train Oscillator
        print("\n[2/3] Training Latent Oscillator...")
        self._train_oscillator(z_all, params_all, n_epochs=n_epochs)

        # 3. Train Transform
        print("\n[3/3] Training Conditional Transform...")
        self._train_transform(z_all, params_all, n_epochs=n_epochs)

    def _train_extractor(self, z_all: torch.Tensor, params_all: torch.Tensor, n_epochs: int):
        """Train the param extractor (linear)."""
        self.extractor.train()
        optimizer = torch.optim.Adam(self.extractor.parameters(), lr=1e-2)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            params_pred = self.extractor(z_all)
            loss = F.mse_loss(params_pred, params_all)
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch:3d}: loss = {loss.item():.6f}")

    def _train_oscillator(self, z_all: torch.Tensor, params_all: torch.Tensor, n_epochs: int):
        """Train the latent oscillator."""
        self.oscillator.train()
        optimizer = torch.optim.AdamW(self.oscillator.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        batch_size = 64
        N = len(z_all)

        for epoch in range(n_epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                z_batch = z_all[idx]
                p_batch = params_all[idx]

                optimizer.zero_grad()
                z_pred = self.oscillator(p_batch)
                loss = F.mse_loss(z_pred, z_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.oscillator.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch:3d}: loss = {epoch_loss/n_batches:.6f}")

    def _train_transform(self, z_all: torch.Tensor, params_all: torch.Tensor, n_epochs: int):
        """Train the conditional transform."""
        self.transform.train()
        optimizer = torch.optim.AdamW(self.transform.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        N = len(z_all)

        for epoch in range(n_epochs):
            # Random pairs
            idx_source = torch.randint(0, N, (N,))
            idx_target = torch.randint(0, N, (N,))

            z_source = z_all[idx_source]
            z_target = z_all[idx_target]
            p_source = params_all[idx_source]
            p_target = params_all[idx_target]

            optimizer.zero_grad()
            z_pred = self.transform(z_source, p_source, p_target)
            loss = F.mse_loss(z_pred, z_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.transform.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch:3d}: loss = {loss.item():.6f}")

    def save(self, path: Path):
        """Save all components."""
        torch.save({
            'extractor': self.extractor.state_dict(),
            'oscillator': self.oscillator.state_dict(),
            'transform': self.transform.state_dict(),
            'param_ranges': self.param_ranges,
        }, str(path))
        print(f"Saved controller to {path}")

    def load(self, path: Path):
        """Load all components."""
        data = torch.load(str(path), map_location=self.device)
        self.extractor.load_state_dict(data['extractor'])
        self.oscillator.load_state_dict(data['oscillator'])
        self.transform.load_state_dict(data['transform'])
        self.param_ranges = data['param_ranges']
        print(f"Loaded controller from {path}")


# ============================================================
# Data Generation
# ============================================================

def generate_saw_with_params(pitch: float, cutoff: float, attack: float = 0.01) -> np.ndarray:
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), dtype=np.float32)

    audio = np.zeros_like(t)
    for h in range(1, 30):
        freq = pitch * h
        if freq > SAMPLE_RATE / 2:
            break
        audio += ((-1) ** h) * np.sin(2 * np.pi * freq * t) / h

    nyq = SAMPLE_RATE / 2
    cutoff_norm = min(cutoff / nyq, 0.99)
    sos = signal.butter(4, cutoff_norm, btype='low', output='sos')
    audio = signal.sosfilt(sos, audio).astype(np.float32)

    env = np.ones_like(t)
    attack_samples = int(attack * SAMPLE_RATE)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    env[-int(0.1 * SAMPLE_RATE):] = np.linspace(1, 0, int(0.1 * SAMPLE_RATE))
    audio = audio * env

    audio = audio / (np.abs(audio).max() + 1e-8) * 0.8
    return audio.astype(np.float32)


def generate_training_data(controller: DCAEController, n_per_dim: int = 6) -> Dict:
    """Generate training data grid."""

    print("Generating training data...")

    pitches = np.linspace(110, 440, n_per_dim)
    cutoffs = np.linspace(500, 4000, n_per_dim)
    attacks = np.linspace(0.01, 0.1, 3)

    z_list = []
    params_list = []

    total = len(pitches) * len(cutoffs) * len(attacks)
    count = 0

    for pitch in pitches:
        for cutoff in cutoffs:
            for attack in attacks:
                audio = generate_saw_with_params(pitch, cutoff, attack)
                z = controller.encode(audio)

                z_list.append(z.cpu())
                params_list.append(controller.normalize_params(pitch, cutoff, attack).cpu())

                count += 1
                if count % 20 == 0:
                    print(f"  {count}/{total}")
                    clear_memory()

    return {
        'z': torch.cat(z_list, dim=0),
        'params': torch.cat(params_list, dim=0),
    }


# ============================================================
# Demo / Test
# ============================================================

def demo(controller: DCAEController, output_dir: Path):
    """Demo all three capabilities."""

    print("\n" + "="*60)
    print("DEMO: DCAE Controller Capabilities")
    print("="*60)

    # 1. GENERATE
    print("\n[1] GENERATE: Create sounds from params")
    for pitch in [110, 220, 440]:
        for cutoff in [500, 2000, 4000]:
            audio = controller.generate(pitch=pitch, cutoff=cutoff, attack=0.01)
            filename = f"gen_p{pitch}_c{cutoff}.wav"
            torchaudio.save(str(output_dir / filename),
                           torch.from_numpy(audio).unsqueeze(0).float(), SAMPLE_RATE)
    print("  Saved 9 generated sounds")

    # 2. ANALYZE
    print("\n[2] ANALYZE: Extract params from audio")
    test_audio = generate_saw_with_params(pitch=220, cutoff=1500, attack=0.05)
    extracted = controller.analyze(test_audio)
    print(f"  Ground truth: pitch=220, cutoff=1500, attack=0.05")
    print(f"  Extracted: pitch={extracted['pitch']:.1f}, cutoff={extracted['cutoff']:.1f}, attack={extracted['attack']:.3f}")

    # 3. EDIT
    print("\n[3] EDIT: Modify existing audio")
    source_audio = generate_saw_with_params(pitch=220, cutoff=500, attack=0.01)
    torchaudio.save(str(output_dir / "edit_source.wav"),
                   torch.from_numpy(source_audio).unsqueeze(0).float(), SAMPLE_RATE)

    # Edit cutoff only
    edited_bright = controller.edit(source_audio, target_cutoff=4000)
    torchaudio.save(str(output_dir / "edit_bright.wav"),
                   torch.from_numpy(edited_bright).unsqueeze(0).float(), SAMPLE_RATE)

    # Edit pitch only
    edited_high = controller.edit(source_audio, target_pitch=440)
    torchaudio.save(str(output_dir / "edit_high_pitch.wav"),
                   torch.from_numpy(edited_high).unsqueeze(0).float(), SAMPLE_RATE)

    # Edit both
    edited_both = controller.edit(source_audio, target_pitch=440, target_cutoff=4000)
    torchaudio.save(str(output_dir / "edit_both.wav"),
                   torch.from_numpy(edited_both).unsqueeze(0).float(), SAMPLE_RATE)

    print("  Saved edit_source.wav, edit_bright.wav, edit_high_pitch.wav, edit_both.wav")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


def main():
    print("="*60)
    print("DCAE Controller - Full Pipeline")
    print("="*60)

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.cuda.empty_cache()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/dcae_controller")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize controller
    controller = DCAEController(device=device)
    print("Controller initialized!")

    # Generate training data
    data = generate_training_data(controller, n_per_dim=6)
    print(f"Training data: z={data['z'].shape}, params={data['params'].shape}")

    # Train all components
    controller.train_all(data, n_epochs=500)

    # Save
    controller.save(output_dir / "dcae_controller.pt")

    # Demo
    demo(controller, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
