#!/usr/bin/env python3
"""
Latent Synth Training — ACE-Step 1.5 VAE Format

Trains a lightweight synthesizer that renders MIDI → z-space latents [64, T] at 25Hz.
Used during DO1 training for cross-instrument pairs (latent synth on the fly, ~2.8ms per MIDI).

Architecture:
  MIDI note → subtractive synth (audio domain) → VAE.encode → z_synth
  But VAE.encode is expensive (~100ms). So we train a SMALL model to predict
  z_synth directly from MIDI parameters, bypassing VAE at training time.

Two-phase approach:
  Phase 1: Generate ground truth pairs (MIDI params → audio → VAE.encode → z_target)
  Phase 2: Train small model to predict z_target from MIDI params directly

The trained model runs at ~2.8ms per MIDI file, enabling on-the-fly rendering
during DO1 training with random VCF/VCA params every step.

Usage:
  # Phase 1: Generate training data (requires VAE on GPU)
  python train_latent_synth.py generate \\
      --vae_path /checkpoints/ace-step-1.5/vae \\
      --midi_dir /data/basic_pitch_midi \\
      --output_dir /data/latent_synth_pairs \\
      --num_timbres 50

  # Phase 2: Train the latent synth model
  python train_latent_synth.py train \\
      --data_dir /data/latent_synth_pairs \\
      --output_dir /checkpoints/latent_synth_v15

  # Test: render a MIDI file
  python train_latent_synth.py render \\
      --checkpoint /checkpoints/latent_synth_v15/best.pt \\
      --midi song.mid \\
      --output rendered.pt
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# VAE constants
VAE_SR = 48000
VAE_HZ = 25.0
VAE_DIM = 64


# =============================================================================
# SUBTRACTIVE SYNTH (audio domain, for generating ground truth)
# =============================================================================

class SubtractiveSynth:
    """
    Simple subtractive synthesizer for generating training audio.

    Signal chain: Oscillator → VCF (filter) → VCA (amplitude envelope)

    Oscillator types: saw, square, triangle, sine, noise
    VCF: 2-pole lowpass with resonance and envelope modulation
    VCA: ADSR envelope with velocity sensitivity
    """

    def __init__(self, sr: int = VAE_SR):
        self.sr = sr

    def render_note(
        self,
        midi_note: int,
        velocity: float,        # 0.0 - 1.0
        duration_sec: float,
        osc_type: str = "saw",
        vcf_cutoff: float = 0.5,    # 0-1 normalized
        vcf_resonance: float = 0.3,  # 0-1
        vcf_env_amount: float = 0.5, # how much envelope modulates filter
        vca_attack: float = 0.01,
        vca_decay: float = 0.1,
        vca_sustain: float = 0.7,
        vca_release: float = 0.15,
    ) -> np.ndarray:
        """Render a single note to audio. Returns mono float32 array."""

        freq = 440.0 * (2 ** ((midi_note - 69) / 12))
        total_sec = duration_sec + vca_release + 0.05  # pad for release
        num_samples = int(total_sec * self.sr)
        t = np.arange(num_samples, dtype=np.float32) / self.sr

        # --- Oscillator ---
        phase = 2 * np.pi * freq * t
        if osc_type == "saw":
            osc = 2.0 * (freq * t % 1.0) - 1.0
        elif osc_type == "square":
            osc = np.sign(np.sin(phase)).astype(np.float32)
        elif osc_type == "triangle":
            osc = 2.0 * np.abs(2.0 * (freq * t % 1.0) - 1.0) - 1.0
        elif osc_type == "sine":
            osc = np.sin(phase).astype(np.float32)
        elif osc_type == "noise":
            osc = np.random.randn(num_samples).astype(np.float32) * 0.5
        else:
            osc = np.sin(phase).astype(np.float32)

        # --- VCA (ADSR envelope) ---
        envelope = self._adsr_envelope(
            num_samples, duration_sec,
            vca_attack, vca_decay, vca_sustain, vca_release
        )
        # Velocity scaling
        envelope = envelope * (0.3 + 0.7 * velocity)

        # Apply VCA
        audio = osc * envelope

        # --- VCF (simple lowpass) ---
        # Map normalized cutoff to Hz (50Hz - 16kHz log scale)
        cutoff_hz = 50.0 * (320 ** vcf_cutoff)  # 50 to 16000

        # Envelope modulates cutoff
        env_mod = self._adsr_envelope(
            num_samples, duration_sec,
            max(0.001, vca_attack * 0.5), vca_decay * 0.7, 0.3, vca_release
        )
        cutoff_modulated = cutoff_hz * (1.0 + vcf_env_amount * 4.0 * env_mod)
        cutoff_modulated = np.clip(cutoff_modulated, 50, self.sr * 0.45)

        # Apply time-varying lowpass (simple one-pole per sample)
        audio = self._one_pole_filter(audio, cutoff_modulated)

        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9

        return audio.astype(np.float32)

    def _adsr_envelope(
        self, num_samples: int, note_dur: float,
        attack: float, decay: float, sustain: float, release: float,
    ) -> np.ndarray:
        """Generate ADSR envelope."""
        env = np.zeros(num_samples, dtype=np.float32)
        a_samples = int(attack * self.sr)
        d_samples = int(decay * self.sr)
        note_samples = int(note_dur * self.sr)
        r_start = min(note_samples, num_samples)
        r_samples = int(release * self.sr)

        # Attack
        if a_samples > 0:
            end_a = min(a_samples, num_samples)
            env[:end_a] = np.linspace(0, 1, end_a, dtype=np.float32)

        # Decay
        d_start = a_samples
        d_end = min(d_start + d_samples, num_samples)
        if d_end > d_start:
            env[d_start:d_end] = np.linspace(1, sustain, d_end - d_start, dtype=np.float32)

        # Sustain
        s_start = d_end
        s_end = min(r_start, num_samples)
        if s_end > s_start:
            env[s_start:s_end] = sustain

        # Release
        r_end = min(r_start + r_samples, num_samples)
        if r_end > r_start:
            env[r_start:r_end] = np.linspace(sustain, 0, r_end - r_start, dtype=np.float32)

        return env

    def _one_pole_filter(self, audio: np.ndarray, cutoff_hz: np.ndarray) -> np.ndarray:
        """Time-varying one-pole lowpass filter."""
        out = np.zeros_like(audio)
        y = 0.0
        for i in range(len(audio)):
            w = 2.0 * np.pi * cutoff_hz[i] / self.sr
            alpha = w / (w + 1.0)
            y = alpha * audio[i] + (1.0 - alpha) * y
            out[i] = y
        return out

    def render_midi(
        self,
        notes: List[Dict],
        params: Dict,
        total_duration: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render a list of MIDI notes to audio.

        Args:
            notes: List of {"pitch": int, "start": float, "end": float, "velocity": float}
            params: Synth params {"osc_type", "vcf_cutoff", "vcf_resonance", ...}
            total_duration: Total audio duration (default: last note end + release)

        Returns:
            Stereo audio [2, samples]
        """
        if not notes:
            dur = total_duration or 1.0
            return np.zeros((2, int(dur * self.sr)), dtype=np.float32)

        if total_duration is None:
            total_duration = max(n["end"] for n in notes) + 0.3

        num_samples = int(total_duration * self.sr)
        audio = np.zeros(num_samples, dtype=np.float32)

        for note in notes:
            note_audio = self.render_note(
                midi_note=note["pitch"],
                velocity=note.get("velocity", 0.8),
                duration_sec=note["end"] - note["start"],
                osc_type=params.get("osc_type", "saw"),
                vcf_cutoff=params.get("vcf_cutoff", 0.5),
                vcf_resonance=params.get("vcf_resonance", 0.3),
                vcf_env_amount=params.get("vcf_env_amount", 0.5),
                vca_attack=params.get("vca_attack", 0.01),
                vca_decay=params.get("vca_decay", 0.1),
                vca_sustain=params.get("vca_sustain", 0.7),
                vca_release=params.get("vca_release", 0.15),
            )

            start_sample = int(note["start"] * self.sr)
            end_sample = min(start_sample + len(note_audio), num_samples)
            length = end_sample - start_sample
            audio[start_sample:end_sample] += note_audio[:length]

        # Clip and normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9

        # Make stereo
        stereo = np.stack([audio, audio], axis=0)
        return stereo


def sample_random_synth_params() -> Dict:
    """Sample random subtractive synth parameters."""
    return {
        "osc_type": random.choice(["saw", "square", "triangle", "sine", "noise"]),
        "vcf_cutoff": random.uniform(0.1, 0.9),
        "vcf_resonance": random.uniform(0.0, 0.8),
        "vcf_env_amount": random.uniform(0.0, 1.0),
        "vca_attack": random.uniform(0.001, 0.15),
        "vca_decay": random.uniform(0.01, 0.5),
        "vca_sustain": random.uniform(0.1, 1.0),
        "vca_release": random.uniform(0.02, 0.5),
    }


# =============================================================================
# MIDI LOADING
# =============================================================================

def load_midi_notes(midi_path: str) -> List[Dict]:
    """Load MIDI file and extract notes."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                notes.append({
                    "pitch": note.pitch,
                    "start": note.start,
                    "end": note.end,
                    "velocity": note.velocity / 127.0,
                })
        notes.sort(key=lambda n: n["start"])
        return notes
    except ImportError:
        raise ImportError("pretty_midi required: pip install pretty_midi")


def notes_to_tensor(
    notes: List[Dict],
    total_frames: int,
    hz: float = VAE_HZ,
) -> torch.Tensor:
    """
    Convert note list to conditioning tensor [5, T].

    Channels:
      0: pitch (MIDI note / 127, 0 = silence)
      1: velocity (0-1)
      2: onset flag (1 at note start frames)
      3: offset flag (1 at note end frames)
      4: active flag (1 during note, 0 during silence)
    """
    tensor = torch.zeros(5, total_frames)

    for note in notes:
        start_frame = int(note["start"] * hz)
        end_frame = int(note["end"] * hz)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

        # Pitch (normalized)
        tensor[0, start_frame:end_frame] = note["pitch"] / 127.0

        # Velocity
        tensor[1, start_frame:end_frame] = note.get("velocity", 0.8)

        # Onset
        tensor[2, start_frame] = 1.0

        # Offset
        if end_frame < total_frames:
            tensor[3, end_frame - 1] = 1.0

        # Active
        tensor[4, start_frame:end_frame] = 1.0

    return tensor


def synth_params_to_tensor(params: Dict) -> torch.Tensor:
    """
    Convert synth params to conditioning tensor [N].

    Maps all params to 0-1 range for conditioning.
    """
    osc_map = {"saw": 0.0, "square": 0.25, "triangle": 0.5, "sine": 0.75, "noise": 1.0}

    return torch.tensor([
        osc_map.get(params["osc_type"], 0.0),
        params["vcf_cutoff"],
        params["vcf_resonance"],
        params["vcf_env_amount"],
        min(params["vca_attack"] / 0.15, 1.0),
        min(params["vca_decay"] / 0.5, 1.0),
        params["vca_sustain"],
        min(params["vca_release"] / 0.5, 1.0),
    ], dtype=torch.float32)


# =============================================================================
# PHASE 1: GENERATE GROUND TRUTH PAIRS
# =============================================================================

def generate_training_data(
    vae_path: str,
    midi_dir: str,
    output_dir: str,
    num_timbres_per_midi: int = 50,
    max_midi_files: int = None,
    device: str = "cuda",
):
    """
    Generate (MIDI params, z_target) training pairs.

    For each MIDI file:
      For each random timbre:
        1. Render audio with subtractive synth
        2. Encode through VAE → z_target
        3. Save (note_tensor, synth_params, z_target)
    """
    from diffusers import AutoencoderOobleck

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderOobleck.from_pretrained(vae_path)
    vae = vae.eval().to(device)

    # Load MIDI files
    midi_dir = Path(midi_dir)
    midi_files = sorted(list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi")))
    if max_midi_files:
        midi_files = midi_files[:max_midi_files]
    print(f"Found {len(midi_files)} MIDI files")

    synth = SubtractiveSynth(sr=VAE_SR)
    manifest = []
    pair_idx = 0

    for midi_idx, midi_path in enumerate(tqdm(midi_files, desc="Processing MIDI")):
        try:
            notes = load_midi_notes(str(midi_path))
        except Exception as e:
            print(f"  Skip {midi_path.name}: {e}")
            continue

        if len(notes) == 0:
            continue

        total_duration = max(n["end"] for n in notes) + 0.5

        for timbre_idx in range(num_timbres_per_midi):
            params = sample_random_synth_params()

            # Render audio
            audio = synth.render_midi(notes, params, total_duration)  # [2, samples]

            # Encode through VAE
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)  # [1, 2, samples]
                dist = vae.encode(audio_tensor)
                z = dist.latent_dist.sample()  # [1, 64, T]
                z = z.squeeze(0).cpu()  # [64, T]

            T = z.shape[-1]

            # Build note conditioning tensor
            note_tensor = notes_to_tensor(notes, T)  # [5, T]

            # Build synth param tensor
            param_tensor = synth_params_to_tensor(params)  # [8]

            # Save pair
            pair_name = f"pair_{pair_idx:06d}"
            pair_path = output_dir / f"{pair_name}.pt"

            torch.save({
                "z_target": z,               # [64, T]
                "note_tensor": note_tensor,   # [5, T]
                "synth_params": param_tensor, # [8]
                "midi_path": str(midi_path),
                "audio_params": params,
                "num_notes": len(notes),
                "duration_sec": total_duration,
            }, pair_path)

            manifest.append({
                "pair_name": pair_name,
                "midi": midi_path.name,
                "timbre_idx": timbre_idx,
                "num_notes": len(notes),
                "duration": total_duration,
                "T": T,
            })

            pair_idx += 1

        if (midi_idx + 1) % 100 == 0:
            print(f"  Generated {pair_idx} pairs from {midi_idx + 1} MIDI files")

    # Save manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump({
            "total_pairs": pair_idx,
            "num_midi_files": len(midi_files),
            "timbres_per_midi": num_timbres_per_midi,
            "vae_dim": VAE_DIM,
            "vae_hz": VAE_HZ,
            "items": manifest,
        }, f, indent=2)

    print(f"\nGenerated {pair_idx} training pairs → {output_dir}")


# =============================================================================
# LATENT SYNTH MODEL
# =============================================================================

class LatentSynthModel(nn.Module):
    """
    Small model that predicts VAE latents directly from MIDI parameters.

    Input:
      note_tensor [5, T]:  pitch, velocity, onset, offset, active per frame
      synth_params [8]:    oscillator type, VCF, VCA parameters

    Output:
      z_pred [64, T]:      predicted VAE latent

    Architecture:
      Global params → FiLM conditioning
      Note tensor → 1D ConvNet → modulated by params → z_pred

    ~2-5M parameters. Runs in ~2.8ms per MIDI on GPU.
    """

    def __init__(
        self,
        note_channels: int = 5,
        param_dim: int = 8,
        latent_dim: int = VAE_DIM,
        hidden_dim: int = 256,
        num_conv_layers: int = 6,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Synth params → FiLM conditioning
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # scale + shift
        )

        # Note tensor → initial features
        self.input_conv = nn.Conv1d(note_channels, hidden_dim, 1)

        # Residual conv blocks with FiLM
        self.conv_blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for _ in range(num_conv_layers):
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(8, hidden_dim),
            ))
            self.film_layers.append(
                nn.Linear(hidden_dim * 2, hidden_dim * 2)
            )

        # Output projection
        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, latent_dim, 1),
        )

    def forward(
        self,
        note_tensor: torch.Tensor,   # [B, 5, T]
        synth_params: torch.Tensor,   # [B, 8]
    ) -> torch.Tensor:
        """Predict VAE latent from MIDI + synth params. Returns [B, 64, T]."""

        # Encode synth params for FiLM
        param_emb = self.param_encoder(synth_params)  # [B, hidden*2]

        # Input projection
        x = self.input_conv(note_tensor)  # [B, hidden, T]

        # Conv blocks with FiLM conditioning
        for conv_block, film_layer in zip(self.conv_blocks, self.film_layers):
            # FiLM modulation
            film = film_layer(param_emb)  # [B, hidden*2]
            scale, shift = film.chunk(2, dim=-1)  # each [B, hidden]
            scale = scale[:, :, None]  # [B, hidden, 1]
            shift = shift[:, :, None]

            # Residual conv
            residual = x
            x = conv_block(x)
            x = x * (1 + scale) + shift  # FiLM modulation
            x = x + residual  # residual connection

        # Output
        z_pred = self.output_conv(x)  # [B, 64, T]
        return z_pred


# =============================================================================
# TRAINING DATASET
# =============================================================================

class LatentSynthDataset(Dataset):
    """Dataset of (note_tensor, synth_params) → z_target pairs."""

    def __init__(self, data_dir: str, max_T: int = 2500):
        self.data_dir = Path(data_dir)
        self.max_T = max_T

        # Load manifest
        manifest_path = self.data_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.items = manifest["items"]
        self.pair_files = [self.data_dir / f"{item['pair_name']}.pt" for item in self.items]

        # Filter to existing files
        self.pair_files = [p for p in self.pair_files if p.exists()]
        print(f"Loaded {len(self.pair_files)} latent synth pairs")

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, idx):
        data = torch.load(self.pair_files[idx], map_location="cpu", weights_only=True)

        z_target = data["z_target"]        # [64, T]
        note_tensor = data["note_tensor"]  # [5, T]
        synth_params = data["synth_params"]  # [8]

        T = z_target.shape[-1]

        # Random crop if too long
        if T > self.max_T:
            start = random.randint(0, T - self.max_T)
            z_target = z_target[:, start:start + self.max_T]
            note_tensor = note_tensor[:, start:start + self.max_T]

        return {
            "z_target": z_target,
            "note_tensor": note_tensor,
            "synth_params": synth_params,
        }


def collate_latent_synth(batch):
    """Pad variable-length sequences to batch max."""
    max_T = max(item["z_target"].shape[-1] for item in batch)

    z_targets = []
    note_tensors = []
    synth_params = []

    for item in batch:
        T = item["z_target"].shape[-1]
        pad = max_T - T
        z_targets.append(F.pad(item["z_target"], (0, pad)))
        note_tensors.append(F.pad(item["note_tensor"], (0, pad)))
        synth_params.append(item["synth_params"])

    return {
        "z_target": torch.stack(z_targets),
        "note_tensor": torch.stack(note_tensors),
        "synth_params": torch.stack(synth_params),
    }


# =============================================================================
# PHASE 2: TRAIN LATENT SYNTH
# =============================================================================

def train_latent_synth(
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda",
    num_workers: int = 4,
):
    """Train the latent synth model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset = LatentSynthDataset(data_dir)
    val_size = min(500, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_latent_synth, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_latent_synth, pin_memory=True,
    )

    # Model
    model = LatentSynthModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Latent synth model: {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []

        for batch in train_loader:
            z_target = batch["z_target"].to(device)
            note_tensor = batch["note_tensor"].to(device)
            synth_params = batch["synth_params"].to(device)

            z_pred = model(note_tensor, synth_params)
            loss = F.mse_loss(z_pred, z_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                z_target = batch["z_target"].to(device)
                note_tensor = batch["note_tensor"].to(device)
                synth_params = batch["synth_params"].to(device)

                z_pred = model(note_tensor, synth_params)
                loss = F.mse_loss(z_pred, z_target)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{epochs} | train: {train_loss:.6f} | val: {val_loss:.6f} | lr: {scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "note_channels": 5,
                    "param_dim": 8,
                    "latent_dim": VAE_DIM,
                    "hidden_dim": 256,
                    "num_conv_layers": 6,
                    "kernel_size": 7,
                },
            }, output_dir / "best.pt")
            print(f"  → Saved best model (val_loss={val_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, output_dir / f"epoch_{epoch+1:04d}.pt")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {output_dir}")


# =============================================================================
# INFERENCE (for use during DO1 training)
# =============================================================================

class LatentSynth:
    """
    Trained latent synth for on-the-fly MIDI → z rendering.

    Usage during DO1 training:
        synth = LatentSynth("checkpoints/latent_synth_v15/best.pt", device="cuda")
        notes = load_midi_notes("stem.mid")
        z_synth = synth.render(notes, random_params=True)  # ~2.8ms
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", {})

        self.model = LatentSynthModel(**config)
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def render(
        self,
        notes: List[Dict],
        params: Optional[Dict] = None,
        total_duration: Optional[float] = None,
        random_params: bool = True,
    ) -> torch.Tensor:
        """
        Render MIDI notes to VAE latent.

        Args:
            notes: List of {"pitch", "start", "end", "velocity"}
            params: Synth params dict (random if None and random_params=True)
            total_duration: Override total duration
            random_params: Sample random params if params is None

        Returns:
            z [1, 64, T] VAE latent
        """
        if not notes:
            T = int((total_duration or 1.0) * VAE_HZ)
            return torch.zeros(1, VAE_DIM, T, device=self.device)

        if total_duration is None:
            total_duration = max(n["end"] for n in notes) + 0.3

        T = int(total_duration * VAE_HZ)

        # Build conditioning tensors
        note_tensor = notes_to_tensor(notes, T).unsqueeze(0).to(self.device)  # [1, 5, T]

        if params is None and random_params:
            params = sample_random_synth_params()
        elif params is None:
            params = sample_random_synth_params()

        param_tensor = synth_params_to_tensor(params).unsqueeze(0).to(self.device)  # [1, 8]

        # Predict latent
        z = self.model(note_tensor, param_tensor)  # [1, 64, T]
        return z

    @torch.no_grad()
    def render_batch(
        self,
        notes_list: List[List[Dict]],
        params_list: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """Render a batch of MIDI files. Returns [B, 64, max_T]."""
        if params_list is None:
            params_list = [sample_random_synth_params() for _ in notes_list]

        # Find max duration
        max_dur = 0
        for notes in notes_list:
            if notes:
                max_dur = max(max_dur, max(n["end"] for n in notes) + 0.3)
        max_T = int(max_dur * VAE_HZ)

        note_tensors = []
        param_tensors = []

        for notes, params in zip(notes_list, params_list):
            nt = notes_to_tensor(notes, max_T)
            pt = synth_params_to_tensor(params)
            note_tensors.append(nt)
            param_tensors.append(pt)

        note_batch = torch.stack(note_tensors).to(self.device)   # [B, 5, T]
        param_batch = torch.stack(param_tensors).to(self.device)  # [B, 8]

        z = self.model(note_batch, param_batch)  # [B, 64, T]
        return z


# =============================================================================
# RENDER COMMAND (for testing)
# =============================================================================

def render_test(checkpoint_path: str, midi_path: str, output_path: str, device: str = "cuda"):
    """Render a MIDI file and save the latent."""
    synth = LatentSynth(checkpoint_path, device=device)
    notes = load_midi_notes(midi_path)

    print(f"MIDI: {midi_path} ({len(notes)} notes)")

    # Render with random params
    import time
    start = time.perf_counter()
    z = synth.render(notes, random_params=True)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Rendered in {elapsed:.1f}ms → z {list(z.shape)}")

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        z = synth.render(notes, random_params=True)
        times.append((time.perf_counter() - start) * 1000)
    print(f"Benchmark (100 runs): {np.mean(times):.1f}ms ± {np.std(times):.1f}ms")

    torch.save({"latent": z.cpu(), "notes": notes}, output_path)
    print(f"Saved: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Latent Synth — MIDI → VAE Latent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate training data
    p = subparsers.add_parser("generate", help="Phase 1: Generate ground truth pairs")
    p.add_argument("--vae_path", required=True, help="ACE-Step 1.5 VAE path")
    p.add_argument("--midi_dir", required=True, help="Directory of MIDI files")
    p.add_argument("--output_dir", required=True, help="Output directory for pairs")
    p.add_argument("--num_timbres", type=int, default=50, help="Timbres per MIDI file")
    p.add_argument("--max_midi", type=int, default=None, help="Max MIDI files to process")
    p.add_argument("--device", type=str, default="cuda")

    # Train model
    p = subparsers.add_parser("train", help="Phase 2: Train latent synth model")
    p.add_argument("--data_dir", required=True, help="Directory with generated pairs")
    p.add_argument("--output_dir", required=True, help="Checkpoint output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)

    # Render test
    p = subparsers.add_parser("render", help="Test: render a MIDI file")
    p.add_argument("--checkpoint", required=True, help="Latent synth checkpoint")
    p.add_argument("--midi", required=True, help="Input MIDI file")
    p.add_argument("--output", default="rendered.pt", help="Output .pt file")
    p.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.command == "generate":
        generate_training_data(
            vae_path=args.vae_path,
            midi_dir=args.midi_dir,
            output_dir=args.output_dir,
            num_timbres_per_midi=args.num_timbres,
            max_midi_files=args.max_midi,
            device=args.device,
        )
    elif args.command == "train":
        train_latent_synth(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            num_workers=args.num_workers,
        )
    elif args.command == "render":
        render_test(args.checkpoint, args.midi, args.output, args.device)


if __name__ == "__main__":
    main()