#!/usr/bin/env python3
"""
Analyze generated latent to check if diffusion produces content throughout.
"""

import sys
sys.path.insert(0, "/home/arlo/Data")
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import torchaudio

# Check if there are saved latents from test runs
test_dir = Path("/home/arlo/Data/sliding_window_test")

print("="*60)
print("ANALYZING OUTPUT AUDIO FILES")
print("="*60)

def analyze_audio_energy(wav_path):
    """Analyze audio energy at 1-second intervals"""
    try:
        wav, sr = torchaudio.load(wav_path)
        wav = wav.squeeze().numpy()

        print(f"\n{wav_path.name}:")
        print(f"  Shape: {wav.shape}, SR: {sr}, Duration: {len(wav)/sr:.2f}s")

        for sec in range(0, int(len(wav) / sr) + 1):
            start = int(sec * sr)
            end = min(int((sec + 1) * sr), len(wav))
            if start < len(wav):
                chunk = wav[start:end]
                rms = np.sqrt(np.mean(chunk**2))
                peak = np.max(np.abs(chunk))
                print(f"    {sec}s - {sec+1}s: RMS = {rms:.4f}, Peak = {peak:.4f}")

        return wav, sr
    except Exception as e:
        print(f"  Error: {e}")
        return None, None

# Analyze all test outputs
for wav_file in sorted(test_dir.glob("*.wav")):
    analyze_audio_energy(wav_file)

# Now let's run a quick generation to capture the latent before decoding
print("\n" + "="*60)
print("RUNNING GENERATION WITH LATENT CAPTURE")
print("="*60)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import genfrominterface
from genfrominterface import Pipeline, load_conditioning, generate, DCAE_SR, DCAE_HOP

# Load model
print("Loading model...")
model = Pipeline(checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints")
model.dcae.to("cuda")
model.ace_step_transformer.to("cuda")

# Load conditioning
cond_dir = Path("/home/arlo/Data/extracted_conditioning/giantsteps_sax/giantsteps")
conds = load_conditioning(cond_dir, window_slow=583)

print(f"\nLoaded conditioning shapes:")
for k, v in conds.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

# Generate with latent return
print("\nGenerating with noise (to capture latent)...")
latent = generate(
    model,
    conds,
    group="saxophone",
    subgroup="saxophone",
    seed=42,
    steps=30,
    cfg_weight=1.0,
    use_ground_truth_latents=False,
    return_latents=True,  # Return latent instead of audio
)

print(f"\nGenerated latent shape: {latent.shape}")

# Analyze latent energy
DCAE_FPS = DCAE_SR / DCAE_HOP  # ~10.77 fps
COND_FPS = 43.066

# The latent is at conditioning fps
T_latent = latent.shape[-1]
print(f"\nLatent temporal dimension: {T_latent} frames")
print(f"At conditioning fps ({COND_FPS:.2f}): {T_latent/COND_FPS:.2f}s")

print(f"\nLatent energy at 1-second intervals (at conditioning fps):")
for sec in range(0, int(T_latent / COND_FPS) + 1):
    start_frame = int(sec * COND_FPS)
    end_frame = min(int((sec + 1) * COND_FPS), T_latent)
    if start_frame < T_latent:
        latent_slice = latent[:, :, :, start_frame:end_frame]
        energy = torch.norm(latent_slice).item()
        mean_abs = torch.mean(torch.abs(latent_slice)).item()
        std = torch.std(latent_slice).item()
        print(f"  {sec}s - {sec+1}s: frames {start_frame}-{end_frame}, L2 = {energy:.2f}, mean_abs = {mean_abs:.4f}, std = {std:.4f}")

# Downsample latent to DCAE fps
target_frames = int(round((T_latent / COND_FPS) * DCAE_FPS))
B, C, H, T = latent.shape
latent_flat = latent.reshape(B, C * H, T)
latent_dcae = F.interpolate(latent_flat, size=target_frames, mode='linear', align_corners=False)
latent_dcae = latent_dcae.reshape(B, C, H, target_frames)

print(f"\nLatent after downsampling to DCAE fps:")
print(f"  Shape: {latent_dcae.shape}")
print(f"  At DCAE fps ({DCAE_FPS:.2f}): {target_frames/DCAE_FPS:.2f}s")

print(f"\nDownsampled latent energy at 1-second intervals:")
for sec in range(0, int(target_frames / DCAE_FPS) + 1):
    start_frame = int(sec * DCAE_FPS)
    end_frame = min(int((sec + 1) * DCAE_FPS), target_frames)
    if start_frame < target_frames:
        latent_slice = latent_dcae[:, :, :, start_frame:end_frame]
        energy = torch.norm(latent_slice).item()
        mean_abs = torch.mean(torch.abs(latent_slice)).item()
        std = torch.std(latent_slice).item()
        print(f"  {sec}s - {sec+1}s: frames {start_frame}-{end_frame}, L2 = {energy:.2f}, mean_abs = {mean_abs:.4f}, std = {std:.4f}")

# Now decode and analyze audio
print("\n" + "="*60)
print("DECODING LATENT AND ANALYZING AUDIO")
print("="*60)

audio_len = int(round((T_latent / COND_FPS) * 44100))
audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=latent_dcae.device)

latent_for_decode = latent_dcae.to(device="cuda", dtype=torch.bfloat16)
sr_pred, wav_pred = model.dcae.decode(latent_for_decode, audio_lengths=audio_lengths, sr=44100)

if isinstance(wav_pred, list):
    wav = wav_pred[0].float().cpu().squeeze().numpy()
else:
    wav = wav_pred[0].float().cpu().squeeze().numpy()

print(f"\nDecoded audio: {len(wav)} samples = {len(wav)/sr_pred:.2f}s at {sr_pred} Hz")

print(f"\nDecoded audio energy at 1-second intervals:")
for sec in range(0, int(len(wav) / sr_pred) + 1):
    start = int(sec * sr_pred)
    end = min(int((sec + 1) * sr_pred), len(wav))
    if start < len(wav):
        chunk = wav[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        peak = np.max(np.abs(chunk))
        print(f"  {sec}s - {sec+1}s: RMS = {rms:.4f}, Peak = {peak:.4f}")

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print("""
If latent has energy throughout but audio doesn't:
  → Issue is in DCAE decoder

If latent already has low energy after 3s:
  → Issue is in diffusion process / conditioning application
""")
