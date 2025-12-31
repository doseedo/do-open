#!/usr/bin/env python3
"""
Trace the full generation + decode pipeline to find where audio energy is lost.
"""

import sys
sys.path.insert(0, "/home/arlo/Data")
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import torch.nn.functional as F
import torchaudio
from pathlib import Path

print("="*60)
print("TRACING GENERATION + DECODE PIPELINE")
print("="*60)

# Import genfrominterface components
from genfrominterface import (
    load_model_any_ckpt, load_conditioning, generate,
    DCAE_SR, DCAE_HOP
)

DCAE_FPS = DCAE_SR / DCAE_HOP  # ~10.766
COND_FPS = 43.066  # Conditioning fps used by the model

# Load model
print("\nLoading model...")
CHECKPOINT = "/home/arlo/Data/ACE-Step/checkpoints"
MANIFEST = "/home/arlo/Data/final_training_manifest_final.json"

model = load_model_any_ckpt(CHECKPOINT, CHECKPOINT, MANIFEST)
model.dcae.to("cuda")
model.ace_step_transformer.to("cuda")
print("Model loaded")

# Load conditioning
cond_dir = Path("/home/arlo/Data/extracted_conditioning/giantsteps_sax/giantsteps")
print(f"\nLoading conditioning from {cond_dir}...")

# Use correct interface for load_conditioning
conds = load_conditioning({"dir": cond_dir.parent, "stem": cond_dir.name}, window_slow=583)
print(f"Conditioning loaded with {len(conds)} items")

# Generate latent (not audio)
print("\n" + "="*60)
print("STEP 1: GENERATING LATENT VIA DIFFUSION")
print("="*60)

latent = generate(
    model,
    conds,
    group="winds",
    subgroup="sax",
    seed=42,
    steps=30,
    cfg_weight=1.0,
    use_ground_truth_latents=False,
    return_latents=True,  # Return latent instead of audio
)

print(f"\nGenerated latent shape: {latent.shape}")  # Should be [1, 8, 16, T] at COND_FPS

# Analyze latent at conditioning fps
T_cond = latent.shape[-1]
duration = T_cond / COND_FPS

print(f"\nLatent at conditioning fps ({COND_FPS:.2f}):")
print(f"  Frames: {T_cond}, Duration: {duration:.2f}s")

print(f"\nLatent energy at 1-second intervals (conditioning fps):")
for sec in range(0, int(duration) + 1):
    start_frame = int(sec * COND_FPS)
    end_frame = min(int((sec + 1) * COND_FPS), T_cond)
    if start_frame < T_cond:
        lat_slice = latent[:, :, :, start_frame:end_frame]
        energy = torch.norm(lat_slice).item()
        mean_abs = torch.mean(torch.abs(lat_slice)).item()
        std = torch.std(lat_slice).item()
        print(f"  {sec}s - {sec+1}s: L2 = {energy:.2f}, mean_abs = {mean_abs:.4f}, std = {std:.4f}")

# Save latent
latent_path = Path("/home/arlo/Data/sliding_window_test/debug_latent_cond_fps.pt")
torch.save(latent.cpu(), latent_path)
print(f"\nSaved latent to: {latent_path}")

# Step 2: Downsample to DCAE fps
print("\n" + "="*60)
print("STEP 2: DOWNSAMPLING LATENT TO DCAE FPS")
print("="*60)

target_frames = int(round(duration * DCAE_FPS))
B, C, H, T = latent.shape
latent_flat = latent.reshape(B, C * H, T)
latent_dcae = F.interpolate(latent_flat, size=target_frames, mode='linear', align_corners=False)
latent_dcae = latent_dcae.reshape(B, C, H, target_frames)

print(f"Downsampled latent shape: {latent_dcae.shape}")
print(f"  Frames: {target_frames}, Duration: {target_frames/DCAE_FPS:.2f}s")

print(f"\nDownsampled latent energy at 1-second intervals (DCAE fps):")
for sec in range(0, int(target_frames / DCAE_FPS) + 1):
    start_frame = int(sec * DCAE_FPS)
    end_frame = min(int((sec + 1) * DCAE_FPS), target_frames)
    if start_frame < target_frames:
        lat_slice = latent_dcae[:, :, :, start_frame:end_frame]
        energy = torch.norm(lat_slice).item()
        mean_abs = torch.mean(torch.abs(lat_slice)).item()
        std = torch.std(lat_slice).item()
        print(f"  {sec}s - {sec+1}s: L2 = {energy:.2f}, mean_abs = {mean_abs:.4f}, std = {std:.4f}")

# Step 3: DCAE Decode
print("\n" + "="*60)
print("STEP 3: DCAE DECODE")
print("="*60)

audio_len = int(round(duration * 44100))
audio_lengths = torch.tensor([audio_len], dtype=torch.long, device="cuda")

# Try with float32
latent_for_decode = latent_dcae.to(device="cuda", dtype=torch.float32)
model.dcae = model.dcae.float()

print(f"Latent dtype: {latent_for_decode.dtype}")
print(f"Expected audio length: {audio_len} samples = {audio_len/44100:.2f}s")

try:
    sr_pred, wav_pred = model.dcae.decode(latent_for_decode, audio_lengths=audio_lengths, sr=44100)

    if isinstance(wav_pred, list):
        wav = wav_pred[0].float().cpu().squeeze().numpy()
    else:
        wav = wav_pred[0].float().cpu().squeeze().numpy()

    print(f"\nDecoded audio: {len(wav)} samples = {len(wav)/sr_pred:.2f}s at {sr_pred} Hz")

    print(f"\nDecoded audio RMS at 1-second intervals:")
    for sec in range(0, int(len(wav) / sr_pred) + 1):
        start = int(sec * sr_pred)
        end = min(int((sec + 1) * sr_pred), len(wav))
        if start < len(wav):
            chunk = wav[start:end]
            rms = np.sqrt(np.mean(chunk**2))
            peak = np.max(np.abs(chunk))
            print(f"  {sec}s - {sec+1}s: RMS = {rms:.4f}, Peak = {peak:.4f}")

    # Save decoded audio
    output_path = Path("/home/arlo/Data/sliding_window_test/debug_decoded.wav")
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)
    torchaudio.save(output_path, wav_tensor, sr_pred)
    print(f"\nSaved decoded audio to: {output_path}")

except Exception as e:
    print(f"\nDecode failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)
print("""
Compare the energy values at each stage:
1. Latent at conditioning fps - should have energy throughout
2. Downsampled latent at DCAE fps - should preserve energy
3. Decoded audio - if RMS drops here, DCAE decoder is the issue

If latent has energy but audio doesn't, check:
- DCAE decoder internal processing
- Audio length mismatch
- Resampling artifacts
""")
