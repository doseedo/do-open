#!/usr/bin/env python3
"""
Diagnostic script to investigate why audio drops off after 3 seconds.
This analyzes conditioning and latent energy across temporal dimension.
"""

import sys
sys.path.insert(0, "/home/arlo/Data")
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

# Load conditioning files
cond_dir = Path("/home/arlo/Data/extracted_conditioning/giantsteps_sax/giantsteps")

print("="*60)
print("CONDITIONING ANALYSIS")
print("="*60)

# Load raw conditioning
amp = np.load(cond_dir / "giantsteps.amp.npy")
pr = np.load(cond_dir / "giantsteps.pianoroll.npy")

print(f"\nRaw conditioning shapes (DCAE fps ~10.77):")
print(f"  amp: {amp.shape}")  # (144,)
print(f"  piano_roll: {pr.shape}")  # (128, 143)

# Check temporal energy distribution in raw conditioning
print(f"\nRaw amp energy at 1-second intervals (at DCAE fps ~10.77):")
dcae_fps = 10.766
for sec in range(0, int(len(amp) / dcae_fps) + 1):
    start_frame = int(sec * dcae_fps)
    end_frame = min(int((sec + 1) * dcae_fps), len(amp))
    if start_frame < len(amp):
        amp_slice = amp[start_frame:end_frame]
        print(f"  {sec}s - {sec+1}s: frames {start_frame}-{end_frame}, mean amp = {np.mean(amp_slice):.4f}")

print(f"\nRaw piano_roll energy at 1-second intervals:")
for sec in range(0, int(pr.shape[1] / dcae_fps) + 1):
    start_frame = int(sec * dcae_fps)
    end_frame = min(int((sec + 1) * dcae_fps), pr.shape[1])
    if start_frame < pr.shape[1]:
        pr_slice = pr[:, start_frame:end_frame]
        active_notes = np.sum(pr_slice > 0)
        print(f"  {sec}s - {sec+1}s: frames {start_frame}-{end_frame}, active_notes = {active_notes}")

# Simulate resampling to conditioning fps
print("\n" + "="*60)
print("RESAMPLING TO CONDITIONING FPS (43.066)")
print("="*60)

window_slow = 583  # Target frames at conditioning fps
cond_fps = 43.066

# Resample amp
amp_tensor = torch.from_numpy(amp).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
amp_resampled = F.interpolate(amp_tensor, size=window_slow, mode='linear', align_corners=False)
amp_resampled = amp_resampled.squeeze().numpy()

# Resample piano roll
pr_tensor = torch.from_numpy(pr).float().unsqueeze(0)  # [1, 128, T]
pr_resampled = F.interpolate(pr_tensor, size=window_slow, mode='nearest')
pr_resampled = pr_resampled.squeeze().numpy()

print(f"\nResampled shapes (conditioning fps 43.066):")
print(f"  amp: {amp_resampled.shape}")
print(f"  piano_roll: {pr_resampled.shape}")
print(f"  Duration: {window_slow / cond_fps:.2f}s")

print(f"\nResampled amp energy at 1-second intervals:")
for sec in range(0, int(window_slow / cond_fps) + 1):
    start_frame = int(sec * cond_fps)
    end_frame = min(int((sec + 1) * cond_fps), window_slow)
    if start_frame < window_slow:
        amp_slice = amp_resampled[start_frame:end_frame]
        print(f"  {sec}s - {sec+1}s: frames {start_frame}-{end_frame}, mean amp = {np.mean(amp_slice):.4f}, max = {np.max(amp_slice):.4f}")

print(f"\nResampled piano_roll energy at 1-second intervals:")
for sec in range(0, int(window_slow / cond_fps) + 1):
    start_frame = int(sec * cond_fps)
    end_frame = min(int((sec + 1) * cond_fps), window_slow)
    if start_frame < window_slow:
        pr_slice = pr_resampled[:, start_frame:end_frame]
        active_notes = np.sum(pr_slice > 0)
        max_activity = np.max(np.sum(pr_slice > 0, axis=0))
        print(f"  {sec}s - {sec+1}s: frames {start_frame}-{end_frame}, total_active = {active_notes}, max_per_frame = {max_activity}")

# Check for zeros or very low values
print("\n" + "="*60)
print("CHECKING FOR DEAD ZONES")
print("="*60)

# Find first zero or near-zero region
threshold = 0.01
silent_frames = np.where(amp_resampled < threshold)[0]
if len(silent_frames) > 0:
    first_silent = silent_frames[0]
    print(f"First near-silent frame (amp < {threshold}): {first_silent} = {first_silent / cond_fps:.2f}s")
else:
    print(f"No near-silent frames found (amp >= {threshold} throughout)")

# Check piano roll
pr_activity = np.sum(pr_resampled > 0, axis=0)
inactive_frames = np.where(pr_activity == 0)[0]
if len(inactive_frames) > 0:
    print(f"Frames with no piano roll activity: {len(inactive_frames)} frames")
    if len(inactive_frames) < 20:
        print(f"  Inactive frame indices: {inactive_frames[:20]}")
else:
    print("Piano roll has activity in every frame")

# Analyze GT latent if available
print("\n" + "="*60)
print("GT LATENT ANALYSIS")
print("="*60)

gt_path = cond_dir / "giantsteps.latent.pt"
if gt_path.exists():
    gt_latent = torch.load(gt_path, weights_only=True)
    print(f"GT latent shape: {gt_latent.shape}")

    # Check energy across time
    if gt_latent.ndim == 3:  # [8, 16, T]
        T_gt = gt_latent.shape[-1]
        gt_fps = dcae_fps  # GT latent is at DCAE fps

        print(f"GT latent energy at 1-second intervals (at DCAE fps):")
        for sec in range(0, int(T_gt / gt_fps) + 1):
            start_frame = int(sec * gt_fps)
            end_frame = min(int((sec + 1) * gt_fps), T_gt)
            if start_frame < T_gt:
                gt_slice = gt_latent[:, :, start_frame:end_frame]
                energy = torch.norm(gt_slice).item()
                mean_abs = torch.mean(torch.abs(gt_slice)).item()
                print(f"  {sec}s - {sec+1}s: frames {start_frame}-{end_frame}, L2 = {energy:.2f}, mean_abs = {mean_abs:.4f}")
else:
    print("GT latent not found")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
Key observations:
1. Raw conditioning has {len(amp)} frames at DCAE fps ({len(amp)/dcae_fps:.2f}s duration)
2. After resampling to conditioning fps: {window_slow} frames ({window_slow/cond_fps:.2f}s duration)
3. The conditioning should cover the full audio duration

If audio drops at 3 seconds ({int(3*cond_fps)} frames at cond fps):
- Check if the model's control branch processes all frames
- Check if the diffusion loop properly uses conditioning for later timesteps
- Check if the DCAE decoder has issues with longer latents
""")
