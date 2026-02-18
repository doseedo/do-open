#!/usr/bin/env python3
"""Debug: compare predicted vs ground truth SMS on a TRAINING sample."""

import numpy as np
import torch
import sys

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/training')
from train_sms_hybrid import HybridSAMIMapper
import orjson

device = 'cuda'

# Load model
model_path = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/sms_hybrid/best_model.pt"
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model = HybridSAMIMapper(n_sines=64, n_noise_bands=8, hidden_dim=512, n_blocks=4).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded, loss was {checkpoint.get('loss', '?'):.4f}")

# Load a training sample
sms_manifest = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_hybrid/sms_manifest.json"
with open(sms_manifest, 'rb') as f:
    manifest = orjson.loads(f.read())

# Find a sample with actual content
for entry in manifest['entries']:
    sms_data_check = torch.load(entry['path'], weights_only=True, map_location='cpu')
    if sms_data_check['amps'].mean() > 0.01:  # Has content
        break
print(f"Found sample with content: amps mean = {sms_data_check['amps'].mean():.3f}")
print(f"\nTraining sample: {entry['path']}")

# Load SMS ground truth
sms_data = torch.load(entry['path'], weights_only=True, map_location='cpu')
gt_freqs = sms_data['freqs'].numpy()  # [T, 64]
gt_amps = sms_data['amps'].numpy()
gt_noise = sms_data['noise_amps'].numpy()

print(f"GT shape: freqs={gt_freqs.shape}, amps={gt_amps.shape}, noise={gt_noise.shape}")
print(f"GT stats:")
print(f"  Freqs: min={gt_freqs.min():.1f}, max={gt_freqs.max():.1f}, mean={gt_freqs[gt_freqs>0].mean():.1f}")
print(f"  Amps: min={gt_amps.min():.3f}, max={gt_amps.max():.3f}, mean={gt_amps.mean():.3f}")
print(f"  Active sines (amp>0.01): {(gt_amps > 0.01).sum(axis=1).mean():.1f}")
print(f"  Noise: min={gt_noise.min():.3f}, max={gt_noise.max():.3f}, mean={gt_noise.mean():.3f}")

# Load latent for this sample
lat_data = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
latent = lat_data.get('latents', lat_data.get('latent'))
print(f"\nLatent shape: {latent.shape}")

# Crop to 22 frames like training does
T = latent.shape[-1]
target_frames = 22
if T > target_frames:
    start = (T - target_frames) // 2
    latent = latent[:, :, start:start + target_frames]
    gt_freqs = gt_freqs[start:start + target_frames]
    gt_amps = gt_amps[start:start + target_frames]
    gt_noise = gt_noise[start:start + target_frames]
print(f"After crop to {target_frames}: latent={latent.shape}, gt_freqs={gt_freqs.shape}")

# Run inference
latent = latent.unsqueeze(0).to(device)  # [1, 8, 16, 22]
with torch.no_grad():
    output = model(latent)
    pred_freqs = output['freqs'][0].cpu().numpy()
    pred_amps = output['amps'][0].cpu().numpy()
    pred_noise = output['noise_amps'][0].cpu().numpy()

print(f"\nPredicted shape: freqs={pred_freqs.shape}")
print(f"Predicted stats:")
print(f"  Freqs: min={pred_freqs.min():.1f}, max={pred_freqs.max():.1f}, mean={pred_freqs.mean():.1f}")
print(f"  Amps: min={pred_amps.min():.3f}, max={pred_amps.max():.3f}, mean={pred_amps.mean():.3f}")
print(f"  Active sines (amp>0.01): {(pred_amps > 0.01).sum(axis=1).mean():.1f}")
print(f"  Noise: min={pred_noise.min():.3f}, max={pred_noise.max():.3f}, mean={pred_noise.mean():.3f}")

# Compare frame by frame for first few frames
print("\n" + "="*60)
print("Frame-by-frame comparison (top 5 sines by GT amp):")
for t in [0, 10, 21]:
    print(f"\nFrame {t}:")
    # Get top GT sines
    top_gt = np.argsort(gt_amps[t])[-5:][::-1]
    print(f"  GT top 5:")
    for i in top_gt:
        print(f"    sine {i}: freq={gt_freqs[t,i]:.1f}Hz, amp={gt_amps[t,i]:.3f}")

    # Get top predicted sines
    top_pred = np.argsort(pred_amps[t])[-5:][::-1]
    print(f"  Pred top 5:")
    for i in top_pred:
        print(f"    sine {i}: freq={pred_freqs[t,i]:.1f}Hz, amp={pred_amps[t,i]:.3f}")
