#!/usr/bin/env python3
"""
Render comparison: original vs DCAE reconstruction.
"""

import torch
import torchaudio
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
import orjson

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("Loading DCAE...")
DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

dcae = MusicDCAE(
    dcae_checkpoint_path=DCAE_PATH,
    vocoder_checkpoint_path=VOCODER_PATH,
)
dcae.dcae.to(device)
dcae.dcae.eval()

# Output directory
out_dir = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/outputs/comparison'
os.makedirs(out_dir, exist_ok=True)

# Load manifest
with open('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json', 'rb') as f:
    manifest = orjson.loads(f.read())

rendered = 0
target = 3

for entry in manifest['entries']:
    if rendered >= target:
        break

    try:
        sms = torch.load(entry['path'], weights_only=True, map_location='cpu')
        audio_path = sms.get('audio_path', '')

        if not audio_path or not os.path.exists(audio_path):
            continue

        # Skip drums
        if any(kw in audio_path.lower() for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
            continue

        name = os.path.basename(audio_path).replace('.wav', '')
        print(f"\nRendering: {name}")

        # Load original
        audio, sr = torchaudio.load(audio_path)
        if sr != 44100:
            audio = torchaudio.functional.resample(audio, sr, 44100)
            sr = 44100

        # Stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        # Crop to 3 seconds
        max_len = sr * 3
        if audio.shape[-1] > max_len:
            audio = audio[:, :max_len]

        # Move to device
        audio_dev = audio.to(device)

        # Encode and decode
        with torch.no_grad():
            audio_lengths = torch.tensor([audio.shape[-1]], device=device)
            z, _ = dcae.encode(audio_dev.unsqueeze(0), audio_lengths=audio_lengths, sr=44100)
            _, reconstructed = dcae.decode(z, audio_lengths=audio_lengths, sr=44100)
            reconstructed = reconstructed[0].cpu()  # [2, samples]

        # Save both
        orig_path = f"{out_dir}/{rendered+1}_{name}_original.wav"
        recon_path = f"{out_dir}/{rendered+1}_{name}_reconstructed.wav"

        torchaudio.save(orig_path, audio, sr)
        torchaudio.save(recon_path, reconstructed, sr)

        print(f"  Saved: {orig_path}")
        print(f"  Saved: {recon_path}")

        rendered += 1

    except Exception as e:
        print(f"  Error: {e}")
        continue

print(f"\nDone! Rendered {rendered} comparison pairs to {out_dir}")
