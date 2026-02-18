#!/usr/bin/env python3
"""Render audio comparison: ground truth mel vs predicted mel through vocoder."""

import torch
import torch.nn.functional as F
import torchaudio
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
from mel_to_sines_mapper import MelMapperV2, ZToMelDataset, collate_fn

from torch.utils.data import DataLoader


def mel_to_audio(dcae, mel, device='cuda'):
    """
    Convert mel spectrogram to audio using the proper vocoder pipeline.

    mel: [B, 128, T] - raw decoder output (in [-1, 1] range)
    """
    # The decoder outputs in [-1, 1] range, need to scale for vocoder
    # From MusicDCAE.decode():
    #   mels = mels * 0.5 + 0.5  # to [0, 1]
    #   mels = mels * (max_mel - min_mel) + min_mel

    mel_scaled = mel * 0.5 + 0.5
    mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value

    # Vocoder expects [B, 128, T], mono
    # Use vocoder.decode() not vocoder()
    wav = dcae.vocoder.decode(mel_scaled).squeeze(1)

    return wav


def main():
    device = 'cuda'
    output_dir = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/audio_comparison'
    os.makedirs(output_dir, exist_ok=True)

    # Load DCAE + Vocoder
    print("Loading DCAE + Vocoder...")
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH)
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    print(f"  Mel value range: [{dcae.min_mel_value}, {dcae.max_mel_value}]")
    print(f"  Scale factor: {dcae.scale_factor}, Shift factor: {dcae.shift_factor}")

    # Load mel mapper
    print("Loading mel mapper...")
    checkpoint = torch.load(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_mapper/best_model.pt',
        weights_only=True
    )
    model = MelMapperV2(hidden_dim=checkpoint.get('hidden_dim', 256)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded model with loss {checkpoint['loss']:.4f}")

    # Load a few samples
    print("Loading dataset...")
    dataset = ZToMelDataset(
        dcae=dcae,
        sms_manifest_path='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        max_samples=50,
        skip_drums=True,
        device=device,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"\nRendering comparisons to {output_dir}/")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Render 5 samples
                break

            z = batch['z'].to(device)

            # Ground truth: z → decoder → mel → vocoder → audio
            # Use the proper denormalization from MusicDCAE
            z_denorm = z / dcae.scale_factor + dcae.shift_factor

            # Use dcae.decoder (not dcae.decode which returns sample object)
            mel_true = dcae.dcae.decoder(z_denorm)  # [B, C, 128, T]
            mel_true = mel_true.mean(dim=1)  # [B, 128, T] - average stereo to mono

            # Predicted: z → mapper → mel → vocoder → audio
            mel_pred = model(z).permute(0, 2, 1)  # [B, 128, T]

            # Match lengths
            min_T = min(mel_pred.shape[-1], mel_true.shape[-1])
            mel_pred = mel_pred[..., :min_T]
            mel_true = mel_true[..., :min_T]

            # Check mel ranges
            print(f"  Sample {i}: mel_true range [{mel_true.min():.2f}, {mel_true.max():.2f}], mel_pred range [{mel_pred.min():.2f}, {mel_pred.max():.2f}]")

            # Run through vocoder with proper scaling
            audio_true = mel_to_audio(dcae, mel_true, device)
            audio_pred = mel_to_audio(dcae, mel_pred, device)

            # Ensure same length
            min_len = min(audio_true.shape[-1], audio_pred.shape[-1])
            audio_true = audio_true[..., :min_len].squeeze()
            audio_pred = audio_pred[..., :min_len].squeeze()

            # Normalize
            audio_true = audio_true / (audio_true.abs().max() + 1e-8) * 0.9
            audio_pred = audio_pred / (audio_pred.abs().max() + 1e-8) * 0.9

            # Save
            torchaudio.save(f'{output_dir}/sample_{i:02d}_ground_truth.wav',
                          audio_true.unsqueeze(0).cpu(), 44100)
            torchaudio.save(f'{output_dir}/sample_{i:02d}_predicted.wav',
                          audio_pred.unsqueeze(0).cpu(), 44100)

            # Also save concatenated A/B comparison
            # Add 0.5s silence between
            silence = torch.zeros(22050)
            combined = torch.cat([audio_true.cpu(), silence, audio_pred.cpu()])
            torchaudio.save(f'{output_dir}/sample_{i:02d}_AB_comparison.wav',
                          combined.unsqueeze(0), 44100)

            print(f"    Saved (duration: {audio_true.shape[0]/44100:.2f}s)")

    print(f"\nDone! Files in {output_dir}/")
    print("  *_ground_truth.wav = decoder mel → vocoder")
    print("  *_predicted.wav = mapper mel → vocoder")
    print("  *_AB_comparison.wav = ground truth, then predicted")


if __name__ == "__main__":
    main()
