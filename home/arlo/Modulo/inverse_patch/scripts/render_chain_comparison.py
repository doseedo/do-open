#!/usr/bin/env python3
"""
Render comparison of all stages in the white-box chain:

1. Original (DCAE vocoder)
2. Predicted mel → vocoder
3. Predicted sines → additive synth
4. Program (MDL) → sines → additive synth
"""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
import orjson

from full_whitebox_chain import (
    MelMapperV2, SMSMapperV2,
    MDLProgramExtractor, ProgramSynthesizer, AdditiveSynth
)


def render_all_stages(z, dcae, mel_mapper, sms_mapper, device='cuda'):
    """Render audio from all stages of the chain."""

    results = {}

    # ========================================
    # 1. ORIGINAL: DCAE decoder → vocoder
    # ========================================
    with torch.no_grad():
        z_denorm = z / dcae.scale_factor + dcae.shift_factor
        mel_dcae = dcae.dcae.decoder(z_denorm).mean(dim=1)  # [B, 128, T]
        mel_scaled = mel_dcae * 0.5 + 0.5
        mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value
        audio_original = dcae.vocoder.decode(mel_scaled).squeeze()
        audio_original = audio_original / (audio_original.abs().max() + 1e-8) * 0.9

    results['original'] = audio_original.cpu().numpy()
    results['mel_dcae'] = mel_dcae[0].cpu().numpy()  # [128, T]

    # ========================================
    # 2. PREDICTED MEL → vocoder
    # ========================================
    with torch.no_grad():
        mel_pred = mel_mapper(z)  # [B, T, 128]
        mel_pred_t = mel_pred.permute(0, 2, 1)  # [B, 128, T]

        # Scale to vocoder range
        mel_for_vocoder = mel_pred_t * 0.5 + 0.5
        mel_for_vocoder = mel_for_vocoder * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value

        # Match length to original
        target_len = mel_scaled.shape[-1]
        if mel_for_vocoder.shape[-1] != target_len:
            mel_for_vocoder = F.interpolate(mel_for_vocoder, size=target_len, mode='linear', align_corners=False)

        audio_mel_pred = dcae.vocoder.decode(mel_for_vocoder).squeeze()
        audio_mel_pred = audio_mel_pred / (audio_mel_pred.abs().max() + 1e-8) * 0.9

    results['mel_predicted'] = audio_mel_pred.cpu().numpy()
    results['mel_pred_spec'] = mel_pred[0].cpu().numpy()  # [T, 128]

    # ========================================
    # 3. PREDICTED SINES → additive synth
    # ========================================
    with torch.no_grad():
        freqs, amps = sms_mapper(mel_pred)  # [B, T, n_sines]

    freqs_np = freqs[0].cpu()  # [T, 64]
    amps_np = amps[0].cpu()

    # Additive synthesis
    synth = AdditiveSynth(sr=44100, hop_length=512)
    audio_sines = synth.synthesize(freqs_np, amps_np)

    results['sines_predicted'] = audio_sines.numpy()
    results['freqs'] = freqs_np.numpy()
    results['amps'] = amps_np.numpy()

    # ========================================
    # 4. PROGRAM (MDL) → sines → additive synth
    # ========================================
    mdl = MDLProgramExtractor()
    program = mdl.extract(freqs_np, amps_np)

    prog_synth = ProgramSynthesizer(n_sines=64)
    prog_freqs, prog_amps = prog_synth.synthesize(program)

    audio_program = synth.synthesize(prog_freqs, prog_amps)

    results['program_rendered'] = audio_program.numpy()
    results['program'] = program
    results['prog_freqs'] = prog_freqs.numpy()
    results['prog_amps'] = prog_amps.numpy()

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load DCAE
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
        vocoder_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    )
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    # Load trained mappers
    print("Loading mappers...")
    mel_mapper = MelMapperV2().to(device)
    sms_mapper = SMSMapperV2().to(device)

    mel_ckpt = torch.load('checkpoints/mel_mapper/best_model.pt', weights_only=True)
    mel_mapper.load_state_dict(mel_ckpt['model_state_dict'])
    mel_mapper.eval()

    sms_ckpt = torch.load('checkpoints/mel_to_sms/best_model_v2.pt', weights_only=True)
    sms_mapper.load_state_dict(sms_ckpt['model_state_dict'])
    sms_mapper.eval()

    # Load test latent
    manifest_path = 'data/sms_v4/sms_manifest.json'
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    z = None
    sample_path = None
    for entry in manifest['entries'][:100]:
        path = entry['path']
        if 'drum' in path.lower():
            continue
        try:
            data = torch.load(path, weights_only=True, map_location='cpu')
            lat_path = data.get('latent_path')
            if lat_path and os.path.exists(lat_path):
                lat_data = torch.load(lat_path, weights_only=True, map_location='cpu')
                z = lat_data.get('latents', lat_data)
                if z.dim() == 3:
                    z = z.unsqueeze(0)
                z = z[:, :, :, :32].to(device)
                sample_path = path
                break
        except:
            continue

    if z is None:
        print("No latent found!")
        return

    print(f"Sample: {sample_path}")
    print(f"z shape: {z.shape}")

    # Render all stages
    print("\nRendering all stages...")
    results = render_all_stages(z, dcae, mel_mapper, sms_mapper, device)

    # Save audio files
    out_dir = 'outputs/chain_comparison'
    os.makedirs(out_dir, exist_ok=True)

    print("\nSaving audio files:")

    sf.write(f'{out_dir}/1_original_dcae.wav', results['original'], 44100)
    print(f"  1. Original (DCAE): {out_dir}/1_original_dcae.wav")

    sf.write(f'{out_dir}/2_predicted_mel.wav', results['mel_predicted'], 44100)
    print(f"  2. Predicted mel → vocoder: {out_dir}/2_predicted_mel.wav")

    sf.write(f'{out_dir}/3_predicted_sines.wav', results['sines_predicted'], 44100)
    print(f"  3. Predicted sines → additive: {out_dir}/3_predicted_sines.wav")

    sf.write(f'{out_dir}/4_program_rendered.wav', results['program_rendered'], 44100)
    print(f"  4. Program (MDL) → sines: {out_dir}/4_program_rendered.wav")

    # Print program summary
    prog = results['program']
    print(f"\nExtracted Program:")
    print(f"  Harmonic groups: {len(prog['harmonic_groups'])}")
    for i, g in enumerate(prog['harmonic_groups'][:5]):
        partials = [f"{p[0]}x" for p in g['partials'][:4]]
        print(f"    {i}: f0={g['f0']:.1f}Hz, partials={partials}")
    print(f"  Independent sines: {len(prog['independent_sines'])}")
    print(f"  Damped sines: {len(prog['damped_sines'])}")

    # Compression
    raw = prog['n_frames'] * 64 * 2
    compressed = (len(prog['harmonic_groups']) * 8 +
                  len(prog['independent_sines']) * 2 +
                  len(prog['damped_sines']) * 2)
    print(f"\n  Compression: {raw} → {compressed} ({raw/(compressed+1):.1f}x)")

    # Audio lengths
    print(f"\nAudio lengths:")
    print(f"  Original: {len(results['original'])/44100:.2f}s")
    print(f"  Mel pred: {len(results['mel_predicted'])/44100:.2f}s")
    print(f"  Sines: {len(results['sines_predicted'])/44100:.2f}s")
    print(f"  Program: {len(results['program_rendered'])/44100:.2f}s")

    print(f"\nDone! Compare the 4 audio files in {out_dir}/")


if __name__ == "__main__":
    main()
