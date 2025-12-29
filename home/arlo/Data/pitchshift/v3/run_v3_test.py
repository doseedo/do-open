#!/usr/bin/env python3
"""
Test V3 Range-Group Pitch Shift Model.

Takes an input audio, pitch shifts it, then applies V3 correction.
Compares: original, librosa pitch-shifted, V3-corrected.

Usage:
    python run_v3_test.py --input audio.wav --checkpoint best.pt --segments segments.json
"""

import sys
from pathlib import Path
import argparse
import json

import torch
import torchaudio
import numpy as np

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/pitchshift/v3')
sys.path.insert(0, '/home/arlo/Data/dø')

from models_v3 import RangeGroupTranslator, RangeGroupTranslatorDirect, RangeGroupTranslatorAdaptive


def load_dcae():
    """Load DCAE model."""
    from do.pipeline_do import DoTrainComponents
    comps = DoTrainComponents(checkpoint_dir='/home/arlo/Data/ACE-Step/checkpoints', dtype="float32")
    dcae = comps.load_dcae()
    return dcae.cuda().eval()


def load_v3_model(checkpoint_path: str, num_groups: int, model_type: str = 'residual'):
    """Load V3 model from checkpoint."""
    if model_type == 'direct':
        model = RangeGroupTranslatorDirect(num_groups=num_groups)
    elif model_type == 'adaptive':
        model = RangeGroupTranslatorAdaptive(num_groups=num_groups)
    else:
        model = RangeGroupTranslator(num_groups=num_groups)

    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.cuda().eval()


def pitch_shift_audio(audio: torch.Tensor, sr: int, semitones: int) -> torch.Tensor:
    """Pitch shift using torchaudio."""
    # torchaudio PitchShift expects [channel, time]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    pitch_shift = torchaudio.transforms.PitchShift(sr, semitones)
    shifted = pitch_shift(audio)

    return shifted.squeeze(0)


def get_range_group(midi_pitch: float, group_size: int = 6, base_pitch: int = 48) -> int:
    """Get range group for a MIDI pitch."""
    return int((midi_pitch - base_pitch) // group_size)


def estimate_pitch(audio: torch.Tensor, sr: int) -> float:
    """Estimate dominant pitch of audio in MIDI using autocorrelation."""
    audio_np = audio.numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=0)

    # Simple autocorrelation-based pitch estimation
    # Look for peaks in autocorrelation between 80Hz and 1000Hz
    min_period = int(sr / 1000)  # 1000 Hz
    max_period = int(sr / 80)    # 80 Hz

    # Take a chunk from the middle
    chunk_size = min(len(audio_np), sr)  # 1 second or less
    start = (len(audio_np) - chunk_size) // 2
    chunk = audio_np[start:start + chunk_size]

    # Normalize
    chunk = chunk - chunk.mean()
    if chunk.std() < 1e-6:
        return 60.0  # Silent, return middle C

    chunk = chunk / chunk.std()

    # Autocorrelation
    corr = np.correlate(chunk, chunk, mode='full')
    corr = corr[len(corr)//2:]  # Take positive lags only

    # Find peak in valid range
    valid_corr = corr[min_period:max_period]
    if len(valid_corr) == 0:
        return 60.0

    peak_idx = np.argmax(valid_corr) + min_period
    f0 = sr / peak_idx

    # Convert to MIDI
    midi = 69 + 12 * np.log2(f0 / 440.0)
    return float(midi)


@torch.inference_mode()
def run_test(
    input_wav: str,
    output_dir: str,
    checkpoint_path: str,
    segments_json: str,
    shift_amounts: list = [-12, -6, 6, 12],
):
    """Run V3 test on input audio."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load segments config
    with open(segments_json, 'r') as f:
        seg_data = json.load(f)

    config = seg_data['config']
    num_groups = config['num_groups']
    group_size = config['group_size']
    base_pitch = config['base_pitch']

    print(f"Config: {num_groups} groups, size={group_size}, base={base_pitch}")

    # Load models
    print("Loading DCAE...")
    dcae = load_dcae()

    print("Loading V3 model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model_type = checkpoint.get('model_type', 'residual')
    model = load_v3_model(checkpoint_path, num_groups, model_type)

    print(f"Model type: {model_type}")
    if model_type == 'residual':
        print(f"Scales: dry={model.dry_scale.item():.3f}, residual={model.residual_scale.item():.3f}")
    elif model_type == 'adaptive':
        print(f"Alpha bounds: [{model.alpha_min:.2f}, {model.alpha_max:.2f}], bias={model.alpha_bias.item():.2f}")

    # Load audio
    print(f"\nLoading: {input_wav}")
    audio, sr = torchaudio.load(input_wav)
    audio = audio.mean(0)  # Mono

    # Resample to 44100 if needed
    if sr != 44100:
        audio = torchaudio.functional.resample(audio, sr, 44100)
        sr = 44100

    # Estimate original pitch
    original_pitch = estimate_pitch(audio, sr)
    original_group = get_range_group(original_pitch, group_size, base_pitch)
    print(f"Original pitch: MIDI {original_pitch:.1f}, group {original_group}")

    # Save original
    stem = Path(input_wav).stem
    orig_path = output_dir / f"{stem}_00_original.wav"
    torchaudio.save(str(orig_path), audio.unsqueeze(0), sr)
    print(f"Saved: {orig_path}")

    # Process each shift amount
    for shift in shift_amounts:
        print(f"\n{'='*50}")
        print(f"Processing shift: {shift:+d} semitones")

        # Pitch shift with librosa
        shifted_audio = pitch_shift_audio(audio, sr, shift)

        # Determine target group (where the shifted audio SHOULD sound like)
        shifted_pitch = original_pitch + shift
        target_group = get_range_group(shifted_pitch, group_size, base_pitch)
        target_group = max(0, min(target_group, num_groups - 1))

        print(f"  Shifted pitch: MIDI {shifted_pitch:.1f}")
        print(f"  Target group: {target_group}")

        # Save pitch-shifted
        ps_path = output_dir / f"{stem}_{shift:+03d}_pitchshift.wav"
        torchaudio.save(str(ps_path), shifted_audio.unsqueeze(0), sr)
        print(f"  Saved: {ps_path}")

        # Encode with DCAE - needs stereo input [B, 2, T] at 48kHz
        # Resample to 48kHz for DCAE
        shifted_48k = torchaudio.functional.resample(shifted_audio, sr, 48000)
        # shifted_audio is [T], make it [1, 2, T] for DCAE
        shifted_stereo = shifted_48k.unsqueeze(0).unsqueeze(0).expand(1, 2, -1).cuda()  # [1, 2, T]

        # Pad to multiple of 8*512 = 4096
        pad_len = (4096 - shifted_stereo.shape[-1] % 4096) % 4096
        if pad_len > 0:
            shifted_stereo = torch.nn.functional.pad(shifted_stereo, (0, pad_len))

        latent, latent_lengths = dcae.encode(shifted_stereo, sr=48000)
        print(f"  Latent shape: {latent.shape}")

        # Apply V3 model
        target_group_tensor = torch.tensor([target_group], device='cuda')
        corrected_latent = model(latent, target_group_tensor)

        # Check correlation
        corr = torch.corrcoef(torch.stack([
            latent.flatten(),
            corrected_latent.flatten()
        ]))[0, 1].item()
        print(f"  Input↔Output correlation: {corr:.3f}")

        # Decode - returns (sr_out, list of [2, T] wavs)
        sr_out, pred_wavs = dcae.decode(corrected_latent, sr=sr)
        # Take first channel of first wav
        corrected_audio = pred_wavs[0][0]  # [T]

        # Trim to original length
        corrected_audio = corrected_audio[:len(shifted_audio)]

        # Save V3-corrected
        v3_path = output_dir / f"{stem}_{shift:+03d}_v3corrected.wav"
        torchaudio.save(str(v3_path), corrected_audio.unsqueeze(0), sr)
        print(f"  Saved: {v3_path}")

    print(f"\n{'='*50}")
    print(f"All outputs saved to: {output_dir}")
    print("\nCompare the files to evaluate quality:")
    print("  - *_original.wav: Original audio")
    print("  - *_pitchshift.wav: Librosa pitch-shifted (has artifacts)")
    print("  - *_v3corrected.wav: V3 model corrected")


def main():
    parser = argparse.ArgumentParser(description="Test V3 Range-Group Pitch Shift")

    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file')
    parser.add_argument('--output_dir', type=str, default='/home/arlo/Data/pitchshift/v3_test_output',
                        help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--segments', type=str, required=True,
                        help='Segments JSON from segment_by_range.py')
    parser.add_argument('--shifts', type=int, nargs='+', default=[-12, -6, 6, 12],
                        help='Shift amounts in semitones')

    args = parser.parse_args()

    run_test(
        input_wav=args.input,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        segments_json=args.segments,
        shift_amounts=args.shifts,
    )


if __name__ == "__main__":
    main()
