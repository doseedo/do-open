#!/usr/bin/env python3
"""
Inference Pipeline

Convert dry trumpet audio to muted using the trained models.

Supports two modes:
1. Teacher mode: Use translator + DCAE (higher quality, slower)
2. Student mode: Use student model directly (faster, VST-ready)

Usage:
    # Teacher mode
    python inference.py --mode teacher \
        --translator_checkpoint ./checkpoints/best.pt \
        --input ./dry_trumpet.wav \
        --output ./muted_trumpet.wav

    # Student mode
    python inference.py --mode student \
        --student_checkpoint ./student_checkpoints/best.pt \
        --input ./dry_trumpet.wav \
        --output ./muted_trumpet.wav
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torchaudio
import numpy as np

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/mute_translator')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/dø')

from models import MuteTranslator, MuteTranslatorLarge
from train_student import MelStudentModel, WaveformStudentModel, MelSpectrogramTransform


class TeacherInference:
    """
    Inference using translator + DCAE.

    Pipeline: audio → DCAE.encode → translator → DCAE.decode → audio
    """

    def __init__(
        self,
        translator_checkpoint: str,
        dcae_checkpoint: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load DCAE
        print("Loading DCAE...")
        from do.pipeline_do import DoTrainComponents
        comps = DoTrainComponents(checkpoint_dir=dcae_checkpoint, dtype="float32")
        self.dcae = comps.load_dcae()
        self.dcae = self.dcae.to(self.device).eval()

        # Load translator
        print("Loading translator...")
        checkpoint = torch.load(translator_checkpoint, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        if 'input_proj.weight' in state_dict:
            in_channels = state_dict['input_proj.weight'].shape[1]
            if in_channels == 8:
                self.translator = MuteTranslator()
            else:
                self.translator = MuteTranslatorLarge()
        else:
            self.translator = MuteTranslator()

        self.translator.load_state_dict(state_dict)
        self.translator = self.translator.to(self.device).eval()

        print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")

    @torch.no_grad()
    def convert(self, input_path: str, output_path: str) -> str:
        """Convert dry audio to muted."""
        # Load audio
        audio, sr = torchaudio.load(input_path)

        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        audio = audio.unsqueeze(0).to(self.device)

        # Encode
        latent = self.dcae.encode(audio, sr=sr)

        # Translate
        muted_latent = self.translator(latent)

        # Decode
        muted_audio = self.dcae.decode(muted_latent)

        # Save
        muted_audio = muted_audio.squeeze(0).cpu()
        muted_audio = muted_audio / (muted_audio.abs().max() + 1e-8) * 0.9

        torchaudio.save(output_path, muted_audio, 44100)

        print(f"Saved: {output_path}")
        return output_path


class StudentInference:
    """
    Inference using student model directly.

    Pipeline: audio → student_model → audio
    Much faster, suitable for real-time/VST use.
    """

    def __init__(
        self,
        student_checkpoint: str,
        model_type: str = "mel",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Load model
        print(f"Loading student model ({model_type})...")
        checkpoint = torch.load(student_checkpoint, map_location=self.device)

        if model_type == "mel":
            self.model = MelStudentModel()
            self.mel_transform = MelSpectrogramTransform()
        else:
            self.model = WaveformStudentModel()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device).eval()

        print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")

    @torch.no_grad()
    def convert(self, input_path: str, output_path: str) -> str:
        """Convert dry audio to muted."""
        # Load audio
        audio, sr = torchaudio.load(input_path)

        # Resample to 44100 if needed
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio = resampler(audio)
            sr = 44100

        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        if self.model_type == "mel":
            # Process in mel domain
            mono = audio.mean(dim=0, keepdim=True)
            dry_mel = self.mel_transform.audio_to_mel(mono)
            dry_mel = dry_mel.unsqueeze(0).to(self.device)

            muted_mel = self.model(dry_mel)

            muted_mel = muted_mel.squeeze(0).cpu()
            muted_audio = self.mel_transform.mel_to_audio(muted_mel)

            # Make stereo
            muted_audio = muted_audio.repeat(2, 1)

        else:
            # Process in waveform domain
            audio = audio.unsqueeze(0).to(self.device)
            muted_audio = self.model(audio)
            muted_audio = muted_audio.squeeze(0).cpu()

        # Normalize
        muted_audio = muted_audio / (muted_audio.abs().max() + 1e-8) * 0.9

        torchaudio.save(output_path, muted_audio, 44100)

        print(f"Saved: {output_path}")
        return output_path


class BatchProcessor:
    """Process multiple files."""

    def __init__(self, converter):
        self.converter = converter

    def process_directory(self, input_dir: str, output_dir: str, extension: str = ".wav"):
        """Process all audio files in directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob(f"*{extension}"))
        print(f"Found {len(files)} files")

        for f in files:
            output_path = output_dir / f"{f.stem}_muted{extension}"
            try:
                self.converter.convert(str(f), str(output_path))
            except Exception as e:
                print(f"Error processing {f}: {e}")

        print(f"Done! Output: {output_dir}")


def export_onnx(model, output_path: str, model_type: str = "mel"):
    """Export model to ONNX for VST/deployment."""
    print(f"Exporting to ONNX: {output_path}")

    model.eval()

    if model_type == "mel":
        # Mel model input: [1, 1, n_mels, time_frames]
        dummy_input = torch.randn(1, 1, 128, 256)
    else:
        # Waveform model input: [1, 2, samples]
        dummy_input = torch.randn(1, 2, 44100 * 3)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['dry_input'],
        output_names=['muted_output'],
        dynamic_axes={
            'dry_input': {0: 'batch', 3 if model_type == "mel" else 2: 'time'},
            'muted_output': {0: 'batch', 3 if model_type == "mel" else 2: 'time'},
        },
        opset_version=14,
    )

    print(f"Exported: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Mute Converter Inference")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['teacher', 'student'],
                        help='Inference mode')

    # Teacher mode args
    parser.add_argument('--translator_checkpoint', type=str,
                        help='Path to translator checkpoint (teacher mode)')
    parser.add_argument('--dcae_checkpoint', type=str,
                        default='/home/arlo/Data/ACE-Step/checkpoints',
                        help='Path to DCAE checkpoint directory')

    # Student mode args
    parser.add_argument('--student_checkpoint', type=str,
                        help='Path to student checkpoint (student mode)')
    parser.add_argument('--model_type', type=str, default='mel',
                        choices=['mel', 'waveform'],
                        help='Student model type')

    # I/O args
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output audio file or directory')
    parser.add_argument('--batch', action='store_true',
                        help='Process entire directory')

    # Export
    parser.add_argument('--export_onnx', type=str,
                        help='Export model to ONNX file')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # Create converter
    if args.mode == 'teacher':
        if not args.translator_checkpoint:
            parser.error("--translator_checkpoint required for teacher mode")
        converter = TeacherInference(
            translator_checkpoint=args.translator_checkpoint,
            dcae_checkpoint=args.dcae_checkpoint,
            device=args.device,
        )
    else:
        if not args.student_checkpoint:
            parser.error("--student_checkpoint required for student mode")
        converter = StudentInference(
            student_checkpoint=args.student_checkpoint,
            model_type=args.model_type,
            device=args.device,
        )

    # Export if requested
    if args.export_onnx:
        if args.mode == 'student':
            export_onnx(converter.model.cpu(), args.export_onnx, args.model_type)
        else:
            print("ONNX export only supported for student mode")

    # Process
    if args.batch:
        processor = BatchProcessor(converter)
        processor.process_directory(args.input, args.output)
    else:
        converter.convert(args.input, args.output)


if __name__ == "__main__":
    main()
