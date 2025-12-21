#!/usr/bin/env python3
"""
Inference Pipeline for Register-Aware Pitch Shift

Apply pitch shift with register-aware timbre correction.

Modes:
1. Teacher mode: Translator + DCAE (higher quality, slower)
2. Student mode: Student model directly (faster, VST-ready)

Usage:
    # Shift up 5 semitones with register correction
    python inference_pitch_shift.py \
        --mode teacher \
        --translator_checkpoint ./checkpoints/best.pt \
        --input ./trumpet.wav \
        --output ./trumpet_shifted.wav \
        --shift 5

    # Batch processing
    python inference_pitch_shift.py \
        --mode teacher \
        --translator_checkpoint ./checkpoints/best.pt \
        --input ./audio_folder \
        --output ./output_folder \
        --shift 7 \
        --batch
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
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/do')

from models_pitch_shift import (
    RegisterAwareTranslator,
    RegisterAwareTranslatorDirect,
    RegisterAwareTranslatorLarge,
    PitchShiftStudentModel,
)


class TeacherInference:
    """
    Inference using translator + DCAE.

    Pipeline:
    1. Load audio
    2. Pitch shift audio (librosa/rubberband)
    3. Encode to DCAE latent
    4. Apply register-aware correction with translator
    5. Decode back to audio
    """

    def __init__(
        self,
        translator_checkpoint: str,
        dcae_checkpoint: str,
        device: str = "cuda",
        pitch_shift_backend: str = "librosa",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pitch_shift_backend = pitch_shift_backend

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
        config = checkpoint.get('config', {})

        # Determine model type
        model_type = config.get('model_type', 'residual')
        if model_type == 'direct':
            self.translator = RegisterAwareTranslatorDirect()
        elif model_type == 'large':
            self.translator = RegisterAwareTranslatorLarge()
        else:
            self.translator = RegisterAwareTranslator()

        self.translator.load_state_dict(state_dict)
        self.translator = self.translator.to(self.device).eval()

        print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")
        print(f"Model type: {model_type}")

    def _pitch_shift_audio(
        self,
        audio: torch.Tensor,
        sr: int,
        shift_semitones: int,
    ) -> torch.Tensor:
        """Apply pitch shift to audio."""
        if shift_semitones == 0:
            return audio

        import librosa

        audio_np = audio.numpy()
        shifted_channels = []

        for ch in range(audio_np.shape[0]):
            shifted = librosa.effects.pitch_shift(
                audio_np[ch],
                sr=sr,
                n_steps=shift_semitones,
            )
            shifted_channels.append(shifted)

        return torch.from_numpy(np.stack(shifted_channels))

    def _estimate_pitch(self, audio: torch.Tensor, sr: int) -> int:
        """Estimate the dominant pitch of the audio."""
        try:
            import librosa
            # Use librosa's pitch detection
            mono = audio.mean(dim=0).numpy()
            pitches, magnitudes = librosa.piptrack(y=mono, sr=sr)

            # Get the most confident pitch
            pitch_idx = magnitudes.argmax()
            pitch_hz = pitches.flatten()[pitch_idx]

            if pitch_hz > 0:
                # Convert Hz to MIDI
                midi_pitch = int(12 * np.log2(pitch_hz / 440) + 69)
                return max(24, min(96, midi_pitch))
        except Exception:
            pass

        # Default to middle C if detection fails
        return 60

    @torch.no_grad()
    def convert(
        self,
        input_path: str,
        output_path: str,
        shift_semitones: int = 0,
        target_pitch: int = None,
    ) -> str:
        """
        Apply register-aware pitch shift.

        Args:
            input_path: Input audio file
            output_path: Output audio file
            shift_semitones: Shift amount in semitones
            target_pitch: Target MIDI pitch (auto-detected if None)

        Returns:
            Output path
        """
        # Load audio
        audio, sr = torchaudio.load(input_path)

        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        # Estimate source pitch if target not specified
        if target_pitch is None:
            source_pitch = self._estimate_pitch(audio, sr)
            target_pitch = source_pitch + shift_semitones
            target_pitch = max(24, min(96, target_pitch))
        else:
            source_pitch = target_pitch - shift_semitones

        print(f"  Source pitch: {source_pitch} (MIDI), Target pitch: {target_pitch}")

        # Apply pitch shift to audio
        shifted_audio = self._pitch_shift_audio(audio, sr, shift_semitones)

        # Encode to latent
        shifted_audio = shifted_audio.unsqueeze(0).to(self.device)
        latent = self.dcae.encode(shifted_audio, sr=sr)

        # Apply register-aware correction
        target_pitch_tensor = torch.tensor([target_pitch], device=self.device)
        shift_tensor = torch.tensor([float(shift_semitones)], device=self.device)

        corrected_latent = self.translator(latent, target_pitch_tensor, shift_tensor)

        # Decode to audio
        corrected_audio = self.dcae.decode(corrected_latent)

        # Normalize and save
        corrected_audio = corrected_audio.squeeze(0).cpu()
        corrected_audio = corrected_audio / (corrected_audio.abs().max() + 1e-8) * 0.9

        torchaudio.save(output_path, corrected_audio, 44100)

        print(f"  Saved: {output_path}")
        return output_path

    @torch.no_grad()
    def convert_with_comparison(
        self,
        input_path: str,
        output_dir: str,
        shift_semitones: int = 0,
    ) -> dict:
        """
        Generate comparison outputs:
        - Original
        - Naive pitch shift (no correction)
        - Register-aware pitch shift (with correction)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(input_path).stem

        # Load audio
        audio, sr = torchaudio.load(input_path)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        # Estimate pitch
        source_pitch = self._estimate_pitch(audio, sr)
        target_pitch = max(24, min(96, source_pitch + shift_semitones))

        # 1. Save original
        original_path = output_dir / f"{stem}_original.wav"
        torchaudio.save(str(original_path), audio, sr)

        # 2. Naive pitch shift (no correction)
        shifted_audio = self._pitch_shift_audio(audio, sr, shift_semitones)
        naive_path = output_dir / f"{stem}_naive_shift{shift_semitones:+d}.wav"
        shifted_normalized = shifted_audio / (shifted_audio.abs().max() + 1e-8) * 0.9
        torchaudio.save(str(naive_path), shifted_normalized, sr)

        # 3. Register-aware pitch shift
        shifted_audio = shifted_audio.unsqueeze(0).to(self.device)
        latent = self.dcae.encode(shifted_audio, sr=sr)

        target_pitch_tensor = torch.tensor([target_pitch], device=self.device)
        shift_tensor = torch.tensor([float(shift_semitones)], device=self.device)

        corrected_latent = self.translator(latent, target_pitch_tensor, shift_tensor)
        corrected_audio = self.dcae.decode(corrected_latent)

        corrected_audio = corrected_audio.squeeze(0).cpu()
        corrected_audio = corrected_audio / (corrected_audio.abs().max() + 1e-8) * 0.9

        corrected_path = output_dir / f"{stem}_corrected_shift{shift_semitones:+d}.wav"
        torchaudio.save(str(corrected_path), corrected_audio, 44100)

        return {
            'original': str(original_path),
            'naive': str(naive_path),
            'corrected': str(corrected_path),
            'source_pitch': source_pitch,
            'target_pitch': target_pitch,
            'shift': shift_semitones,
        }


class StudentInference:
    """
    Inference using student model directly.

    Much faster, suitable for real-time/VST use.
    """

    def __init__(
        self,
        student_checkpoint: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        print("Loading student model...")
        checkpoint = torch.load(student_checkpoint, map_location=self.device)

        self.model = PitchShiftStudentModel()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device).eval()

        print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )

        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=1025,
            n_mels=128,
        )

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=2048,
            hop_length=512,
        )

    @torch.no_grad()
    def convert(
        self,
        input_path: str,
        output_path: str,
        shift_semitones: int = 0,
        target_pitch: int = 60,
    ) -> str:
        """Apply register-aware pitch shift using student model."""
        # Load audio
        audio, sr = torchaudio.load(input_path)

        # Resample if needed
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio = resampler(audio)
            sr = 44100

        # Convert to mono for mel
        mono = audio.mean(dim=0, keepdim=True)

        # To mel spectrogram
        mel = self.mel_transform(mono)  # [1, n_mels, T]
        mel = mel.unsqueeze(0).to(self.device)  # [1, 1, n_mels, T]

        # Apply model
        target_pitch_tensor = torch.tensor([target_pitch], device=self.device)
        shift_tensor = torch.tensor([float(shift_semitones)], device=self.device)

        corrected_mel = self.model(mel, target_pitch_tensor, shift_tensor)

        # Convert back to audio
        corrected_mel = corrected_mel.squeeze(0).squeeze(0).cpu()  # [n_mels, T]

        # Inverse mel
        spec = self.inverse_mel(corrected_mel)
        corrected_audio = self.griffin_lim(spec)

        # Make stereo
        corrected_audio = corrected_audio.unsqueeze(0).repeat(2, 1)

        # Normalize
        corrected_audio = corrected_audio / (corrected_audio.abs().max() + 1e-8) * 0.9

        torchaudio.save(output_path, corrected_audio, 44100)

        print(f"Saved: {output_path}")
        return output_path


class BatchProcessor:
    """Process multiple files."""

    def __init__(self, converter):
        self.converter = converter

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        shift_semitones: int = 0,
        extension: str = ".wav",
    ):
        """Process all audio files in directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.glob(f"*{extension}"))
        print(f"Found {len(files)} files")

        results = []
        for f in files:
            output_path = output_dir / f"{f.stem}_shift{shift_semitones:+d}{extension}"
            try:
                result = self.converter.convert(
                    str(f),
                    str(output_path),
                    shift_semitones=shift_semitones,
                )
                results.append({'input': str(f), 'output': result, 'success': True})
            except Exception as e:
                print(f"Error processing {f}: {e}")
                results.append({'input': str(f), 'error': str(e), 'success': False})

        print(f"\nDone! Output: {output_dir}")
        print(f"Successful: {sum(1 for r in results if r['success'])}/{len(results)}")

        return results


def export_onnx(model, output_path: str):
    """Export student model to ONNX for VST/deployment."""
    print(f"Exporting to ONNX: {output_path}")

    model.eval().cpu()

    # Dummy inputs
    mel = torch.randn(1, 1, 128, 256)
    target_pitch = torch.tensor([60])
    shift_amount = torch.tensor([0.0])

    torch.onnx.export(
        model,
        (mel, target_pitch, shift_amount),
        output_path,
        input_names=['mel_input', 'target_pitch', 'shift_amount'],
        output_names=['mel_output'],
        dynamic_axes={
            'mel_input': {0: 'batch', 3: 'time'},
            'mel_output': {0: 'batch', 3: 'time'},
        },
        opset_version=14,
    )

    print(f"Exported: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Register-Aware Pitch Shift Inference")

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

    # I/O args
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output audio file or directory')
    parser.add_argument('--batch', action='store_true',
                        help='Process entire directory')

    # Pitch shift args
    parser.add_argument('--shift', type=int, default=0,
                        help='Pitch shift in semitones')
    parser.add_argument('--target_pitch', type=int, default=None,
                        help='Target MIDI pitch (auto-detected if not specified)')

    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison outputs (original, naive, corrected)')

    # Export
    parser.add_argument('--export_onnx', type=str,
                        help='Export model to ONNX file (student mode only)')

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
            device=args.device,
        )

    # Export if requested
    if args.export_onnx:
        if args.mode == 'student':
            export_onnx(converter.model, args.export_onnx)
        else:
            print("ONNX export only supported for student mode")
        return

    # Process
    if args.compare and args.mode == 'teacher':
        result = converter.convert_with_comparison(
            args.input,
            args.output,
            shift_semitones=args.shift,
        )
        print("\nComparison outputs:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    elif args.batch:
        processor = BatchProcessor(converter)
        processor.process_directory(
            args.input,
            args.output,
            shift_semitones=args.shift,
        )
    else:
        converter.convert(
            args.input,
            args.output,
            shift_semitones=args.shift,
            target_pitch=args.target_pitch,
        )


if __name__ == "__main__":
    main()
