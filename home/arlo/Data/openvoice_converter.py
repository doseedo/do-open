#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenVoice Voice Conversion Wrapper

Zero-shot voice conversion using OpenVoice V2.
Converts input audio to match a reference speaker's voice.

Usage:
    python openvoice_converter.py \
        --input generated_vocals.wav \
        --speaker-ref my_voice.wav \
        --output converted.wav
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add OpenVoice to path
sys.path.insert(0, '/home/arlo/Data/OpenVoice')


def convert_voice_openvoice(
    input_audio_path: str,
    speaker_ref_path: str,
    output_path: str,
    language: str = 'EN',
    device: str = 'cuda:0'
):
    """
    Convert input audio to match reference speaker voice using OpenVoice.

    Args:
        input_audio_path: Path to input audio to convert
        speaker_ref_path: Path to reference speaker audio
        output_path: Path to save converted audio
        language: Language code (EN, ES, FR, ZH, JP, KR)
        device: Torch device

    Returns:
        Path to converted audio file
    """
    start_time = time.time()

    print("=" * 80)
    print("OpenVoice V2 Voice Conversion")
    print("=" * 80)
    print(f"Input Audio:      {input_audio_path}")
    print(f"Speaker Ref:      {speaker_ref_path}")
    print(f"Output:           {output_path}")
    print(f"Language:         {language}")
    print(f"Device:           {device}")
    print("=" * 80)

    # Validate inputs
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

    if not os.path.exists(speaker_ref_path):
        raise FileNotFoundError(f"Speaker reference not found: {speaker_ref_path}")

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    try:
        # Import OpenVoice
        print("\n[1/4] Loading OpenVoice V2...")
        load_start = time.time()

        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter

        # Initialize models
        ckpt_converter = '/home/arlo/Data/OpenVoice/checkpoints/checkpoints/converter'
        device = device
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        load_time = time.time() - load_start
        print(f"   ✓ OpenVoice loaded in {load_time:.2f}s")

        # Extract speaker embedding from reference
        print("\n[2/4] Extracting speaker embedding...")
        embed_start = time.time()

        reference_speaker = speaker_ref_path
        target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

        embed_time = time.time() - embed_start
        print(f"   ✓ Speaker embedding extracted in {embed_time:.2f}s")

        # Convert voice
        print("\n[3/4] Converting voice...")
        convert_start = time.time()

        # Get source speaker embedding
        source_se, _ = se_extractor.get_se(input_audio_path, tone_color_converter, target_dir='processed', vad=True)

        # Convert tone color
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=input_audio_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            message=encode_message
        )

        convert_time = time.time() - convert_start
        print(f"   ✓ Voice converted in {convert_time:.2f}s")

        # Verify output
        print("\n[4/4] Verifying output...")
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"   ✓ Output file created: {file_size:,} bytes")
        else:
            raise FileNotFoundError(f"Output file was not created: {output_path}")

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"✅ Voice conversion complete!")
        print(f"   Output: {output_path}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   - Model load: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
        print(f"   - Speaker embed: {embed_time:.2f}s ({embed_time/total_time*100:.1f}%)")
        print(f"   - Conversion: {convert_time:.2f}s ({convert_time/total_time*100:.1f}%)")
        print("=" * 80)

        return output_path

    except Exception as e:
        print(f"\n❌ Voice conversion failed: {e}")
        print(f"\nNote: Make sure OpenVoice is properly installed and models are downloaded.")
        print(f"Run: cd /home/arlo/Data/OpenVoice && python -m download_models")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='OpenVoice V2 Voice Conversion - Zero-shot voice cloning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python openvoice_converter.py \
      --input generated_vocals.wav \
      --speaker-ref my_voice.wav \
      --output converted.wav

  # Convert to Spanish speaker
  python openvoice_converter.py \
      --input generated_vocals.wav \
      --speaker-ref spanish_speaker.wav \
      --language ES \
      --output converted.wav

  # Use specific GPU
  python openvoice_converter.py \
      --input generated_vocals.wav \
      --speaker-ref my_voice.wav \
      --output converted.wav \
      --device cuda:1
        """
    )

    # Required
    parser.add_argument('--input', type=str, required=True, help='Input audio file to convert')
    parser.add_argument('--speaker-ref', type=str, required=True, help='Reference speaker audio for voice cloning')
    parser.add_argument('--output', type=str, required=True, help='Output audio file path')

    # Optional
    parser.add_argument('--language', type=str, default='EN',
                       choices=['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'],
                       help='Language code (default: EN for English)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda:0)')

    args = parser.parse_args()

    # Convert voice
    try:
        convert_voice_openvoice(
            input_audio_path=args.input,
            speaker_ref_path=args.speaker_ref,
            output_path=args.output,
            language=args.language,
            device=args.device
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
