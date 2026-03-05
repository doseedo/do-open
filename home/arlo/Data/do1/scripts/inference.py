#!/usr/bin/env python3
"""
DO1 Inference — All Use Cases

Every use case maps to the same three tensors:
  x_cond [B, 64, T]   — what to work with
  x_ref  [B, 64, T']  — what it should sound like
  mask   [B, 1, T]     — 1=preserve, 0=generate

Usage:
  python inference.py separate --mix song.wav --ref violin_sample.wav
  python inference.py transfer --source guitar.wav --ref trumpet_sample.wav
  python inference.py generate --ref violin_sample.wav --duration 10
  python inference.py midi2audio --midi melody.mid --ref violin_sample.wav
  python inference.py fx_remove --input wet_vocal.wav --ref dry_vocal_sample.wav
  python inference.py fx_match --input dry_vocal.wav --ref reference_production.wav
  python inference.py voice_convert --input singer_a.wav --ref singer_b_sample.wav
  python inference.py inpaint --input recording.wav --start 2.0 --end 4.0 --ref same_session.wav
  python inference.py continue --input 8bars.wav --duration 5 --ref same_session.wav
  python inference.py drums --midi drum_pattern.mid --ref drum_kit_sample.wav
  python inference.py accompaniment --context drums.wav --ref bass_sample.wav
  python inference.py session --mix full_song.wav --out session_dir/

Requires:
  - Trained DO1 checkpoint
  - ACE-Step 1.5 VAE checkpoint
  - For MIDI tasks: latent synth checkpoint
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torchaudio
import numpy as np

# VAE format: [B, 64, T] at 25Hz, 48kHz stereo
VAE_SR = 48000
VAE_HZ = 25.0
VAE_DIM = 64


# ============================================================================
# VAE ENCODE / DECODE
# ============================================================================

class AudioCodec:
    """Wraps ACE-Step 1.5 VAE for encode/decode."""

    def __init__(self, vae_path: str, device: str = "cuda"):
        from diffusers import AutoencoderOobleck
        self.device = torch.device(device)
        self.vae = AutoencoderOobleck.from_pretrained(vae_path)
        self.vae = self.vae.eval().to(self.device)

    @torch.no_grad()
    def encode(self, audio_path: str) -> torch.Tensor:
        """audio file → z [1, 64, T]"""
        waveform, sr = torchaudio.load(audio_path)
        # Resample to 48kHz if needed
        if sr != VAE_SR:
            waveform = torchaudio.functional.resample(waveform, sr, VAE_SR)
        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)
        # Encode
        waveform = waveform.unsqueeze(0).to(self.device)  # [1, 2, samples]
        dist = self.vae.encode(waveform)
        z = dist.latent_dist.sample()  # [1, 64, T]
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor, output_path: str):
        """z [1, 64, T] → audio file"""
        z = z.to(self.device)
        audio = self.vae.decode(z).sample  # [1, 2, samples]
        audio = audio.squeeze(0).cpu()  # [2, samples]
        # Normalize
        audio = audio / (audio.abs().max() + 1e-8) * 0.95
        torchaudio.save(output_path, audio, VAE_SR)
        return output_path

    def encode_numpy(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """numpy array → z [1, 64, T]"""
        waveform = torch.from_numpy(audio).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if sr != VAE_SR:
            waveform = torchaudio.functional.resample(waveform, sr, VAE_SR)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        waveform = waveform / (waveform.abs().max() + 1e-8)
        waveform = waveform.unsqueeze(0).to(self.device)
        dist = self.vae.encode(waveform)
        return dist.latent_dist.sample()


# ============================================================================
# DO1 MODEL WRAPPER
# ============================================================================

class DO1:
    """Wraps DO1 model for inference."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        # TODO: load from actual checkpoint once architecture is updated to 1.5 VAE
        # For now this is the interface
        self.model = self._load_model(checkpoint_path)
        self.model = self.model.eval().to(self.device)

    def _load_model(self, path):
        """Load DO1 model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            config = {}
        # TODO: instantiate model with correct config for 1.5 VAE format
        # model = DO1Model(**config)
        # model.load_state_dict(state_dict)
        # return model
        raise NotImplementedError("Update model architecture for 1.5 VAE format [64, T]")

    @torch.no_grad()
    def __call__(
        self,
        x_cond: torch.Tensor,
        x_ref: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 50,
        cfg_scale: float = 2.0,
        sigma: float = 0.0,
    ) -> torch.Tensor:
        """
        Run DO1 inference.

        Args:
            x_cond:    [B, 64, T]  content/context
            x_ref:     [B, 64, T'] style/timbre reference
            mask:      [B, 1, T]   1=preserve, 0=generate
            num_steps: ODE solver steps
            cfg_scale: classifier-free guidance scale
            sigma:     expression noise on x_cond (0=exact, 0.3=expressive)

        Returns:
            z_output:  [B, 64, T]
        """
        x_cond = x_cond.to(self.device)
        x_ref = x_ref.to(self.device)
        mask = mask.to(self.device)

        # Expression control: add noise to x_cond
        if sigma > 0:
            x_cond = x_cond + sigma * torch.randn_like(x_cond)

        B, C, T = x_cond.shape

        # Start from noise
        x = torch.randn(B, C, T, device=self.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=self.device)

            # Conditional pass
            v_cond = self.model(x, x_cond, x_ref, mask, t)

            if cfg_scale > 1.0:
                # Unconditional pass (x_ref zeroed)
                v_uncond = self.model(x, x_cond, torch.zeros_like(x_ref), mask, t)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            x = x + v * dt

        return x


# ============================================================================
# UTILITY
# ============================================================================

def seconds_to_frames(seconds: float) -> int:
    return int(seconds * VAE_HZ)

def frames_to_seconds(frames: int) -> float:
    return frames / VAE_HZ

def match_length(z: torch.Tensor, target_T: int) -> torch.Tensor:
    """Truncate or pad z along time dimension."""
    T = z.shape[-1]
    if T >= target_T:
        return z[..., :target_T]
    return torch.nn.functional.pad(z, (0, target_T - T))

def ones_mask(T: int) -> torch.Tensor:
    return torch.ones(1, 1, T)

def zeros_mask(T: int) -> torch.Tensor:
    return torch.zeros(1, 1, T)

def partial_mask(T: int, start_frame: int, end_frame: int) -> torch.Tensor:
    """1 everywhere except zeros in [start_frame, end_frame]."""
    mask = torch.ones(1, 1, T)
    mask[:, :, start_frame:end_frame] = 0.0
    return mask


# ============================================================================
# USE CASES
# ============================================================================

def cmd_separate(args, do1: DO1, codec: AudioCodec):
    """Source separation: extract one instrument from a mix."""
    z_mix = codec.encode(args.mix)
    z_ref = codec.encode(args.ref)
    T = z_mix.shape[-1]

    # Step 1: Rough separation
    z_rough = do1(
        x_cond=z_mix,
        x_ref=z_ref,
        mask=zeros_mask(T),
        num_steps=args.steps,
        cfg_scale=args.cfg,
    )

    if args.repair:
        # Step 2: Repair pass
        z_clean = do1(
            x_cond=z_rough,
            x_ref=z_ref,
            mask=ones_mask(T),
            num_steps=args.steps,
            cfg_scale=args.cfg,
        )
    else:
        z_clean = z_rough

    codec.decode(z_clean, args.output)
    print(f"Separated stem saved to {args.output}")


def cmd_transfer(args, do1: DO1, codec: AudioCodec):
    """Timbre transfer: change instrument on existing recording."""
    z_source = codec.encode(args.source)
    z_ref = codec.encode(args.ref)
    T = z_source.shape[-1]

    z_output = do1(
        x_cond=z_source,
        x_ref=z_ref,
        mask=ones_mask(T),
        num_steps=args.steps,
        cfg_scale=args.cfg,
        sigma=args.expression,
    )

    codec.decode(z_output, args.output)
    print(f"Transferred output saved to {args.output}")


def cmd_generate(args, do1: DO1, codec: AudioCodec):
    """Unconditional generation from reference."""
    z_ref = codec.encode(args.ref)
    T = seconds_to_frames(args.duration)

    z_output = do1(
        x_cond=torch.zeros(1, VAE_DIM, T, device=do1.device),
        x_ref=z_ref,
        mask=zeros_mask(T),
        num_steps=args.steps,
        cfg_scale=args.cfg,
    )

    codec.decode(z_output, args.output)
    print(f"Generated {args.duration}s audio saved to {args.output}")


def cmd_midi2audio(args, do1: DO1, codec: AudioCodec):
    """MIDI → realistic instrument audio via latent synth."""
    z_ref = codec.encode(args.ref)

    # Render MIDI through latent synth
    # TODO: load latent synth and render
    # z_synth = latent_synth.render(midi_path, random_vcf=False)
    # For now, placeholder:
    raise NotImplementedError(
        "Latent synth needs retraining for 1.5 VAE format. "
        "Once retrained: midi → latent_synth → z_synth → DO1 with mask=ones"
    )

    # T = z_synth.shape[-1]
    # z_output = do1(
    #     x_cond=z_synth,
    #     x_ref=z_ref,
    #     mask=ones_mask(T),
    #     num_steps=args.steps,
    #     cfg_scale=args.cfg,
    #     sigma=args.expression,
    # )
    # codec.decode(z_output, args.output)


def cmd_fx_remove(args, do1: DO1, codec: AudioCodec):
    """FX removal: strip reverb/delay/distortion."""
    z_wet = codec.encode(args.input)
    z_ref = codec.encode(args.ref)  # dry reference
    T = z_wet.shape[-1]

    z_dry = do1(
        x_cond=z_wet,
        x_ref=z_ref,
        mask=ones_mask(T),
        num_steps=args.steps,
        cfg_scale=args.cfg,
    )

    codec.decode(z_dry, args.output)
    print(f"FX removed, saved to {args.output}")


def cmd_fx_match(args, do1: DO1, codec: AudioCodec):
    """FX matching: make input sound like reference's production."""
    z_input = codec.encode(args.input)
    z_ref = codec.encode(args.ref)  # reference with desired FX character
    T = z_input.shape[-1]

    z_matched = do1(
        x_cond=z_input,
        x_ref=z_ref,
        mask=ones_mask(T),
        num_steps=args.steps,
        cfg_scale=args.cfg,
    )

    codec.decode(z_matched, args.output)
    print(f"FX matched, saved to {args.output}")


def cmd_voice_convert(args, do1: DO1, codec: AudioCodec):
    """Voice conversion: change singer identity."""
    z_singer_a = codec.encode(args.input)
    z_singer_b = codec.encode(args.ref)
    T = z_singer_a.shape[-1]

    z_converted = do1(
        x_cond=z_singer_a,
        x_ref=z_singer_b,
        mask=ones_mask(T),
        num_steps=args.steps,
        cfg_scale=args.cfg,
    )

    codec.decode(z_converted, args.output)
    print(f"Voice converted, saved to {args.output}")


def cmd_inpaint(args, do1: DO1, codec: AudioCodec):
    """Inpainting: regenerate a section of audio."""
    z_input = codec.encode(args.input)
    z_ref = codec.encode(args.ref)
    T = z_input.shape[-1]

    start_frame = seconds_to_frames(args.start)
    end_frame = seconds_to_frames(args.end)

    # Zero out the inpaint region in x_cond
    z_cond = z_input.clone()
    z_cond[:, :, start_frame:end_frame] = 0.0

    mask = partial_mask(T, start_frame, end_frame)

    z_output = do1(
        x_cond=z_cond,
        x_ref=z_ref,
        mask=mask,
        num_steps=args.steps,
        cfg_scale=args.cfg,
    )

    codec.decode(z_output, args.output)
    print(f"Inpainted {args.start:.1f}s-{args.end:.1f}s, saved to {args.output}")


def cmd_continue(args, do1: DO1, codec: AudioCodec):
    """Continuation: extend a recording."""
    z_input = codec.encode(args.input)
    z_ref = codec.encode(args.ref) if args.ref else z_input
    T_original = z_input.shape[-1]
    T_extend = seconds_to_frames(args.duration)
    T_total = T_original + T_extend

    # Pad input with zeros for the extension
    z_cond = torch.nn.functional.pad(z_input, (0, T_extend))

    # Mask: ones for original, zeros for extension
    mask = torch.ones(1, 1, T_total)
    mask[:, :, T_original:] = 0.0

    z_output = do1(
        x_cond=z_cond,
        x_ref=z_ref,
        mask=mask,
        num_steps=args.steps,
        cfg_scale=args.cfg,
    )

    codec.decode(z_output, args.output)
    original_sec = frames_to_seconds(T_original)
    total_sec = frames_to_seconds(T_total)
    print(f"Extended {original_sec:.1f}s → {total_sec:.1f}s, saved to {args.output}")


def cmd_drums(args, do1: DO1, codec: AudioCodec):
    """Drums from MIDI grid: render drum pattern with specific kit."""
    z_ref = codec.encode(args.ref)

    # Same as midi2audio but specifically for drums
    # TODO: latent synth with drum-appropriate synthesis
    raise NotImplementedError(
        "Latent synth needs retraining for 1.5 VAE format. "
        "Once retrained: drum MIDI → latent_synth (noise bursts for hits) → DO1"
    )


def cmd_accompaniment(args, do1: DO1, codec: AudioCodec):
    """Generate accompaniment that fits existing audio."""
    z_context = codec.encode(args.context)
    z_ref = codec.encode(args.ref)
    T = z_context.shape[-1]

    z_accompaniment = do1(
        x_cond=z_context,
        x_ref=z_ref,
        mask=zeros_mask(T),
        num_steps=args.steps,
        cfg_scale=args.cfg,
        sigma=args.expression,
    )

    codec.decode(z_accompaniment, args.output)
    print(f"Accompaniment saved to {args.output}")


def cmd_session(args, do1: DO1, codec: AudioCodec):
    """Full audio-to-session pipeline."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {args.mix}")
    z_mix = codec.encode(args.mix)
    T = z_mix.shape[-1]
    duration = frames_to_seconds(T)
    print(f"Duration: {duration:.1f}s ({T} frames)")

    # Step 1: Classify instruments present
    # TODO: lightweight classifier on z_mix
    # For now use a default set or user-specified instruments
    if args.instruments:
        instruments = args.instruments.split(",")
    else:
        instruments = ["vocals", "drums", "bass", "other"]
    print(f"Separating: {instruments}")

    # Step 2: Reference library — user provides or we use defaults
    ref_dir = Path(args.ref_dir) if args.ref_dir else None

    stems = {}
    for inst in instruments:
        print(f"\n--- Separating: {inst} ---")

        # Get reference for this instrument
        if ref_dir and (ref_dir / f"{inst}.wav").exists():
            z_ref = codec.encode(str(ref_dir / f"{inst}.wav"))
        else:
            print(f"  No reference for {inst}, using generic separation")
            # Fallback: use a portion of the mix as self-reference
            # This is suboptimal but functional
            z_ref = z_mix[:, :, :min(T, seconds_to_frames(5))]

        # Separate
        z_rough = do1(
            x_cond=z_mix,
            x_ref=z_ref,
            mask=zeros_mask(T),
            num_steps=args.steps,
            cfg_scale=args.cfg,
        )

        # Repair
        z_clean = do1(
            x_cond=z_rough,
            x_ref=z_ref,
            mask=ones_mask(T),
            num_steps=args.steps,
            cfg_scale=args.cfg,
        )

        # Save stem
        stem_path = out_dir / f"{inst}.wav"
        codec.decode(z_clean, str(stem_path))
        stems[inst] = z_clean
        print(f"  Saved: {stem_path}")

    # Step 3: Canonicalize for MIDI extraction (optional)
    if args.extract_midi:
        print("\n--- Extracting MIDI ---")
        for inst, z_clean in stems.items():
            if inst in ["vocals", "drums", "bass", "other"]:
                # Decode to audio for basic_pitch
                temp_path = out_dir / f"{inst}_canonical.wav"
                codec.decode(z_clean, str(temp_path))

                try:
                    from basic_pitch.inference import predict
                    model_output, midi_data, note_events = predict(str(temp_path))
                    midi_path = out_dir / f"{inst}.mid"
                    midi_data.write(str(midi_path))
                    print(f"  MIDI saved: {midi_path}")
                except ImportError:
                    print(f"  basic_pitch not installed, skipping MIDI for {inst}")
                except Exception as e:
                    print(f"  MIDI extraction failed for {inst}: {e}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Session exported to: {out_dir}")
    print(f"Stems: {list(stems.keys())}")
    print(f"Duration: {duration:.1f}s")


def cmd_style_transfer(args, do1: DO1, codec: AudioCodec):
    """Style transfer: keep melody, change playing style."""
    z_source = codec.encode(args.source)
    z_ref = codec.encode(args.ref)
    T = z_source.shape[-1]

    z_output = do1(
        x_cond=z_source,
        x_ref=z_ref,
        mask=ones_mask(T),
        num_steps=args.steps,
        cfg_scale=args.cfg,
        sigma=args.expression,
    )

    codec.decode(z_output, args.output)
    print(f"Style transferred, saved to {args.output}")


def cmd_reharmonize(args, do1: DO1, codec: AudioCodec):
    """Reharmonize: change chords under a melody."""
    # New harmony provided as audio (e.g. from synth render of new chords)
    z_new_harmony = codec.encode(args.harmony)
    z_ref = codec.encode(args.ref)  # original character to preserve
    T = z_new_harmony.shape[-1]

    z_output = do1(
        x_cond=z_new_harmony,
        x_ref=z_ref,
        mask=ones_mask(T),
        num_steps=args.steps,
        cfg_scale=args.cfg,
    )

    codec.decode(z_output, args.output)
    print(f"Reharmonized, saved to {args.output}")


# ============================================================================
# CLI
# ============================================================================

def add_common_args(parser):
    parser.add_argument("--model", type=str, required=True, help="DO1 checkpoint path")
    parser.add_argument("--vae", type=str, required=True, help="ACE-Step 1.5 VAE path")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output path")
    parser.add_argument("--steps", type=int, default=50, help="ODE solver steps")
    parser.add_argument("--cfg", type=float, default=2.0, help="CFG scale (1.0=no guidance)")
    parser.add_argument("--expression", type=float, default=0.0, help="Expression noise σ (0=exact)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")


def main():
    parser = argparse.ArgumentParser(description="DO1 Inference — Universal Latent Audio Operator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # separate
    p = subparsers.add_parser("separate", help="Extract instrument from mix")
    add_common_args(p)
    p.add_argument("--mix", required=True, help="Input mix audio")
    p.add_argument("--ref", required=True, help="Reference of target instrument")
    p.add_argument("--no-repair", dest="repair", action="store_false", help="Skip repair pass")

    # transfer
    p = subparsers.add_parser("transfer", help="Change instrument timbre")
    add_common_args(p)
    p.add_argument("--source", required=True, help="Source audio (content)")
    p.add_argument("--ref", required=True, help="Reference audio (target timbre)")

    # generate
    p = subparsers.add_parser("generate", help="Generate from reference")
    add_common_args(p)
    p.add_argument("--ref", required=True, help="Reference audio (style/timbre)")
    p.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")

    # midi2audio
    p = subparsers.add_parser("midi2audio", help="MIDI → realistic audio")
    add_common_args(p)
    p.add_argument("--midi", required=True, help="Input MIDI file")
    p.add_argument("--ref", required=True, help="Reference instrument audio")

    # fx_remove
    p = subparsers.add_parser("fx_remove", help="Remove FX from audio")
    add_common_args(p)
    p.add_argument("--input", required=True, help="Wet input audio")
    p.add_argument("--ref", required=True, help="Dry reference audio")

    # fx_match
    p = subparsers.add_parser("fx_match", help="Match FX character of reference")
    add_common_args(p)
    p.add_argument("--input", required=True, help="Input audio")
    p.add_argument("--ref", required=True, help="Reference with desired FX character")

    # voice_convert
    p = subparsers.add_parser("voice_convert", help="Change singer identity")
    add_common_args(p)
    p.add_argument("--input", required=True, help="Input vocal (singer A)")
    p.add_argument("--ref", required=True, help="Reference vocal (singer B)")

    # inpaint
    p = subparsers.add_parser("inpaint", help="Regenerate a section")
    add_common_args(p)
    p.add_argument("--input", required=True, help="Input audio with bad section")
    p.add_argument("--ref", required=True, help="Style reference")
    p.add_argument("--start", type=float, required=True, help="Start of region to inpaint (seconds)")
    p.add_argument("--end", type=float, required=True, help="End of region to inpaint (seconds)")

    # continue
    p = subparsers.add_parser("continue", help="Extend a recording")
    add_common_args(p)
    p.add_argument("--input", required=True, help="Input audio to extend")
    p.add_argument("--ref", type=str, default=None, help="Style reference (default: input itself)")
    p.add_argument("--duration", type=float, default=5.0, help="Seconds to add")

    # drums
    p = subparsers.add_parser("drums", help="Drums from MIDI grid")
    add_common_args(p)
    p.add_argument("--midi", required=True, help="Drum MIDI pattern")
    p.add_argument("--ref", required=True, help="Drum kit reference audio")

    # accompaniment
    p = subparsers.add_parser("accompaniment", help="Generate part that fits existing audio")
    add_common_args(p)
    p.add_argument("--context", required=True, help="Existing audio (e.g. drums)")
    p.add_argument("--ref", required=True, help="Reference for new instrument (e.g. bass sample)")

    # session
    p = subparsers.add_parser("session", help="Full audio-to-session pipeline")
    p.add_argument("--model", type=str, required=True, help="DO1 checkpoint path")
    p.add_argument("--vae", type=str, required=True, help="ACE-Step 1.5 VAE path")
    p.add_argument("--mix", required=True, help="Input full mix")
    p.add_argument("--out", required=True, help="Output session directory")
    p.add_argument("--ref-dir", type=str, default=None, help="Directory with instrument references")
    p.add_argument("--instruments", type=str, default=None, help="Comma-separated instruments to extract")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--extract-midi", action="store_true", help="Extract MIDI from separated stems")
    p.add_argument("--device", type=str, default="cuda")

    # style_transfer
    p = subparsers.add_parser("style_transfer", help="Keep melody, change playing style")
    add_common_args(p)
    p.add_argument("--source", required=True, help="Source audio (content)")
    p.add_argument("--ref", required=True, help="Reference audio (target style)")

    # reharmonize
    p = subparsers.add_parser("reharmonize", help="Change chords under melody")
    add_common_args(p)
    p.add_argument("--harmony", required=True, help="New harmony audio (e.g. synth render)")
    p.add_argument("--ref", required=True, help="Original character reference")

    args = parser.parse_args()

    # Load models
    print("Loading VAE...")
    codec = AudioCodec(args.vae, device=args.device)
    print("Loading DO1...")
    do1 = DO1(args.model, device=args.device)

    # Dispatch
    commands = {
        "separate": cmd_separate,
        "transfer": cmd_transfer,
        "generate": cmd_generate,
        "midi2audio": cmd_midi2audio,
        "fx_remove": cmd_fx_remove,
        "fx_match": cmd_fx_match,
        "voice_convert": cmd_voice_convert,
        "inpaint": cmd_inpaint,
        "continue": cmd_continue,
        "drums": cmd_drums,
        "accompaniment": cmd_accompaniment,
        "session": cmd_session,
        "style_transfer": cmd_style_transfer,
        "reharmonize": cmd_reharmonize,
    }

    cmd_fn = commands[args.command]
    cmd_fn(args, do1, codec)


if __name__ == "__main__":
    main()