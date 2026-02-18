#!/usr/bin/env python3
"""
Gradio Web UI for ACE-Step generation (ControlBranch-ready)

- Loads the same Pipeline you trained with and restores hparams from the Lightning .ckpt
- Works with ctrl_enc + ctrlnet residual injection (ControlBranch1D)
- Instrument-token CFG (ON vs OFF) + sharper PR masking like previews
- Proper EnCodec gating (keeps tokens LongTensor)
"""

import sys, os, argparse, subprocess, json, random, time, shutil, tempfile, hashlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
import torchaudio
import gradio as gr
import pretty_midi
import mido
from mido import MidiFile, MidiTrack, MetaMessage

torch.set_float32_matmul_precision("high")

# ------------------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------------------
sys.path.append('/home/arlo/Data')  # folder that has trainer_performer.py
sys.path.append('/home/arlo/Data/ACE-Step')  # Add ACE-Step directory for acestep imports

try:
    from trainer_performer_backup import Pipeline  # if you kept a backup
except Exception:
    from trainer_performer import Pipeline

from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000,  320

# Soundfont mapping for different instrument groups
INSTRUMENT_SOUNDFONTS = {
    # Brass
    "trombone": "/home/arlo/Data/soundfonts/trombone.sf2",
    "trumpet": "/home/arlo/Data/soundfonts/trumpet.sf2",
    "french_horn": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add french_horn.sf2
    "tuba": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add tuba.sf2

    # Winds
    "sax": "/home/arlo/Data/soundfonts/sax.sf2",
    "bassoon": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add bassoon.sf2
    "clarinet": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add clarinet.sf2
    "flute": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add flute.sf2
    "oboe": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add oboe.sf2

    # Strings
    "violin": "/home/arlo/Data/soundfonts/violin.sf2",
    "viola": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add viola.sf2
    "cello": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add cello.sf2

    # Piano
    "acoustic_piano": "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "keys": "/usr/share/sounds/sf2/FluidR3_GM.sf2",

    # Guitar
    "acoustic_guitar": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add acoustic_guitar.sf2
    "electric_guitar": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add electric_guitar.sf2
    "plucked": "/usr/share/sounds/sf2/FluidR3_GM.sf2",

    # Bass
    "electric_bass": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add electric_bass.sf2
    "upright_bass": "/usr/share/sounds/sf2/FluidR3_GM.sf2",  # TODO: add upright_bass.sf2

    "undefined": "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "default": "/usr/share/sounds/sf2/FluidR3_GM.sf2"  # fallback
}

# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
MODEL: Union[Pipeline, None] = None
GROUP_NAMES: list = []
SUBGROUP_NAMES: list = []
MANIFEST_PATHS: list = []
MANIFEST_DATA: list = []

# Cache for conditioning extractions
CONDITIONING_CACHE: dict = {}

# Cache for ground truth latents
LATENT_CACHE: dict = {}

# ------------------------------------------------------------------------------
# Sample Recreation Enhancement Helpers
# ------------------------------------------------------------------------------
def detect_onsets_from_piano_roll(piano_roll: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Detect note onsets from piano roll for time-varying noise and encodec weighting.

    Args:
        piano_roll: [B, 128, T] piano roll tensor
        threshold: minimum value to consider as a note

    Returns:
        onset_mask: [B, 1, T] tensor with 1.0 at onsets, decaying afterwards
    """
    B, P, T = piano_roll.shape
    if T <= 1:
        return torch.zeros(B, 1, T, device=piano_roll.device, dtype=piano_roll.dtype)

    # Detect onsets: note starts where previous frame was silent
    is_active = (piano_roll > threshold).float()  # [B, 128, T]
    was_silent = F.pad(is_active[:, :, :-1] <= threshold, (1, 0), value=1.0)  # [B, 128, T]
    onset = (is_active * was_silent).amax(dim=1, keepdim=True)  # [B, 1, T]

    # Create decay envelope: strong at onset, decay over next ~100ms
    decay_frames = 5  # Roughly 100ms at 43fps
    onset_envelope = onset.clone()
    for i in range(1, decay_frames):
        if i < T:
            shifted = F.pad(onset[:, :, :-i], (i, 0), value=0.0)
            onset_envelope = torch.maximum(onset_envelope, shifted * (1.0 - i / decay_frames))

    # Debug: log onset statistics
    num_onsets = (onset > 0.5).sum().item()
    onset_coverage = (onset_envelope > 0.1).float().mean().item()
    print(f"   🎯 Detected {num_onsets} onsets, coverage: {onset_coverage*100:.1f}% of frames")

    return onset_envelope


def apply_time_varying_noise(
    gt_latents: torch.Tensor,
    noise: torch.Tensor,
    base_noise_level: float,
    onset_mask: torch.Tensor,
    onset_preservation: float = 0.7
) -> torch.Tensor:
    """
    Apply time-varying noise that preserves attacks/transients.

    Args:
        gt_latents: Ground truth latents [B, C, H, T]
        noise: Random noise tensor [B, C, H, T]
        base_noise_level: Base noise level (0-1)
        onset_mask: [B, 1, T] mask indicating onsets (1.0 at attacks)
        onset_preservation: How much to reduce noise at onsets (0-1)

    Returns:
        Noisy latents with preserved attacks
    """
    # Expand onset mask to match latent dimensions
    onset_mask_expanded = onset_mask.unsqueeze(2)  # [B, 1, 1, T]

    # Reduce noise at onsets
    time_varying_noise_level = base_noise_level * (1.0 - onset_preservation * onset_mask_expanded)

    # Debug: show noise level variation
    min_noise = time_varying_noise_level.min().item()
    max_noise = time_varying_noise_level.max().item()
    print(f"   📊 Noise level range: {min_noise:.3f} (at attacks) to {max_noise:.3f} (elsewhere)")

    # Mix with time-varying noise level
    x = (1.0 - time_varying_noise_level) * gt_latents + time_varying_noise_level * noise

    return x


def apply_multiresolution_latent_mixing(
    gt_latents: torch.Tensor,
    noise: torch.Tensor,
    base_noise_level: float,
    freq_split_dim: int = 2  # Height dimension
) -> torch.Tensor:
    """
    Preserve low-frequency structure, only vary high-frequency details.

    Args:
        gt_latents: [B, C, H, T] ground truth latents
        noise: Random noise [B, C, H, T]
        base_noise_level: Noise level for high frequencies
        freq_split_dim: Dimension to split (2 = height)

    Returns:
        Mixed latents with preserved low frequencies
    """
    B, C, H, T = gt_latents.shape

    # Split along height dimension (frequency-like)
    split_point = H // 2

    # Low frequency (bottom half of height): keep mostly intact
    gt_low = gt_latents[:, :, :split_point, :]
    noise_low = noise[:, :, :split_point, :]
    low_noise_level = base_noise_level * 0.3  # Much less noise in low freqs
    mixed_low = (1.0 - low_noise_level) * gt_low + low_noise_level * noise_low

    # High frequency (top half): apply full noise
    gt_high = gt_latents[:, :, split_point:, :]
    noise_high = noise[:, :, split_point:, :]
    mixed_high = (1.0 - base_noise_level) * gt_high + base_noise_level * noise_high

    # Recombine
    x = torch.cat([mixed_low, mixed_high], dim=2)

    return x


def weight_encodec_by_onsets(
    encodec_tokens: torch.Tensor,
    piano_roll: torch.Tensor,
    onset_boost: float = 2.0
) -> torch.Tensor:
    """
    Boost encodec token influence at note onsets to preserve attack timbre.

    Args:
        encodec_tokens: [B, C, T_enc] encodec tokens
        piano_roll: [B, 128, T_pr] piano roll
        onset_boost: Multiplier for encodec at onsets

    Returns:
        Weighted encodec tokens with boosted onsets
    """
    # Detect onsets from piano roll
    onset_mask = detect_onsets_from_piano_roll(piano_roll)  # [B, 1, T_pr]

    # Resize to match encodec temporal dimension
    T_enc = encodec_tokens.shape[-1]
    if onset_mask.shape[-1] != T_enc:
        onset_mask = F.interpolate(onset_mask, size=T_enc, mode='nearest')

    # Weight encodec tokens: 1.0 baseline, up to (1.0 + onset_boost) at onsets
    onset_weight = 1.0 + onset_boost * onset_mask  # [B, 1, T_enc]

    # Debug: show weight range
    min_weight = onset_weight.min().item()
    max_weight = onset_weight.max().item()
    print(f"   📊 Encodec weight range: {min_weight:.2f}x (baseline) to {max_weight:.2f}x (at attacks)")

    # Apply weighting (note: encodec tokens are integers, so we'd modify the conditioning later)
    # For now, return the weight mask to apply during ctrl_enc
    return onset_weight


# ------------------------------------------------------------------------------
# Test-Time Adaptation for Sample Recreation
# ------------------------------------------------------------------------------
@torch.no_grad()
def create_model_copy_for_adaptation(model):
    """Create a lightweight copy of model with only adapter layers for adaptation"""
    import copy
    # We'll adapt in-place but save original weights to restore later
    original_adapter_state = {}
    for name, param in model.cond_adapter.named_parameters():
        original_adapter_state[name] = param.data.clone()
    return original_adapter_state


def restore_model_weights(model, original_state):
    """Restore model weights from saved state"""
    for name, param in model.cond_adapter.named_parameters():
        if name in original_state:
            param.data.copy_(original_state[name])


def adapt_model_to_sample(
    model,
    audio_file: str,
    piano_roll: np.ndarray,
    amp: np.ndarray,
    rframe: np.ndarray,
    rbend: np.ndarray,
    encodec_tokens: torch.Tensor,
    group: str,
    subgroup: str,
    num_steps: int = 10,
    learning_rate: float = 1e-4,
    device: torch.device = None
):
    """
    Temporarily fine-tune model on a specific audio sample for better reconstruction.

    This implements test-time adaptation: we do a few gradient steps to adapt the model
    to the specific characteristics of the input sample, then generate variations.

    Args:
        model: The Pipeline model
        audio_file: Path to audio file to adapt to
        piano_roll, amp, rframe, rbend, encodec_tokens: Conditioning features
        group, subgroup: Instrument identifiers
        num_steps: Number of adaptation steps (5-20 recommended)
        learning_rate: Learning rate for adaptation (1e-4 to 1e-3)
        device: Device to run on

    Returns:
        original_state: Dict of original weights to restore later
    """
    if device is None:
        device = next(model.parameters()).device

    print(f"\n{'='*80}")
    print(f"🧠 TEST-TIME ADAPTATION")
    print(f"{'='*80}")
    print(f"Adapting model to sample: {Path(audio_file).name}")
    print(f"Steps: {num_steps}, Learning rate: {learning_rate}")

    # Save original adapter weights
    original_state = create_model_copy_for_adaptation(model)

    # Extract ground truth latents
    gt_latents = extract_ground_truth_latents(audio_file, model)
    if gt_latents is None:
        print("⚠️ Could not extract ground truth latents, skipping adaptation")
        return original_state

    gt_latents = gt_latents.to(device, dtype=next(model.parameters()).dtype)
    if gt_latents.ndim == 3:
        gt_latents = gt_latents.unsqueeze(0)  # [B, C, H, T]

    print(f"✅ Extracted GT latents: {gt_latents.shape}")

    # Prepare conditioning
    g2i = getattr(model, "group2id", None)
    s2i = getattr(model, "subgroup2id", None)
    if g2i is None or s2i is None:
        from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS
        g2i = {g:i for i,g in enumerate(list(APPROVED_GROUPS) if not isinstance(APPROVED_GROUPS, dict) else APPROVED_GROUPS.keys())}
        s2i = {}
        i = 0
        for gs in APPROVED_SUBGROUPS.values():
            for sg in gs:
                if sg not in s2i:
                    s2i[sg] = i; i += 1

    gid, sgid = int(g2i[group]), int(s2i[subgroup])

    conds = {
        "piano_roll": torch.from_numpy(piano_roll).float().unsqueeze(0).to(device),
        "amp": torch.from_numpy(amp).float().unsqueeze(0).to(device),
        "rframe": torch.from_numpy(rframe).float().unsqueeze(0).to(device),
        "rbend": torch.from_numpy(rbend).float().unsqueeze(0).to(device),
        "rbend_mask": torch.from_numpy((rframe > 0.5).astype(np.float32)).bool().unsqueeze(0).to(device),
        "encodec_tokens": encodec_tokens.to(device),
        "group_id": torch.tensor([gid], dtype=torch.long, device=device),
        "subgroup_id": torch.tensor([sgid], dtype=torch.long, device=device),
    }

    # Only adapt the cond_adapter (lightweight adaptation)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.cond_adapter.parameters():
        param.requires_grad = True

    # Optimizer for adaptation
    optimizer = torch.optim.Adam(
        [p for p in model.cond_adapter.parameters() if p.requires_grad],
        lr=learning_rate
    )

    model.train()  # Enable training mode for adaptation

    # Adaptation loop - must enable gradients even if called from @torch.no_grad() context
    print("\n🔄 Adapting model...")
    with torch.enable_grad():  # Override no_grad context from generate()
        for step in range(num_steps):
            optimizer.zero_grad()

            # Get conditioning tokens
            tokens, _ = model.ctrl_enc(**conds)

            # Get predicted latents through cond_adapter
            tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)

            # Resize to match GT latents temporal dimension
            T_target = gt_latents.shape[-1]
            pred_patch = model.cond_adapter(tokens_adapt, T_out=T_target, scale=1.0)

            # Loss: MSE between predicted conditioning patch and GT latents structure
            # We're not trying to predict exact latents, but adapt the conditioning to this sample
            loss = F.mse_loss(pred_patch, gt_latents)

            loss.backward()
            optimizer.step()

            if step % 2 == 0 or step == num_steps - 1:
                print(f"   Step {step+1}/{num_steps}: loss={loss.item():.6f}")

    model.eval()  # Back to eval mode

    # Freeze parameters again
    for param in model.parameters():
        param.requires_grad = False

    print(f"✅ Adaptation complete!")
    print(f"{'='*80}\n")

    return original_state


# ------------------------------------------------------------------------------
# Best-of-N Sampling with Reranking
# ------------------------------------------------------------------------------

def compute_mfcc_similarity(ref_mfcc: torch.Tensor, cand_mfcc: torch.Tensor) -> float:
    """
    Compute MFCC similarity using cosine similarity over time-averaged features
    """
    # Time-average MFCCs
    ref_avg = ref_mfcc.mean(dim=0)  # [num_coeffs]
    cand_avg = cand_mfcc.mean(dim=0)

    # Cosine similarity
    sim = F.cosine_similarity(ref_avg.unsqueeze(0), cand_avg.unsqueeze(0)).item()
    return max(0.0, sim)  # Clamp to [0, 1]


def detect_clipping(audio: torch.Tensor, threshold: float = 0.99) -> float:
    """
    Detect clipping - returns penalty in [0, 1] where 1 = heavily clipped
    """
    max_val = audio.abs().max().item()
    if max_val > threshold:
        # Count samples near max
        near_max = (audio.abs() > threshold).float().mean().item()
        return min(1.0, near_max * 10)  # Scale up the penalty
    return 0.0


def compute_loudness_normality(audio: torch.Tensor, sr: int = 44100) -> float:
    """
    Check if loudness is in normal range. Returns score in [0, 1] where 1 = good
    Returns penalty if too quiet or too loud
    """
    # RMS loudness
    rms = torch.sqrt(torch.mean(audio ** 2)).item()
    rms_db = 20 * np.log10(rms + 1e-10)

    # Ideal range: -20 to -10 dB
    if -20 <= rms_db <= -10:
        return 1.0
    elif rms_db < -30:  # Too quiet
        return 0.3
    elif rms_db > -5:  # Too loud (likely clipping)
        return 0.2
    else:
        # Gradual penalty
        return 1.0 - abs(rms_db + 15) / 20  # -15 dB is center


def compute_spectral_flatness_penalty(audio: torch.Tensor, sr: int = 44100) -> float:
    """
    Spectral flatness: 1 = white noise (bad), 0 = tonal (good)
    Returns penalty where high flatness = high penalty
    """
    # Compute magnitude spectrum
    spec = torch.stft(audio[0] if audio.ndim > 1 else audio,
                      n_fft=2048, hop_length=512, return_complex=True).abs()

    # Geometric mean / arithmetic mean
    geo_mean = torch.exp(torch.log(spec + 1e-10).mean(dim=0))
    arith_mean = spec.mean(dim=0)
    flatness = (geo_mean / (arith_mean + 1e-10)).mean().item()

    # If flatness > 0.3, it's noisy
    if flatness > 0.3:
        return min(1.0, (flatness - 0.3) / 0.3)
    return 0.0


def compute_onset_correlation(audio: torch.Tensor, piano_roll: np.ndarray,
                              sr: int = 44100, fps: float = 43.066) -> float:
    """
    Compute correlation between audio onsets and piano roll onsets
    """
    # Detect onsets in audio using energy
    hop_length = 512
    energy = torch.stft(audio[0] if audio.ndim > 1 else audio,
                       n_fft=2048, hop_length=hop_length, return_complex=True).abs()
    energy = energy.mean(dim=0)  # Average across frequency

    # Onset strength = derivative of energy
    onset_strength = torch.diff(energy, prepend=energy[:1])
    onset_strength = torch.relu(onset_strength)  # Only positive changes

    # Detect onsets from piano roll
    pr_onset = np.diff(piano_roll.max(axis=0), prepend=0)
    pr_onset = np.maximum(pr_onset, 0)

    # Resample to same length
    audio_frames = len(onset_strength)
    pr_frames = len(pr_onset)

    if pr_frames != audio_frames:
        # Resample piano roll onsets to match audio frames
        pr_onset_tensor = torch.from_numpy(pr_onset).float().unsqueeze(0).unsqueeze(0)
        pr_onset_resampled = F.interpolate(pr_onset_tensor, size=audio_frames, mode='linear', align_corners=False)
        pr_onset = pr_onset_resampled[0, 0].numpy()

    # Normalize
    onset_strength_np = onset_strength.cpu().numpy()
    onset_strength_np = onset_strength_np / (np.max(onset_strength_np) + 1e-10)
    pr_onset = pr_onset / (np.max(pr_onset) + 1e-10)

    # Correlation
    corr = np.corrcoef(onset_strength_np, pr_onset)[0, 1]
    return max(0.0, corr)  # Clamp to [0, 1]


def score_candidate(
    output_path: str,
    ref_audio_path: str,
    ref_encodec_tokens: torch.Tensor,
    piano_roll: np.ndarray,
    model,
    weights: dict = None
) -> dict:
    """
    Comprehensive scoring of a candidate against reference sample

    Args:
        output_path: Path to generated audio
        ref_audio_path: Path to reference/input audio
        ref_encodec_tokens: Pre-extracted encodec tokens from reference
        piano_roll: Piano roll conditioning used for generation
        model: Model (for encodec extraction)
        weights: Dict of score weights (optional)

    Returns:
        Dict with individual scores and total score
    """
    if weights is None:
        weights = {
            'encodec': 0.4,      # Timbre match (most important)
            'mfcc': 0.15,        # Timbral features
            'spectral': 0.15,    # Spectral similarity
            'onset': 0.1,        # Rhythm alignment
            'loudness': 0.1,     # Loudness normality
            'clip_penalty': 0.3, # Clipping penalty
            'flatness_penalty': 0.2  # Spectral flatness penalty
        }

    try:
        # Load reference audio
        ref_audio, ref_sr = torchaudio.load(ref_audio_path)

        # Load candidate audio
        cand_audio, cand_sr = torchaudio.load(output_path)

        # Resample candidate to match reference if needed
        if cand_sr != ref_sr:
            cand_audio = torchaudio.functional.resample(cand_audio, cand_sr, ref_sr)

        # Match lengths (use shorter)
        min_len = min(ref_audio.shape[-1], cand_audio.shape[-1])
        ref_audio = ref_audio[..., :min_len]
        cand_audio = cand_audio[..., :min_len]

        scores = {}

        # 1. Encodec similarity (timbre match) - MOST IMPORTANT
        try:
            cand_encodec = extract_encodec_tokens_from_audio(cand_audio, ref_sr, model)
            if cand_encodec is not None and ref_encodec_tokens is not None:
                # Flatten and compute cosine similarity
                ref_flat = ref_encodec_tokens.flatten(1).float()
                cand_flat = cand_encodec.flatten(1).float()

                # Match dimensions if needed
                min_dim = min(ref_flat.shape[-1], cand_flat.shape[-1])
                ref_flat = ref_flat[..., :min_dim]
                cand_flat = cand_flat[..., :min_dim]

                encodec_sim = F.cosine_similarity(ref_flat, cand_flat, dim=-1).mean().item()
                scores['encodec'] = max(0.0, min(1.0, encodec_sim))
            else:
                scores['encodec'] = 0.5  # Neutral if extraction fails
        except Exception as e:
            print(f"   ⚠️ Encodec scoring failed: {e}")
            scores['encodec'] = 0.5

        # 2. MFCC similarity
        try:
            ref_mfcc = torchaudio.compliance.kaldi.mfcc(ref_audio, sample_frequency=ref_sr)
            cand_mfcc = torchaudio.compliance.kaldi.mfcc(cand_audio, sample_frequency=ref_sr)
            scores['mfcc'] = compute_mfcc_similarity(ref_mfcc, cand_mfcc)
        except Exception as e:
            print(f"   ⚠️ MFCC scoring failed: {e}")
            scores['mfcc'] = 0.5

        # 3. Spectral similarity
        try:
            ref_spec = torch.stft(ref_audio[0], n_fft=2048, hop_length=512, return_complex=True).abs()
            cand_spec = torch.stft(cand_audio[0], n_fft=2048, hop_length=512, return_complex=True).abs()

            # Flatten and compute cosine similarity
            spec_sim = F.cosine_similarity(
                ref_spec.flatten().unsqueeze(0),
                cand_spec.flatten().unsqueeze(0),
                dim=-1
            ).item()
            scores['spectral'] = max(0.0, min(1.0, spec_sim))
        except Exception as e:
            print(f"   ⚠️ Spectral scoring failed: {e}")
            scores['spectral'] = 0.5

        # 4. Onset correlation (rhythm alignment)
        try:
            scores['onset'] = compute_onset_correlation(cand_audio, piano_roll, sr=ref_sr)
        except Exception as e:
            print(f"   ⚠️ Onset scoring failed: {e}")
            scores['onset'] = 0.5

        # 5. Loudness normality
        try:
            scores['loudness'] = compute_loudness_normality(cand_audio, sr=ref_sr)
        except Exception as e:
            print(f"   ⚠️ Loudness scoring failed: {e}")
            scores['loudness'] = 0.7

        # 6. Clipping penalty
        try:
            scores['clip_penalty'] = detect_clipping(cand_audio)
        except Exception as e:
            print(f"   ⚠️ Clipping detection failed: {e}")
            scores['clip_penalty'] = 0.0

        # 7. Spectral flatness penalty
        try:
            scores['flatness_penalty'] = compute_spectral_flatness_penalty(cand_audio, sr=ref_sr)
        except Exception as e:
            print(f"   ⚠️ Flatness scoring failed: {e}")
            scores['flatness_penalty'] = 0.0

        # Compute weighted total score
        total_score = (
            weights['encodec'] * scores['encodec'] +
            weights['mfcc'] * scores['mfcc'] +
            weights['spectral'] * scores['spectral'] +
            weights['onset'] * scores['onset'] +
            weights['loudness'] * scores['loudness'] -
            weights['clip_penalty'] * scores['clip_penalty'] -
            weights['flatness_penalty'] * scores['flatness_penalty']
        )

        scores['total'] = total_score

        return scores

    except Exception as e:
        print(f"   ❌ Candidate scoring failed: {e}")
        # Return neutral scores
        return {
            'encodec': 0.5, 'mfcc': 0.5, 'spectral': 0.5, 'onset': 0.5,
            'loudness': 0.5, 'clip_penalty': 0.5, 'flatness_penalty': 0.5,
            'total': 0.0
        }


def extract_encodec_tokens_from_audio(audio: torch.Tensor, sr: int, model) -> torch.Tensor:
    """Extract encodec tokens from audio tensor"""
    try:
        # Use model's encodec encoder if available
        if hasattr(model, 'ctrl_enc') and hasattr(model.ctrl_enc, 'encodec'):
            encodec = model.ctrl_enc.encodec

            # Prepare audio
            if audio.ndim == 1:
                audio = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            elif audio.ndim == 2:
                audio = audio.unsqueeze(0)  # [1, C, T]

            # Resample to 24kHz if needed (encodec native rate)
            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)

            # Extract tokens
            device = next(encodec.parameters()).device
            audio = audio.to(device)

            with torch.no_grad():
                encoded = encodec.encode(audio)
                if isinstance(encoded, tuple):
                    tokens = encoded[0]  # Usually (tokens, scale)
                else:
                    tokens = encoded

            return tokens.cpu()
        else:
            return None
    except Exception as e:
        print(f"   ⚠️ Encodec extraction failed: {e}")
        return None


def generate_best_of_n(
    model,
    audio_file: str,
    piano_roll: np.ndarray,
    amp: np.ndarray,
    rframe: np.ndarray,
    rbend: np.ndarray,
    encodec_tokens: torch.Tensor,
    group: str,
    subgroup: str,
    base_seed: int = 0,
    n_candidates: int = 12,
    **generation_args
) -> tuple:
    """
    Best-of-N sampling with comprehensive reranking

    Generates multiple candidates with parameter variations and returns the one
    that best matches the input audio sample.

    Args:
        model: Pipeline model
        audio_file: Path to reference audio (sample to match)
        piano_roll, amp, rframe, rbend, encodec_tokens: Conditioning inputs
        group, subgroup: Instrument identifiers
        base_seed: Base seed for candidate generation
        n_candidates: Number of candidates to generate (default: 12)
        **generation_args: Additional generation parameters

    Returns:
        (best_output_path, all_candidates_info)
    """
    print(f"\n{'='*80}")
    print(f"🎲 BEST-OF-N SAMPLING (N={n_candidates})")
    print(f"{'='*80}")
    print(f"Reference: {Path(audio_file).name}")
    print(f"Generating {n_candidates} candidates with parameter variations...")

    # Pre-extract reference encodec tokens for scoring
    ref_audio, ref_sr = torchaudio.load(audio_file)
    ref_encodec = extract_encodec_tokens_from_audio(ref_audio, ref_sr, model)

    # Define parameter sweep configurations with SAMPLER DIVERSITY
    # Include: noise levels, instBoost, AND step counts for artifact reduction
    base_steps = generation_args.get('steps', 20)

    param_variations = [
        # Core noise/boost variations
        {"t0": 0.6, "noise_level": 0.6, "inst_boost": 2.5, "steps": base_steps},
        {"t0": 0.7, "noise_level": 0.7, "inst_boost": 2.5, "steps": base_steps},
        {"t0": 0.75, "noise_level": 0.75, "inst_boost": 2.5, "steps": base_steps},
        {"t0": 0.8, "noise_level": 0.8, "inst_boost": 2.5, "steps": base_steps},
        {"t0": 0.7, "noise_level": 0.7, "inst_boost": 3.0, "steps": base_steps},
        {"t0": 0.7, "noise_level": 0.7, "inst_boost": 2.0, "steps": base_steps},

        # STEP-COUNT SWEEPS (sampler diversity)
        # More steps can reduce artifacts, fewer steps can be faster
        {"t0": 0.7, "noise_level": 0.7, "inst_boost": 2.5, "steps": max(15, base_steps - 10)},
        {"t0": 0.7, "noise_level": 0.7, "inst_boost": 2.5, "steps": base_steps + 10},
        {"t0": 0.7, "noise_level": 0.7, "inst_boost": 2.5, "steps": base_steps + 20},

        # Combined variations
        {"t0": 0.65, "noise_level": 0.65, "inst_boost": 2.5, "steps": base_steps + 10},
        {"t0": 0.75, "noise_level": 0.75, "inst_boost": 2.5, "steps": base_steps + 15},
        {"t0": 0.7, "noise_level": 0.7, "inst_boost": 2.8, "steps": base_steps + 5},
    ]

    # Expand with seed variations
    all_configs = []
    seeds_per_config = max(1, n_candidates // len(param_variations))

    for config in param_variations:
        for i in range(seeds_per_config):
            config_copy = config.copy()
            config_copy['seed'] = base_seed + len(all_configs) * 1000
            all_configs.append(config_copy)
            if len(all_configs) >= n_candidates:
                break
        if len(all_configs) >= n_candidates:
            break

    # Trim to exact count
    all_configs = all_configs[:n_candidates]

    print(f"\n📋 Parameter sweep (with sampler diversity):")
    print(f"   {len(set(c['t0'] for c in all_configs))} noise levels")
    print(f"   {len(set(c['inst_boost'] for c in all_configs))} instBoost values")
    print(f"   {len(set(c['steps'] for c in all_configs))} step counts: {sorted(set(c['steps'] for c in all_configs))}")
    print(f"   {len(all_configs)} total candidates\n")

    # Generate all candidates
    candidates = []

    for idx, config in enumerate(all_configs, 1):
        print(f"   [{idx}/{len(all_configs)}] Generating candidate...")
        print(f"      t0={config['t0']}, noise={config['noise_level']}, instBoost={config['inst_boost']}, steps={config['steps']}, seed={config['seed']}")

        # Merge with base generation args
        gen_args = generation_args.copy()
        gen_args.update(config)

        # Generate
        try:
            output_path = generate(
                model=model,
                piano_roll=piano_roll,
                amp=amp,
                rframe=rframe,
                rbend=rbend,
                encodec_tokens=encodec_tokens,
                group=group,
                subgroup=subgroup,
                audio_file=audio_file,
                **gen_args
            )

            # Score this candidate
            scores = score_candidate(
                output_path=output_path,
                ref_audio_path=audio_file,
                ref_encodec_tokens=ref_encodec,
                piano_roll=piano_roll,
                model=model
            )

            candidates.append({
                'path': output_path,
                'config': config,
                'scores': scores,
                'total_score': scores['total']
            })

            print(f"      ✅ Score: {scores['total']:.3f} (enc={scores['encodec']:.2f}, mfcc={scores['mfcc']:.2f}, spec={scores['spectral']:.2f})")

        except Exception as e:
            print(f"      ❌ Generation failed: {e}")
            continue

    if not candidates:
        raise RuntimeError("All candidates failed to generate")

    # Sort by total score (descending)
    candidates.sort(key=lambda x: x['total_score'], reverse=True)

    # Print ranking
    print(f"\n{'='*80}")
    print(f"🏆 CANDIDATE RANKING")
    print(f"{'='*80}")
    for idx, cand in enumerate(candidates[:5], 1):  # Top 5
        config = cand['config']
        scores = cand['scores']
        print(f"   {idx}. Score: {cand['total_score']:.3f}")
        print(f"      Config: t0={config['t0']}, noise={config['noise_level']}, instBoost={config['inst_boost']}, steps={config['steps']}, seed={config['seed']}")
        print(f"      Scores: enc={scores['encodec']:.2f}, mfcc={scores['mfcc']:.2f}, spec={scores['spectral']:.2f}, onset={scores['onset']:.2f}")
        print(f"      Quality: loud={scores['loudness']:.2f}, clip_pen={scores['clip_penalty']:.2f}, flat_pen={scores['flatness_penalty']:.2f}")
        print(f"      File: {Path(cand['path']).name}")
        print()

    best = candidates[0]
    print(f"✅ Returning best candidate: {Path(best['path']).name}")
    print(f"   Total score: {best['total_score']:.3f}")
    print(f"{'='*80}\n")

    return best['path'], candidates


# ------------------------------------------------------------------------------
# Caching helpers
# ------------------------------------------------------------------------------
def _get_file_cache_key(audio_path: str, extra_context: str = None) -> str:
    """Generate a cache key based on file path, size, modification time, and optional context."""
    try:
        stat = os.stat(audio_path)
        # Use path, size, mtime, and optional context for cache key
        key_data = f"{os.path.abspath(audio_path)}_{stat.st_size}_{stat.st_mtime}"
        if extra_context:
            key_data += f"_{extra_context}"
        return hashlib.md5(key_data.encode()).hexdigest()
    except (OSError, IOError):
        # If we can't stat the file, use just the path and context
        key_data = os.path.abspath(audio_path)
        if extra_context:
            key_data += f"_{extra_context}"
        return hashlib.md5(key_data.encode()).hexdigest()

def _is_cache_valid(cache_entry: dict, audio_path: str) -> bool:
    """Check if a cache entry is still valid."""
    try:
        # Check if all cached files still exist
        if "paths" in cache_entry:
            paths = cache_entry["paths"]
            valid = all(
                paths.get(k) and os.path.exists(paths[k])
                for k in ["piano_roll", "amp", "rframe", "rbend", "encodec"]
            )
            print(f"🔍 Cache validation (manifest paths): {valid}")
            if not valid:
                missing = [k for k in ["piano_roll", "amp", "rframe", "rbend", "encodec"]
                          if not (paths.get(k) and os.path.exists(paths[k]))]
                print(f"⚠️ Missing paths: {missing}")
            return valid
        elif "dir" in cache_entry:
            out_dir = Path(cache_entry["dir"])
            stem = cache_entry["stem"]
            req = [
                out_dir / f"{stem}.pianoroll.npy",
                out_dir / f"{stem}.amp.npy",
                out_dir / f"{stem}.rframe.npy",
                out_dir / f"{stem}.rbend.npy",
                out_dir / f"{stem}.encodec.pt"
            ]
            valid = all(x.exists() for x in req)
            print(f"🔍 Cache validation (disk cache): {valid}")
            if not valid:
                missing = [str(x) for x in req if not x.exists()]
                print(f"⚠️ Missing files: {missing[:2]}{'...' if len(missing) > 2 else ''}")
            return valid
    except Exception as e:
        print(f"⚠️ Cache validation error: {e}")
        pass
    return False

def clear_conditioning_cache():
    """Clear the conditioning cache."""
    global CONDITIONING_CACHE
    CONDITIONING_CACHE.clear()
    print("🗑️ Conditioning cache cleared.")

def clear_latent_cache():
    """Clear the latent cache."""
    global LATENT_CACHE
    LATENT_CACHE.clear()
    print("🗑️ Latent cache cleared.")

def clear_all_caches():
    """Clear both conditioning and latent caches."""
    clear_conditioning_cache()
    clear_latent_cache()

def get_cache_stats() -> dict:
    """Get cache statistics."""
    return {
        "conditioning_entries": len(CONDITIONING_CACHE),
        "latent_entries": len(LATENT_CACHE),
        "conditioning_keys": list(CONDITIONING_CACHE.keys())[:3],
        "latent_keys": list(LATENT_CACHE.keys())[:3]
    }

# ------------------------------------------------------------------------------
# Manifest helpers
# ------------------------------------------------------------------------------
def _find_manifest_record_by_audio(audio_path: str):
    if not MANIFEST_DATA:
        return None
    ap = Path(audio_path)
    # full path
    for it in MANIFEST_DATA:
        p = it.get("audio_path")
        if p and os.path.abspath(p) == os.path.abspath(audio_path):
            return it
    # basename
    for it in MANIFEST_DATA:
        p = it.get("audio_path")
        if p and os.path.basename(p) == ap.name:
            return it
    return None

# ------------------------------------------------------------------------------
# Compatibility helpers
# ------------------------------------------------------------------------------
def _resize_like(src_t: torch.Tensor, target_param: torch.Tensor) -> torch.Tensor:
    src = src_t.detach().cpu()
    tgt = target_param.detach().cpu().clone()
    if tuple(src.shape) == tuple(tgt.shape):
        return src
    common = tuple(min(a, b) for a, b in zip(src.shape, tgt.shape))
    slicers = tuple(slice(0, x) for x in common)
    tgt[slicers] = src[slicers]
    print(f"[compat] resized tensor: ckpt {tuple(src.shape)} -> model {tuple(tgt.shape)}")
    return tgt

def _pipeline_ctor_kwargs_from_ckpt_hparams(hp: dict) -> dict:
    import inspect
    sig = inspect.signature(Pipeline.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    out = {}
    for k, v in (hp or {}).items():
        if k in allowed:
            out[k] = v
    return out

def load_model_any_ckpt(ckpt_path: str, checkpoint_dir: str, manifest_json: str) -> Pipeline:
    print(f"Loading checkpoint: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu")

    hp = blob.get("hyper_parameters", {})
    ctor_kwargs = _pipeline_ctor_kwargs_from_ckpt_hparams(hp)
    ctor_kwargs["checkpoint_dir"] = checkpoint_dir
    ctor_kwargs["manifest_json"]  = manifest_json

    print("Instantiating Pipeline with restored hyperparameters:")
    for k in sorted(ctor_kwargs.keys()):
        print(f"  - {k} = {ctor_kwargs[k]}")
    model = Pipeline(**ctor_kwargs).eval()

    sd = blob.get("state_dict", blob)

    # Patch keys that commonly change size across runs
    def _safe_getattr_weight(obj, attr, sub_attr=None):
        module = getattr(obj, attr, None)
        if module is None:
            return None
        if sub_attr is not None:
            if hasattr(module, '__getitem__'):  # for things like sclr_proj[0]
                try:
                    module = module[0]
                except (IndexError, TypeError):
                    return None
        return getattr(module, "weight", None) if sub_attr == "weight" else getattr(module, "bias", None)
    
    patch_keys = [
        ("ctrl_enc.subgroup_emb.weight",  _safe_getattr_weight(model.ctrl_enc, "subgroup_emb", "weight")),
        ("ctrl_enc.group_emb.weight",     _safe_getattr_weight(model.ctrl_enc, "group_emb", "weight")),
        ("group_head.weight",             _safe_getattr_weight(model, "group_head", "weight")),
        ("group_head.bias",               _safe_getattr_weight(model, "group_head", "bias")),
        ("sub_head.weight",               _safe_getattr_weight(model, "sub_head", "weight")),
        ("sub_head.bias",                 _safe_getattr_weight(model, "sub_head", "bias")),
        ("ctrl_enc.sclr_proj.0.weight",   _safe_getattr_weight(model.ctrl_enc, "sclr_proj", "weight")),
        ("ctrl_enc.sclr_proj.0.bias",     _safe_getattr_weight(model.ctrl_enc, "sclr_proj", "bias")),
    ]
    for k, target in patch_keys:
        if target is None:
            continue
        if k in sd and tuple(sd[k].shape) != tuple(target.shape):
            sd[k] = _resize_like(sd[k], target)

    print("Loading state dict (strict=False)...")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[compat] Missing keys ({len(missing)}). Example: {missing[:8]}...")
    if unexpected:
        print(f"[compat] Unexpected keys ({len(unexpected)}). Example: {unexpected[:8]}...")

    return model

# ------------------------------------------------------------------------------
# Audio Utility Functions
# ------------------------------------------------------------------------------
def apply_tape_speed_sox(input_path: str, output_path: str, speed: float) -> str:
    """
    Apply tape-style speed change (varispeed) using sox.
    This changes both tempo and pitch like a tape machine.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        speed: Speed factor (e.g., 0.8 = slower, 1.25 = faster)

    Returns:
        Path to the output file
    """
    import subprocess

    if speed == 1.0:
        # No change needed, just copy
        shutil.copy(input_path, output_path)
        return output_path

    print(f"🎞️ Applying tape speed {speed}x: {Path(input_path).name} → {Path(output_path).name}")

    # Use sox speed effect for tape-style slowdown/speedup
    cmd = ["sox", input_path, output_path, "speed", str(speed)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        print(f"✅ Tape speed applied successfully")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Sox command failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        raise RuntimeError(f"Failed to apply tape speed: {e}")
    except FileNotFoundError:
        raise RuntimeError("Sox not found. Please install sox: sudo apt-get install sox")

def apply_time_stretch_sox(input_path: str, output_path: str, speed: float) -> str:
    """
    Apply time-stretching (pitch-preserving speed change) using sox.
    This changes tempo but preserves pitch using time-stretching algorithms.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        speed: Speed factor (e.g., 0.8 = slower, 1.25 = faster)

    Returns:
        Path to the output file
    """
    import subprocess

    if speed == 1.0:
        # No change needed, just copy
        shutil.copy(input_path, output_path)
        return output_path

    print(f"🎼 Applying time-stretch {speed}x (pitch preserved): {Path(input_path).name} → {Path(output_path).name}")

    # Use sox tempo effect for pitch-preserving slowdown/speedup
    # tempo changes the speed without changing pitch (time-stretching)
    cmd = ["sox", input_path, output_path, "tempo", str(speed)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        print(f"✅ Time-stretch applied successfully")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Sox command failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        raise RuntimeError(f"Failed to apply time-stretch: {e}")
    except FileNotFoundError:
        raise RuntimeError("Sox not found. Please install sox: sudo apt-get install sox")

def apply_pitch_shift_sox(input_path: str, output_path: str, semitones: int) -> str:
    """
    Apply pitch shift using sox.
    This changes pitch without changing duration.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        semitones: Number of semitones to shift (e.g., 12 = up one octave, -12 = down one octave)

    Returns:
        Path to the output file
    """
    import subprocess

    if semitones == 0:
        # No change needed, just copy
        shutil.copy(input_path, output_path)
        return output_path

    print(f"🎹 Pitch shifting {semitones:+d} semitones: {Path(input_path).name} → {Path(output_path).name}")

    # Use sox pitch effect (pitch takes cents, 100 cents = 1 semitone)
    cents = semitones * 100
    cmd = ["sox", input_path, output_path, "pitch", str(cents)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        print(f"✅ Pitch shift applied successfully")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Sox command failed: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        raise RuntimeError(f"Failed to apply pitch shift: {e}")
    except FileNotFoundError:
        raise RuntimeError("Sox not found. Please install sox: sudo apt-get install sox")

def sum_audio_tracks(audio_file_paths: list, output_path: str, normalize: bool = True) -> str:
    """
    Sum/mix multiple audio tracks into a single master track using torchaudio.

    Args:
        audio_file_paths: List of paths to audio files to mix
        output_path: Path where the mixed track should be saved
        normalize: Whether to normalize the output to prevent clipping (default: True)

    Returns:
        Path to the generated master track
    """
    if not audio_file_paths:
        raise ValueError("No audio files provided")

    if len(audio_file_paths) == 1:
        # If only one track, just copy it
        shutil.copy(audio_file_paths[0], output_path)
        return output_path

    print(f"🎚️ Mixing {len(audio_file_paths)} audio tracks...")

    # Load all audio files and find the longest duration
    audio_tensors = []
    sample_rate = None
    max_length = 0

    for i, file_path in enumerate(audio_file_paths):
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found: {file_path}")
            continue

        wav, sr = torchaudio.load(file_path)

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            # Resample if needed
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        audio_tensors.append(wav)
        max_length = max(max_length, wav.shape[-1])
        print(f"   Track {i+1}: {wav.shape[-1]} samples ({wav.shape[-1]/sr:.2f}s)")

    if not audio_tensors:
        raise ValueError("No valid audio files found")

    # Pad all tracks to the same length
    padded_tensors = []
    for wav in audio_tensors:
        if wav.shape[-1] < max_length:
            pad_length = max_length - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad_length))
        padded_tensors.append(wav)

    # Sum all tracks
    mixed = torch.stack(padded_tensors).sum(dim=0)

    # Normalize if requested
    if normalize:
        max_val = mixed.abs().max()
        if max_val > 0:
            # Leave some headroom (0.95 to avoid clipping)
            mixed = mixed * (0.95 / max_val)
            print(f"   Normalized: peak reduced from {max_val:.3f} to {mixed.abs().max():.3f}")

    # Save the mixed track
    torchaudio.save(output_path, mixed, sample_rate)

    print(f"✅ Mixed track saved: {output_path}")
    print(f"   Duration: {max_length / sample_rate:.2f}s")
    print(f"   Channels: {mixed.shape[0]}")

    return output_path

# ------------------------------------------------------------------------------
# MIDI Processing and FluidSynth Rendering
# ------------------------------------------------------------------------------

def modify_midi_tempo(input_midi_path: str, output_midi_path: str, tempo_scale: float) -> str:
    """
    Modify MIDI file tempo by scaling all tempo change events.

    Args:
        input_midi_path: Path to input MIDI file
        output_midi_path: Path to save modified MIDI file
        tempo_scale: Tempo scaling factor (e.g., 0.75 = 75% speed, slower)

    Returns:
        Path to the modified MIDI file
    """
    import mido

    if tempo_scale == 1.0:
        # No change needed, just copy
        shutil.copy(input_midi_path, output_midi_path)
        return output_midi_path

    print(f"🎼 Modifying MIDI tempo: {tempo_scale}x")
    print(f"   Input:  {Path(input_midi_path).name}")
    print(f"   Output: {Path(output_midi_path).name}")

    midi = mido.MidiFile(input_midi_path)

    # Scale all tempo messages
    # Note: MIDI tempo is in microseconds per quarter note
    # To slow down by factor X, we multiply tempo by 1/X (increase microseconds per beat)
    tempo_multiplier = 1.0 / tempo_scale

    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                original_tempo = msg.tempo
                msg.tempo = int(original_tempo * tempo_multiplier)
                print(f"   Modified tempo: {original_tempo} → {msg.tempo} µs/qn")

    midi.save(output_midi_path)
    print(f"   ✅ Saved modified MIDI with {tempo_scale}x tempo")

    return output_midi_path

def is_midi_file(file_path: str) -> bool:
    """Check if file is a MIDI file based on extension."""
    if not file_path:
        return False
    return Path(file_path).suffix.lower() in ['.mid', '.midi']

def is_multitrack_midi(midi_path: str) -> tuple:
    """
    Check if MIDI file contains multiple tracks/instruments.
    Returns:
        tuple: (is_multitrack: bool, track_count: int, non_drum_tracks: list)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        non_drum_instruments = [inst for inst in midi_data.instruments if not inst.is_drum]
        track_count = len(non_drum_instruments)

        # Consider multitrack if more than 1 non-drum instrument
        is_multitrack = track_count > 1

        print(f"🎼 MIDI Analysis: {Path(midi_path).name}")
        print(f"   Total instruments: {len(midi_data.instruments)}")
        print(f"   Non-drum tracks: {track_count}")
        print(f"   Multitrack: {'Yes' if is_multitrack else 'No'}")

        return is_multitrack, track_count, non_drum_instruments
    except Exception as e:
        print(f"⚠️ Error analyzing MIDI file: {e}")
        return False, 0, []

def is_monophonic_track(instrument, fps: float = 43.066, overlap_threshold: float = 0.01) -> bool:
    """
    Check if a MIDI instrument/track is monophonic (no overlapping notes).

    Args:
        instrument: pretty_midi.Instrument object
        fps: Frames per second for time resolution
        overlap_threshold: Minimum overlap time (seconds) to consider polyphonic

    Returns:
        bool: True if track is monophonic, False if polyphonic
    """
    if not instrument.notes or len(instrument.notes) < 2:
        return True  # No notes or single note = monophonic

    # Sort notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

    # Check for overlapping notes
    for i in range(len(sorted_notes) - 1):
        current_note = sorted_notes[i]
        next_note = sorted_notes[i + 1]

        # Check if current note ends after next note starts (overlap)
        overlap = current_note.end - next_note.start
        if overlap > overlap_threshold:
            return False  # Found overlapping notes = polyphonic

    return True  # No significant overlaps found = monophonic

def extract_midi_tempo(midi_path: str) -> float:
    """
    Extract tempo from MIDI file, handling non-standard tempo placement.

    Args:
        midi_path: Path to MIDI file

    Returns:
        float: Tempo in BPM (defaults to 120.0 if not found)
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # Method 1: Check for tempo changes (standard location)
        if hasattr(midi_data, 'tempo_changes') and midi_data.tempo_changes:
            initial_tempo = midi_data.tempo_changes[0].tempo
            print(f"🎵 Found tempo from tempo changes: {initial_tempo:.1f} BPM")
            return initial_tempo

        # Method 2: Estimate from note timing patterns
        all_notes = []
        for inst in midi_data.instruments:
            if not inst.is_drum and inst.notes:
                all_notes.extend([note.start for note in inst.notes])

        if len(all_notes) > 10:  # Need enough notes for estimation
            all_notes.sort()
            intervals = []

            # Look for regular time intervals (beats)
            for i in range(1, min(50, len(all_notes))):  # Check first 50 notes
                interval = all_notes[i] - all_notes[i-1]
                if 0.1 < interval < 2.0:  # Reasonable beat intervals
                    intervals.append(interval)

            if intervals:
                import statistics
                median_interval = statistics.median(intervals)
                # Assume the median interval represents quarter notes
                estimated_tempo = 60.0 / median_interval

                # Clamp to reasonable range
                if 40 <= estimated_tempo <= 200:
                    print(f"🎵 Estimated tempo from note timing: {estimated_tempo:.1f} BPM")
                    return estimated_tempo

        print("🎵 No tempo found, using default: 120.0 BPM")
        return 120.0

    except Exception as e:
        print(f"⚠️ Error extracting tempo: {e}, using default: 120.0 BPM")
        return 120.0

def analyze_track_polyphony(midi_path: str) -> dict:
    """
    Analyze polyphony for each track in a MIDI file.

    Returns:
        dict: {
            'total_tracks': int,
            'monophonic_tracks': int,
            'polyphonic_tracks': int,
            'track_analysis': [{'name': str, 'is_monophonic': bool, 'note_count': int}]
        }
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        non_drum_instruments = [inst for inst in midi_data.instruments if not inst.is_drum]

        track_analysis = []
        monophonic_count = 0

        for i, instrument in enumerate(non_drum_instruments):
            is_mono = is_monophonic_track(instrument)
            track_name = instrument.name if instrument.name else f"Track {i+1}"
            note_count = len(instrument.notes)

            track_analysis.append({
                'name': track_name,
                'is_monophonic': is_mono,
                'note_count': note_count,
                'program': instrument.program
            })

            if is_mono:
                monophonic_count += 1

        total_tracks = len(non_drum_instruments)
        polyphonic_count = total_tracks - monophonic_count

        print(f"🎼 Polyphony Analysis: {Path(midi_path).name}")
        print(f"   Total tracks: {total_tracks}")
        print(f"   Monophonic tracks: {monophonic_count}")
        print(f"   Polyphonic tracks: {polyphonic_count}")

        return {
            'total_tracks': total_tracks,
            'monophonic_tracks': monophonic_count,
            'polyphonic_tracks': polyphonic_count,
            'track_analysis': track_analysis
        }

    except Exception as e:
        print(f"⚠️ Error analyzing track polyphony: {e}")
        return {
            'total_tracks': 0,
            'monophonic_tracks': 0,
            'polyphonic_tracks': 0,
            'track_analysis': []
        }

def render_midi_to_audio(midi_path: str, output_dir: str = "./temp_audio", instrument_group: str = None) -> str:
    """
    Render MIDI file to audio using FluidSynth with appropriate soundfont.
    Args:
        midi_path: Path to MIDI file
        output_dir: Directory to save rendered audio
        instrument_group: Instrument group/subgroup to determine soundfont
    Returns:
        Path to rendered audio file
    """
    midi_path = Path(midi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with instrument group
    if instrument_group:
        audio_filename = f"{midi_path.stem}_{instrument_group}_rendered.wav"
    else:
        audio_filename = f"{midi_path.stem}_default_rendered.wav"
    audio_path = output_dir / audio_filename

    # Choose appropriate soundfont based on instrument group
    soundfont_path = INSTRUMENT_SOUNDFONTS.get("default")  # default fallback
    matched_instrument = "default"
    if instrument_group:
        # Check if the instrument group matches any of our specific soundfonts
        for instrument, sf_path in INSTRUMENT_SOUNDFONTS.items():
            if instrument != "default" and instrument.lower() in instrument_group.lower():
                soundfont_path = sf_path
                matched_instrument = instrument
                break

    print(f"🎵 Rendering MIDI to audio: {midi_path.name} -> {audio_filename}")
    if instrument_group:
        print(f"   Instrument group: '{instrument_group}' -> Matched: '{matched_instrument}'")
        print(f"   Using soundfont: {soundfont_path}")

    try:
        # Try fluidsynth first (preferred)
        cmd = [
            "fluidsynth",
            "-ni",  # no interactive mode
            "-g", "0.5",  # gain
            "-F", str(audio_path),  # output file
            soundfont_path,  # soundfont based on instrument group
            str(midi_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0 and audio_path.exists():
            print(f"✅ FluidSynth rendering successful: {audio_path}")
            return str(audio_path)
        else:
            print(f"⚠️ FluidSynth failed with return code {result.returncode}")
            print(f"   STDOUT: {result.stdout[:200]}")
            print(f"   STDERR: {result.stderr[:200]}")
            print(f"   Trying alternative...")
            # Remove failed file
            if audio_path.exists():
                audio_path.unlink()

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"⚠️ FluidSynth error: {e}, trying alternative...")

    # Fallback: Create sine wave audio from MIDI using pretty_midi
    try:
        print("🔄 Fallback: Creating sine wave audio from MIDI...")
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))

        # Synthesize with sine waves (simple but clean)
        audio_data = midi_data.synthesize(fs=44100, wave=np.sin)

        # Normalize audio
        if audio_data.max() > 0:
            audio_data = audio_data / np.abs(audio_data).max() * 0.9

        # Save as WAV using torchaudio
        audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)  # Add channel dimension
        torchaudio.save(str(audio_path), audio_tensor, 44100)

        print(f"✅ Sine wave synthesis successful: {audio_path}")
        return str(audio_path)

    except Exception as e:
        print(f"❌ All audio rendering methods failed: {e}")
        raise RuntimeError(f"Could not render MIDI to audio: {e}")

def split_midi_into_track_files(midi_path: str, output_dir: str = "./temp_tracks") -> list:
    """
    Split a multitrack MIDI file into separate MIDI files, one per track.
    Returns list of individual MIDI file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    midi_stem = Path(midi_path).stem

    track_files = []

    for i, instrument in enumerate(midi_data.instruments):
        # Process ALL tracks including drums (user may want to process drum patterns)
        # if instrument.is_drum:
        #     continue  # Skip drum tracks

        # Create a new MIDI file with just this track
        single_track_midi = pretty_midi.PrettyMIDI()
        single_track_midi.instruments.append(instrument)

        # Copy basic timing information
        single_track_midi.resolution = midi_data.resolution

        # Save individual track MIDI file
        track_name = instrument.name if instrument.name else f"Track_{i+1}"
        track_file = output_dir / f"{midi_stem}_{track_name}_track{i+1}.mid"
        single_track_midi.write(str(track_file))

        track_files.append(str(track_file))
        print(f"   Created track MIDI: {track_file.name}")

    return track_files

def process_multitrack_midi_simple(midi_path: str, progress=None, **generation_args) -> tuple:
    """
    Process multitrack MIDI using the simple approach:
    1. Split MIDI into individual track files
    2. Render each track to FluidSynth audio
    3. Process each audio file individually (like audio upload)
    4. Sum the results

    Returns: (mixed_audio_path, individual_audio_paths, info_text)
    """
    print(f"🎼 Processing multitrack MIDI (simple approach): {Path(midi_path).name}")

    if progress:
        progress(0.1, desc="Splitting MIDI into tracks...")

    # Step 1: Split MIDI into individual track files
    track_midi_files = split_midi_into_track_files(midi_path)
    track_count = len(track_midi_files)

    if track_count == 0:
        raise ValueError("No non-drum tracks found in MIDI file")

    print(f"   Split into {track_count} individual track MIDI files")

    if progress:
        progress(0.2, desc="Rendering tracks to audio...")

    # Step 2: Render each track to FluidSynth audio
    individual_audio_files = []
    for i, track_midi in enumerate(track_midi_files):
        if progress:
            progress(0.2 + (i / track_count) * 0.2, desc=f"Rendering track {i+1}/{track_count}...")

        track_name = Path(track_midi).stem
        print(f"🎵 Rendering track {i+1}: {track_name}")

        # Render this track to audio using existing function
        # Pass the subgroup from generation_args for appropriate soundfont selection
        instrument_subgroup = generation_args.get('subgroup', None)
        track_audio = render_midi_to_audio(track_midi, instrument_group=instrument_subgroup)
        individual_audio_files.append(track_audio)
        print(f"   → {Path(track_audio).name}")

    if progress:
        progress(0.4, desc="Processing individual tracks...")

    # Step 3: Process each audio file individually (like regular audio upload)
    individual_outputs = []
    for i, audio_file in enumerate(individual_audio_files):
        if progress:
            progress(0.4 + (i / track_count) * 0.5, desc=f"Generating from track {i+1}/{track_count}...")

        print(f"🎼 Processing track {i+1} audio: {Path(audio_file).name}")

        # Process this audio file using the standard audio processing pipeline
        # Note: We'll call the regular generation function for this audio file
        track_output = process_single_audio_file(audio_file, **generation_args)
        individual_outputs.append(track_output)
        print(f"   → Generated: {Path(track_output).name}")

    if progress:
        progress(0.9, desc="Summing tracks for mixed output...")

    # Step 4: Sum all the individual outputs
    print("🔄 Summing individual tracks to create mixed output...")
    mixed_audio = sum_audio_files(individual_outputs)

    if progress:
        progress(1.0, desc="Done!")

    info_text = f"Generated {track_count} individual tracks from multitrack MIDI (simple approach)"
    return mixed_audio, individual_outputs, info_text

def process_single_audio_file(audio_file: str, **generation_args) -> str:
    """
    Process a single audio file using the standard audio upload pipeline.
    Returns the path to the generated audio file.
    """
    # Extract conditioning from the audio file
    instrument_subgroup = generation_args.get('subgroup', None)
    extraction = extract_conditioning_from_audio(audio_file, instrument_group=instrument_subgroup)
    pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=generation_args.get('win_slow', 1024))

    # Get the original audio length for correct timing
    try:
        wav, sr = torchaudio.load(audio_file)
        orig_len = wav.shape[-1]
    except Exception:
        orig_len = None

    # Generate using the standard generate function
    output_path = generate(
        generation_args['MODEL'], pr, amp, rfr, rbd, enc,
        generation_args['group'], generation_args['subgroup'],
        generation_args['steps'], generation_args['seed'],
        generation_args['adapter_scale'], generation_args['cfg_weight'],
        generation_args['t0'], sr_out=32000,
        instrument_strength=generation_args.get('instrument_strength', 1.0),
        inst_boost=generation_args.get('inst_boost', 2.5),
        piano_roll_gain=generation_args.get('piano_roll_gain', 1.0),
        amp_gain=generation_args.get('amp_gain', 1.0),
        rframe_gain=generation_args.get('rframe_gain', 1.0),
        rbend_gain=generation_args.get('rbend_gain', 1.0),
        encodec_gain=generation_args.get('encodec_gain', 1.0),
        use_overlap_decoder=generation_args.get('use_overlap_decoder', True),
        original_audio_length=orig_len,
        pitch_fidelity_boost=generation_args.get('pitch_fidelity_boost', 1.0),
        onset_guidance_boost=generation_args.get('onset_guidance_boost', 2.0),
        pitch_snap_strength=generation_args.get('pitch_snap_strength', 0.5),
        noise_level=generation_args.get('noise_level', 1.0),
        audio_file=audio_file,
        use_time_varying_noise=generation_args.get('use_time_varying_noise', False),
        onset_preservation=generation_args.get('onset_preservation', 0.7),
        use_multiresolution_mixing=generation_args.get('use_multiresolution_mixing', False),
        use_onset_weighted_encodec=generation_args.get('use_onset_weighted_encodec', False),
        encodec_onset_boost=generation_args.get('encodec_onset_boost', 2.0),
        use_test_time_adaptation=generation_args.get('use_test_time_adaptation', False),
        adaptation_steps=generation_args.get('adaptation_steps', 10),
        adaptation_learning_rate=generation_args.get('adaptation_learning_rate', 1e-4)
    )

    return output_path

def sum_audio_files(audio_file_paths: list) -> str:
    """
    Load multiple audio files and sum them to create a mixed output.
    Returns the path to the mixed audio file.
    """
    if not audio_file_paths:
        raise ValueError("No audio files to sum")

    print(f"🔄 Summing {len(audio_file_paths)} audio files...")

    mixed_audio = None
    sample_rate = None

    for i, audio_path in enumerate(audio_file_paths):
        print(f"   Loading file {i+1}: {Path(audio_path).name}")

        # Load audio file
        audio, sr = torchaudio.load(audio_path)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            print(f"⚠️ Sample rate mismatch: {sr} vs {sample_rate}")

        # Convert to numpy and ensure 1D
        audio_numpy = audio.squeeze().numpy()
        if audio_numpy.ndim > 1:
            # If still multi-dimensional, take first channel
            audio_numpy = audio_numpy[0] if audio_numpy.shape[0] < audio_numpy.shape[1] else audio_numpy.flatten()

        if mixed_audio is None:
            mixed_audio = audio_numpy.copy()
        else:
            # Ensure same length (pad shorter one if needed)
            if len(audio_numpy) != len(mixed_audio):
                max_len = max(len(audio_numpy), len(mixed_audio))
                if len(audio_numpy) < max_len:
                    audio_numpy = np.pad(audio_numpy, (0, max_len - len(audio_numpy)))
                if len(mixed_audio) < max_len:
                    mixed_audio = np.pad(mixed_audio, (0, max_len - len(mixed_audio)))

            mixed_audio += audio_numpy

    # Save the mixed output
    mixed_path = Path("./generated_ui") / f"mixed_{int(time.time())}_multitrack_simple.wav"
    mixed_path.parent.mkdir(exist_ok=True)

    # Ensure mixed_audio is 1D and convert to proper 2D tensor for saving
    if mixed_audio.ndim > 1:
        mixed_audio = mixed_audio.flatten()

    # Convert to tensor with proper shape [channels, samples]
    mixed_tensor = torch.from_numpy(mixed_audio).unsqueeze(0)  # Shape: [1, samples]
    torchaudio.save(str(mixed_path), mixed_tensor, sample_rate)

    print(f"✅ Mixed output saved: {mixed_path.name}")
    return str(mixed_path)

def render_multitrack_debug_audio(track_midi_paths: list, debug_dir: str, audio_stem: str, instrument_group: str = None) -> list:
    """
    Render individual MIDI tracks to audio using FluidSynth for debugging multitrack performance.

    Args:
        track_midi_paths: List of paths to individual track MIDI files
        debug_dir: Directory to save debug audio files
        audio_stem: Base name for audio files
        instrument_group: Instrument group/subgroup to determine soundfont

    Returns:
        list: Paths to rendered audio files
    """
    if not track_midi_paths:
        print("🎵 No MIDI tracks to render for debugging")
        return []

    print(f"🎵 Rendering {len(track_midi_paths)} individual tracks with FluidSynth for debugging...")

    try:
        audio_dir = Path(debug_dir) / "debug_audio"
        audio_dir.mkdir(exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create debug audio directory: {e}")
        return []

    rendered_audio_paths = []

    for i, midi_path in enumerate(track_midi_paths):
        if not midi_path or not Path(midi_path).exists():
            print(f"   ⚠️ Track {i+1}: MIDI file not found: {midi_path}")
            continue

        midi_file = Path(midi_path)
        track_name = midi_file.stem.replace(f"{audio_stem}_track_", "").replace(f"{audio_stem}_", "")

        print(f"   Rendering track {i+1}: {track_name}")

        try:
            # Use the existing render_midi_to_audio function which handles FluidSynth properly
            rendered_audio = render_midi_to_audio(
                midi_path,
                output_dir=str(audio_dir),
                instrument_group=instrument_group
            )

            if rendered_audio and Path(rendered_audio).exists():
                print(f"   ✅ Track {i+1} rendered successfully")
                rendered_audio_paths.append(rendered_audio)
            else:
                print(f"   ❌ Track {i+1} rendering failed")

        except Exception as e:
            print(f"   ❌ Track {i+1} rendering error: {e}")

    print(f"🎵 Successfully rendered {len(rendered_audio_paths)}/{len(track_midi_paths)} tracks")
    return rendered_audio_paths

def midi_to_piano_roll_conditioning(midi_path: str, window_slow: int = 1024, fps: float = 43.066, tempo_override: float = None) -> tuple:
    """
    Convert MIDI file directly to piano roll conditioning (no other conditioning).
    Args:
        midi_path: Path to MIDI file
        window_slow: Target length for conditioning
        fps: Frames per second for piano roll
    Returns:
        tuple: (piano_roll, empty_amp, empty_rframe, empty_rbend, empty_encodec)
    """
    print(f"🎼 Converting MIDI to piano roll conditioning: {Path(midi_path).name}")

    # Load MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    if not midi_data.instruments:
        raise ValueError("MIDI file contains no instruments")

    # Use tempo override if provided, otherwise extract from MIDI
    if tempo_override is not None:
        actual_tempo = tempo_override
        print(f"🎵 Using tempo override: {actual_tempo:.1f} BPM")
    else:
        actual_tempo = extract_midi_tempo(midi_path)
        print(f"🎵 Detected tempo: {actual_tempo:.1f} BPM")

    base_tempo = 120.0  # fps=43.066 seems to be calibrated for 120 BPM
    tempo_ratio = actual_tempo / base_tempo
    adjusted_fps = fps * tempo_ratio

    print(f"🎵 Tempo adjustment: {actual_tempo:.1f} BPM (ratio: {tempo_ratio:.3f}, adjusted fps: {adjusted_fps:.3f})")

    # Get duration and create time grid with tempo-adjusted fps
    duration = max(midi_data.get_end_time(), 1.0)  # At least 1 second
    time_steps = int(duration * adjusted_fps) + 1

    # Create piano roll
    piano_roll = np.zeros((128, time_steps))

    # Convert all instruments to piano roll (merge them)
    total_notes = 0
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue  # Skip drum tracks

        for note in instrument.notes:
            start_frame = int(note.start * adjusted_fps)
            end_frame = int(note.end * adjusted_fps)
            start_frame = max(0, min(start_frame, time_steps - 1))
            end_frame = max(start_frame + 1, min(end_frame, time_steps))

            # Use velocity for intensity (normalized to 0-1)
            intensity = note.velocity / 127.0
            piano_roll[note.pitch, start_frame:end_frame] = intensity
            total_notes += 1

    print(f"✅ Created piano roll: {piano_roll.shape}, {total_notes} notes, {duration:.2f}s")

    # Resize to target window (preserve full MIDI duration)
    # TIMING FIX: Don't truncate MIDI - preserve full length for proper timing
    original_frames = piano_roll.shape[1]
    target_length = max(window_slow, original_frames)  # Always preserve full length

    if piano_roll.shape[1] != target_length:
        # Only pad if too short, never truncate
        pad_width = target_length - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        print(f"🎵 Padded piano roll from {original_frames} to {target_length} frames")

    # Create empty conditioning for other modalities (match piano roll length)
    final_length = piano_roll.shape[1]
    empty_amp = np.zeros(final_length)
    empty_rframe = np.zeros(final_length)
    empty_rbend = np.zeros(final_length)

    # Create minimal encodec tokens (all zeros - will be ignored if encodec_gain=0)
    encodec_length = final_length // 4  # Typical encodec downsampling
    empty_encodec = torch.zeros((1, 8, encodec_length), dtype=torch.long)

    print(f"✅ MIDI conditioning ready: PR={piano_roll.shape}, others empty")

    return piano_roll, empty_amp, empty_rframe, empty_rbend, empty_encodec

def midi_to_multitrack_piano_rolls(midi_path: str, window_slow: int = 1024, fps: float = 43.066, tempo_override: Optional[float] = None) -> dict:
    """
    Convert multitrack MIDI file to separate piano rolls for each track/voice.
    Args:
        midi_path: Path to MIDI file
        window_slow: Target length for conditioning
        fps: Frames per second for piano roll
    Returns:
        dict: {
            'track_piano_rolls': [piano_roll_per_track],
            'track_info': [{'name': str, 'program': int, 'note_count': int}],
            'combined_piano_roll': combined_piano_roll,
            'empty_conditioning': (empty_amp, empty_rframe, empty_rbend, empty_encodec)
        }
    """
    print(f"🎼 Converting multitrack MIDI to separate piano rolls: {Path(midi_path).name}")

    # Load MIDI and check if multitrack
    is_multi, track_count, non_drum_instruments = is_multitrack_midi(midi_path)

    # Extract tempo from original MIDI or use override
    if tempo_override is not None:
        original_tempo = tempo_override
        print(f"🎵 Using tempo override: {original_tempo:.1f} BPM")
    else:
        original_tempo = extract_midi_tempo(midi_path)
        print(f"🎵 Detected tempo: {original_tempo:.1f} BPM")

    # Analyze polyphony for each track
    polyphony_analysis = analyze_track_polyphony(midi_path)

    if not is_multi:
        # Fallback to single track processing
        print("   Single track detected, using standard processing")
        piano_roll, empty_amp, empty_rframe, empty_rbend, empty_encodec = midi_to_piano_roll_conditioning(midi_path, window_slow, fps, tempo_override=tempo_override)
        return {
            'track_piano_rolls': [piano_roll],
            'track_info': [{'name': 'Track 1', 'program': 0, 'note_count': int(np.sum(piano_roll > 0.1))}],
            'combined_piano_roll': piano_roll,
            'empty_conditioning': (empty_amp, empty_rframe, empty_rbend, empty_encodec)
        }

    # Process multitrack MIDI
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    duration = max(midi_data.get_end_time(), 1.0)

    # Apply tempo adjustment (same as in midi_to_piano_roll_conditioning)
    base_tempo = 120.0  # fps=43.066 seems to be calibrated for 120 BPM
    tempo_ratio = original_tempo / base_tempo
    adjusted_fps = fps * tempo_ratio

    print(f"🎵 Multitrack tempo adjustment: {original_tempo:.1f} BPM (ratio: {tempo_ratio:.3f}, adjusted fps: {adjusted_fps:.3f})")

    time_steps = int(duration * adjusted_fps) + 1

    track_piano_rolls = []
    track_info = []
    combined_piano_roll = np.zeros((128, time_steps))

    print(f"   Processing {track_count} tracks, duration: {duration:.2f}s")

    for i, instrument in enumerate(non_drum_instruments):
        # Create piano roll for this track
        track_piano_roll = np.zeros((128, time_steps))
        note_count = 0

        for note in instrument.notes:
            start_frame = int(note.start * adjusted_fps)
            end_frame = int(note.end * adjusted_fps)
            start_frame = max(0, min(start_frame, time_steps - 1))
            end_frame = max(start_frame + 1, min(end_frame, time_steps))

            # Use velocity for intensity
            intensity = note.velocity / 127.0
            track_piano_roll[note.pitch, start_frame:end_frame] = intensity
            combined_piano_roll[note.pitch, start_frame:end_frame] = intensity
            note_count += 1

        # Note: Individual track resizing will be done after all tracks are processed
        # to ensure consistent length across all tracks

        track_piano_rolls.append(track_piano_roll)

        # Get track info with polyphony status
        track_name = instrument.name if instrument.name else f"Track {i+1}"

        # Find polyphony info for this track
        is_monophonic = False
        if i < len(polyphony_analysis['track_analysis']):
            is_monophonic = polyphony_analysis['track_analysis'][i]['is_monophonic']

        track_info.append({
            'name': track_name,
            'program': instrument.program,
            'note_count': note_count,
            'is_monophonic': is_monophonic
        })

        mono_status = "monophonic" if is_monophonic else "polyphonic"
        print(f"   Track {i+1}: {track_name} (Program {instrument.program}) - {note_count} notes [{mono_status}]")

    # Resize all piano rolls (individual tracks and combined) consistently
    # Use standard window_slow (1024) for model compatibility
    original_frames = combined_piano_roll.shape[1]
    target_length = window_slow  # Use model training length

    # Resize combined piano roll (truncate or pad to target_length)
    if combined_piano_roll.shape[1] != target_length:
        if combined_piano_roll.shape[1] > target_length:
            # Truncate to target length
            combined_piano_roll = combined_piano_roll[:, :target_length]
        else:
            # Pad to target length
            pad_width = target_length - combined_piano_roll.shape[1]
            combined_piano_roll = np.pad(combined_piano_roll, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    # Resize all individual track piano rolls to match (truncate or pad to target_length)
    for i, track_pr in enumerate(track_piano_rolls):
        if track_pr.shape[1] != target_length:
            if track_pr.shape[1] > target_length:
                # Truncate to target length
                track_piano_rolls[i] = track_pr[:, :target_length]
            else:
                # Pad to target length
                pad_width = target_length - track_pr.shape[1]
                track_piano_rolls[i] = np.pad(track_pr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

    # Create empty conditioning for other modalities (match final length)
    final_length = target_length
    empty_amp = np.zeros(final_length)
    empty_rframe = np.zeros(final_length)
    empty_rbend = np.zeros(final_length)
    encodec_length = final_length // 4
    empty_encodec = torch.zeros((1, 8, encodec_length), dtype=torch.long)

    print(f"✅ Multitrack processing complete: {len(track_piano_rolls)} tracks processed")

    return {
        'track_piano_rolls': track_piano_rolls,
        'track_info': track_info,
        'combined_piano_roll': combined_piano_roll,
        'empty_conditioning': (empty_amp, empty_rframe, empty_rbend, empty_encodec),
        'polyphony_analysis': polyphony_analysis,
        'original_tempo': original_tempo
    }

# ------------------------------------------------------------------------------
# MIDI Generation Functions (from ac.py midigen feature)
# ------------------------------------------------------------------------------
SOUNDFONT_PATH = "/home/arlo/.local/lib/python3.9/site-packages/pretty_midi/TimGM6mb.sf2"
MIDI_CHORD_FOLDER = '/home/arlo/free-midi-chords/output/01 - C Major - A minor/4 Progression/Major'

def get_random_transposed_midi_wav(tempo: int = 80):
    """
    Select a random MIDI chord progression and set its tempo.
    Returns paths to both MIDI and WAV files.
    """
    output_midi = f'/tmp/transposed_output_{os.getpid()}_{int(time.time())}.mid'
    output_wav = f'/tmp/transposed_output_{os.getpid()}_{int(time.time())}.wav'

    midi_files = [f for f in os.listdir(MIDI_CHORD_FOLDER) if f.endswith('.mid')]
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {MIDI_CHORD_FOLDER}")

    midi_file = random.choice(midi_files)
    midi_path = os.path.join(MIDI_CHORD_FOLDER, midi_file)
    print(f"🎼 Selected MIDI: {midi_file} at {tempo} BPM")

    # Load and process with pretty_midi
    midi = pretty_midi.PrettyMIDI(midi_path)
    temp_mid_path = f'/tmp/temp_pretty_{os.getpid()}_{int(time.time())}.mid'
    midi.write(temp_mid_path)

    # Reopen with mido to inject fixed tempo
    mid = MidiFile(temp_mid_path)
    new_mid = MidiFile()
    tempo_meta = MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0)

    for i, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        if i == 0:
            new_track.append(tempo_meta)
        for msg in track:
            if not (msg.type == 'set_tempo'):
                new_track.append(msg)
        new_mid.tracks.append(new_track)

    new_mid.save(output_midi)

    # Convert to WAV
    subprocess.run([
        "fluidsynth", "-T", "wav", "-F", output_wav, SOUNDFONT_PATH, output_midi
    ], check=True)

    return output_midi, output_wav


def compute_best_tempos(scene_changes: list) -> list:
    """
    Compute optimal tempo for each scene to align scene boundaries with musical bars.
    """
    MIN_TEMPO = 70
    MAX_TEMPO = 160
    MAX_TEMPO_JUMP = 20

    tempos = []

    for i in range(len(scene_changes)):
        if i == len(scene_changes) - 1:
            break
        duration = scene_changes[i+1] - scene_changes[i]
        best = None
        best_score = float('inf')

        for bpm in range(MIN_TEMPO, MAX_TEMPO + 1):
            seconds_per_beat = 60 / bpm
            beats = duration / seconds_per_beat

            # Prefer durations that land near exact beats
            residual = abs(round(beats) - beats)
            full_bars = beats / 4
            bar_residual = abs(round(full_bars) - full_bars)

            score = residual + (bar_residual * 0.5)

            # Penalize large tempo jumps
            if tempos:
                jump = abs(bpm - tempos[-1])
                if jump > MAX_TEMPO_JUMP:
                    score += (jump - MAX_TEMPO_JUMP) * 0.3

            if score < best_score:
                best = bpm
                best_score = score

        tempos.append(best)

    return tempos


def apply_automation_to_midi(
    midi_path: str,
    scene_start: float,
    scene_duration: float,
    track_automation: list,
    output_path: str,
    total_duration: float,
    scene_tempo: float
):
    """
    Apply volume automation to MIDI notes within a specific scene window.
    Time values in track_automation are normalized 0-1 relative to scene duration.
    """
    # 1. Convert normalized automation times to absolute seconds
    absolute_automation = [
        (scene_start + (t * scene_duration), v)
        for t, v in track_automation
    ]

    # 2. Ensure coverage of full scene duration
    if not absolute_automation:
        absolute_automation = [
            (scene_start, 0.5),
            (scene_start + scene_duration, 0.5)
        ]
    else:
        # Add start point if missing
        if absolute_automation[0][0] > scene_start:
            absolute_automation.insert(0, (scene_start, absolute_automation[0][1]))
        # Add end point if missing
        if absolute_automation[-1][0] < scene_start + scene_duration:
            absolute_automation.append((scene_start + scene_duration, absolute_automation[-1][1]))

    print(f"🎛 Scene {scene_start:.1f}-{scene_start+scene_duration:.1f}s automation:")
    for t, v in absolute_automation:
        print(f"  {t:.2f}s: {v:.2f}")

    # 3. Process MIDI with tempo and automation
    mid = MidiFile(midi_path)
    new_mid = MidiFile()
    ticks_per_beat = mid.ticks_per_beat

    for track in mid.tracks:
        new_track = MidiTrack()
        abs_time = 0.0  # Tracks absolute time in seconds

        # Inject scene tempo at start
        new_track.append(MetaMessage(
            'set_tempo',
            tempo=mido.bpm2tempo(scene_tempo),
            time=0
        ))
        for msg in track:
            # Skip original tempo messages
            if msg.type == 'set_tempo':
                continue

            # Convert delta time to seconds
            delta_seconds = mido.tick2second(msg.time, ticks_per_beat, mido.bpm2tempo(scene_tempo))
            abs_time += delta_seconds

            # CRITICAL FIX: The incoming MIDI starts at time=0 (it was just generated for this scene)
            # We should keep ALL notes and only check if we're within the scene DURATION
            # Map abs_time (0-based) to scene_time (scene_start-based) for automation lookup
            scene_time = scene_start + abs_time

            if abs_time <= scene_duration:
                # Apply volume to note_on messages using scene_time for automation
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Find surrounding automation points
                    before = [p for p in absolute_automation if p[0] <= scene_time]
                    after = [p for p in absolute_automation if p[0] > scene_time]

                    # Calculate current volume
                    if before and after:
                        prev_time, prev_vol = max(before, key=lambda x: x[0])
                        next_time, next_vol = min(after, key=lambda x: x[0])
                        if next_time > prev_time:
                            ratio = (scene_time - prev_time) / (next_time - prev_time)
                            volume = prev_vol + ratio * (next_vol - prev_vol)
                        else:
                            volume = prev_vol
                    elif before:
                        volume = before[-1][1]
                    else:
                        volume = after[0][1] if after else 0.5

                    # Scale to MIDI velocity (1-127)
                    new_velocity = max(1, min(127, int(volume * 127)))
                    msg = msg.copy(velocity=new_velocity)

                new_track.append(msg)

        new_mid.tracks.append(new_track)

    new_mid.save(output_path)
    print(f"✅ Saved automated MIDI: {output_path}")


def concatenate_midi_scenes(
    scene_midi_paths: dict,
    scene_durations: list,
    output_path: str,
    soundfont_path: str = None
) -> str:
    """
    Concatenate multiple scene MIDIs into one long MIDI file.
    Each scene MIDI is trimmed/padded to match its exact duration.

    Args:
        scene_midi_paths: Dict mapping scene_idx -> midi_file_path
        scene_durations: List of scene durations in seconds
        output_path: Where to save the concatenated MIDI
        soundfont_path: Optional soundfont for debug WAV rendering

    Returns:
        Path to concatenated MIDI file
    """
    from mido import Message

    combined_midi = MidiFile(ticks_per_beat=480)
    combined_track = MidiTrack()
    combined_midi.tracks.append(combined_track)

    print(f"\n🎼 Concatenating {len(scene_midi_paths)} scene MIDIs")
    cumulative_duration = 0.0
    cumulative_ticks = 0

    for scene_idx in sorted(scene_midi_paths.keys()):
        midi_path = scene_midi_paths[scene_idx]
        duration_sec = scene_durations[scene_idx]

        print(f"\n{'─'*60}")
        print(f"🎬 Scene {scene_idx}: Duration = {duration_sec:.2f}s")
        print(f"   MIDI path: {midi_path}")
        print(f"   Concatenation point: {cumulative_duration:.3f}s → {cumulative_duration + duration_sec:.3f}s")
        print(f"   Cumulative ticks before: {cumulative_ticks}")

        # Load scene MIDI
        scene_midi = MidiFile(midi_path)

        # Get tempo from first track
        tempo = 500000  # Default 120 BPM
        for msg in scene_midi.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

        bpm = mido.tempo2bpm(tempo)
        print(f"   Tempo: {bpm:.1f} BPM")

        # CRITICAL FIX: Add tempo change message at the START of each scene
        # For scene 0: time=0 (first message)
        # For scene 1+: time=0 (happens immediately after last note of previous scene)
        if scene_idx == 0:
            combined_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        else:
            # Insert tempo change with time=0 (relative to end of previous scene)
            combined_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
            print(f"   🎵 Inserted tempo change: {bpm:.1f} BPM at scene boundary")

        # Calculate max ticks for this scene
        ticks_per_second = bpm * combined_midi.ticks_per_beat / 60.0
        max_ticks = int(duration_sec * ticks_per_second)
        print(f"   Max ticks for scene: {max_ticks} ({duration_sec:.3f}s at {ticks_per_second:.1f} ticks/s)")
        current_ticks = 0

        # Track if this is the first message of this scene
        first_message_in_scene = True
        messages_copied = 0

        # Copy messages from scene MIDI - MERGE all tracks into one
        for track in scene_midi.tracks:
            for msg in track:
                if msg.is_meta and msg.type == 'set_tempo':
                    continue  # Skip tempo messages (already added)

                if msg.is_meta:
                    continue  # Skip other meta messages

                # Limit message time to stay within scene duration
                msg_time = msg.time

                # CRITICAL FIX: First message of each scene (except scene 0) should have time=0
                # This ensures the new scene starts immediately after the previous one
                if first_message_in_scene:
                    if scene_idx > 0:
                        print(f"   🎯 First message of scene {scene_idx}: type={msg.type}, setting time=0 (was {msg.time})")
                        msg_time = 0
                    first_message_in_scene = False

                if current_ticks + msg_time > max_ticks:
                    msg_time = max(0, max_ticks - current_ticks)

                msg_copy = msg.copy(time=msg_time)
                if hasattr(msg_copy, 'channel'):
                    msg_copy.channel = 0  # Force channel 0
                combined_track.append(msg_copy)
                current_ticks += msg_time
                messages_copied += 1

                if current_ticks >= max_ticks:
                    print(f"   ✅ Reached max_ticks after {messages_copied} messages")
                    break
            # Don't break between tracks - continue copying from all tracks
            # if current_ticks >= max_ticks:
            #     break

        print(f"   📊 Copied {messages_copied} messages from {len(scene_midi.tracks)} tracks")

        # Pad with silence if scene ended early
        if current_ticks < max_ticks:
            padding_ticks = max_ticks - current_ticks
            combined_track.append(Message('note_off', note=0, velocity=0, time=padding_ticks, channel=0))
            print(f"   ⚠️ Padded with {padding_ticks} ticks ({padding_ticks/ticks_per_second:.3f}s)")
            current_ticks = max_ticks

        # Flush all notes at scene boundary
        for note in range(128):
            combined_track.append(Message('note_off', note=note, velocity=0, time=0, channel=0))

        # Update cumulative counters
        cumulative_ticks += current_ticks
        cumulative_duration += duration_sec
        actual_duration = current_ticks / ticks_per_second
        print(f"   ✅ Scene {scene_idx} complete:")
        print(f"      Ticks copied: {current_ticks} (target: {max_ticks})")
        print(f"      Actual duration: {actual_duration:.3f}s (target: {duration_sec:.3f}s)")
        print(f"      Cumulative ticks: {cumulative_ticks}")
        print(f"      Cumulative duration: {cumulative_duration:.3f}s")

        combined_track.append(Message('control_change', control=123, value=0, time=0, channel=0))

    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 CONCATENATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total scenes concatenated: {len(scene_midi_paths)}")
    print(f"Final cumulative duration: {cumulative_duration:.3f}s")
    print(f"Final cumulative ticks: {cumulative_ticks}")
    print(f"{'='*60}\n")

    # Save concatenated MIDI
    combined_midi.save(output_path)
    print(f"✅ Saved concatenated MIDI: {output_path}")

    # Render debug WAV with selected soundfont
    if soundfont_path:
        debug_wav_path = output_path.replace('.mid', '_debug.wav')
        print(f"\n🎼 Rendering debug WAV with soundfont: {Path(soundfont_path).name}")
        try:
            subprocess.run([
                "fluidsynth", "-ni", "-g", "0.5", "-F", debug_wav_path,
                soundfont_path, output_path
            ], check=True, capture_output=True)
            print(f"✅ Debug WAV saved: {debug_wav_path}")
        except Exception as e:
            print(f"⚠️ Failed to render debug WAV: {e}")

    # Return MIDI path (no rendering - will be handled by caller with selected soundfont)
    return output_path

# ------------------------------------------------------------------------------
# Conditioning I/O
# ------------------------------------------------------------------------------
def extract_conditioning_from_audio(audio_path: str, output_dir: str = "./extracted_conditioning", instrument_group: str = None) -> dict:
    # Check memory cache first (include instrument group in cache key)
    cache_context = f"instrument_{instrument_group}" if instrument_group else "no_instrument"
    cache_key = _get_file_cache_key(audio_path, cache_context)
    print(f"🔍 Cache key for {Path(audio_path).name} ({cache_context}): {cache_key[:8]}...")
    print(f"🔍 Cache contains {len(CONDITIONING_CACHE)} entries: {list(CONDITIONING_CACHE.keys())}")

    if cache_key in CONDITIONING_CACHE:
        cached_result = CONDITIONING_CACHE[cache_key]
        if _is_cache_valid(cached_result, audio_path):
            print(f"✅ Using cached conditioning from memory for: {Path(audio_path).name}")
            return cached_result
        else:
            # Remove invalid cache entry
            print(f"⚠️ Cache entry invalid for {Path(audio_path).name}, removing...")
            del CONDITIONING_CACHE[cache_key]

    # Check manifest paths
    rec = _find_manifest_record_by_audio(audio_path)
    if rec:
        paths = {}
        prp = rec.get("piano_roll_path") or rec.get("pianoroll_path")
        c = rec.get("conditioning_paths", {}) or {}
        paths["piano_roll"] = prp or c.get("piano_roll") or c.get("pianoroll")
        paths["amp"]        = c.get("amp")
        paths["rframe"]     = c.get("rframe")
        paths["rbend"]      = c.get("rbend")
        paths["encodec"]    = rec.get("encodec_path") or c.get("encodec")
        if all(paths.get(k) and os.path.exists(paths[k]) for k in ["piano_roll","amp","rframe","rbend","encodec"]):
            result = {"paths": paths}
            CONDITIONING_CACHE[cache_key] = result
            print("✅ Using conditioning from manifest paths.")
            return result
        print("⚠️ Manifest record found but some files are missing; falling back to local extraction.")

    # Check disk cache (include instrument group in directory name)
    base_stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in Path(audio_path).stem)[:128] or "audio"
    dir_stem = f"{base_stem}_{instrument_group}" if instrument_group else base_stem
    out_dir = Path(output_dir) / dir_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    req = [out_dir/f"{base_stem}.pianoroll.npy", out_dir/f"{base_stem}.amp.npy", out_dir/f"{base_stem}.rframe.npy",
           out_dir/f"{base_stem}.rbend.npy", out_dir/f"{base_stem}.encodec.pt"]
    if all(x.exists() for x in req):
        result = {"dir": str(out_dir), "stem": base_stem}
        CONDITIONING_CACHE[cache_key] = result
        print(f"✅ Using disk-cached conditioning: {out_dir}")
        return result

    # Extract new conditioning
    print(f"🔄 Extracting conditioning for: {Path(audio_path).name}")
    cmd = ["python", "test_extract_local.py", "--input", str(audio_path), "--output", str(out_dir)]
    print(f"Running extraction: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr)
        raise RuntimeError("Extraction failed.")

    result = {"dir": str(out_dir), "stem": base_stem}
    CONDITIONING_CACHE[cache_key] = result
    print("✅ Conditioning extracted successfully.")
    return result

def _np_load_first(*candidates):
    for p in candidates:
        if p is not None and os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(f"None of: {candidates}")

def load_conditioning(extraction: dict, window_slow: int):
    if "paths" in extraction:
        paths = extraction["paths"]
        pr  = _np_load_first(paths.get("piano_roll"), paths.get("pianoroll"))
        amp = _np_load_first(paths.get("amp"))
        rfr = _np_load_first(paths.get("rframe"))
        rbd = _np_load_first(paths.get("rbend"))
        enc_data = torch.load(paths["encodec"], map_location="cpu")
    else:
        out_dir = Path(extraction["dir"]); stem = extraction["stem"]
        nested = out_dir / stem
        pr  = _np_load_first(out_dir/f"{stem}.pianoroll.npy", out_dir/f"{stem}.piano_roll.npy",
                             nested/f"{stem}.pianoroll.npy", nested/f"{stem}.piano_roll.npy")
        amp = _np_load_first(out_dir/f"{stem}.amp.npy", nested/f"{stem}.amp.npy")
        rfr = _np_load_first(out_dir/f"{stem}.rframe.npy", nested/f"{stem}.rframe.npy")
        rbd = _np_load_first(out_dir/f"{stem}.rbend.npy", nested/f"{stem}.rbend.npy")
        enc_path = out_dir/f"{stem}.encodec.pt"
        if not enc_path.exists():
            for cand in [out_dir/f"{stem}.encodec_tokens.pt", nested/f"{stem}.encodec.pt", nested/f"{stem}.encodec_tokens.pt"]:
                if cand.exists():
                    enc_path = cand; break
        enc_data = torch.load(enc_path, map_location="cpu")

    # standardize encodec tensor to [1,C,T] long
    if isinstance(enc_data, (list, tuple)):
        enc = None
        for obj in (enc_data, len(enc_data) and enc_data[0], len(enc_data) and isinstance(enc_data[0], (list,tuple)) and enc_data[0][0]):
            if torch.is_tensor(obj):
                enc = obj; break
        if enc is None: raise RuntimeError("Unrecognized encodec token structure")
    else:
        enc = enc_data
    if enc.ndim == 2:
        enc = enc.unsqueeze(0)

    # pad/trim to window_slow
    def _pad_arr(x, L):
        if x.shape[-1] >= L: return x[..., :L]
        pad = [(0,0)]*(x.ndim-1) + [(0, L - x.shape[-1])]
        return np.pad(x, pad, mode="constant")

    pr  = _pad_arr(pr,  window_slow)
    amp = _pad_arr(amp, window_slow)
    rfr = _pad_arr(rfr, window_slow)
    rbd = _pad_arr(rbd, window_slow)

    return pr, amp, rfr, rbd, enc.long()

# ------------------------------------------------------------------------------
# Latent extraction helpers
# ------------------------------------------------------------------------------
def extract_ground_truth_latents(audio_path: str, model: Pipeline) -> torch.Tensor:
    """Extract ground truth latents from audio using the DCAE encoder."""
    # Check latent cache first
    cache_key = _get_file_cache_key(audio_path)
    if cache_key in LATENT_CACHE:
        print(f"✅ Using cached ground truth latents for: {Path(audio_path).name}")
        return LATENT_CACHE[cache_key]

    try:
        print(f"🔄 Extracting ground truth latents for: {Path(audio_path).name}")

        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to stereo if mono
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Normalize audio
        waveform = waveform / (waveform.abs().max() + 1e-8)

        # Move to device and add batch dimension
        device = next(model.parameters()).device
        waveform = waveform.to(device)
        audio_batch = waveform.unsqueeze(0).float()
        audio_lengths = torch.tensor([waveform.shape[-1]], device=device)

        # Extract latents using DCAE
        with torch.no_grad():
            latents, latent_lengths = model.dcae.encode(
                audios=audio_batch,
                audio_lengths=audio_lengths,
                sr=sr
            )

        # Remove batch dimension
        latents = latents.squeeze(0)

        # Cache the latents (keep on CPU to save GPU memory)
        LATENT_CACHE[cache_key] = latents.cpu()
        print(f"✅ Extracted and cached ground truth latents: {latents.shape}")
        return latents

    except Exception as e:
        print(f"⚠️ Failed to extract ground truth latents: {e}")
        print("   Falling back to pure noise generation")
        return None

# ------------------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------------------
def _bank_softplus_resized_compat(model, H: int, device, dtype):
    if hasattr(model, "_bank_softplus_resized"):
        return model._bank_softplus_resized(H, device, dtype)
    W = getattr(model, "pitch2h_bank", None)
    if W is None:
        W = torch.ones(H, 128, device=device, dtype=dtype) * 0.01
    else:
        W = W.to(device=device, dtype=dtype)
    if W.shape[0] != H:
        W = F.interpolate(W.T.unsqueeze(0), size=H, mode="linear", align_corners=False).squeeze(0).T
    return F.softplus(W)

def _adapter_gain_scale_compat(model) -> float:
    if hasattr(model, "_adapter_gain_scale"):
        return model._adapter_gain_scale()
    steps = int(getattr(model, "adapter_warmup_steps", 2000))
    gstep = int(getattr(model, "global_step", 0))
    return float(min(1.0, (gstep + 1) / max(1, steps)))

@torch.no_grad()
def _prep_ctrl_residuals_if_enabled(model: Pipeline, pr_128: torch.Tensor, amp_1t: torch.Tensor, T_lat: int):
    if not getattr(model.hparams, "use_ctrl_branch", False):
        return None
    if not hasattr(model, "ctrlnet"):
        return None
    if amp_1t.shape[-1] != pr_128.shape[-1]:
        amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
    ctrl_in = torch.cat([pr_128, amp_1t], dim=1)  # [B,129,T]
    res_list = model.ctrlnet(ctrl_in, T_out_list=[T_lat] * len(model.ctrlnet.to_blocks))
    scale = float(getattr(model.hparams, "control_scale", 1.0))
    return [r * scale for r in res_list]

# ------------------------------------------------------------------------------
# MIDI conversion and voice separation
# ------------------------------------------------------------------------------
def piano_roll_to_midi(piano_roll, output_path, fps=43.066, program=0, velocity=80, min_note_duration=0.1, tempo=120.0):
    """
    Convert piano roll to MIDI file.
    Args:
        piano_roll: numpy array of shape [128, T] representing MIDI piano roll
        output_path: path to save MIDI file
        fps: frames per second (matches conditioning extraction fps)
        program: MIDI program number (instrument)
        velocity: MIDI velocity
        min_note_duration: minimum note duration in seconds to filter false detections
        tempo: BPM tempo for the MIDI file
    Returns:
        path to saved MIDI file
    """
    # Create a PrettyMIDI object with specified tempo
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Create an instrument instance
    instrument = pretty_midi.Instrument(program=program)

    # Convert piano roll to notes
    notes_added = 0
    notes_filtered = 0

    for pitch in range(128):
        # Find note onsets and offsets
        note_events = piano_roll[pitch] > 0.1  # threshold for note detection

        if not np.any(note_events):
            continue

        # Find transitions
        diff = np.diff(np.concatenate(([False], note_events, [False])).astype(int))
        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]

        # Create notes with length filtering
        for onset, offset in zip(onsets, offsets):
            start_time = onset / fps
            end_time = offset / fps
            note_duration = end_time - start_time

            # Filter out very short notes (false detections)
            if note_duration < min_note_duration:
                notes_filtered += 1
                continue

            # Ensure minimum duration for valid notes
            if note_duration < 0.05:
                end_time = start_time + 0.05

            note = pretty_midi.Note(
                velocity=int(velocity * piano_roll[pitch, onset:offset].mean()),
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
            notes_added += 1

    # Add instrument to MIDI
    midi.instruments.append(instrument)

    # Save MIDI file
    midi.write(str(output_path))
    print(f"✅ Saved MIDI: {output_path} ({notes_added} notes, filtered {notes_filtered} short notes)")
    return str(output_path)

def save_basic_pitch_midi_with_voices(audio_file, subgroup=None, progress=None, tempo=120.0):
    """
    Save Basic Pitch MIDI from conditioning extraction with voice separation.
    Args:
        audio_file: input audio file path
        subgroup: instrument subgroup for soundfont selection
        progress: optional progress callback
        tempo: BPM tempo for output MIDI files
    Returns:
        dict with main MIDI path and voice MIDI paths
    """
    if audio_file is None:
        raise gr.Error("Please upload an audio file first.")

    if progress:
        progress(0, desc="Extracting conditioning...")

    # Extract conditioning (which includes Basic Pitch piano roll)
    extraction = extract_conditioning_from_audio(audio_file)
    win_slow = 1024  # default window size
    pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

    if progress:
        progress(0.3, desc="Creating output directory...")

    # CRITICAL FIX: Always create output in /home/arlo/Data/miditest/ for debugging
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    audio_stem = Path(audio_file).stem
    debug_dir = Path("/home/arlo/Data/miditest")
    debug_dir.mkdir(exist_ok=True)
    output_dir = debug_dir / f"{timestamp}_{audio_stem}"
    voices_dir = output_dir / "voices"
    output_dir.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(exist_ok=True)

    # Also create the original location for backward compatibility
    original_output_dir = Path("./midi_exports") / f"{timestamp}_{audio_stem}"
    original_voices_dir = original_output_dir / "voices"
    original_output_dir.mkdir(parents=True, exist_ok=True)
    original_voices_dir.mkdir(exist_ok=True)

    if progress:
        progress(0.5, desc="Saving main MIDI...")

    # Save main MIDI file (both locations)
    main_midi_path = output_dir / f"{audio_stem}_basicpitch.mid"
    piano_roll_to_midi(pr, main_midi_path, tempo=tempo)

    # Also save to original location
    original_main_midi = original_output_dir / f"{audio_stem}_basicpitch.mid"
    piano_roll_to_midi(pr, original_main_midi, tempo=tempo)

    if progress:
        progress(0.6, desc="Separating voices...")

    # Separate voices using existing function
    voices = separate_piano_roll_voices(pr)

    if progress:
        progress(0.8, desc="Saving voice MIDI files...")

    # Save individual voice MIDI files with note length filtering (both locations)
    voice_midi_paths = []
    for i, voice_pr in enumerate(voices):
        # Debug location (primary)
        voice_path = voices_dir / f"{audio_stem}_voice_{i+1}.mid"
        piano_roll_to_midi(voice_pr, voice_path, program=0, velocity=80, min_note_duration=0.1, tempo=tempo)
        voice_midi_paths.append(str(voice_path))

        # Original location (backup)
        original_voice_path = original_voices_dir / f"{audio_stem}_voice_{i+1}.mid"
        piano_roll_to_midi(voice_pr, original_voice_path, program=0, velocity=80, min_note_duration=0.1, tempo=tempo)

    if progress:
        progress(0.9, desc="Rendering FluidSynth debug audio for voices...")

    # Render FluidSynth debug audio for each voice
    debug_audio_paths = render_multitrack_debug_audio(voice_midi_paths, str(output_dir), audio_stem, instrument_group=subgroup)

    if progress:
        progress(1.0, desc="Done!")

    result = {
        "main_midi": str(main_midi_path),
        "voice_midis": voice_midi_paths,
        "debug_audio_paths": debug_audio_paths,
        "output_dir": str(output_dir),
        "voice_count": len(voices),
        "debug_dir": str(output_dir),  # For debugging reference
        "original_dir": str(original_output_dir)  # For backward compatibility
    }

    print(f"🎼 Saved {len(voices)} voice MIDI files to: {voices_dir}")
    return result

def create_voices_zip(voice_midi_paths, output_dir):
    """Create a ZIP file containing all voice MIDI files."""
    import zipfile

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    zip_path = Path(output_dir) / f"voices_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for voice_path in voice_midi_paths:
            voice_file = Path(voice_path)
            if voice_file.exists():
                # Add file to zip with just the filename (no path)
                zipf.write(voice_path, voice_file.name)

    print(f"✅ Created voices ZIP: {zip_path}")
    return str(zip_path)

def create_audio_voices_zip(voice_audio_paths, output_dir="./generated_ui"):
    """Create a ZIP file containing all voice audio files."""
    import zipfile

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    zip_path = Path(output_dir) / f"audio_voices_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for voice_path in voice_audio_paths:
            voice_file = Path(voice_path)
            if voice_file.exists():
                # Add file to zip with just the filename (no path)
                zipf.write(voice_path, voice_file.name)

    print(f"✅ Created audio voices ZIP: {zip_path}")
    return str(zip_path)

def create_combined_voices_midi(voice_midi_paths, output_dir, audio_stem):
    """Create a combined MIDI file with each voice on a separate channel."""
    combined_midi = pretty_midi.PrettyMIDI()

    # Load each voice and assign to different channels/instruments
    for i, voice_path in enumerate(voice_midi_paths):
        try:
            voice_midi = pretty_midi.PrettyMIDI(voice_path)
            if voice_midi.instruments:
                # Copy the first instrument from each voice MIDI
                voice_instrument = voice_midi.instruments[0]

                # Create new instrument with different program and channel
                combined_instrument = pretty_midi.Instrument(
                    program=i % 128,  # Different program for each voice
                    is_drum=False,
                    name=f"Voice {i+1}"
                )

                # Copy all notes
                combined_instrument.notes = voice_instrument.notes.copy()
                combined_midi.instruments.append(combined_instrument)

        except Exception as e:
            print(f"⚠️ Error loading voice {i+1}: {e}")
            continue

    # CRITICAL FIX: Always save to /home/arlo/Data/miditest/ for debugging
    debug_dir = Path("/home/arlo/Data/miditest")
    debug_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    combined_path = debug_dir / f"{audio_stem}_combined_voices_{timestamp}.mid"
    combined_midi.write(str(combined_path))

    # Also save to the original output directory
    original_path = Path(output_dir) / f"{audio_stem}_combined_voices_{timestamp}.mid"
    combined_midi.write(str(original_path))

    print(f"✅ Created combined voices MIDI: {combined_path}")
    print(f"✅ Also saved to: {original_path}")
    return str(combined_path)  # Return debug path for easier access

def handle_midi_download(audio_file, subgroup=None, progress=gr.Progress(track_tqdm=True)):
    """Handle MIDI download button click."""
    try:
        # If input is already MIDI, handle differently
        if is_midi_file(audio_file):
            progress(0.1, desc="Analyzing MIDI file...")

            # Check if multitrack
            is_multi, track_count, _ = is_multitrack_midi(audio_file)

            # Create output directory
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            audio_stem = Path(audio_file).stem
            debug_dir = Path("/home/arlo/Data/miditest")
            debug_dir.mkdir(exist_ok=True)
            output_dir = debug_dir / f"{timestamp}_{audio_stem}_midi_input"
            voices_dir = output_dir / "voices"
            tracks_dir = output_dir / "tracks"
            output_dir.mkdir(parents=True, exist_ok=True)
            voices_dir.mkdir(exist_ok=True)
            tracks_dir.mkdir(exist_ok=True)

            if is_multi:
                progress(0.2, desc=f"Processing {track_count} tracks...")

                # Use multitrack processing
                win_slow = 1024
                multitrack_data = midi_to_multitrack_piano_rolls(audio_file, win_slow)

                progress(0.4, desc="Saving individual track MIDI files...")
                # Save each track as a separate MIDI file
                track_midi_paths = []
                original_tempo = multitrack_data.get('original_tempo', 120.0)
                for i, (track_pr, track_info) in enumerate(zip(multitrack_data['track_piano_rolls'], multitrack_data['track_info'])):
                    track_name = track_info['name'].replace(' ', '_').replace('/', '_')
                    track_path = tracks_dir / f"{audio_stem}_track_{i+1}_{track_name}.mid"
                    piano_roll_to_midi(track_pr, track_path, program=track_info['program'], velocity=80, min_note_duration=0.1, tempo=original_tempo)
                    track_midi_paths.append(str(track_path))
                    print(f"   Saved track {i+1}: {track_info['name']} ({track_info['note_count']} notes)")

                # For multitrack MIDI, always use each track as one voice
                # Don't apply separation logic - each track is already a separate voice
                polyphony_data = multitrack_data.get('polyphony_analysis', {})

                # For info purposes only - we don't use this for separation decision
                all_monophonic = polyphony_data.get('polyphonic_tracks', 1) == 0

                progress(0.6, desc="Using each track as one voice...")
                print(f"🎵 Multitrack MIDI - using {len(track_midi_paths)} tracks as {len(track_midi_paths)} voices (no voice separation)")
                print("   Each track will be treated as one voice, regardless of internal polyphony")

                # Use track files as voice files - one track = one voice
                voice_midi_paths = track_midi_paths.copy()

                # Save main MIDI (copy original)
                main_midi_path = output_dir / f"{audio_stem}_original.mid"
                shutil.copy(audio_file, main_midi_path)

                # Render FluidSynth debug audio for each track
                progress(0.8, desc="Rendering FluidSynth debug audio for individual tracks...")
                debug_audio_paths = render_multitrack_debug_audio(track_midi_paths, str(output_dir), audio_stem, instrument_group=subgroup)

                result = {
                    "main_midi": str(main_midi_path),
                    "voice_midis": voice_midi_paths,
                    "track_midis": track_midi_paths,
                    "debug_audio_paths": debug_audio_paths,
                    "track_info": multitrack_data['track_info'],
                    "polyphony_analysis": polyphony_data,
                    "debug_dir": str(output_dir),
                    "voice_count": len(voice_midi_paths),
                    "track_count": track_count,
                    "is_multitrack": True,
                    "all_monophonic": all_monophonic
                }

            else:
                progress(0.2, desc="Processing single track MIDI file...")
                # Use standard single track processing
                win_slow = 1024
                pr, _, _, _, _ = midi_to_piano_roll_conditioning(audio_file, win_slow)

                # Extract tempo for single track processing too
                original_tempo = extract_midi_tempo(audio_file)

                progress(0.4, desc="Separating voices...")
                voices = separate_piano_roll_voices(pr)

                progress(0.6, desc="Saving voice MIDI files...")
                # Save individual voice MIDI files
                voice_midi_paths = []
                for i, voice_pr in enumerate(voices):
                    voice_path = voices_dir / f"{audio_stem}_voice_{i+1}.mid"
                    piano_roll_to_midi(voice_pr, voice_path, program=0, velocity=80, min_note_duration=0.1, tempo=original_tempo)
                    voice_midi_paths.append(str(voice_path))

                # Save main MIDI (copy original)
                main_midi_path = output_dir / f"{audio_stem}_original.mid"
                shutil.copy(audio_file, main_midi_path)

                # Render FluidSynth debug audio for each voice
                progress(0.8, desc="Rendering FluidSynth debug audio for individual voices...")
                debug_audio_paths = render_multitrack_debug_audio(voice_midi_paths, str(output_dir), audio_stem, instrument_group=subgroup)

                result = {
                    "main_midi": str(main_midi_path),
                    "voice_midis": voice_midi_paths,
                    "track_midis": [],
                    "debug_audio_paths": debug_audio_paths,
                    "track_info": [],
                    "debug_dir": str(output_dir),
                    "voice_count": len(voices),
                    "track_count": 1,
                    "is_multitrack": False
                }

        else:
            # Original audio file processing (use default tempo for audio)
            result = save_basic_pitch_midi_with_voices(audio_file, subgroup=subgroup, progress=progress, tempo=120.0)
            audio_stem = Path(audio_file).stem

        if progress:
            progress(0.9, desc="Creating combined files...")

        # Create ZIP file for voices (use debug directory)
        voices_zip = create_voices_zip(result["voice_midis"], result["debug_dir"])

        # Create combined voices MIDI for comparison (use debug directory)
        combined_midi = create_combined_voices_midi(result["voice_midis"], result["debug_dir"], audio_stem)

        # Create info text
        if result.get("is_multitrack", False):
            # Build track info with polyphony status
            track_info_text = "\n".join(
                f"• {info['name']}: {info['note_count']} notes (Program {info['program']}) "
                f"[{'monophonic' if info.get('is_monophonic', False) else 'polyphonic'}]"
                for info in result["track_info"]
            )
            track_files_text = "\n".join(f"• {Path(p).name}" for p in result["track_midis"])

            # Polyphony summary
            polyphony_data = result.get("polyphony_analysis", {})
            mono_count = polyphony_data.get("monophonic_tracks", 0)
            poly_count = polyphony_data.get("polyphonic_tracks", 0)
            all_mono = result.get("all_monophonic", False)

            # For multitrack MIDI, tracks are always used as voices (no separation)
            voice_section = f"""
🎵 Multitrack Processing: Each track is one voice!
Track files are used directly as voice files (no voice separation applied).
Each track will be treated as a separate voice, regardless of internal polyphony.

Individual track files (serving as voice files):
{track_files_text}"""

            info_text = f"""MIDI Export Complete!
📂 Output Directory: {Path(result['debug_dir']).name}
🎼 Original MIDI: {Path(result['main_midi']).name}
🎹 Combined Voices: {Path(combined_midi).name}
🎵 Voice Count: {result['voice_count']}
🎶 Track Count: {result['track_count']}
📦 Voices ZIP: {Path(voices_zip).name}

🎼 MULTITRACK MIDI DETECTED!
This file contains {result['track_count']} separate tracks/instruments.
Polyphony: {mono_count} monophonic, {poly_count} polyphonic

Track Information:
{track_info_text}
{voice_section}"""
        else:
            info_text = f"""MIDI Export Complete!
📂 Output Directory: {Path(result['debug_dir']).name}
🎼 Original MIDI: {Path(result['main_midi']).name}
🎹 Combined Voices: {Path(combined_midi).name}
🎵 Voice Count: {result['voice_count']}
📦 Voices ZIP: {Path(voices_zip).name}

Compare the original vs. voice-separated results:
• Original: All notes on single track
• Combined: Each voice on separate MIDI channel
• Individual: Separate files for each voice

Individual voice files:
""" + "\n".join(f"• {Path(p).name}" for p in result["voice_midis"])

        if progress:
            progress(1.0, desc="Done!")

        # Prepare debug audio files for UI (up to 6 files)
        debug_audio_outputs = []
        debug_audio_paths = result.get("debug_audio_paths", [])

        debug_info_text = f"FluidSynth rendered {len(debug_audio_paths)} individual files for debugging"
        if debug_audio_paths:
            debug_info_text += ":\n" + "\n".join(f"• {Path(p).name}" for p in debug_audio_paths)

        # Create gr.update() objects for each debug audio file slot
        for i in range(6):
            if i < len(debug_audio_paths):
                # Make visible and set the file
                debug_audio_outputs.append(gr.update(value=debug_audio_paths[i], visible=True))
            else:
                # Keep hidden
                debug_audio_outputs.append(gr.update(value=None, visible=False))

        return (result["main_midi"], voices_zip, combined_midi, info_text, debug_info_text) + tuple(debug_audio_outputs)

    except Exception as e:
        error_msg = f"Error exporting MIDI: {str(e)}"
        print(f"❌ {error_msg}")
        # Return proper gr.update() objects for error case
        error_debug_outputs = [gr.update(value=None, visible=False) for _ in range(6)]
        return (None, None, None, error_msg, "Error occurred - no debug audio available") + tuple(error_debug_outputs)

# ------------------------------------------------------------------------------
# Voice separation for monophonic mode - NOTE-BASED APPROACH
# ------------------------------------------------------------------------------

# Helper classes and functions for note-based processing
class NoteEvent:
    def __init__(self, pitch, start_time, end_time, velocity=80):
        self.pitch = pitch
        self.start_time = start_time
        self.end_time = end_time
        self.velocity = velocity

    def __repr__(self):
        return f"Note({self.pitch}, {self.start_time:.3f}-{self.end_time:.3f})"

def extract_note_events_from_piano_roll(piano_roll, fps=43.066):
    """Extract actual note events from piano roll"""
    notes = []

    for pitch in range(128):
        # Find note onsets and offsets
        note_events = piano_roll[pitch] > 0.1
        if not np.any(note_events):
            continue

        # Find transitions
        diff = np.diff(np.concatenate(([False], note_events, [False])).astype(int))
        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]

        # Create note events
        for onset, offset in zip(onsets, offsets):
            start_time = onset / fps
            end_time = offset / fps
            if end_time - start_time >= 0.02:  # Minimum duration
                notes.append(NoteEvent(pitch, start_time, end_time))

    return sorted(notes, key=lambda n: n.start_time)

def group_notes_by_chord_changes(note_events, tolerance=0.05):
    """Group notes that start at approximately the same time into chords"""
    if not note_events:
        return []

    chords = []
    current_chord = [note_events[0]]
    current_time = note_events[0].start_time

    for note in note_events[1:]:
        if abs(note.start_time - current_time) <= tolerance:
            # Same chord
            current_chord.append(note)
        else:
            # New chord
            if current_chord:
                chords.append(current_chord)
            current_chord = [note]
            current_time = note.start_time

    # Add last chord
    if current_chord:
        chords.append(current_chord)

    return chords

def assign_first_chord_by_register(pitches, num_voices):
    """Assign first chord by register separation"""
    assignments = {i: None for i in range(num_voices)}
    sorted_pitches = sorted(pitches, reverse=True)  # Highest first

    for i, pitch in enumerate(sorted_pitches):
        if i < num_voices:
            assignments[i] = pitch

    return assignments

def assign_note_to_voice(voice_piano_roll, note, original_piano_roll):
    """Assign a complete note to a voice piano roll"""
    fps = 43.066  # Should match the original fps
    start_frame = int(note.start_time * fps)
    end_frame = int(note.end_time * fps)

    # Copy the entire note duration from original to voice
    for frame in range(start_frame, min(end_frame, original_piano_roll.shape[1])):
        if original_piano_roll[note.pitch, frame] > 0.1:
            voice_piano_roll[note.pitch, frame] = original_piano_roll[note.pitch, frame]

def separate_piano_roll_voices_new(piano_roll):
    """
    NEW NOTE-BASED voice separation that processes actual note events.
    CRITICAL FIX: Processes notes instead of individual frames.
    """
    print(f"🎼 Input piano roll shape: {piano_roll.shape}")

    # Extract actual note events instead of processing every frame
    note_events = extract_note_events_from_piano_roll(piano_roll)
    if len(note_events) == 0:
        return [piano_roll]

    print(f"🎼 Found {len(note_events)} note events for voice separation")

    # Group note events by chord changes (simultaneous onsets)
    chord_changes = group_notes_by_chord_changes(note_events)
    print(f"🎼 Detected {len(chord_changes)} chord changes")

    # Analyze chord structure from note events
    from collections import Counter
    chord_sizes = [len(chord) for chord in chord_changes]
    max_voices = max(chord_sizes) if chord_sizes else 1

    # Get all unique pitches from note events
    all_pitches = sorted(set(note.pitch for note in note_events))
    pitch_range = max(all_pitches) - min(all_pitches) if all_pitches else 0

    chord_counts = Counter(chord_sizes)
    common_chord_size = chord_counts.most_common(1)[0][0] if chord_counts else 1

    print(f"🎼 Chord sizes: min={min(chord_sizes)}, max={max_voices}, common={common_chord_size}")
    print(f"🎼 Pitch range: {min(all_pitches) if all_pitches else 0}-{max(all_pitches) if all_pitches else 0} ({pitch_range} semitones)")

    # FIXED: Use only as many voices as needed for simultaneous notes
    target_voices = max_voices  # Use exactly the number of voices needed
    print(f"🎼 Using {target_voices} voices")

    # Initialize voices
    voices = [np.zeros_like(piano_roll) for _ in range(target_voices)]

    # Track voice assignments across chord changes
    voice_assignments = {}
    voice_identities = {}  # For long-term tracking

    print(f"🎼 Processing {len(chord_changes)} chord changes...")

    for i, current_chord in enumerate(chord_changes):
        current_time = current_chord[0].start_time  # Use first note's start time
        current_pitches = sorted([note.pitch for note in current_chord])
        print(f"\\n--- Chord {i}: time {current_time:.3f}s, pitches {current_pitches} ---")

        if i == 0:
            # First chord: assign by register
            assignments = assign_first_chord_by_register(current_pitches, target_voices)

            # Apply assignments to piano roll
            for voice_idx, pitch in assignments.items():
                if pitch is not None:
                    # Find all notes in current chord with this pitch and assign them
                    for note in current_chord:
                        if note.pitch == pitch:
                            assign_note_to_voice(voices[voice_idx], note, piano_roll)
                            print(f"   Initial: Voice {voice_idx} <- Pitch {pitch}")

                    # Track for continuity
                    voice_identities[voice_idx] = [(current_time, pitch)]

            voice_assignments[i] = assignments
        else:
            # Subsequent chords: simple register-based assignment for consistent voice leading
            assignments = assign_first_chord_by_register(current_pitches, target_voices)

            # Apply assignments
            for voice_idx, pitch in assignments.items():
                if pitch is not None and voice_idx < len(voices):
                    # Find all notes in current chord with this pitch and assign them
                    for note in current_chord:
                        if note.pitch == pitch:
                            assign_note_to_voice(voices[voice_idx], note, piano_roll)

                    # Track for continuity
                    if voice_idx not in voice_identities:
                        voice_identities[voice_idx] = []
                    voice_identities[voice_idx].append((current_time, pitch))
                    print(f"   Voice {voice_idx}: -> {pitch}")

            voice_assignments[i] = assignments

    # Notes are already fully assigned by assign_note_to_voice function
    print("🎼 Note assignment complete (no frame-level sustaining needed)")

    # Remove empty voices and verify note preservation
    final_voices = []
    voice_stats = []

    for i, voice in enumerate(voices):
        note_count = np.sum(voice > 0.1)
        if note_count > 0:
            voice_stats.append((i, note_count, voice))

    # Keep ALL voices with content to preserve all notes
    for voice_idx, note_count, voice in voice_stats:
        final_voices.append(voice)
        print(f"🎼 Voice {len(final_voices)}: {note_count} note events")

    if len(final_voices) == 0:
        final_voices = [piano_roll]

    # VERIFICATION: Check that we preserved all notes
    total_original_notes = np.sum(piano_roll > 0.1)
    total_separated_notes = sum(np.sum(voice > 0.1) for voice in final_voices)
    print(f"🔍 VERIFICATION: Original {total_original_notes} note events -> Separated {total_separated_notes} note events")
    if total_separated_notes < total_original_notes:
        missing_events = total_original_notes - total_separated_notes
        print(f"❌ STILL MISSING {missing_events} note events after separation!")
    else:
        print(f"✅ All note events preserved in voice separation")

    print(f"🎵 Successfully separated {len(final_voices)} voices from piano roll")
    return final_voices
def separate_piano_roll_voices(piano_roll):
    """
    Separate piano roll into individual voices using NOTE-BASED processing.
    CRITICAL FIX: Processes actual note events instead of individual frames.
    Args:
        piano_roll: numpy array of shape [128, T] representing MIDI piano roll
    Returns:
        list of piano roll arrays, each containing one voice
    """
    # Use the new note-based approach
    return separate_piano_roll_voices_new(piano_roll)

def assign_pitches_to_voices(current_pitches, prev_assignments, max_voices):
    """
    Assign current pitches to voices using Hungarian algorithm with strict register boundaries.
    """
    # Use the same algorithm as the enhanced continuity function
    return solve_voice_assignment(current_pitches, prev_assignments, {}, 0)

def solve_voice_assignment(current_pitches, prev_assignments, voice_identities, time_step):
    """
    Solve voice assignment using Hungarian algorithm with strict register boundaries.
    Completely prevents octave jumps by enforcing register-based voice separation.
    FIXED: Now ensures ALL pitches get assigned by allowing multiple pitches per voice.
    """
    num_voices = max(len(prev_assignments), len(current_pitches))  # Use only as many voices as needed
    num_pitches = len(current_pitches)

    if num_pitches == 0:
        return {i: None for i in range(num_voices)}

    # Create cost matrix
    cost_matrix = np.zeros((num_voices, num_pitches))

    for voice_idx in range(num_voices):
        prev_pitch = prev_assignments.get(voice_idx)

        if prev_pitch is None:
            # No previous assignment - use OVERLAPPING register-based assignment
            for pitch_idx, pitch in enumerate(current_pitches):
                # Same overlapping register boundaries as with previous assignments
                if pitch >= 80:
                    if voice_idx == 0:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx == 1:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 70:
                    if voice_idx == 1:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [0, 2]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 65:
                    if voice_idx == 2:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [1, 3]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 60:
                    if voice_idx == 3:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [2, 4]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 55:
                    if voice_idx == 4:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [3, 5]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                elif pitch >= 50:
                    if voice_idx == 5:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx in [4, 6]:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
                else:
                    if voice_idx == 6:
                        cost_matrix[voice_idx, pitch_idx] = 1
                    elif voice_idx == 5:
                        cost_matrix[voice_idx, pitch_idx] = 51
                    else:
                        cost_matrix[voice_idx, pitch_idx] = 201
        else:
            # Has previous assignment - favor closest pitches with strict register enforcement
            for pitch_idx, pitch in enumerate(current_pitches):
                distance = abs(pitch - prev_pitch)
                cost = distance

                # Apply OVERLAPPING register boundaries with preferences (not absolute exclusions)
                register_penalty = 0

                # Very high pitches (80+) prefer Voice 0, but can use Voice 1
                if pitch >= 80:
                    if voice_idx == 0:
                        register_penalty = 0  # Perfect match
                    elif voice_idx == 1:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # High pitches (70-79) prefer Voice 1, but can use Voice 0 or 2
                elif pitch >= 70:
                    if voice_idx == 1:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [0, 2]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Upper mid pitches (65-69) prefer Voice 2, but can use Voice 1 or 3
                elif pitch >= 65:
                    if voice_idx == 2:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [1, 3]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Mid pitches (60-64) prefer Voice 3, but can use Voice 2 or 4
                elif pitch >= 60:
                    if voice_idx == 3:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [2, 4]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Lower mid pitches (55-59) prefer Voice 4, but can use Voice 3 or 5
                elif pitch >= 55:
                    if voice_idx == 4:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [3, 5]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Low pitches (50-54) prefer Voice 5, but can use Voice 4 or 6
                elif pitch >= 50:
                    if voice_idx == 5:
                        register_penalty = 0  # Perfect match
                    elif voice_idx in [4, 6]:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                # Very low pitches (<50) prefer Voice 6, but can use Voice 5
                else:
                    if voice_idx == 6:
                        register_penalty = 0  # Perfect match
                    elif voice_idx == 5:
                        register_penalty = 50  # Acceptable but penalized
                    else:
                        register_penalty = 200  # Heavy penalty but not impossible

                cost += register_penalty

                # Add historical affinity bonus
                identity_key = f"voice_{voice_idx}"
                if identity_key in voice_identities:
                    pitch_history = voice_identities[identity_key]
                    if pitch in pitch_history:
                        affinity_bonus = min(50, pitch_history[pitch] * 5)  # Cap at 50
                        cost = max(0, cost - affinity_bonus)

                # Virtually impossible penalty for jumps >= 12 semitones (octave or more)
                if distance >= 12:
                    cost += 100000  # Make octave jumps virtually impossible

                # Heavy penalty for jumps >= 7 semitones (perfect 5th or more)
                elif distance >= 7:
                    cost += 1000 * (distance - 6)

                # Medium penalty for jumps >= 4 semitones (major 3rd or more)
                elif distance >= 4:
                    cost += 100 * (distance - 3)

                cost_matrix[voice_idx, pitch_idx] = cost

    # Solve assignment using Hungarian algorithm, but only assign finite costs
    # Filter out infinite costs to avoid impossible assignments
    finite_assignments = []
    for voice_idx in range(num_voices):
        for pitch_idx in range(num_pitches):
            if cost_matrix[voice_idx, pitch_idx] != float('inf'):
                finite_assignments.append((voice_idx, pitch_idx, cost_matrix[voice_idx, pitch_idx]))

    # CRITICAL FIX: Use Hungarian algorithm properly to ensure ALL pitches get assigned
    if finite_assignments:
        from scipy.optimize import linear_sum_assignment

        # Create a proper cost matrix for Hungarian algorithm
        cost_matrix_hungarian = np.full((num_voices, num_pitches), 1000000.0)

        for voice_idx, pitch_idx, cost in finite_assignments:
            cost_matrix_hungarian[voice_idx, pitch_idx] = cost

        # If we have more pitches than voices, expand to ensure all pitches can be assigned
        if num_pitches > num_voices:
            # Add extra virtual voices to handle overflow
            extra_voices = num_pitches - num_voices
            extra_matrix = np.full((extra_voices, num_pitches), 1000.0)
            cost_matrix_hungarian = np.vstack([cost_matrix_hungarian, extra_matrix])
            num_voices_expanded = num_voices + extra_voices
        else:
            num_voices_expanded = num_voices

        # Solve Hungarian assignment
        voice_indices, pitch_indices = linear_sum_assignment(cost_matrix_hungarian)

        # Create assignment ensuring all pitches are included
        assignment = {i: None for i in range(num_voices)}

        for voice_idx, pitch_idx in zip(voice_indices, pitch_indices):
            if pitch_idx < num_pitches and cost_matrix_hungarian[voice_idx, pitch_idx] < 999999:
                if voice_idx < num_voices:
                    # Normal voice assignment
                    assignment[voice_idx] = current_pitches[pitch_idx]
                else:
                    # Overflow - find best available voice for this pitch
                    best_voice = None
                    best_cost = float('inf')
                    for v in range(num_voices):
                        if assignment[v] is None:
                            cost = cost_matrix_hungarian[v, pitch_idx]
                            if cost < best_cost:
                                best_voice = v
                                best_cost = cost
                    if best_voice is not None:
                        assignment[best_voice] = current_pitches[pitch_idx]
                    else:
                        # Force assignment to voice with lowest cost
                        voice_costs = [(v, cost_matrix_hungarian[v, pitch_idx]) for v in range(num_voices)]
                        voice_costs.sort(key=lambda x: x[1])
                        assignment[voice_costs[0][0]] = current_pitches[pitch_idx]

        # Ensure we didn't miss any pitches
        assigned_pitches = {p for p in assignment.values() if p is not None}
        missing_pitches = set(current_pitches) - assigned_pitches

        if missing_pitches:
            print(f"⚠️  CRITICAL: Still missing pitches after Hungarian: {sorted(missing_pitches)}")
            # Force assign missing pitches to available voices
            available_voices = [v for v in range(num_voices) if assignment[v] is None]
            missing_list = sorted(missing_pitches)

            for i, pitch in enumerate(missing_list):
                if i < len(available_voices):
                    assignment[available_voices[i]] = pitch
                    print(f"   Forced assignment: Voice {available_voices[i]} <- Pitch {pitch}")
    else:
        assignment = {i: None for i in range(num_voices)}

    # Fill in None for voices without assignments but preserve previous pitch memory
    for voice_idx in range(num_voices):
        if voice_idx not in assignment:
            assignment[voice_idx] = None

    # VERIFICATION: Ensure all pitches are assigned
    assigned_pitches = {p for p in assignment.values() if p is not None}
    if len(assigned_pitches) != len(current_pitches):
        missing = set(current_pitches) - assigned_pitches
        print(f"⚠️  ASSIGNMENT VERIFICATION FAILED: Missing {len(missing)} pitches: {sorted(missing)}")

    return assignment

def assign_pitches_to_voices_with_continuity(current_pitches, prev_assignments, max_voices, voice_identities, current_time):
    """
    Enhanced voice assignment using Hungarian algorithm with strict register boundaries.
    Completely prevents octave jumps by enforcing register-based voice separation.
    """
    if not current_pitches:
        return {i: None for i in range(max_voices)}

    current_pitches = sorted(current_pitches)

    # Use the Hungarian algorithm with strict register boundaries
    return solve_voice_assignment(current_pitches, prev_assignments, voice_identities, current_time)

def mix_audio_files(audio_files, output_path):
    """
    Mix multiple audio files into a single output.
    Args:
        audio_files: list of file paths to mix
        output_path: path for the mixed output
    Returns:
        path to the mixed audio file
    """
    if not audio_files:
        raise ValueError("No audio files to mix")

    if len(audio_files) == 1:
        # If only one file, just copy it
        shutil.copy(audio_files[0], output_path)
        return output_path

    # Load all audio files
    mixed_audio = None
    sample_rate = None

    for audio_path in audio_files:
        try:
            audio, sr = torchaudio.load(audio_path)

            if sample_rate is None:
                sample_rate = sr
                mixed_audio = audio
            elif sr == sample_rate:
                # Ensure same length by padding/trimming
                min_len = min(mixed_audio.shape[-1], audio.shape[-1])
                mixed_audio = mixed_audio[..., :min_len]
                audio = audio[..., :min_len]
                mixed_audio = mixed_audio + audio
            else:
                print(f"⚠️ Skipping {audio_path} due to sample rate mismatch: {sr} vs {sample_rate}")

        except Exception as e:
            print(f"⚠️ Error loading {audio_path}: {e}")
            continue

    if mixed_audio is not None:
        # Normalize to prevent clipping
        mixed_audio = mixed_audio / (mixed_audio.abs().max() + 1e-8) * 0.9

        # Apply final audio processing (compression + high-pass filter) - DISABLED to match training previews
        # mixed_audio = apply_final_audio_processing(mixed_audio, sample_rate=sample_rate)

        torchaudio.save(output_path, mixed_audio, sample_rate)
        print(f"✅ Mixed {len(audio_files)} files into: {output_path}")

    return output_path

def normalize_audio_lengths(audio_files, target_duration=None):
    """
    Ensure all audio files have the same duration by padding or trimming.
    Args:
        audio_files: list of file paths
        target_duration: target duration in samples, or None to use the longest
    Returns:
        list of normalized file paths
    """
    if len(audio_files) <= 1:
        return audio_files

    print("🎵 Normalizing audio lengths...")

    # Load all files to determine target length
    audio_data = []
    sample_rates = []

    for file_path in audio_files:
        try:
            audio, sr = torchaudio.load(file_path)
            audio_data.append((audio, sr, file_path))
            sample_rates.append(sr)
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            continue

    if not audio_data:
        return audio_files

    # Check sample rates are consistent
    if len(set(sample_rates)) > 1:
        print(f"⚠️ Inconsistent sample rates: {set(sample_rates)}")

    # Determine target length
    if target_duration is None:
        target_length = max(audio.shape[-1] for audio, sr, path in audio_data)
    else:
        target_length = int(target_duration * sample_rates[0])

    print(f"🎵 Target length: {target_length} samples")

    # Normalize all files to target length
    normalized_files = []
    for audio, sr, file_path in audio_data:
        if audio.shape[-1] == target_length:
            normalized_files.append(file_path)
            continue

        # Pad or trim to target length
        if audio.shape[-1] < target_length:
            # Pad with zeros
            padding = target_length - audio.shape[-1]
            audio_normalized = F.pad(audio, (0, padding), mode='constant', value=0)
        else:
            # Trim to target length
            audio_normalized = audio[..., :target_length]

        # Save normalized version
        path = Path(file_path)
        normalized_path = path.parent / f"{path.stem}_normalized{path.suffix}"
        torchaudio.save(str(normalized_path), audio_normalized, sr)
        normalized_files.append(str(normalized_path))

        print(f"🎵 Normalized {path.name}: {audio.shape[-1]} → {target_length} samples")

    return normalized_files

def apply_dynamic_resonance_suppression(audio: torch.Tensor, sample_rate: int = 32000,
                                        sensitivity: float = 1.5, strength: float = 0.6) -> torch.Tensor:
    """
    Dynamic resonance suppressor (similar to Oeksound Soothe).
    Detects and reduces harsh/resonant frequencies that stick out.

    Args:
        audio: Audio tensor [channels, samples]
        sample_rate: Sample rate in Hz
        sensitivity: How aggressive to detect resonances (1.0-3.0, higher = more aggressive)
        strength: How much to reduce detected resonances (0.0-1.0, higher = more reduction)

    Returns:
        Audio with resonances suppressed
    """
    import torch
    import torch.nn.functional as F_torch

    # STFT parameters
    n_fft = 2048
    hop_length = 512
    win_length = 2048

    # Process each channel separately
    processed_channels = []

    for ch in range(audio.shape[0]):
        channel = audio[ch]

        # Forward STFT
        window = torch.hann_window(win_length, device=channel.device)
        stft = torch.stft(
            channel,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )

        # Get magnitude and phase
        magnitude = torch.abs(stft)  # [freq_bins, time_frames]
        phase = torch.angle(stft)

        # Calculate spectral statistics for resonance detection
        # Average magnitude per frequency bin across time
        avg_magnitude = magnitude.mean(dim=1, keepdim=True)  # [freq_bins, 1]

        # Standard deviation per frequency bin
        std_magnitude = magnitude.std(dim=1, keepdim=True)

        # Detect resonances: bins that are significantly above average
        # Threshold is mean + (sensitivity * std)
        resonance_threshold = avg_magnitude + (sensitivity * std_magnitude)

        # Calculate gain reduction mask
        # Frequencies above threshold get reduced proportionally
        excess = (magnitude - resonance_threshold).clamp(min=0)
        max_excess = excess.max(dim=1, keepdim=True)[0] + 1e-8

        # Normalize excess to 0-1 range
        normalized_excess = excess / max_excess

        # Apply gain reduction (more excess = more reduction)
        gain_reduction = 1.0 - (normalized_excess * strength)

        # Smooth the gain reduction across frequency bins to avoid artifacts
        # Use a small averaging kernel
        kernel_size = 5
        padding = kernel_size // 2
        gain_reduction_smooth = gain_reduction.unsqueeze(0).unsqueeze(0)  # [1, 1, freq, time]
        kernel = torch.ones(1, 1, kernel_size, 1, device=gain_reduction.device) / kernel_size
        gain_reduction_smooth = F_torch.conv2d(
            gain_reduction_smooth,
            kernel,
            padding=(padding, 0)
        ).squeeze(0).squeeze(0)

        # Apply gain reduction to magnitude
        magnitude_suppressed = magnitude * gain_reduction_smooth

        # Reconstruct complex spectrum
        stft_suppressed = magnitude_suppressed * torch.exp(1j * phase)

        # Inverse STFT
        audio_suppressed = torch.istft(
            stft_suppressed,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            length=channel.shape[0]
        )

        processed_channels.append(audio_suppressed)

    # Stack channels back together
    result = torch.stack(processed_channels, dim=0)

    return result

def apply_final_audio_processing(audio: torch.Tensor, sample_rate: int = 32000) -> torch.Tensor:
    """
    Apply comprehensive audio processing: HPF, EQ, dynamic resonance suppression, compression, and limiting.

    Args:
        audio: Audio tensor [channels, samples]
        sample_rate: Sample rate in Hz

    Returns:
        Processed audio tensor with:
        - High-pass filter (removes 0-100Hz rumble)
        - Mid-frequency reduction (-2.5 dB at 300-400Hz)
        - Dynamic resonance suppression (Soothe-like - targets harsh frequencies)
        - Dynamic range compression (tames abnormal spikes)
        - Soft limiting (prevents clipping)
    """
    import torch
    import torchaudio.functional as F

    # Ensure audio is float and on CPU
    audio = audio.float().cpu()

    # STEP 1: Aggressive high-pass filter to remove 0-100Hz rumble
    # Use cascaded filters for steep rolloff (24 dB/octave)
    cutoff_freq = 100.0

    # Apply 4 cascaded high-pass filters for very steep rolloff
    for i in range(4):
        # Stagger cutoffs: 100Hz, 85Hz, 72Hz, 61Hz
        current_cutoff = cutoff_freq * (0.85 ** i)
        audio = F.highpass_biquad(audio, sample_rate, current_cutoff, Q=0.707)

    # STEP 2: Mid-frequency reduction (-2.5 dB around 300-400Hz)
    # This reduces muddiness and boxy resonances
    mid_freq = 350.0  # Center frequency for mids
    mid_reduction_db = -2.5  # Gain in dB (negative = reduction)
    mid_q = 1.0  # Moderate Q for smooth reduction

    # Apply peaking EQ (bell filter) to reduce mids
    # Note: equalizer_biquad expects gain in dB, not linear
    audio = F.equalizer_biquad(audio, sample_rate, mid_freq, mid_reduction_db, mid_q)

    # STEP 2.5: Dynamic resonance suppression (Soothe-like) - DISABLED (too slow)
    # Detects and reduces harsh/resonant frequencies that stick out
    # print("   Applying dynamic resonance suppression...")
    # audio = apply_dynamic_resonance_suppression(
    #     audio,
    #     sample_rate=sample_rate,
    #     sensitivity=1.5,  # Moderate detection (1.0-3.0)
    #     strength=0.6      # Moderate reduction (0.0-1.0)
    # )

    # STEP 3: Multi-stage compression to tame abnormal spikes
    # Stage 1: Gentle compression for overall dynamics
    threshold_gentle = 0.6
    ratio_gentle = 3.0

    # Stage 2: Aggressive compression for sudden spikes
    threshold_aggressive = 0.8
    ratio_aggressive = 8.0

    # Convert to mono for level detection
    if audio.shape[0] > 1:
        level_detect = audio.mean(dim=0)
    else:
        level_detect = audio[0]

    abs_audio = torch.abs(level_detect)

    # Stage 1: Gentle compression
    gain_reduction_gentle = torch.ones_like(abs_audio)
    over_gentle = abs_audio > threshold_gentle

    if over_gentle.any():
        excess = abs_audio[over_gentle] - threshold_gentle
        compressed_excess = excess / ratio_gentle
        gain_reduction_gentle[over_gentle] = (threshold_gentle + compressed_excess) / abs_audio[over_gentle]

    # Stage 2: Aggressive spike compression
    gain_reduction_aggressive = torch.ones_like(abs_audio)
    over_aggressive = abs_audio > threshold_aggressive

    if over_aggressive.any():
        excess = abs_audio[over_aggressive] - threshold_aggressive
        compressed_excess = excess / ratio_aggressive
        gain_reduction_aggressive[over_aggressive] = (threshold_aggressive + compressed_excess) / abs_audio[over_aggressive]

    # Combine both stages (multiply gain reductions)
    gain_reduction = gain_reduction_gentle * gain_reduction_aggressive

    # Smooth gain reduction with attack/release envelope
    attack = 0.002   # 2ms attack (fast for transients)
    release = 0.080  # 80ms release (moderate for musical feel)

    attack_samples = max(1, int(attack * sample_rate))
    release_samples = max(1, int(release * sample_rate))

    # Apply exponential smoothing for more natural attack/release
    smoothed_gain = torch.zeros_like(gain_reduction)
    attack_coef = 1.0 - torch.exp(torch.tensor(-1.0 / attack_samples))
    release_coef = 1.0 - torch.exp(torch.tensor(-1.0 / release_samples))

    envelope = 1.0
    for i in range(len(gain_reduction)):
        target = gain_reduction[i].item()
        if target < envelope:
            # Attack (gain reduction increases)
            envelope = envelope * (1.0 - attack_coef) + target * attack_coef
        else:
            # Release (gain reduction decreases)
            envelope = envelope * (1.0 - release_coef) + target * release_coef
        smoothed_gain[i] = envelope

    # Apply gain reduction to all channels
    for ch in range(audio.shape[0]):
        audio[ch] = audio[ch] * smoothed_gain

    # STEP 4: Adaptive makeup gain (compensate for compression)
    # Calculate average gain reduction to determine makeup gain
    avg_reduction = smoothed_gain.mean().item()
    makeup_gain = 1.0 / (avg_reduction + 0.1)  # Inverse of average reduction
    makeup_gain = min(makeup_gain, 1.5)  # Cap at 1.5x to avoid over-boosting

    audio = audio * makeup_gain

    # STEP 5: Soft brick-wall limiter to prevent any clipping
    # This catches any remaining spikes that got through
    limit_threshold = 0.95
    limit_ratio = 20.0  # Very high ratio for hard limiting

    abs_limited = torch.abs(audio)
    over_limit = abs_limited > limit_threshold

    if over_limit.any():
        # Apply soft knee limiting
        for ch in range(audio.shape[0]):
            ch_abs = torch.abs(audio[ch])
            ch_over = ch_abs > limit_threshold
            if ch_over.any():
                excess = ch_abs[ch_over] - limit_threshold
                limited_excess = excess / limit_ratio
                new_magnitude = limit_threshold + limited_excess
                # Preserve sign
                sign = torch.sign(audio[ch, ch_over])
                audio[ch, ch_over] = sign * new_magnitude

    # Final safety clip (should rarely be needed after limiting)
    audio = torch.clamp(audio, -0.98, 0.98)

    print(f"🎛️ Audio processing: HPF @{cutoff_freq}Hz | Mid EQ {mid_reduction_db}dB @{mid_freq}Hz | 2-stage compression | Soft limiter")
    return audio

# ------------------------------------------------------------------------------
# Sampler
# ------------------------------------------------------------------------------
@torch.no_grad()
def generate(
    model: Pipeline, piano_roll, amp, rframe, rbend, encodec_tokens,
    group, subgroup, steps, seed, adapter_scale, cfg_weight, t0, sr_out,
    instrument_strength=1.0, inst_boost=2.5,
    piano_roll_gain=1.0, amp_gain=1.0, rframe_gain=1.0, rbend_gain=1.0, encodec_gain=1.0,
    use_overlap_decoder=True, original_audio_length=None,
    pitch_fidelity_boost=1.0, onset_guidance_boost=2.0, pitch_snap_strength=0.5,
    noise_level=1.0, audio_file=None,
    # NEW: Sample recreation enhancement features
    use_time_varying_noise=False, onset_preservation=0.7,
    use_multiresolution_mixing=False,
    use_onset_weighted_encodec=False, encodec_onset_boost=2.0,
    # NEW: Test-time adaptation
    use_test_time_adaptation=False, adaptation_steps=10, adaptation_learning_rate=1e-4,
    # NEW: Self-consistency ensembling
    use_self_consistency=False, consistency_samples=3, consistency_noise_scale=0.05
):
    device = next(model.parameters()).device
    model.eval()

    # ids
    g2i = getattr(model, "group2id", None)
    s2i = getattr(model, "subgroup2id", None)
    if g2i is None or s2i is None:
        # fallback to approved lists
        g2i = {g:i for i,g in enumerate(list(APPROVED_GROUPS) if not isinstance(APPROVED_GROUPS, dict) else APPROVED_GROUPS.keys())}
        s2i = {}
        i = 0
        for gs in APPROVED_SUBGROUPS.values():
            for sg in gs:
                if sg not in s2i:
                    s2i[sg] = i; i += 1

    if subgroup not in APPROVED_SUBGROUPS.get(group, []):
        raise ValueError(f"Subgroup '{subgroup}' is not valid for group '{group}'. "
                         f"Valid for {group}: {APPROVED_SUBGROUPS.get(group, [])}")

    gid, sgid = int(g2i[group]), int(s2i[subgroup])
    print(f"[ids] {group}->{gid}  {subgroup}->{sgid}")

    # T grid
    T_slow = int(piano_roll.shape[1])

    # build conds
    conds = {
        "piano_roll": torch.from_numpy(piano_roll).float().unsqueeze(0).to(device),  # [1,128,T]
        "amp":        torch.from_numpy(amp).float().unsqueeze(0).to(device),         # [1,T]
        "rframe":     torch.from_numpy(rframe).float().unsqueeze(0).to(device),      # [1,T]
        "rbend":      torch.from_numpy(rbend).float().unsqueeze(0).to(device),       # [1,T]
        "rbend_mask": torch.from_numpy((rframe > 0.5).astype(np.float32)).bool().unsqueeze(0).to(device),
        "encodec_tokens": encodec_tokens.to(device),                                  # [1,C,Tf]
        "group_id":   torch.tensor([gid], dtype=torch.long, device=device),
        "subgroup_id":torch.tensor([sgid], dtype=torch.long, device=device),
    }

    # stream gains (continuous only)
    audio_red = 2.0 - float(instrument_strength) if instrument_strength > 1.0 else 1.0
    audio_red = max(0.1, audio_red)
    conds["piano_roll"] *= float(piano_roll_gain) * audio_red
    conds["amp"]        *= float(amp_gain)        * audio_red
    conds["rframe"]     *= float(rframe_gain)     * audio_red
    conds["rbend"]      *= float(rbend_gain)      * audio_red

    # encodec gating (keep long)
    enc = conds["encodec_tokens"].clone()
    if encodec_gain <= 0.0:
        enc.zero_()
    elif encodec_gain < 1.0:
        keep = (torch.rand_like(enc.float()) < float(encodec_gain))
        enc = torch.where(keep, enc, enc.new_zeros(()).expand_as(enc))
    conds["encodec_tokens"] = enc

    # Compute onset weight for encodec if using onset-weighted feature
    onset_weight_mask = None
    if use_onset_weighted_encodec and encodec_gain > 0.0:
        onset_weight_mask = weight_encodec_by_onsets(
            conds["encodec_tokens"],
            conds["piano_roll"],
            onset_boost=float(encodec_onset_boost)
        )  # [1, 1, T_enc]
        print(f"✅ Using onset-weighted encodec (boost={encodec_onset_boost:.2f})")

    # ctrl_enc strengths (temporarily scaled by instrument_strength / encodec_gain)
    orig_inst = getattr(model.ctrl_enc, 'inst_strength', 3.0)
    orig_film = getattr(model.ctrl_enc, 'film_strength', 1.0)
    orig_ch   = getattr(model.ctrl_enc, 'channel_mod_strength', 1.0)
    try:
        model.ctrl_enc.inst_strength = orig_inst * float(instrument_strength)
        if encodec_gain <= 0.0:
            model.ctrl_enc.film_strength = 0.0
            model.ctrl_enc.channel_mod_strength = 0.0
        else:
            # If using onset weighting, boost the base strength and apply weighting to tokens after
            base_film = orig_film * float(encodec_gain)
            base_ch = orig_ch * float(encodec_gain)

            if use_onset_weighted_encodec:
                # Apply onset weighting through film/channel strengths
                # We'll weight the tokens after encoding
                model.ctrl_enc.film_strength = base_film
                model.ctrl_enc.channel_mod_strength = base_ch
            else:
                model.ctrl_enc.film_strength = base_film
                model.ctrl_enc.channel_mod_strength = base_ch

        tokens, _ = model.ctrl_enc(**conds)

        # Apply onset weighting to tokens if enabled
        if onset_weight_mask is not None:
            # Resize onset weight to match token temporal dimension
            if onset_weight_mask.shape[-1] != tokens.shape[-1]:
                onset_weight_mask_resized = F.interpolate(
                    onset_weight_mask, size=tokens.shape[-1], mode='nearest'
                )
            else:
                onset_weight_mask_resized = onset_weight_mask

            # Apply weighting to all token channels
            # onset_weight_mask_resized is [B, 1, T], tokens is [B, C, T]
            # Broadcasting will expand [B, 1, T] to [B, C, T] automatically
            tokens = tokens * onset_weight_mask_resized  # [B, C, T] * [B, 1, T] = [B, C, T]
            print(f"✅ Applied onset weighting to ctrl_enc tokens")

    finally:
        model.ctrl_enc.inst_strength = orig_inst
        model.ctrl_enc.film_strength = orig_film
        model.ctrl_enc.channel_mod_strength = orig_ch

    # cond adapter sample / latents init
    tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)
    sample_patch = model.cond_adapter(tokens_adapt, T_out=T_slow, scale=_adapter_gain_scale_compat(model))
    if int(seed) <= 0:
        seed = torch.seed() % 2**31
    torch.manual_seed(int(seed))

    # Initialize latents based on noise level
    if float(noise_level) >= 1.0:
        # Pure noise (original behavior)
        x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))
    else:
        # Try to extract ground truth latents for proper noise mixing
        gt_latents = None
        if audio_file is not None:
            gt_latents = extract_ground_truth_latents(audio_file, model)

        if gt_latents is not None:
            # Resize ground truth latents to match expected shape
            gt_latents = gt_latents.to(device=device, dtype=tokens.dtype)
            target_shape = sample_patch.shape  # [1, 8, 16, T]

            # Add batch dimension if missing
            if gt_latents.ndim == 3:  # [8, 16, T] -> [1, 8, 16, T]
                gt_latents = gt_latents.unsqueeze(0)

            # Pad or crop temporal dimension to match target
            if gt_latents.shape[-1] != target_shape[-1]:
                if gt_latents.shape[-1] < target_shape[-1]:
                    # Pad if too short
                    pad_size = target_shape[-1] - gt_latents.shape[-1]
                    gt_latents = F.pad(gt_latents, (0, pad_size), mode='constant', value=0)
                else:
                    # Crop if too long
                    gt_latents = gt_latents[..., :target_shape[-1]]

            # Ensure dimensions match
            if gt_latents.shape != target_shape:
                print(f"⚠️ GT latent shape mismatch: {gt_latents.shape} vs {target_shape}, using noise")
                x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))
            else:
                if float(noise_level) <= 0.0:
                    # Pure ground truth latents
                    x = gt_latents
                    print(f"✅ Using pure ground truth latents: {x.shape}")
                else:
                    # Mix ground truth latents with noise
                    noise = torch.randn_like(gt_latents)

                    # Enhanced sample recreation features
                    if use_time_varying_noise or use_multiresolution_mixing:
                        pr_for_onset = conds["piano_roll"]  # [1, 128, T]

                        if use_time_varying_noise:
                            # Detect onsets for time-varying noise
                            onset_mask = detect_onsets_from_piano_roll(pr_for_onset)  # [1, 1, T_pr]

                            # Resize onset mask to match latent temporal dimension
                            if onset_mask.shape[-1] != gt_latents.shape[-1]:
                                onset_mask = F.interpolate(onset_mask, size=gt_latents.shape[-1], mode='nearest')

                            # Apply time-varying noise (preserves attacks)
                            x = apply_time_varying_noise(
                                gt_latents, noise, float(noise_level),
                                onset_mask, onset_preservation=float(onset_preservation)
                            )
                            print(f"✅ Mixed GT latents with time-varying noise (onset preservation={onset_preservation:.2f}): {x.shape}")

                        elif use_multiresolution_mixing:
                            # Apply multi-resolution mixing (preserves low frequencies)
                            x = apply_multiresolution_latent_mixing(
                                gt_latents, noise, float(noise_level)
                            )
                            print(f"✅ Mixed GT latents with multi-resolution noise: {x.shape}")
                    else:
                        # Standard uniform noise mixing
                        x = (1.0 - float(noise_level)) * gt_latents + float(noise_level) * noise
                        print(f"✅ Mixed GT latents with {float(noise_level):.2f} noise: {x.shape}")
        else:
            # Fallback to pure noise if no ground truth available
            print("⚠️ No ground truth latents available, using pure noise")
            x = torch.randn_like(sample_patch.to(device=device, dtype=tokens.dtype))

    # Test-time adaptation: DISABLED - conceptually flawed
    # The cond_adapter produces conditioning signals, not latents themselves
    # Training it to match GT latents breaks the model's conditioning
    original_state = None
    if use_test_time_adaptation and audio_file is not None:
        print("⚠️ Test-time adaptation is currently DISABLED due to implementation issues")
        print("   For better sample recreation, try:")
        print("   - Lower t0/noiseLevel (0.6-0.8) for closer match to sample")
        print("   - instBoost: 2.5 (this was the key from monophonic mode)")
        print("   - encodec_gain: 1.0")
        print("   - More steps (40-50)")

    # Control residuals (constant across loop)
    pr_128 = conds["piano_roll"].to(dtype=x.dtype)
    amp_1t = conds["amp"].unsqueeze(1).to(dtype=x.dtype)
    ctrl_res_list = _prep_ctrl_residuals_if_enabled(model, pr_128, amp_1t, T_lat=T_slow)

    # scheduler mapping
    T_train = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
    steps   = max(1, int(steps))
    dt      = float(t0) / float(steps)  # Use t0 to match noise level with timestep schedule

    for i in range(steps, 0, -1):
        t_cont = torch.full((x.shape[0],), i * dt, device=device, dtype=torch.float32)
        t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        # instrument ON/OFF patches
        tokens_adapt = tokens.to(dtype=next(model.cond_adapter.parameters()).dtype)

        # Simplified: when cfg_weight=1 and no onset boost, use single conditioning like trainer
        use_cfg = float(cfg_weight) > 1.0 or float(onset_guidance_boost) > 0.0

        if use_cfg:
            tokens_on  = tokens_adapt.clone(); tokens_on[:, 0, :] *= float(inst_boost)
            tokens_off = tokens_adapt.clone(); tokens_off[:, 0, :].zero_()
            cond_on  = model.cond_adapter(tokens_on,  T_out=x.shape[-1], scale=float(adapter_scale)).to(x)
            cond_off = model.cond_adapter(tokens_off, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)
        else:
            # Single conditioning path (matches trainer preview behavior)
            cond_patch = model.cond_adapter(tokens_adapt, T_out=x.shape[-1], scale=float(adapter_scale)).to(x)

        # PR-guided masking/sharpening
        B, C, H, T_lat = x.shape
        pr = conds["piano_roll"].to(device=x.device, dtype=x.dtype)
        if pr.shape[-1] != T_lat:
            pr = F.interpolate(pr, size=T_lat, mode="nearest")

        W_hp = _bank_softplus_resized_compat(model, H, device=x.device, dtype=x.dtype)

        if use_cfg:
            # CFG mode: use normalized piano roll with pitch snap
            pr_norm = pr / (pr.amax(dim=1, keepdim=True) + 1e-6)
            pr_target = pr_norm.clone()
            if hasattr(model, 'pr_head') and i < steps * 0.8:
                try:
                    with torch.no_grad():
                        x_flat = x.reshape(B, C*H, T_lat)
                        pr_logits = model.pr_head(x_flat)
                        pr_prob = pr_logits.sigmoid()
                        snap = float(pitch_snap_strength)
                        pr_target = ((1.0 - snap) * pr_target + snap * pr_prob).detach()
                except Exception as e:
                    if i == steps:
                        print(f"[PitchSnap] disabled: {e}")
            pr_high = pr_target * (1.0 + float(pitch_fidelity_boost))
            pr_low  = pr_target * (1.0 - float(pitch_fidelity_boost) * 0.5)
            H_on  = torch.einsum('bpt,hp->bht', pr_high, W_hp)
            H_off = torch.einsum('bpt,hp->bht', pr_low,  W_hp)

            sharp = 1.0 + float(pitch_fidelity_boost) * 0.5
            H_on  = (H_on  + 1e-6).pow(sharp)
            H_off = (H_off + 1e-6).pow(sharp)
            H_on  = H_on  / (H_on.amax(dim=1, keepdim=True)  + 1e-6)
            H_off = H_off / (H_off.amax(dim=1, keepdim=True) + 1e-6)

            # Simplified conditioning application to match trainer (no active mask or adaptive scaling)
            # This avoids hard discontinuities that can cause clicking artifacts
            cond_on  = cond_on  * H_on.unsqueeze(1)
            cond_off = cond_off * H_off.unsqueeze(1)
        else:
            # Simple pitch-height masking (matches trainer - uses normalized piano roll if trained that way)
            pr_for_mask = pr / (pr.amax(dim=1, keepdim=True) + 1e-6)  # Normalize to match training
            Hmap = torch.einsum('bpt,hp->bht', pr_for_mask, W_hp)
            cond_patch = cond_patch * Hmap.unsqueeze(1)

        # transformer with ControlBranch residuals
        if ctrl_res_list is not None:
            model._ctrl_residuals = ctrl_res_list

        if use_cfg:
            # onset-weighted guidance for CFG
            if pr_target.shape[-1] > 1:
                onset = (pr_target[:, :, 1:] > 0.1) & (pr_target[:, :, :-1] <= 0.1)
                onset = F.pad(onset.float().amax(dim=1, keepdim=True), (1,0))
            else:
                onset = torch.zeros_like(pr_target[:, :1, :])
            if onset.shape[-1] != T_lat:
                onset = F.interpolate(onset, size=T_lat, mode="nearest")
            base_guid = max(1.0, float(cfg_weight))
            step_guid = base_guid * (1.0 + float(onset_guidance_boost) * onset)  # [B,1,T]

            v_un = model._call_transformer_no_xattn(latents=x + cond_off, t=t_idx)
            v_co = model._call_transformer_no_xattn(latents=x + cond_on,  t=t_idx)
            v_pred = v_un + step_guid.unsqueeze(1) * (v_co - v_un)
        else:
            # Simple single-pass like trainer
            x_in = x + cond_patch
            v_pred = model._call_transformer_no_xattn(latents=x_in, t=t_idx)

        # SELF-CONSISTENCY ENSEMBLING: In final 10-20% of steps, run multiple predictions and average
        if use_self_consistency and i <= max(1, int(steps * 0.2)):  # Last 20% of steps
            if i == max(1, int(steps * 0.2)):  # Print once at start of ensembling
                print(f"🔄 Self-consistency ensembling active (last {int(steps * 0.2)} steps, {consistency_samples} samples)")

            # Store original prediction
            predictions = [v_pred]

            # Generate additional predictions with small noise perturbations
            for _ in range(consistency_samples - 1):
                # Add small noise to x for diversity
                x_noisy = x + torch.randn_like(x) * consistency_noise_scale

                if use_cfg:
                    v_un_noisy = model._call_transformer_no_xattn(latents=x_noisy + cond_off, t=t_idx)
                    v_co_noisy = model._call_transformer_no_xattn(latents=x_noisy + cond_on,  t=t_idx)
                    v_pred_noisy = v_un_noisy + step_guid.unsqueeze(1) * (v_co_noisy - v_un_noisy)
                else:
                    x_in_noisy = x_noisy + cond_patch
                    v_pred_noisy = model._call_transformer_no_xattn(latents=x_in_noisy, t=t_idx)

                predictions.append(v_pred_noisy)

            # Average all predictions (not the noisy samples!)
            v_pred = torch.stack(predictions).mean(dim=0)

        x = x - dt * v_pred  # Use dt to match timestep schedule
        if i == steps:
            if use_cfg:
                print(f"[CondEnergy] on={cond_on.norm().item():.3f} off={cond_off.norm().item():.3f}")
            else:
                print(f"[CondEnergy] patch={cond_patch.norm().item():.3f}")

    model._ctrl_residuals = None
    print("Decoding audio...")

    # decode
    if original_audio_length is not None:
        audio_len = int(round(original_audio_length * sr_out / DCAE_SR))
    else:
        # Calculate correct audio length from latent frames using DCAE hop size
        # This matches the trainer's calculation exactly
        audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
        print(f"🎵 Calculated audio length: {T_slow} frames * {DCAE_HOP} hop * ({sr_out}/{DCAE_SR}) = {audio_len} samples")

    p = next(model.dcae.parameters(), None)
    dev = p.device if p is not None else getattr(model.dcae, "device", device)
    dtype = p.dtype if p is not None else torch.float32
    x_for_dcae = x[:1].to(device=dev, dtype=dtype)
    audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=dev)

    if use_overlap_decoder and hasattr(model.dcae, 'decode_overlap'):
        print("🔊 Using overlap decoder")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type=="cuda")):
            sr_pred, wav_pred = model.dcae.decode_overlap(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)
    else:
        sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

    wav = wav_pred[0].float().cpu()

    # Apply final audio processing (compression + high-pass filter) - DISABLED to match training previews
    # wav = apply_final_audio_processing(wav, sample_rate=sr_pred)

    out_dir = Path("./generated_ui"); out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}_cfg{cfg_weight:.1f}.wav"
    torchaudio.save(str(out_path), wav, sr_pred)
    print(f"✅ Wrote: {out_path}")

    # Restore original model weights if test-time adaptation was used
    if original_state is not None:
        print("🔄 Restoring original model weights...")
        restore_model_weights(model, original_state)
        print("✅ Model weights restored")

    return str(out_path)

@torch.no_grad()
def generate_monophonic_multiple(
    model, piano_roll, amp, rframe, rbend, encodec_tokens,
    group, subgroup, seed, steps, adapter_scale, cfg_weight, t0, sr_out,
    instrument_strength=1.0, inst_boost=2.5,
    piano_roll_gain=1.0, amp_gain=1.0, rframe_gain=1.0, rbend_gain=1.0, encodec_gain=1.0,
    use_overlap_decoder=True, original_audio_length=None,
    pitch_fidelity_boost=1.0, onset_guidance_boost=2.0, pitch_snap_strength=0.5,
    noise_level=1.0, audio_file=None, progress=None, voice_complete_callback=None,
    enable_voice_separation=True, fatten_mode=False, fatten_type="fake"
):
    """
    Generate multiple monophonic outputs from separated voices and create a mixed sum.

    Args:
        enable_voice_separation: If True, separate voices from piano roll. If False, assume
                                piano roll already contains separated tracks (e.g., multi-track MIDI)
        fatten_mode: If True, create octave-up versions of each voice to double the track count
        fatten_type: "real" (transpose piano roll) or "fake" (pitch shift audio output)
    """
    print("🎵 Starting monophonic multiple voice generation")
    print(f"   Voice separation enabled: {enable_voice_separation}")
    print(f"   🎚️ Fatten mode: {fatten_mode}, Type: {fatten_type}")

    # Separate the piano roll into voices (only if voice separation is enabled)
    if enable_voice_separation:
        print("   Separating piano roll into voices...")
        voices = separate_piano_roll_voices(piano_roll)
    else:
        print("   Skipping voice separation - using tracks as-is (multi-track MIDI)")
        # For multi-track MIDI, piano roll already contains separated tracks
        # We just need to wrap it in a list to process as a single "voice"
        voices = [piano_roll]

    if len(voices) == 1:
        print("⚠️ Only one voice detected, falling back to regular generation")
        return generate(
            model, piano_roll, amp, rframe, rbend, encodec_tokens,
            group, subgroup, steps, seed, adapter_scale, cfg_weight, t0, sr_out,
            instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
            use_overlap_decoder, original_audio_length, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
            noise_level, audio_file,
            use_time_varying_noise=False, onset_preservation=0.7,
            use_multiresolution_mixing=False, use_onset_weighted_encodec=False, encodec_onset_boost=2.0,
            use_test_time_adaptation=False, adaptation_steps=10, adaptation_learning_rate=1e-4
        )

    # Generate each voice separately
    voice_outputs = []
    base_seed = int(seed) if seed > 0 else torch.seed() % 2**31

    out_dir = Path("./generated_ui"); out_dir.mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')

    for i, voice_pr in enumerate(voices):
        if progress:
            progress_val = 0.5 + (i / len(voices)) * 0.4  # 50-90% range
            progress(progress_val, desc=f"Generating voice {i+1}/{len(voices)}...")

        print(f"🎼 Generating voice {i+1}/{len(voices)}")

        # Use different seed for each voice for variety
        voice_seed = base_seed + i * 1000

        voice_output = generate(
            model, voice_pr, amp, rframe, rbend, encodec_tokens,
            group, subgroup, steps, voice_seed, adapter_scale, cfg_weight, t0, sr_out,
            instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
            use_overlap_decoder, original_audio_length, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
            noise_level, audio_file,
            use_time_varying_noise=False, onset_preservation=0.7,
            use_multiresolution_mixing=False, use_onset_weighted_encodec=False, encodec_onset_boost=2.0,
            use_test_time_adaptation=False, adaptation_steps=10, adaptation_learning_rate=1e-4
        )

        # Rename the output to include voice number
        voice_path = out_dir / f"{timestamp}_voice{i+1}_seed{voice_seed}_cfg{cfg_weight:.1f}.wav"
        shutil.move(voice_output, str(voice_path))
        voice_outputs.append(str(voice_path))
        print(f"✅ Voice {i+1} saved: {voice_path.name}")

        # Notify callback that this voice is complete (for incremental updates)
        if voice_complete_callback:
            voice_complete_callback(i, str(voice_path), len(voices))

    # Fatten mode: create octave-up versions of each voice
    if fatten_mode:
        print(f"\n🎚️ FATTEN MODE ({fatten_type.upper()}): Creating octave-up voices...")
        original_voice_count = len(voice_outputs)

        if fatten_type == "real":
            # Real mode: transpose piano roll up 12 semitones and generate again
            print(f"   Generating {original_voice_count} octave-up voices from transposed piano rolls...")
            for i, voice_pr in enumerate(voices):
                if progress:
                    progress_val = 0.5 + ((len(voice_outputs) / (len(voices) * 2))) * 0.4
                    progress(progress_val, desc=f"Generating octave-up voice {i+1}/{len(voices)}...")

                print(f"   🎼 Generating octave-up voice {i+1}/{len(voices)}")

                # Transpose piano roll up one octave (shift all pitch indices up by 12)
                octave_voice_pr = voice_pr.clone()
                # Piano roll shape: (batch, time, pitch)
                # We need to shift the pitch dimension up by 12 semitones
                # Create a new tensor shifted up
                shifted_pr = torch.zeros_like(octave_voice_pr)
                if len(octave_voice_pr.shape) == 3:
                    # Shift pitch up by 12 (move data from index i to i+12)
                    shifted_pr[:, :, 12:] = octave_voice_pr[:, :, :-12]
                else:
                    # Handle 2D case (time, pitch)
                    shifted_pr[:, 12:] = octave_voice_pr[:, :-12]

                octave_voice_pr = shifted_pr

                # Use different seed for octave voice
                octave_seed = base_seed + (original_voice_count + i) * 1000

                octave_output = generate(
                    model, octave_voice_pr, amp, rframe, rbend, encodec_tokens,
                    group, subgroup, steps, octave_seed, adapter_scale, cfg_weight, t0, sr_out,
                    instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
                    use_overlap_decoder, original_audio_length, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength,
                    noise_level, audio_file,
                    use_time_varying_noise=False, onset_preservation=0.7,
                    use_multiresolution_mixing=False, use_onset_weighted_encodec=False, encodec_onset_boost=2.0,
                    use_test_time_adaptation=False, adaptation_steps=10, adaptation_learning_rate=1e-4
                )

                # Rename the output to include voice number
                octave_path = out_dir / f"{timestamp}_voice{original_voice_count+i+1}_octave_seed{octave_seed}_cfg{cfg_weight:.1f}.wav"
                shutil.move(octave_output, str(octave_path))
                voice_outputs.append(str(octave_path))
                print(f"   ✅ Octave-up voice {i+1} saved: {octave_path.name}")

                # Notify callback for octave voice
                if voice_complete_callback:
                    voice_complete_callback(original_voice_count + i, str(octave_path), len(voices) * 2)

        else:  # fake mode
            # Fake mode: pitch shift existing outputs up an octave
            print(f"   Pitch shifting {original_voice_count} voices up an octave...")
            import librosa
            import soundfile as sf

            for i, voice_path in enumerate(list(voice_outputs)):  # Use list() to avoid modifying during iteration
                print(f"   🎚️ Pitch shifting voice {i+1}/{original_voice_count}...")

                # Load audio
                audio, sr = librosa.load(voice_path, sr=None)

                # Pitch shift up 12 semitones (one octave)
                shifted_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=12)

                # Save shifted version
                octave_path = out_dir / f"{timestamp}_voice{original_voice_count+i+1}_octave_fake.wav"
                sf.write(str(octave_path), shifted_audio, sr)
                voice_outputs.append(str(octave_path))
                print(f"   ✅ Octave-up voice {i+1} saved: {octave_path.name}")

                # Notify callback for octave voice
                if voice_complete_callback:
                    voice_complete_callback(original_voice_count + i, str(octave_path), len(voices) * 2)

        print(f"   ✅ Fatten mode complete: {original_voice_count} original + {original_voice_count} octave = {len(voice_outputs)} total voices")

    # Normalize all voice outputs to same length
    if progress:
        progress(0.90, desc="Normalizing voice lengths...")

    normalized_voice_outputs = normalize_audio_lengths(voice_outputs)

    # Create mixed output
    if progress:
        progress(0.95, desc="Mixing voices...")

    mixed_path = out_dir / f"{timestamp}_mixed_seed{base_seed}_cfg{cfg_weight:.1f}.wav"
    mix_audio_files(normalized_voice_outputs, str(mixed_path))

    if progress:
        progress(1.0, desc="Done!")

    # Return all outputs as a tuple: (mixed_output, individual_voices)
    return {
        "mixed": str(mixed_path),
        "voices": normalized_voice_outputs,
        "voice_count": len(voices)
    }

# ------------------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------------------
def run_generation(
    audio_file, group, subgroup, seed, steps, adapter_scale, cfg_weight, t0,
    instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
    use_overlap_decoder, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength, noise_level,
    monophonic_mode, midi_mode, render_and_extract, tempo_override, progress=gr.Progress(track_tqdm=True)
):
    if audio_file is None:
        raise gr.Error("Please upload an audio or MIDI file or pick a random one.")

    # Check if input is MIDI file
    is_midi = is_midi_file(audio_file)

    # CRITICAL FIX: Calculate window_slow based on actual file duration
    # Don't hardcode to 1024 frames (23.78s limit)
    # Use 43.066 fps to match piano roll frame rate
    fps = 43.066
    if is_midi:
        # For MIDI files, get actual duration from MIDI
        try:
            import pretty_midi
            midi_data = pretty_midi.PrettyMIDI(audio_file)
            actual_duration = max(midi_data.get_end_time(), 1.0)
            win_slow = int(actual_duration * fps)
            print(f"🎵 MIDI duration: {actual_duration:.2f}s → window_slow = {win_slow} frames (at {fps} fps)")
        except Exception as e:
            print(f"⚠️ Could not determine MIDI duration: {e}, using default 1024")
            win_slow = int(getattr(MODEL.hparams, "window_slow", 1024))
    else:
        # For audio files, get actual duration from audio
        try:
            import torchaudio
            wav, sr = torchaudio.load(audio_file)
            actual_duration = wav.shape[-1] / sr
            win_slow = int(actual_duration * fps)
            print(f"🎵 Audio duration: {actual_duration:.2f}s → window_slow = {win_slow} frames (at {fps} fps)")
        except Exception as e:
            print(f"⚠️ Could not determine audio duration: {e}, using default 1024")
            win_slow = int(getattr(MODEL.hparams, "window_slow", 1024))

    if is_midi and midi_mode:
        progress(0, desc="Processing MIDI file…")

        if render_and_extract:
            # Mode 1: Render MIDI to audio, then extract full conditioning
            progress(0.1, desc="Rendering MIDI to audio…")
            rendered_audio = render_midi_to_audio(audio_file, instrument_group=subgroup)

            progress(0.3, desc="Extracting conditioning from rendered audio…")
            extraction = extract_conditioning_from_audio(rendered_audio, instrument_group=subgroup)

            # But use original MIDI for piano roll
            progress(0.5, desc="Loading MIDI piano roll…")
            # TIMING FIX: Don't use tempo adjustment when using audio conditioning
            # to ensure piano roll and audio conditioning have matching timing scales
            pr_midi, _, _, _, _ = midi_to_piano_roll_conditioning(audio_file, win_slow, fps=43.066, tempo_override=None)
            pr_duration = pr_midi.shape[1] / (43.066 * (tempo_override or 120.0) / 120.0) if tempo_override else pr_midi.shape[1] / 43.066
            print(f"🎵 Piano roll: {pr_midi.shape}, estimated duration: {pr_duration:.2f}s")

            # Load conditioning from rendered audio
            _, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

            # TIMING FIX: Use MIDI piano roll to match individual tracks
            # Resize MIDI piano roll to match conditioning length if needed
            conditioning_length = amp.shape[-1]
            if pr_midi.shape[1] != conditioning_length:
                print(f"🎵 Resizing MIDI piano roll: {pr_midi.shape[1]} → {conditioning_length} frames")
                if pr_midi.shape[1] > conditioning_length:
                    pr_midi = pr_midi[:, :conditioning_length]  # truncate
                else:
                    pad_width = conditioning_length - pr_midi.shape[1]
                    pr_midi = np.pad(pr_midi, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)  # pad

            pr = pr_midi
            print(f"🎵 Using MIDI piano roll: {pr.shape} (to match individual tracks)")

            # TIMING DEBUG: Force use of piano roll calculation instead of FluidSynth audio length
            # Get original length from rendered audio
            try:
                wav, sr = torchaudio.load(rendered_audio)
                fluidsynth_len = wav.shape[-1]
                fluidsynth_duration = fluidsynth_len / sr
                print(f"🎵 FluidSynth audio length: {fluidsynth_len} samples ({fluidsynth_duration:.2f}s at {sr}Hz)")

                # Calculate what the length SHOULD be based on piano roll
                expected_duration = pr.shape[1] / 43.066
                expected_len = int(expected_duration * sr)
                print(f"🎵 Expected audio length: {expected_len} samples ({expected_duration:.2f}s)")

                # FORCE use of piano roll-based calculation
                orig_len = None  # This will force the piano roll calculation path
                print(f"🎵 FORCING piano roll calculation instead of FluidSynth length")

            except Exception:
                orig_len = None
                print("⚠️ Could not determine rendered audio length")

            print("🎼 Mode: MIDI -> Audio -> Full Conditioning (with original MIDI PR)")

        else:
            # Mode 2: Use MIDI directly for piano roll only, empty other conditioning
            progress(0.2, desc="Converting MIDI to piano roll…")
            pr, amp, rfr, rbd, enc = midi_to_piano_roll_conditioning(audio_file, win_slow, tempo_override=tempo_override)
            orig_len = None  # No audio length reference
            print("🎼 Mode: MIDI -> Piano Roll Only")

    else:
        # Original audio file processing
        progress(0, desc="Extracting conditioning from audio…")

        # original len (for exact decode length)
        try:
            wav, sr = torchaudio.load(audio_file)
            orig_len = wav.shape[-1]
        except Exception:
            orig_len = None

        extraction = extract_conditioning_from_audio(audio_file, instrument_group=subgroup)
        progress(0.25, desc="Loading conditioning…")
        pr, amp, rfr, rbd, enc = load_conditioning(extraction, window_slow=win_slow)

    # Check if we should disable voice separation for multitrack MIDI
    should_skip_voice_separation = False
    if midi_mode and is_midi_file(audio_file):
        is_multi, track_count, _ = is_multitrack_midi(audio_file)
        if is_multi:
            should_skip_voice_separation = True
            print(f"🎼 Multitrack MIDI detected ({track_count} tracks) - DISABLING voice separation")
            print("   Individual tracks should be processed separately, not combined and re-separated")

    if monophonic_mode and not should_skip_voice_separation:
        progress(0.5, desc="Generating multiple voices…")
        result = generate_monophonic_multiple(
            MODEL, pr, amp, rfr, rbd, enc,
            group, subgroup, int(seed), int(steps), float(adapter_scale), float(cfg_weight), float(t0),
            sr_out=32000, instrument_strength=float(instrument_strength), inst_boost=float(inst_boost),
            piano_roll_gain=float(piano_roll_gain), amp_gain=float(amp_gain),
            rframe_gain=float(rframe_gain), rbend_gain=float(rbend_gain), encodec_gain=float(encodec_gain),
            use_overlap_decoder=bool(use_overlap_decoder), original_audio_length=orig_len,
            pitch_fidelity_boost=float(pitch_fidelity_boost), onset_guidance_boost=float(onset_guidance_boost),
            pitch_snap_strength=float(pitch_snap_strength), noise_level=float(noise_level),
            audio_file=audio_file, progress=progress
        )
        # Return mixed output, individual voices, and info
        if isinstance(result, dict):
            info_text = f"Generated {result['voice_count']} voices and mixed them."
            # Return mixed, all individual voices, and info
            return result["mixed"], result["voices"], info_text
        else:
            # Fallback case - single output
            return result, [], "Monophonic mode active but only single voice detected."
    else:
        progress(0.5, desc="Generating…")
        out = generate(
            MODEL, pr, amp, rfr, rbd, enc,
            group, subgroup, int(steps), int(seed), float(adapter_scale), float(cfg_weight), float(t0),
            sr_out=32000, instrument_strength=float(instrument_strength), inst_boost=float(inst_boost),
            piano_roll_gain=float(piano_roll_gain), amp_gain=float(amp_gain),
            rframe_gain=float(rframe_gain), rbend_gain=float(rbend_gain), encodec_gain=float(encodec_gain),
            use_overlap_decoder=bool(use_overlap_decoder), original_audio_length=orig_len,
            pitch_fidelity_boost=float(pitch_fidelity_boost), onset_guidance_boost=float(onset_guidance_boost),
            pitch_snap_strength=float(pitch_snap_strength), noise_level=float(noise_level),
            audio_file=audio_file,
            use_time_varying_noise=False, onset_preservation=0.7,
            use_multiresolution_mixing=False, use_onset_weighted_encodec=False, encodec_onset_boost=2.0,
            use_test_time_adaptation=False, adaptation_steps=10, adaptation_learning_rate=1e-4
        )
        progress(1.0, desc="Done!")

        # Check if this was a multitrack MIDI case - use simple approach
        if midi_mode and is_midi_file(audio_file):
            is_multi, track_count, _ = is_multitrack_midi(audio_file)
            if is_multi:
                print(f"🎼 Multitrack MIDI detected - using simple approach")

                # Use the new simple approach
                generation_args = {
                    'MODEL': MODEL,
                    'group': group,
                    'subgroup': subgroup,
                    'steps': int(steps),
                    'seed': int(seed),
                    'adapter_scale': float(adapter_scale),
                    'cfg_weight': float(cfg_weight),
                    't0': float(t0),
                    'instrument_strength': float(instrument_strength),
                    'inst_boost': float(inst_boost),
                    'piano_roll_gain': float(piano_roll_gain),
                    'amp_gain': float(amp_gain),
                    'rframe_gain': float(rframe_gain),
                    'rbend_gain': float(rbend_gain),
                    'encodec_gain': float(encodec_gain),
                    'use_overlap_decoder': bool(use_overlap_decoder),
                    'pitch_fidelity_boost': float(pitch_fidelity_boost),
                    'onset_guidance_boost': float(onset_guidance_boost),
                    'pitch_snap_strength': float(pitch_snap_strength),
                    'noise_level': float(noise_level),
                    'win_slow': win_slow
                }

                return process_multitrack_midi_simple(audio_file, progress=progress, **generation_args)

        return out, [], "Single generation completed."

def select_random_file():
    if not MANIFEST_PATHS:
        raise gr.Error("Manifest not loaded.")
    src = Path(random.choice(MANIFEST_PATHS))
    tmp = Path(tempfile.mkdtemp())
    return str(Path(shutil.copy(src, tmp)))

def select_random_file_by_group(target_group):
    if not MANIFEST_DATA:
        raise gr.Error("Manifest not loaded.")
    pool = [it["audio_path"] for it in MANIFEST_DATA if it.get("group")==target_group and it.get("audio_path")]
    if not pool:
        raise gr.Error(f"No files for group '{target_group}'")
    src = Path(random.choice(pool))
    tmp = Path(tempfile.mkdtemp())
    return str(Path(shutil.copy(src, tmp)))

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown("### dø stem — ControlBranch Pipeline")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(
                    label="Upload Audio or MIDI File",
                    file_types=[".wav", ".mp3", ".flac", ".m4a", ".mid", ".midi"],
                    type="filepath"
                )
                random_btn = gr.Button("🎤 Random from Manifest", variant="secondary")
                random_group_btn = gr.Button("🎯 Random from Current Group", variant="secondary")

                # MIDI Processing Options (auto-detected)
                gr.Markdown("#### 🎼 Processing Mode")
                file_type_info = gr.Markdown("**Upload a file to see processing mode**")

                midi_mode = gr.Checkbox(
                    label="MIDI Processing Active",
                    value=False,
                    interactive=False,
                    info="Automatically enabled when MIDI file is detected"
                )
                render_and_extract = gr.Checkbox(
                    label="🎵 Render & Extract Full Conditioning",
                    value=True,  # Default to full conditioning
                    info="MIDI → audio → full conditioning + original MIDI piano roll. Uncheck for piano roll only."
                )

                # Tempo Control
                with gr.Row():
                    detected_tempo_display = gr.Textbox(
                        label="Detected Tempo",
                        value="120.0",
                        interactive=False,
                        scale=1,
                        info="Auto-detected from MIDI file"
                    )
                    tempo_override = gr.Number(
                        label="Tempo Override (BPM)",
                        value=120.0,
                        minimum=40.0,
                        maximum=200.0,
                        step=0.1,
                        scale=1,
                        info="Override detected tempo if needed"
                    )

                group_dd = gr.Dropdown(GROUP_NAMES, label="Instrument Group",
                                       value=GROUP_NAMES[0] if GROUP_NAMES else None)
                subgroup_dd = gr.Dropdown(SUBGROUP_NAMES, label="Instrument Subgroup",
                                          value=SUBGROUP_NAMES[0] if SUBGROUP_NAMES else None)

            with gr.Column(scale=2):
                with gr.Row():
                    seed_slider  = gr.Slider(0, 10000, value=0, step=1, label="Seed (0 = random)")
                    steps_slider = gr.Slider(10, 100, value=40, step=1, label="Steps")
                with gr.Row():
                    adapter_slider = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Adapter Scale")
                    cfg_slider     = gr.Slider(1.0, 6.0, value=1.0, step=0.1, label="Instrument CFG")
                t0_slider = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="T0 (denoising range: 1.0=full, 0.8=partial)")

                instrument_strength = gr.Slider(0.0, 5.0, value=1.0, step=0.1, label="Instrument Conditioning Strength")
                inst_boost = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Instrument Token Boost")

                with gr.Row():
                    piano_roll_gain = gr.Slider(0.0, 4.0, value=1.0, step=0.1, label="Piano Roll Gain")
                    amp_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Amplitude Gain")
                with gr.Row():
                    rframe_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="RFrame Gain")
                    rbend_gain  = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="RBend Gain")
                encodec_gain = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="EnCodec Gain")

                pitch_fidelity_boost = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Pitch Fidelity Boost")
                onset_guidance_boost = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Onset Guidance Boost")
                pitch_snap_strength  = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Pitch Snap Strength")

                noise_level = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Noise Level (0=pure conditioning, 1=pure noise)")

                use_overlap_decoder = gr.Checkbox(label="Use Overlap Decoder", value=True)
                monophonic_mode = gr.Checkbox(label="🎵 Monophonic Mode (separate voices)", value=False)
                generate_btn = gr.Button("🎹 Generate", variant="primary")
                midi_download_btn = gr.Button("🎼 Download Basic Pitch MIDI + Voices", variant="secondary")

        with gr.Row():
            audio_output = gr.Audio(label="🎵 Mixed/Main Output", type="filepath")

        # Individual voice players (max 6 voices as per our limit)
        gr.Markdown("### Individual Voices")
        voice_players = []
        for i in range(6):
            voice_audio = gr.Audio(label=f"🎼 Voice {i+1}", type="filepath", visible=False)
            voice_players.append(voice_audio)

        # Audio voices ZIP download
        with gr.Row():
            audio_voices_zip_file = gr.File(label="🎵 Download All Generated Voices (ZIP)", type="filepath")

        info_output = gr.Textbox(label="Generation Info", interactive=False, lines=2)

        # MIDI Download section
        gr.Markdown("### MIDI Files")
        with gr.Row():
            midi_info = gr.Textbox(label="MIDI Export Info", interactive=False, lines=3)
        with gr.Row():
            main_midi_file = gr.File(label="🎼 Original MIDI (Basic Pitch)", type="filepath")
            voices_zip_file = gr.File(label="🎵 Voice MIDI Files (ZIP)", type="filepath")
        with gr.Row():
            combined_midi_file = gr.File(label="🎹 Combined Voices MIDI", type="filepath")

        # Debug Audio section
        gr.Markdown("### 🔍 Debug Audio (FluidSynth Renders)")
        debug_audio_info = gr.Textbox(label="FluidSynth Debug Info", interactive=False, lines=2,
                                     value="Individual FluidSynth renders will appear here to help debug multitrack performance")
        debug_audio_files = []
        for i in range(6):  # Support up to 6 debug audio files like voice players
            debug_audio = gr.File(label=f"🎵 Track/Voice {i+1} FluidSynth Render", type="filepath", visible=False)
            debug_audio_files.append(debug_audio)

        # Function to handle generation and update all outputs
        def handle_generation_with_voices(*args):
            # Call run_generation with all arguments
            mixed_audio, voice_files, info_text = run_generation(*args)

            # Prepare outputs for all voice players (up to 6)
            voice_outputs = [None] * 6  # Start with all None
            voice_visibility = [False] * 6  # Start with all hidden
            audio_zip_path = None

            if voice_files and len(voice_files) > 1:
                info_text += f"\n\nGenerated {len(voice_files)} individual voices:"
                for i, voice_file in enumerate(voice_files):
                    if i < 6:  # Don't exceed our maximum
                        voice_outputs[i] = voice_file
                        voice_visibility[i] = True
                        info_text += f"\n• Voice {i+1}: {Path(voice_file).name}"

                # Create ZIP file with all voice audio files
                audio_zip_path = create_audio_voices_zip(voice_files)
                info_text += f"\n\n📦 All voices packaged in ZIP file"

            # Create return tuple: mixed_audio, info_text, then updates for all 6 voice players, then ZIP file
            result = [mixed_audio]
            for i in range(6):
                # Return gr.update() to modify existing components
                result.append(gr.update(value=voice_outputs[i], visible=voice_visibility[i]))
            result.append(audio_zip_path)  # Add ZIP file
            result.append(info_text)  # Add info text last

            return tuple(result)

        # events
        random_btn.click(fn=select_random_file, inputs=[], outputs=[audio_input])
        random_group_btn.click(fn=select_random_file_by_group, inputs=[group_dd], outputs=[audio_input])

        # dynamic subgroup options by selected group
        def _opts_for_group(g):
            return gr.Dropdown(choices=sorted(APPROVED_SUBGROUPS.get(g, [])),
                               value=(sorted(APPROVED_SUBGROUPS.get(g, [])) or [None])[0])
        group_dd.change(_opts_for_group, inputs=[group_dd], outputs=[subgroup_dd])

        # Auto-detect file type and update UI accordingly
        def _detect_and_update_file_type(file_path):
            if not file_path:
                return (
                    gr.update(value="**Upload a file to see processing mode**"),  # file_type_info
                    gr.update(value=False),  # midi_mode
                    gr.update(value=1.0),    # piano_roll_gain
                    gr.update(value=1.0),    # amp_gain
                    gr.update(value=1.0),    # rframe_gain
                    gr.update(value=1.0),    # rbend_gain
                    gr.update(value=1.0),    # encodec_gain
                    gr.update(value="120.0"),  # detected_tempo_display
                    gr.update(value=120.0)     # tempo_override
                )

            is_midi = is_midi_file(file_path)
            file_name = Path(file_path).name

            if is_midi:
                # Extract tempo for MIDI files
                detected_tempo = extract_midi_tempo(file_path)
                info_text = f"**🎼 MIDI File Detected**: `{file_name}`  \n✅ **MIDI processing automatically enabled**  \n🎵 **Detected Tempo**: {detected_tempo:.1f} BPM"
                # Auto-enable MIDI mode with smart gains
                return (
                    gr.update(value=info_text),   # file_type_info
                    gr.update(value=True),        # midi_mode (auto-enable)
                    gr.update(value=1.2),         # piano_roll_gain (emphasize)
                    gr.update(value=0.8),         # amp_gain
                    gr.update(value=0.8),         # rframe_gain
                    gr.update(value=0.8),         # rbend_gain
                    gr.update(value=0.5),         # encodec_gain (lower for MIDI)
                    gr.update(value=f"{detected_tempo:.1f}"),  # detected_tempo_display
                    gr.update(value=float(detected_tempo))     # tempo_override
                )
            else:
                info_text = f"**🎵 Audio File Detected**: `{file_name}`  \n✅ **Standard audio processing mode**"
                # Audio mode - standard gains
                return (
                    gr.update(value=info_text),   # file_type_info
                    gr.update(value=False),       # midi_mode (disable)
                    gr.update(value=1.0),         # piano_roll_gain
                    gr.update(value=1.0),         # amp_gain
                    gr.update(value=1.0),         # rframe_gain
                    gr.update(value=1.0),         # rbend_gain
                    gr.update(value=1.0),         # encodec_gain
                    gr.update(value="N/A"),       # detected_tempo_display
                    gr.update(value=120.0)        # tempo_override (default)
                )

        # Smart gain adjustments when render mode changes (for MIDI only)
        def _update_gains_for_render_mode(midi_enabled, render_enabled):
            if not midi_enabled:
                return tuple(gr.update() for _ in range(5))  # No changes for audio files

            if render_enabled:
                # MIDI render mode - full conditioning available
                return (
                    gr.update(value=1.2),  # piano_roll_gain
                    gr.update(value=0.8),  # amp_gain
                    gr.update(value=0.8),  # rframe_gain
                    gr.update(value=0.8),  # rbend_gain
                    gr.update(value=0.5)   # encodec_gain
                )
            else:
                # MIDI direct mode - only piano roll
                return (
                    gr.update(value=1.0),  # piano_roll_gain
                    gr.update(value=0.0),  # amp_gain = 0
                    gr.update(value=0.0),  # rframe_gain = 0
                    gr.update(value=0.0),  # rbend_gain = 0
                    gr.update(value=0.0)   # encodec_gain = 0
                )

        # Auto-detect file type when file is uploaded
        audio_input.change(
            fn=_detect_and_update_file_type,
            inputs=[audio_input],
            outputs=[file_type_info, midi_mode, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain, detected_tempo_display, tempo_override]
        )

        # Update gains when render mode changes (for MIDI files)
        render_and_extract.change(
            fn=_update_gains_for_render_mode,
            inputs=[midi_mode, render_and_extract],
            outputs=[piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain]
        )

        inputs = [audio_input, group_dd, subgroup_dd, seed_slider, steps_slider, adapter_slider, cfg_slider, t0_slider,
                  instrument_strength, inst_boost, piano_roll_gain, amp_gain, rframe_gain, rbend_gain, encodec_gain,
                  use_overlap_decoder, pitch_fidelity_boost, onset_guidance_boost, pitch_snap_strength, noise_level,
                  monophonic_mode, midi_mode, render_and_extract, tempo_override]

        # Create outputs list: mixed output, all 6 voice players, ZIP file, info
        outputs = [audio_output] + voice_players + [audio_voices_zip_file, info_output]

        generate_btn.click(fn=handle_generation_with_voices, inputs=inputs, outputs=outputs)

        # MIDI download button event
        midi_download_btn.click(
            fn=handle_midi_download,
            inputs=[audio_input, subgroup_dd],
            outputs=[main_midi_file, voices_zip_file, combined_midi_file, midi_info, debug_audio_info] + debug_audio_files
        )

    return iface

# ------------------------------------------------------------------------------
# FastAPI Endpoints
# ------------------------------------------------------------------------------
from fastapi import FastAPI, HTTPException, Form, UploadFile, File, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from celery import Celery
from typing import Optional
import uuid
import logging

# Create FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Celery configuration
celery_app = Celery("ace_step_tasks", broker="pyamqp://guest:guest@localhost//", backend="redis://localhost:6379/0")
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Only prefetch 1 task at a time for long-running tasks
    broker_heartbeat=0,  # Disable heartbeat to prevent timeouts on long tasks
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=10,
    broker_pool_limit=None,  # Unlimited connections
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=2400,  # 40 minutes hard limit
    result_expires=3600,
    task_track_started=True,
    task_send_sent_event=True,
    # Broker transport options for RabbitMQ to prevent connection resets
    broker_transport_options={
        'confirm_publish': True,
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.5,
        'socket_keepalive': True,
        'socket_keepalive_options': {
            1: 1,   # TCP_KEEPIDLE
            2: 10,  # TCP_KEEPINTVL
            3: 6    # TCP_KEEPCNT
        }
    },
    # Result backend options
    result_backend_transport_options={
        'socket_keepalive': True,
        'socket_connect_timeout': 30
    },
    # Task result settings
    task_ignore_result=False,
    result_persistent=True
)

logging.basicConfig(level=logging.INFO)

# Celery worker initialization - load model when worker starts
from celery.signals import worker_process_init

@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize model when Celery worker starts"""
    global MODEL, GROUP_NAMES, SUBGROUP_NAMES, MANIFEST_DATA

    # Get checkpoint paths from environment variables (set in start-logs.sh)
    checkpoint = os.environ.get('ACE_CHECKPOINT', '/mnt/msdd/exps/logs_v2/checkpoints/NEWRUN/epoch=85-step=50000.ckpt')
    checkpoint_dir = os.environ.get('ACE_CHECKPOINT_DIR', '/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c')
    manifest = os.environ.get('ACE_MANIFEST', '/home/arlo/Data/final_training_manifest_final.json')

    print(f"🔧 Initializing ACE-Step model in Celery worker...")
    print(f"   Checkpoint: {checkpoint}")
    print(f"   Checkpoint Dir: {checkpoint_dir}")
    print(f"   Manifest: {manifest}")

    # Load manifest
    with open(manifest, "r") as f:
        MANIFEST_DATA = json.load(f)

    # Load model
    MODEL = load_model_any_ckpt(checkpoint, checkpoint_dir, manifest)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL.to(dev).eval()

    GROUP_NAMES = list(APPROVED_GROUPS) if not isinstance(APPROVED_GROUPS, dict) else list(APPROVED_GROUPS.keys())
    SUBGROUP_NAMES = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})

    print(f"✅ Model loaded in Celery worker on {dev}")
    print(f"   Groups: {len(GROUP_NAMES)} | Subgroups: {len(SUBGROUP_NAMES)}")

# Celery task for audio generation
@celery_app.task(bind=True, name="generate_ace_step_task")
def generate_ace_step_task(
    self,
    audio_file_path: Optional[str],
    description: str,
    duration: float,
    steps: int,
    seed: int,
    adapter_scale: float,
    cfg_weight: float,
    instrument_strength: float,
    noise_level: float,
    piano_roll_gain: float,
    amp_gain: float,
    rframe_gain: float,
    rbend_gain: float,
    encodec_gain: float,
    pitch_fidelity_boost: float,
    onset_guidance_boost: float,
    pitch_snap_strength: float,
    instrument_group: str = None,
    instrument_subgroup: str = None,
    monophonic_mode: bool = False,
    fatten_mode: bool = False,
    fatten_type: str = "fake",
    enable_voice_separation: bool = False,
    scene_durations: Optional[list] = None,
    automation_data: Optional[str] = None,
    tape_speed: float = 1.0,
    slowdown_method: str = "stretch",
    use_overlap_decoder: bool = True,
    use_time_varying_noise: bool = False,
    onset_preservation: float = 0.7,
    use_multiresolution_mixing: bool = False,
    use_onset_weighted_encodec: bool = False,
    encodec_onset_boost: float = 2.0,
    use_test_time_adaptation: bool = False,
    adaptation_steps: int = 10,
    adaptation_learning_rate: float = 1e-4,
    use_best_of_n: bool = False,
    n_candidates: int = 12,
    use_self_consistency: bool = False,
    consistency_samples: int = 3,
    consistency_noise_scale: float = 0.05,
    transpose_up_octave: bool = False
):
    """Celery task for ACE-Step audio generation"""
    try:
        global MODEL
        if MODEL is None:
            raise RuntimeError("Model not loaded. Please start the server first.")

        print(f"\n{'='*80}")
        print(f"🎵 ACE-STEP GENERATION TASK STARTED")
        print(f"{'='*80}")
        print(f"Description: {description}")
        print(f"Steps: {steps}, Seed: {seed}, CFG: {cfg_weight}")
        print(f"Monophonic mode: {monophonic_mode}, Voice separation: {enable_voice_separation}")
        print(f"🎚️ FATTEN MODE: {fatten_mode}, Type: {fatten_type}")
        print(f"Duration parameter: {duration}s")
        print(f"Audio file: {audio_file_path}")
        print(f"🎞️ TAPE SPEED: {tape_speed}x, Method: {slowdown_method}")
        print(f"🔊 USE OVERLAP DECODER: {use_overlap_decoder}")

        # CRITICAL: Log scene_durations received
        print(f"\n📥 SCENE DATA RECEIVED:")
        print(f"   scene_durations type: {type(scene_durations)}")
        print(f"   scene_durations value: {scene_durations}")
        if scene_durations:
            print(f"   scene_durations length: {len(scene_durations)}")
            print(f"   Will use scene-aware generation: {len(scene_durations) > 1}")
        else:
            print(f"   scene_durations is None or empty - using simple generation")
        print(f"{'='*80}\n")

        # Use provided instrument group/subgroup or fall back to defaults
        group = instrument_group if instrument_group else (GROUP_NAMES[0] if GROUP_NAMES else "piano")
        subgroup = instrument_subgroup if instrument_subgroup else (SUBGROUP_NAMES[0] if SUBGROUP_NAMES else "piano")

        print(f"Instrument: {group} / {subgroup}")

        # Parse automation data if provided
        global_automation = []
        if automation_data:
            try:
                automation_json = json.loads(automation_data)
                if isinstance(automation_json, dict) and 'points' in automation_json:
                    global_automation = [
                        (float(point['time']), float(point['volume']))
                        for point in automation_json['points']
                    ]
                    print(f"🎛 Loaded {len(global_automation)} global automation points")
            except Exception as e:
                print(f"⚠️ Automation parse error: {e}")

        # Check if audio file is provided for conditioning
        if audio_file_path is None:
            # No audio file provided - generate MIDI conditioning using midigen feature
            print("🎹 No audio file provided - generating MIDI conditioning...")

            # Check if we have scene data for multi-scene MIDI generation (ac.py style)
            if scene_durations and len(scene_durations) > 1:
                print(f"\n{'='*80}")
                print(f"🎬 SCENE-AWARE MIDI GENERATION")
                print(f"{'='*80}")
                print(f"📥 Received {len(scene_durations)} scene durations from frontend:")
                for i, dur in enumerate(scene_durations):
                    print(f"   Scene {i}: {dur:.3f}s")
                total_duration = sum(scene_durations)
                print(f"   TOTAL DURATION: {total_duration:.3f}s")

                # 1. Reconstruct scene_changes from durations
                scene_changes = [0.0]
                for dur in scene_durations:
                    scene_changes.append(scene_changes[-1] + dur)

                print(f"\n📍 Scene change timestamps:")
                for i, timestamp in enumerate(scene_changes):
                    print(f"   Scene {i} starts at: {timestamp:.3f}s")

                # 2. Compute optimal tempo for EACH scene
                tempos = compute_best_tempos(scene_changes)
                print(f"\n🎵 Computed tempos for each scene:")
                for i, tempo in enumerate(tempos):
                    print(f"   Scene {i}: {tempo} BPM")

                # 3. Log tempo analysis
                print("\n🎼 Tempo & Beat Alignment Analysis:")
                for i in range(len(scene_durations)):
                    bpm = tempos[i]
                    seconds_per_beat = 60 / bpm
                    scene_start = scene_changes[i]
                    scene_end = scene_changes[i + 1]
                    scene_duration = scene_end - scene_start

                    beats_in_scene = scene_duration / seconds_per_beat
                    total_beats_before_next_scene = scene_end / seconds_per_beat

                    print(f"▶ Scene {i}")
                    print(f"   Duration: {scene_duration:.3f}s  →  {beats_in_scene:.2f} beats at {bpm} BPM")
                    print(f"   Next scene lands on beat {total_beats_before_next_scene:.2f}")

                # 4. Generate different MIDI for each scene + apply automation
                print(f"\n{'='*80}")
                print(f"🎹 GENERATING MIDI FOR EACH SCENE")
                print(f"{'='*80}")
                scene_midi_paths = {}
                os.makedirs("/tmp/midi_processing", exist_ok=True)

                for scene_idx, scene_dur in enumerate(scene_durations):
                    if scene_dur <= 0:
                        print(f"\n⚠️ Scene {scene_idx}: Skipping (duration {scene_dur:.3f}s <= 0)")
                        continue

                    # Generate unique MIDI at this scene's tempo
                    scene_tempo = tempos[scene_idx]
                    print(f"\n🎼 Scene {scene_idx}:")
                    print(f"   Generating MIDI at {scene_tempo} BPM...")
                    original_midi, _ = get_random_transposed_midi_wav(tempo=scene_tempo)

                    # Get MIDI duration
                    import pretty_midi
                    midi_data = pretty_midi.PrettyMIDI(original_midi)
                    original_midi_duration = midi_data.get_end_time()
                    print(f"   Generated MIDI duration: {original_midi_duration:.3f}s")
                    print(f"   Target scene duration: {scene_dur:.3f}s")

                    # Calculate scene timing
                    scene_start = sum(scene_durations[:scene_idx])
                    scene_end = scene_start + scene_dur
                    print(f"   Scene window: {scene_start:.3f}s → {scene_end:.3f}s")

                    # Extract automation points for this scene
                    scene_automation = []
                    if global_automation:
                        # Get automation points within scene window
                        for t, v in global_automation:
                            if scene_start <= t <= scene_end:
                                # Normalize time to [0, 1] relative to scene
                                normalized_time = (t - scene_start) / scene_dur
                                scene_automation.append((normalized_time, v))

                        # Ensure we have start/end points
                        if not scene_automation or scene_automation[0][0] > 0:
                            scene_automation.insert(0, (0.0, 0.5))
                        if not scene_automation or scene_automation[-1][0] < 1.0:
                            scene_automation.append((1.0, scene_automation[-1][1] if scene_automation else 0.5))

                    # Apply automation to MIDI
                    processed_midi_path = f"/tmp/midi_processing/scene_{scene_idx}_automated.mid"
                    apply_automation_to_midi(
                        midi_path=original_midi,
                        scene_start=scene_start,
                        scene_duration=scene_dur,
                        track_automation=scene_automation,
                        output_path=processed_midi_path,
                        total_duration=sum(scene_durations),
                        scene_tempo=scene_tempo
                    )

                    # Verify processed MIDI duration
                    processed_midi_data = pretty_midi.PrettyMIDI(processed_midi_path)
                    processed_duration = processed_midi_data.get_end_time()
                    print(f"   ✅ Processed MIDI saved: {Path(processed_midi_path).name}")
                    print(f"   ✅ Processed MIDI duration: {processed_duration:.3f}s")

                    scene_midi_paths[scene_idx] = processed_midi_path

                # 5. Choose soundfont based on selected instrument
                soundfont_path = INSTRUMENT_SOUNDFONTS.get(subgroup, INSTRUMENT_SOUNDFONTS.get("default"))
                print(f"\n🎹 Selected soundfont for '{subgroup}': {soundfont_path}")

                # 6. Concatenate all scene MIDIs into one long file
                print(f"\n{'='*80}")
                print(f"🔗 CONCATENATING SCENE MIDIs")
                print(f"{'='*80}")
                print(f"Concatenating {len(scene_midi_paths)} scene MIDIs...")
                for idx in sorted(scene_midi_paths.keys()):
                    print(f"   Scene {idx}: {scene_durations[idx]:.3f}s → {scene_midi_paths[idx]}")

                concatenated_midi_path = f"/tmp/midi_processing/concatenated_{time.time()}.mid"
                concatenated_midi_path = concatenate_midi_scenes(
                    scene_midi_paths=scene_midi_paths,
                    scene_durations=scene_durations,
                    output_path=concatenated_midi_path,
                    soundfont_path=soundfont_path  # Pass soundfont for debug render
                )

                # Verify final concatenated MIDI
                final_midi_data = pretty_midi.PrettyMIDI(concatenated_midi_path)
                final_midi_duration = final_midi_data.get_end_time()
                print(f"\n✅ Concatenated MIDI saved: {concatenated_midi_path}")
                print(f"✅ Concatenated MIDI duration: {final_midi_duration:.3f}s")
                print(f"{'='*80}\n")

                # Handle monophonic mode: check if MIDI has multiple tracks
                if monophonic_mode:
                    print(f"\n🎵 MONOPHONIC MODE: Analyzing MIDI structure...")

                    # Create debug folder for voice renders
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    voice_debug_dir = Path("/home/arlo/Data/voice_debug") / timestamp
                    voice_debug_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n{'='*80}")
                    print(f"📁 DEBUG FILES SAVED TO: {voice_debug_dir}")
                    print(f"{'='*80}")

                    # Copy concatenated MIDI to debug folder
                    debug_concat_midi = voice_debug_dir / "concatenated_master.mid"
                    shutil.copy(concatenated_midi_path, str(debug_concat_midi))
                    print(f"   Copied concatenated MIDI: {debug_concat_midi.name}")

                    # Split MIDI into tracks/voices using pretty_midi
                    pm = pretty_midi.PrettyMIDI(concatenated_midi_path)
                    non_empty_instruments = [inst for inst in pm.instruments if len(inst.notes) > 0]

                    if len(non_empty_instruments) > 1:
                        # Multitrack MIDI: use each track as a voice
                        print(f"   Found {len(non_empty_instruments)} tracks - using each as a voice")
                        voice_midi_paths = []
                        voice_audio_paths = []

                        # Extract tempo changes from concatenated MIDI using mido
                        import mido
                        concat_mido = mido.MidiFile(concatenated_midi_path)
                        tempo_messages = []
                        for track in concat_mido.tracks:
                            for msg in track:
                                if msg.type == 'set_tempo':
                                    tempo_messages.append(msg.copy())
                        print(f"   Extracted {len(tempo_messages)} tempo changes from concatenated MIDI")

                        for i, inst in enumerate(non_empty_instruments):
                            # Create voice MIDI with pretty_midi
                            voice_midi = pretty_midi.PrettyMIDI()
                            voice_midi.instruments.append(inst)

                            # Save temporarily
                            temp_voice_path = voice_debug_dir / f"voice_{i+1}_temp.mid"
                            voice_midi.write(str(temp_voice_path))

                            # Load with mido to add tempo messages
                            voice_mido = mido.MidiFile(str(temp_voice_path))

                            # Insert tempo messages into first track
                            if len(voice_mido.tracks) > 0:
                                # Remove existing tempo messages
                                voice_mido.tracks[0] = mido.MidiTrack([
                                    msg for msg in voice_mido.tracks[0]
                                    if msg.type != 'set_tempo'
                                ])
                                # Insert all tempo messages from concatenated MIDI at the beginning
                                for tempo_msg in reversed(tempo_messages):
                                    voice_mido.tracks[0].insert(0, tempo_msg.copy())

                            # Save final voice MIDI with tempo changes
                            voice_midi_path = voice_debug_dir / f"voice_{i+1}_input.mid"
                            voice_mido.save(str(voice_midi_path))
                            voice_midi_paths.append(str(voice_midi_path))
                            print(f"   💾 Voice {i+1}: {len(inst.notes)} notes + {len(tempo_messages)} tempo changes → {voice_midi_path}")

                            # Clean up temp file
                            temp_voice_path.unlink()

                        # Render each voice with selected soundfont
                        print(f"\n🎼 Rendering {len(voice_midi_paths)} voices with {subgroup} soundfont...")
                        for i, voice_midi in enumerate(voice_midi_paths):
                            voice_audio_path = voice_debug_dir / f"voice_{i+1}_input_render.wav"
                            print(f"   Rendering voice {i+1}/{len(voice_midi_paths)}...")
                            subprocess.run([
                                "fluidsynth", "-ni", "-g", "0.5", "-F", str(voice_audio_path),
                                soundfont_path, voice_midi
                            ], check=True, capture_output=True)
                            voice_audio_paths.append(str(voice_audio_path))
                            print(f"   🎵 Voice {i+1} rendered: {voice_audio_path}")

                        # Fatten mode: create octave-up versions
                        if fatten_mode and fatten_type == "real":
                            print(f"\n🎚️ FATTEN MODE (REAL): Creating octave-up voices...")
                            original_voice_count = len(voice_midi_paths)
                            octave_midi_paths = []
                            octave_audio_paths = []

                            for i, voice_midi in enumerate(voice_midi_paths):
                                # Load MIDI and transpose up 12 semitones
                                pm_voice = pretty_midi.PrettyMIDI(voice_midi)
                                for inst in pm_voice.instruments:
                                    for note in inst.notes:
                                        note.pitch += 12  # Transpose up one octave

                                # Save transposed MIDI
                                octave_midi_path = voice_debug_dir / f"voice_{i+1}_octave.mid"
                                pm_voice.write(str(octave_midi_path))
                                octave_midi_paths.append(str(octave_midi_path))
                                print(f"   💾 Created octave-up voice {i+1}: {octave_midi_path.name}")

                                # Render octave-up version
                                octave_audio_path = voice_debug_dir / f"voice_{i+1}_octave_render.wav"
                                subprocess.run([
                                    "fluidsynth", "-ni", "-g", "0.5", "-F", str(octave_audio_path),
                                    soundfont_path, str(octave_midi_path)
                                ], check=True, capture_output=True)
                                octave_audio_paths.append(str(octave_audio_path))
                                print(f"   🎵 Octave-up voice {i+1} rendered: {octave_audio_path.name}")

                            # Add octave voices to the lists
                            voice_midi_paths.extend(octave_midi_paths)
                            voice_audio_paths.extend(octave_audio_paths)
                            print(f"\n✅ Doubled to {len(voice_audio_paths)} total voices ({original_voice_count} original + {original_voice_count} octave-up)")

                        # Set audio_file_path to None to trigger voice-by-voice generation
                        audio_file_path = None
                        print(f"\n{'='*80}")
                        print(f"📂 DEBUG FILES SUMMARY:")
                        print(f"   Master MIDI: {debug_concat_midi}")
                        for i, (midi_path, audio_path) in enumerate(zip(voice_midi_paths, voice_audio_paths), 1):
                            print(f"   Voice {i}: {Path(midi_path).name} → {Path(audio_path).name}")
                        print(f"{'='*80}\n")

                    else:
                        # Single track MIDI: split MIDI into voices FIRST, then render each voice
                        print(f"   Single track detected - splitting MIDI into voices...")

                        # Load the concatenated MIDI
                        pm_concat = pretty_midi.PrettyMIDI(concatenated_midi_path)

                        # Split into voices using voice separation algorithm
                        # Get the single instrument
                        if len(pm_concat.instruments) > 0 and len(pm_concat.instruments[0].notes) > 0:
                            instrument = pm_concat.instruments[0]

                            # Sort notes by start time
                            sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

                            # Simple voice separation: group overlapping notes into different voices
                            voices = []
                            for note in sorted_notes:
                                # Find a voice where this note doesn't overlap
                                assigned = False
                                for voice_notes in voices:
                                    # Check if this note overlaps with any note in this voice
                                    if all(note.start >= existing.end for existing in voice_notes):
                                        voice_notes.append(note)
                                        assigned = True
                                        break

                                if not assigned:
                                    # Create new voice
                                    voices.append([note])

                            print(f"   Split into {len(voices)} voices")

                            # Create voice MIDI files and render each
                            voice_midi_paths = []
                            voice_audio_paths = []

                            # Get tempo changes from original MIDI
                            import mido
                            concat_mido = mido.MidiFile(concatenated_midi_path)
                            tempo_messages = []
                            for track in concat_mido.tracks:
                                for msg in track:
                                    if msg.type == 'set_tempo':
                                        tempo_messages.append(msg.copy())

                            for voice_idx, voice_notes in enumerate(voices):
                                # Create MIDI for this voice
                                voice_pm = pretty_midi.PrettyMIDI()
                                voice_inst = pretty_midi.Instrument(program=0)
                                voice_inst.notes = voice_notes
                                voice_pm.instruments.append(voice_inst)

                                # Save voice MIDI
                                voice_midi_path = voice_debug_dir / f"voice_{voice_idx + 1}.mid"
                                voice_pm.write(str(voice_midi_path))
                                voice_midi_paths.append(str(voice_midi_path))
                                print(f"   💾 Saved voice {voice_idx + 1} MIDI: {voice_midi_path}")

                                # Add tempo changes to voice MIDI
                                if tempo_messages:
                                    voice_mido = mido.MidiFile(str(voice_midi_path))
                                    if len(voice_mido.tracks) > 0:
                                        for tempo_msg in tempo_messages:
                                            voice_mido.tracks[0].insert(0, tempo_msg)
                                        voice_mido.save(str(voice_midi_path))

                                # Render voice with FluidSynth
                                voice_audio_path = voice_debug_dir / f"voice_{voice_idx + 1}_rendered.wav"
                                subprocess.run([
                                    "fluidsynth", "-ni", "-g", "0.5", "-F", str(voice_audio_path),
                                    soundfont_path, str(voice_midi_path)
                                ], check=True, capture_output=True)
                                voice_audio_paths.append(str(voice_audio_path))
                                print(f"   🎵 Rendered voice {voice_idx + 1}: {voice_audio_path}")

                            print(f"\n✅ Split and rendered {len(voices)} voices")

                            # Fatten mode: create octave-up versions
                            if fatten_mode and fatten_type == "real":
                                print(f"\n🎚️ FATTEN MODE (REAL): Creating octave-up voices...")
                                original_voice_count = len(voice_midi_paths)
                                octave_midi_paths = []
                                octave_audio_paths = []

                                for voice_idx, voice_midi in enumerate(voice_midi_paths):
                                    # Load MIDI and transpose up 12 semitones
                                    pm_voice = pretty_midi.PrettyMIDI(voice_midi)
                                    for inst in pm_voice.instruments:
                                        for note in inst.notes:
                                            note.pitch += 12  # Transpose up one octave

                                    # Save transposed MIDI
                                    octave_midi_path = voice_debug_dir / f"voice_{voice_idx+1}_octave.mid"
                                    pm_voice.write(str(octave_midi_path))
                                    octave_midi_paths.append(str(octave_midi_path))
                                    print(f"   💾 Created octave-up voice {voice_idx+1}: {octave_midi_path.name}")

                                    # Render octave-up version
                                    octave_audio_path = voice_debug_dir / f"voice_{voice_idx+1}_octave_rendered.wav"
                                    subprocess.run([
                                        "fluidsynth", "-ni", "-g", "0.5", "-F", str(octave_audio_path),
                                        soundfont_path, str(octave_midi_path)
                                    ], check=True, capture_output=True)
                                    octave_audio_paths.append(str(octave_audio_path))
                                    print(f"   🎵 Octave-up voice {voice_idx+1} rendered: {octave_audio_path.name}")

                                # Add octave voices to the lists
                                voice_midi_paths.extend(octave_midi_paths)
                                voice_audio_paths.extend(octave_audio_paths)
                                print(f"\n✅ Doubled to {len(voice_audio_paths)} total voices ({original_voice_count} original + {original_voice_count} octave-up)")

                            print(f"\n{'='*80}")
                            print(f"📂 DEBUG FILES SUMMARY:")
                            print(f"   Master MIDI: {debug_concat_midi}")
                            for i, (midi_path, audio_path) in enumerate(zip(voice_midi_paths, voice_audio_paths), 1):
                                print(f"   Voice {i}: {Path(midi_path).name} → {Path(audio_path).name}")
                            print(f"{'='*80}\n")
                            # Set audio_file_path to None to trigger voice-by-voice generation
                            audio_file_path = None
                        else:
                            print(f"   ERROR: No notes found in concatenated MIDI")
                            # Fallback to rendering full MIDI
                            audio_file_path = concatenated_midi_path.replace('.mid', '.wav')
                            subprocess.run([
                                "fluidsynth", "-ni", "-g", "0.5", "-F", audio_file_path,
                                soundfont_path, concatenated_midi_path
                            ], check=True, capture_output=True)

                else:
                    # Normal mode: render concatenated MIDI with selected soundfont
                    print(f"\n🎼 Rendering concatenated MIDI with {subgroup} soundfont...")
                    audio_file_path = concatenated_midi_path.replace('.mid', '.wav')
                    subprocess.run([
                        "fluidsynth", "-ni", "-g", "0.5", "-F", audio_file_path,
                        soundfont_path, concatenated_midi_path
                    ], check=True, capture_output=True)
                    print(f"✅ Rendered audio: {audio_file_path}")

            else:
                # Simple single-scene MIDI generation (original behavior)
                scene_changes = [0.0, float(duration)]

                # Compute optimal tempo for the duration
                tempos = compute_best_tempos(scene_changes)
                selected_tempo = tempos[0] if tempos else 120

                print(f"🎼 Computed tempo: {selected_tempo} BPM for {duration}s duration")

                # Generate random MIDI file at the computed tempo
                midi_path, wav_path = get_random_transposed_midi_wav(tempo=selected_tempo)

                # Use the generated WAV as conditioning
                audio_file_path = wav_path
                print(f"✅ Generated MIDI conditioning: {audio_file_path}")

        # Setup output directory
        process_id = str(uuid.uuid4())
        output_dir = Path("/home/arlo/ScoreAI/audiofiles") / f"ace_step_output_{process_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Handle monophonic mode with scene changes (pre-rendered voices)
        if monophonic_mode and audio_file_path is None and 'voice_audio_paths' in locals():
            print(f"\n🎵 MONOPHONIC + SCENE CHANGES MODE: Generating {len(voice_audio_paths)} voices...")
            if tape_speed < 1.0:
                print(f"🎞️ TAPE SPEED SLOWDOWN ENABLED: {tape_speed}x using {slowdown_method} method")
                print(f"   Will slow down FluidSynth-rendered voices, process, then speed back up")

            # Track completed voices for incremental updates
            completed_voices = []
            fps = 43.066

            for voice_idx, voice_audio in enumerate(voice_audio_paths):
                print(f"\n🎼 Processing voice {voice_idx + 1}/{len(voice_audio_paths)}")
                print(f"   Original FluidSynth render: {voice_audio}")

                # Apply speed slowdown if needed
                if tape_speed < 1.0:
                    print(f"🎞️ Applying {slowdown_method} slowdown ({tape_speed}x) to voice {voice_idx + 1}...")
                    slowed_voice_path = str(Path(voice_audio).parent / f"slowed_{Path(voice_audio).name}")
                    print(f"   Input:  {voice_audio}")
                    print(f"   Output: {slowed_voice_path}")
                    if slowdown_method == "stretch":
                        apply_time_stretch_sox(voice_audio, slowed_voice_path, tape_speed)
                    else:  # tape
                        apply_tape_speed_sox(voice_audio, slowed_voice_path, tape_speed)
                    voice_audio = slowed_voice_path
                    print(f"✅ Voice {voice_idx + 1} slowed file saved: {slowed_voice_path}")
                    print(f"   Now using slowed version for conditioning extraction")

                # Determine duration from voice audio
                import torchaudio
                wav, sr = torchaudio.load(voice_audio)
                actual_duration = wav.shape[-1] / sr
                window_slow = int(actual_duration * fps)
                print(f"   Voice duration: {actual_duration:.2f}s ({window_slow} frames)")

                # Extract conditioning from this voice
                extraction = extract_conditioning_from_audio(
                    voice_audio,
                    instrument_group=subgroup
                )

                # Load conditioning data
                piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning(extraction, window_slow)

                # Generate audio for this voice
                voice_seed = seed + (voice_idx * 1000)
                print(f"   Generating with seed {voice_seed}...")

                voice_output = generate(
                    model=MODEL,
                    piano_roll=piano_roll,
                    amp=amp,
                    rframe=rframe,
                    rbend=rbend,
                    encodec_tokens=encodec_tokens,
                    group=group,
                    subgroup=subgroup,
                    steps=steps,
                    seed=voice_seed,
                    adapter_scale=adapter_scale,
                    cfg_weight=cfg_weight,
                    t0=1.0,
                    sr_out=44100,
                    instrument_strength=instrument_strength,
                    inst_boost=2.5,
                    piano_roll_gain=piano_roll_gain,
                    amp_gain=amp_gain,
                    rframe_gain=rframe_gain,
                    rbend_gain=rbend_gain,
                    encodec_gain=encodec_gain,
                    use_overlap_decoder=use_overlap_decoder,
                    original_audio_length=int(actual_duration * 44100),
                    pitch_fidelity_boost=pitch_fidelity_boost,
                    onset_guidance_boost=onset_guidance_boost,
                    pitch_snap_strength=pitch_snap_strength,
                    noise_level=noise_level,
                    audio_file=voice_audio,
                    use_time_varying_noise=use_time_varying_noise,
                    onset_preservation=onset_preservation,
                    use_multiresolution_mixing=use_multiresolution_mixing,
                    use_onset_weighted_encodec=use_onset_weighted_encodec,
                    encodec_onset_boost=encodec_onset_boost,
                    use_test_time_adaptation=use_test_time_adaptation,
                    adaptation_steps=adaptation_steps,
                    adaptation_learning_rate=adaptation_learning_rate
                )

                # Copy voice to output directory
                voice_output_path = output_dir / f"{voice_idx + 1}.wav"
                shutil.copy(voice_output, str(voice_output_path))

                # Copy to debug folder if it exists
                if 'voice_debug_dir' in locals() and voice_debug_dir.exists():
                    debug_output_path = voice_debug_dir / f"voice_{voice_idx + 1}_output.wav"
                    shutil.copy(voice_output, str(debug_output_path))
                    print(f"   📁 Debug copy: {debug_output_path.name}")

                # Add to completed list
                download_url = f"/download/{process_id}/{voice_idx + 1}.wav"
                completed_voices.append(download_url)

                # Update Celery task state
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'completed_voices': completed_voices.copy(),
                        'total_voices': len(voice_audio_paths),
                        'progress': len(completed_voices) / len(voice_audio_paths)
                    }
                )
                print(f"   ✅ Voice {voice_idx + 1} complete: {voice_output_path.name}")

            print(f"\n✅ All {len(voice_audio_paths)} voices generated successfully")

            # Print debug folder summary
            if 'voice_debug_dir' in locals() and voice_debug_dir.exists():
                print(f"\n{'='*60}")
                print(f"📁 VOICE DEBUG FILES SAVED")
                print(f"{'='*60}")
                print(f"Location: {voice_debug_dir}")
                print(f"\nMaster file:")
                print(f"  • concatenated_master.mid  (All voices combined with tempo changes)")
                print(f"\nPer-voice files:")
                for i in range(len(voice_audio_paths)):
                    print(f"  Voice {i+1}:")
                    print(f"    • voice_{i+1}_input.mid          (MIDI with tempo changes)")
                    print(f"    • voice_{i+1}_input_render.wav  (Rendered with {subgroup} soundfont)")
                    print(f"    • voice_{i+1}_output.wav        (AI-generated output)")
                print(f"{'='*60}\n")

            # Apply speed up if we slowed down the input
            if tape_speed < 1.0:
                print(f"\n{'='*80}")
                print(f"🎞️ SPEED RESTORATION ({slowdown_method.upper()}): {1.0/tape_speed:.2f}x")
                print(f"{'='*80}")
                print(f"Speeding up all {len(voice_audio_paths)} voice outputs...")

                speedup_factor = 1.0 / tape_speed
                for voice_idx in range(len(voice_audio_paths)):
                    voice_file = output_dir / f"{voice_idx + 1}.wav"
                    if voice_file.exists():
                        print(f"   Processing voice {voice_idx + 1}: {voice_file}")
                        temp_path = str(voice_file.parent / f"temp_{voice_file.name}")
                        print(f"   Input:  {voice_file}")
                        print(f"   Temp:   {temp_path}")
                        if slowdown_method == "stretch":
                            apply_time_stretch_sox(str(voice_file), temp_path, speedup_factor)
                        else:  # tape
                            apply_tape_speed_sox(str(voice_file), temp_path, speedup_factor)
                        shutil.move(temp_path, str(voice_file))
                        print(f"✅ Voice {voice_idx + 1} restored and saved: {voice_file}")

                print(f"✅ All voice outputs restored to original speed")
                print(f"{'='*80}\n")

            return {"file_paths": completed_voices}

        # MULTITRACK MIDI DETECTION: Check if uploaded file is multi-track MIDI with voice separation disabled
        if is_midi_file(audio_file_path) and monophonic_mode and not enable_voice_separation:
            print(f"\n{'='*80}")
            print(f"🎼 MULTI-TRACK MIDI DETECTED (Voice Separation: Disabled)")
            print(f"{'='*80}")
            if tape_speed < 1.0:
                print(f"🎞️ TAPE SPEED ENABLED: {tape_speed}x using {slowdown_method} method")
                print(f"   FluidSynth renders will be slowed, processed, then sped back up")

            is_multi, track_count, _ = is_multitrack_midi(audio_file_path)

            if is_multi:
                print(f"   Detected {track_count} tracks - rendering each with FluidSynth")
                print(f"   Instrument: {subgroup}")

                # Create debug directory in /tmp
                debug_dir = Path(f"/tmp/multitrack_debug_{process_id}")
                debug_dir.mkdir(parents=True, exist_ok=True)
                print(f"   Debug folder: {debug_dir}")

                # Step 1: Split MIDI into separate track files
                print(f"\n📂 Step 1: Splitting MIDI into {track_count} separate files...")
                track_midi_files = split_midi_into_track_files(audio_file_path, output_dir=str(debug_dir))
                print(f"   ✅ Created {len(track_midi_files)} track MIDI files")

                # Step 1.5: Modify MIDI tempo if using stretch mode
                if tape_speed < 1.0 and slowdown_method == "stretch":
                    print(f"\n🎼 Step 1.5: Modifying MIDI tempo ({tape_speed}x) for stretch mode...")
                    slowed_midi_files = []
                    for i, track_midi in enumerate(track_midi_files):
                        slowed_midi_path = str(Path(track_midi).parent / f"slowed_{Path(track_midi).name}")
                        modify_midi_tempo(track_midi, slowed_midi_path, tape_speed)
                        slowed_midi_files.append(slowed_midi_path)
                    track_midi_files = slowed_midi_files
                    print(f"   ✅ All {len(track_midi_files)} MIDI files slowed to {tape_speed}x tempo")
                elif tape_speed < 1.0 and slowdown_method == "tape":
                    print(f"\n🎼 Step 1.5: Skipping MIDI tempo modification (tape mode - will slow renders instead)")

                # Step 2: Render each track with FluidSynth (using correct soundfont)
                print(f"\n🎹 Step 2: Rendering each track with FluidSynth ({subgroup} soundfont)...")
                track_audio_files = []
                for i, track_midi in enumerate(track_midi_files):
                    print(f"   Rendering track {i+1}/{len(track_midi_files)}: {Path(track_midi).name}")
                    track_audio = render_midi_to_audio(
                        track_midi,
                        output_dir=str(debug_dir),
                        instrument_group=subgroup  # Use selected instrument for soundfont
                    )
                    track_audio_files.append(track_audio)
                    print(f"      → {Path(track_audio).name}")

                print(f"   ✅ All tracks rendered")

                # Step 2.5: Real fatten mode - transpose MIDI and render octave-up versions
                if fatten_mode and fatten_type == "real":
                    print(f"\n🎚️ Step 2.5 REAL FATTEN MODE: Creating octave-up MIDI and rendering...")
                    original_track_count = len(track_midi_files)
                    octave_midi_files = []
                    octave_audio_files = []

                    for i, track_midi in enumerate(track_midi_files):
                        # Load MIDI and transpose up 12 semitones (one octave)
                        import pretty_midi
                        pm = pretty_midi.PrettyMIDI(track_midi)
                        for instrument in pm.instruments:
                            for note in instrument.notes:
                                note.pitch += 12  # Transpose up one octave

                        # Save transposed MIDI
                        octave_midi_path = str(Path(track_midi).parent / f"octave_{Path(track_midi).name}")
                        pm.write(octave_midi_path)
                        octave_midi_files.append(octave_midi_path)
                        print(f"   Created octave-up MIDI {i+1}: {Path(octave_midi_path).name}")

                        # Render octave-up version with FluidSynth
                        octave_audio = render_midi_to_audio(
                            octave_midi_path,
                            output_dir=str(debug_dir),
                            instrument_group=subgroup
                        )
                        octave_audio_files.append(octave_audio)
                        print(f"      → Rendered: {Path(octave_audio).name}")

                    # Add octave files to the processing lists
                    track_midi_files.extend(octave_midi_files)
                    track_audio_files.extend(octave_audio_files)
                    print(f"   ✅ Doubled to {len(track_audio_files)} total tracks ({original_track_count} original + {original_track_count} octave-up)")

                # Step 2.6: Apply tape slowdown to renders if using tape mode
                if tape_speed < 1.0 and slowdown_method == "tape":
                    print(f"\n🎞️ Step 2.6: Applying tape slowdown ({tape_speed}x) to FluidSynth renders...")
                    slowed_audio_files = []
                    for i, track_audio in enumerate(track_audio_files):
                        slowed_audio_path = str(Path(track_audio).parent / f"slowed_{Path(track_audio).name}")
                        print(f"   Track {i+1}: {Path(track_audio).name} → {Path(slowed_audio_path).name}")
                        apply_tape_speed_sox(track_audio, slowed_audio_path, tape_speed)
                        slowed_audio_files.append(slowed_audio_path)
                    track_audio_files = slowed_audio_files
                    print(f"   ✅ All {len(track_audio_files)} renders slowed with tape method")
                elif tape_speed < 1.0 and slowdown_method == "stretch":
                    print(f"   (Renders are at {tape_speed}x tempo from MIDI modification)")

                # Step 3: Extract conditioning and generate for each track
                print(f"\n🎵 Step 3: Generating AI audio for each track with full conditioning...")
                print(f"   📊 Total FluidSynth renders to process: {len(track_audio_files)}")
                if fatten_mode and fatten_type == "fake":
                    print(f"   🎚️ FAKE FATTEN MODE: Will pitch shift outputs up an octave (doubling track count)")
                elif fatten_mode and fatten_type == "real":
                    print(f"   🎚️ REAL FATTEN MODE: Octave-up tracks already rendered from transposed MIDI")
                    print(f"   📊 This includes {len(track_audio_files)//2} original + {len(track_audio_files)//2} octave-up tracks")
                completed_voices = []
                voice_outputs = []
                fps = 43.066
                output_track_counter = 1  # Track number for output files (accounts for fatten mode doubling)
                # For real fatten, track_audio_files is already doubled; for fake fatten, we'll double during generation
                total_expected_tracks = len(track_audio_files) * (2 if (fatten_mode and fatten_type == "fake") else 1)
                print(f"   📊 Expected total output tracks: {total_expected_tracks}")

                for track_idx, track_audio in enumerate(track_audio_files):
                    print(f"\n🎼 Processing track {track_idx + 1}/{len(track_audio_files)}")
                    print(f"   Input: {Path(track_audio).name}")

                    # Determine duration from track audio
                    import torchaudio
                    wav, sr = torchaudio.load(track_audio)
                    actual_duration = wav.shape[-1] / sr
                    window_slow = int(actual_duration * fps)
                    print(f"   Duration: {actual_duration:.2f}s ({window_slow} frames)")

                    # Extract conditioning from rendered track
                    print(f"   Extracting conditioning...")
                    extraction = extract_conditioning_from_audio(
                        track_audio,
                        instrument_group=subgroup
                    )

                    # Load conditioning data
                    piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning(extraction, window_slow)
                    print(f"   Loaded conditioning: PR={piano_roll.shape}, amp={amp.shape}")

                    # Generate audio for this track with FULL conditioning
                    voice_seed = seed + (track_idx * 1000)
                    print(f"   Generating with seed {voice_seed}...")

                    voice_output = generate(
                        model=MODEL,
                        piano_roll=piano_roll,
                        amp=amp,  # REAL conditioning
                        rframe=rframe,  # REAL conditioning
                        rbend=rbend,  # REAL conditioning
                        encodec_tokens=encodec_tokens,  # REAL conditioning
                        group=group,
                        subgroup=subgroup,
                        steps=steps,
                        seed=voice_seed,
                        adapter_scale=adapter_scale,
                        cfg_weight=cfg_weight,
                        t0=1.0,
                        sr_out=44100,
                        instrument_strength=instrument_strength,
                        inst_boost=2.5,
                        piano_roll_gain=piano_roll_gain,
                        amp_gain=amp_gain,
                        rframe_gain=rframe_gain,
                        rbend_gain=rbend_gain,
                        encodec_gain=encodec_gain,
                        use_overlap_decoder=use_overlap_decoder,
                        original_audio_length=wav.shape[-1],
                        pitch_fidelity_boost=pitch_fidelity_boost,
                        onset_guidance_boost=onset_guidance_boost,
                        pitch_snap_strength=pitch_snap_strength,
                        noise_level=noise_level,
                        audio_file=track_audio,
                        use_time_varying_noise=use_time_varying_noise,
                        onset_preservation=onset_preservation,
                        use_multiresolution_mixing=use_multiresolution_mixing,
                        use_onset_weighted_encodec=use_onset_weighted_encodec,
                        encodec_onset_boost=encodec_onset_boost,
                        use_test_time_adaptation=use_test_time_adaptation,
                        adaptation_steps=adaptation_steps,
                        adaptation_learning_rate=adaptation_learning_rate
                    )

                    # Speed up this voice immediately if we slowed it down
                    if tape_speed < 1.0:
                        print(f"   🎞️ Restoring speed ({1.0/tape_speed:.2f}x) for track {track_idx + 1}...")
                        speedup_factor = 1.0 / tape_speed
                        temp_output = str(Path(voice_output).parent / f"temp_sped_{Path(voice_output).name}")
                        if slowdown_method == "stretch":
                            apply_time_stretch_sox(voice_output, temp_output, speedup_factor)
                        else:  # tape
                            apply_tape_speed_sox(voice_output, temp_output, speedup_factor)
                        # Replace voice_output with sped-up version
                        shutil.move(temp_output, voice_output)
                        print(f"   ✅ Track {track_idx + 1} restored to original speed")

                    # Copy original voice to output directory
                    voice_output_path = output_dir / f"{output_track_counter}.wav"
                    shutil.copy(voice_output, str(voice_output_path))
                    voice_outputs.append(str(voice_output_path))

                    # Also copy to debug directory for comparison
                    debug_output = debug_dir / f"track{track_idx + 1}_output.wav"
                    shutil.copy(voice_output, str(debug_output))

                    download_url = f"/download/{process_id}/{output_track_counter}.wav"
                    completed_voices.append(download_url)
                    print(f"   ✅ Track {track_idx + 1} saved as output {output_track_counter}: {voice_output_path.name}")
                    output_track_counter += 1

                    # Fake fatten mode: create octave-up version by pitch shifting
                    if fatten_mode and fatten_type == "fake":
                        print(f"   🎚️ FAKE FATTEN MODE: Pitch shifting output up an octave...")
                        octave_output = str(Path(voice_output).parent / f"octave_{Path(voice_output).name}")
                        apply_pitch_shift_sox(voice_output, octave_output, 12)  # +12 semitones = +1 octave

                        # Copy octave version to output directory
                        octave_output_path = output_dir / f"{output_track_counter}.wav"
                        shutil.copy(octave_output, str(octave_output_path))
                        voice_outputs.append(str(octave_output_path))

                        # Also copy to debug directory
                        debug_octave = debug_dir / f"track{track_idx + 1}_octave.wav"
                        shutil.copy(octave_output, str(debug_octave))

                        octave_download_url = f"/download/{process_id}/{output_track_counter}.wav"
                        completed_voices.append(octave_download_url)
                        print(f"   ✅ Octave-up saved as output {output_track_counter}: {octave_output_path.name}")
                        output_track_counter += 1

                    # Update Celery task state with partial results for frontend
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'completed_voices': completed_voices.copy(),
                            'total_voices': total_expected_tracks,
                            'progress': len(completed_voices) / total_expected_tracks
                        }
                    )

                    print(f"   📊 Progress: {len(completed_voices)}/{total_expected_tracks} tracks complete")

                # Step 4: Create mix of all tracks
                print(f"\n🎚️ Step 4: Creating mix of {len(voice_outputs)} tracks...")
                mix_path = output_dir / "0.wav"
                mix_output = sum_audio_tracks(voice_outputs, str(mix_path), normalize=True)

                # Copy mix to debug directory
                debug_mix = debug_dir / "0_mix.wav"
                shutil.copy(str(mix_path), str(debug_mix))

                file_paths = [f"/download/{process_id}/0.wav"] + completed_voices

                # Print debug summary
                print(f"\n{'='*80}")
                print(f"✅ MULTI-TRACK MIDI GENERATION COMPLETE")
                print(f"{'='*80}")
                print(f"📁 Debug files saved to: {debug_dir}")
                print(f"\n📂 Debug folder contents:")
                print(f"   MIDI tracks:")
                for midi_file in track_midi_files:
                    print(f"      • {Path(midi_file).name}")
                print(f"\n   FluidSynth renders:")
                for audio_file in track_audio_files:
                    print(f"      • {Path(audio_file).name}")
                print(f"\n   AI-generated outputs:")
                for i in range(len(voice_outputs)):
                    print(f"      • track{i+1}_output.wav")
                print(f"\n   Final mix:")
                print(f"      • 0_mix.wav")
                print(f"\n🎵 Returning to frontend:")
                print(f"   Total tracks: {len(file_paths)}")
                print(f"   Mix (track 0): {file_paths[0]}")
                print(f"   Individual tracks: {completed_voices}")
                for i, track in enumerate(file_paths):
                    track_type = "MIX (muted)" if i == 0 else f"Voice {i}"
                    print(f"      Track {i}: {track} [{track_type}]")
                print(f"{'='*80}\n")

                if tape_speed < 1.0:
                    print(f"✅ All tracks already restored to original speed (done per-track)")

                return {
                    "file_paths": file_paths,
                    "mainAudio": file_paths[0],
                    "voices": completed_voices  # Individual tracks without mix
                }

        # Apply speed slowdown if needed (before conditioning extraction)
        original_audio_path = audio_file_path
        if tape_speed < 1.0:
            print(f"\n{'='*80}")
            print(f"🎞️ SPEED SLOWDOWN ({slowdown_method.upper()}): {tape_speed}x")
            print(f"{'='*80}")
            print(f"Slowing down input audio for better detail capture...")
            print(f"Method: {slowdown_method} ({'pitch changes' if slowdown_method == 'tape' else 'pitch preserved'})")

            # Create slowed-down version
            slowed_path = str(Path(audio_file_path).parent / f"slowed_{Path(audio_file_path).name}")

            # Apply appropriate slowdown method
            if slowdown_method == "stretch":
                apply_time_stretch_sox(audio_file_path, slowed_path, tape_speed)
            else:  # tape
                apply_tape_speed_sox(audio_file_path, slowed_path, tape_speed)

            # Use slowed version for processing
            audio_file_path = slowed_path
            print(f"✅ Using slowed audio for processing: {Path(audio_file_path).name}")
            print(f"   (Will speed back up to {1.0/tape_speed:.2f}x after generation)")
            print(f"{'='*80}\n")

        # Get the original audio length for correct output timing
        # CRITICAL FIX: Determine actual duration from audio file, not parameter
        try:
            import torchaudio
            wav, sr = torchaudio.load(audio_file_path)
            orig_len = wav.shape[-1]
            actual_duration = orig_len / sr
            print(f"🎵 Original audio length: {orig_len} samples ({actual_duration:.2f}s at {sr}Hz)")

            # Transpose up octave if requested
            if transpose_up_octave:
                print(f"🎵 Transposing input up one octave (+12 semitones)...")
                # Pitch shift up one octave by resampling
                # To pitch up: resample from (sr/2) to sr, treating current samples as if from lower rate
                # This doubles frequency content = up one octave
                wav = torchaudio.functional.resample(wav, int(sr / 2), sr)
                # Save transposed audio to temp file for conditioning extraction
                temp_transposed = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                torchaudio.save(temp_transposed.name, wav, sr)
                audio_file_path = temp_transposed.name
                print(f"   ✅ Transposed audio saved to: {audio_file_path}")
                print(f"   Original duration preserved, pitch shifted up 12 semitones")
        except Exception as e:
            orig_len = None
            actual_duration = duration  # Fallback to parameter
            print(f"⚠️ Could not determine audio length: {e}, using parameter duration {duration}s")

        # Extract conditioning from audio file (either uploaded or generated)
        extraction = extract_conditioning_from_audio(
            audio_file_path,
            instrument_group=subgroup  # Use subgroup as instrument hint
        )

        # Load conditioning data - use ACTUAL duration from audio file
        # Use 43.066 fps to match piano roll frame rate
        fps = 43.066
        window_slow = int(actual_duration * fps)
        print(f"🎵 Using window_slow = {window_slow} frames ({actual_duration:.2f}s at {fps} fps)")
        piano_roll, amp, rframe, rbend, encodec_tokens = load_conditioning(extraction, window_slow)

        # Generate audio - use monophonic mode if enabled
        if monophonic_mode:
            print(f"🎵 Using monophonic mode - separating voices...")

            # Track completed voices for incremental updates
            completed_voices = []

            def voice_complete_callback(voice_idx, voice_path, total_voices):
                """Called when each voice completes generation"""
                # Copy voice to output directory with sequential numbering
                voice_output_path = output_dir / f"{voice_idx + 1}.wav"
                shutil.copy(voice_path, str(voice_output_path))

                # Add to completed list
                download_url = f"/download/{process_id}/{voice_idx + 1}.wav"
                completed_voices.append(download_url)

                # Update Celery task state with partial results
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'completed_voices': completed_voices.copy(),
                        'total_voices': total_voices,
                        'progress': len(completed_voices) / total_voices
                    }
                )
                print(f"📊 Progress update: {len(completed_voices)}/{total_voices} voices completed")

            result = generate_monophonic_multiple(
                model=MODEL,
                piano_roll=piano_roll,
                amp=amp,
                rframe=rframe,
                rbend=rbend,
                encodec_tokens=encodec_tokens,
                group=group,
                subgroup=subgroup,
                steps=steps,
                seed=seed,
                adapter_scale=adapter_scale,
                cfg_weight=cfg_weight,
                t0=1.0,  # FIXME: Should this use user's t0 instead of hardcoded 1.0?
                sr_out=44100,
                instrument_strength=instrument_strength,
                inst_boost=inst_boost,  # Changed from hardcoded 2.5 to use user's value
                piano_roll_gain=piano_roll_gain,
                amp_gain=amp_gain,
                rframe_gain=rframe_gain,
                rbend_gain=rbend_gain,
                encodec_gain=encodec_gain,
                use_overlap_decoder=use_overlap_decoder,
                original_audio_length=orig_len,
                pitch_fidelity_boost=pitch_fidelity_boost,
                onset_guidance_boost=onset_guidance_boost,
                pitch_snap_strength=pitch_snap_strength,
                noise_level=noise_level,
                audio_file=audio_file_path,
                progress=None,
                voice_complete_callback=voice_complete_callback,
                enable_voice_separation=enable_voice_separation,
                fatten_mode=fatten_mode,
                fatten_type=fatten_type
            )

            # Handle monophonic result - individual voices already copied by callback
            # Just need to copy the mixed output as 0.wav
            file_paths = [f"/download/{process_id}/0.wav"] + completed_voices

            if isinstance(result, dict):
                # Copy mixed output as 0.wav
                mixed_path = result.get("mixed")
                if mixed_path and os.path.exists(mixed_path):
                    output_path = output_dir / "0.wav"
                    shutil.copy(mixed_path, str(output_path))
                    print(f"✅ Mixed output saved: {output_path}")
                else:
                    print(f"⚠️ No mixed output found in result")

                logging.info(f"✅ ACE-Step monophonic generation complete: {len(file_paths)} files")
            else:
                # Fallback - single output
                output_path = output_dir / "0.wav"
                shutil.copy(result, str(output_path))
                logging.info(f"✅ ACE-Step generation complete (single voice): {output_path}")

            # Apply speed up if we slowed down the input
            if tape_speed < 1.0:
                print(f"\n{'='*80}")
                print(f"🎞️ SPEED RESTORATION ({slowdown_method.upper()}): {1.0/tape_speed:.2f}x")
                print(f"{'='*80}")
                print(f"Speeding up generated audio back to original tempo...")

                # Speed up all output files
                speedup_factor = 1.0 / tape_speed
                for wav_file in output_dir.glob("*.wav"):
                    temp_path = str(wav_file.parent / f"temp_{wav_file.name}")
                    if slowdown_method == "stretch":
                        apply_time_stretch_sox(str(wav_file), temp_path, speedup_factor)
                    else:  # tape
                        apply_tape_speed_sox(str(wav_file), temp_path, speedup_factor)
                    shutil.move(temp_path, str(wav_file))
                    print(f"✅ Restored speed: {wav_file.name}")

                print(f"✅ All outputs restored to original speed")
                print(f"{'='*80}\n")

            return {"file_paths": file_paths}

        else:
            # Regular single-voice generation
            if use_best_of_n and audio_file_path:
                # Best-of-N sampling with reranking
                output_audio, all_candidates = generate_best_of_n(
                    model=MODEL,
                    audio_file=audio_file_path,
                    piano_roll=piano_roll,
                    amp=amp,
                    rframe=rframe,
                    rbend=rbend,
                    encodec_tokens=encodec_tokens,
                    group=group,
                    subgroup=subgroup,
                    base_seed=seed,
                    n_candidates=n_candidates,
                    # Generation parameters
                    steps=steps,
                    adapter_scale=adapter_scale,
                    cfg_weight=cfg_weight,
                    t0=1.0,
                    sr_out=44100,
                    instrument_strength=instrument_strength,
                    inst_boost=2.5,
                    piano_roll_gain=piano_roll_gain,
                    amp_gain=amp_gain,
                    rframe_gain=rframe_gain,
                    rbend_gain=rbend_gain,
                    encodec_gain=encodec_gain,
                    use_overlap_decoder=use_overlap_decoder,
                    original_audio_length=orig_len,
                    pitch_fidelity_boost=pitch_fidelity_boost,
                    onset_guidance_boost=onset_guidance_boost,
                    pitch_snap_strength=pitch_snap_strength,
                    noise_level=noise_level,
                    use_time_varying_noise=use_time_varying_noise,
                    onset_preservation=onset_preservation,
                    use_multiresolution_mixing=use_multiresolution_mixing,
                    use_onset_weighted_encodec=use_onset_weighted_encodec,
                    encodec_onset_boost=encodec_onset_boost,
                    use_test_time_adaptation=False,  # Disabled for Best-of-N
                    adaptation_steps=adaptation_steps,
                    adaptation_learning_rate=adaptation_learning_rate,
                    use_self_consistency=use_self_consistency,
                    consistency_samples=consistency_samples,
                    consistency_noise_scale=consistency_noise_scale
                )
            else:
                # Standard single generation
                output_audio = generate(
                    model=MODEL,
                    piano_roll=piano_roll,
                    amp=amp,
                    rframe=rframe,
                    rbend=rbend,
                    encodec_tokens=encodec_tokens,
                    group=group,
                    subgroup=subgroup,
                    steps=steps,
                    seed=seed,
                    adapter_scale=adapter_scale,
                    cfg_weight=cfg_weight,
                    t0=1.0,
                    sr_out=44100,
                    instrument_strength=instrument_strength,
                    inst_boost=2.5,
                    piano_roll_gain=piano_roll_gain,
                    amp_gain=amp_gain,
                    rframe_gain=rframe_gain,
                    rbend_gain=rbend_gain,
                    encodec_gain=encodec_gain,
                    use_overlap_decoder=use_overlap_decoder,
                    original_audio_length=orig_len,
                    pitch_fidelity_boost=pitch_fidelity_boost,
                    onset_guidance_boost=onset_guidance_boost,
                    pitch_snap_strength=pitch_snap_strength,
                    noise_level=noise_level,
                    audio_file=audio_file_path,
                    # NEW: Sample recreation enhancement features
                    use_time_varying_noise=use_time_varying_noise,
                    onset_preservation=onset_preservation,
                    use_multiresolution_mixing=use_multiresolution_mixing,
                    use_onset_weighted_encodec=use_onset_weighted_encodec,
                    encodec_onset_boost=encodec_onset_boost,
                    # NEW: Test-time adaptation
                    use_test_time_adaptation=use_test_time_adaptation,
                    adaptation_steps=adaptation_steps,
                    adaptation_learning_rate=adaptation_learning_rate,
                    # NEW: Self-consistency ensembling
                    use_self_consistency=use_self_consistency,
                    consistency_samples=consistency_samples,
                    consistency_noise_scale=consistency_noise_scale
                )

            # Save output
            output_path = output_dir / "0.wav"
            # output_audio is actually a file path returned by generate(), not audio data
            # Copy the generated file to the final output location
            shutil.copy(output_audio, str(output_path))

            # Apply speed up if we slowed down the input
            if tape_speed < 1.0:
                print(f"\n{'='*80}")
                print(f"🎞️ SPEED RESTORATION ({slowdown_method.upper()}): {1.0/tape_speed:.2f}x")
                print(f"{'='*80}")
                print(f"Speeding up generated audio back to original tempo...")

                speedup_factor = 1.0 / tape_speed
                temp_path = str(output_path.parent / f"temp_{output_path.name}")
                if slowdown_method == "stretch":
                    apply_time_stretch_sox(str(output_path), temp_path, speedup_factor)
                else:  # tape
                    apply_tape_speed_sox(str(output_path), temp_path, speedup_factor)
                shutil.move(temp_path, str(output_path))

                print(f"✅ Output restored to original speed: {output_path.name}")
                print(f"{'='*80}\n")

            logging.info(f"✅ ACE-Step generation complete: {output_path}")
            return {"file_paths": [f"/download/{process_id}/0.wav"]}

    except Exception as e:
        logging.error(f"❌ Error in ACE-Step generation: {e}")
        raise

@app.post("/generate")
async def generate_audio(
    # Support both parameter formats: doseedo2.html sends 'params' JSON, javascript2.js sends individual fields
    params: Optional[str] = Form(None),
    description: str = Form(""),
    duration: Optional[float] = Form(None),  # No default - will be derived from audio or scene data
    steps: int = Form(50),
    seed: int = Form(0),
    adapter_scale: float = Form(1.0),
    cfg_weight: float = Form(3.0),
    instrument_strength: float = Form(1.0),
    noise_level: float = Form(1.0),
    piano_roll_gain: float = Form(1.0),
    amp_gain: float = Form(1.0),
    rframe_gain: float = Form(1.0),
    rbend_gain: float = Form(1.0),
    encodec_gain: float = Form(1.0),
    pitch_fidelity_boost: float = Form(1.0),
    onset_guidance_boost: float = Form(2.0),
    pitch_snap_strength: float = Form(0.5),
    # NEW: Sample recreation enhancement features
    use_time_varying_noise: bool = Form(False),
    onset_preservation: float = Form(0.7),
    use_multiresolution_mixing: bool = Form(False),
    use_onset_weighted_encodec: bool = Form(False),
    encodec_onset_boost: float = Form(2.0),
    # NEW: Test-time adaptation
    use_test_time_adaptation: bool = Form(False),
    adaptation_steps: int = Form(10),
    adaptation_learning_rate: float = Form(1e-4),
    # NEW: Best-of-N sampling
    use_best_of_n: bool = Form(False),
    n_candidates: int = Form(12),
    # NEW: Self-consistency ensembling
    use_self_consistency: bool = Form(False),
    consistency_samples: int = Form(3),
    consistency_noise_scale: float = Form(0.05),
    # Transpose input
    transpose_up_octave: bool = Form(False),
    # Other modes
    monophonic_mode: bool = Form(False),
    enable_voice_separation: bool = Form(False),
    # Support both file field names
    audio_file: Optional[UploadFile] = File(None),
    conditioningAudio: Optional[UploadFile] = File(None),
    audioFile: Optional[UploadFile] = File(None),
    # Scene-aware MIDI generation parameters
    scene_durations: Optional[str] = Form(None),
    automation_data: Optional[str] = Form(None)
):
    """FastAPI endpoint for ACE-Step generation"""

    print(f"📥 Received /generate request")
    print(f"   params JSON: {params[:100] if params else 'None'}")
    print(f"   audio_file: {audio_file}")
    print(f"   conditioningAudio: {conditioningAudio}")
    print(f"   audioFile: {audioFile}")

    # Parse params JSON if provided (from doseedo2.html)
    instrument_group = None
    instrument_subgroup = None
    fatten_mode = False
    fatten_type = "fake"
    tape_speed = 1.0
    slowdown_method = "tape"
    use_overlap_decoder = True
    if params:
        params_dict = json.loads(params)
        steps = params_dict.get('steps', steps)
        seed = params_dict.get('seed', seed)
        cfg_weight = params_dict.get('cfgWeight', cfg_weight)
        instrument_strength = params_dict.get('instrumentStrength', instrument_strength)
        noise_level = params_dict.get('noiseLevel', noise_level)
        piano_roll_gain = params_dict.get('pianoRollGain', piano_roll_gain)
        amp_gain = params_dict.get('ampGain', amp_gain)
        rframe_gain = params_dict.get('rframeGain', rframe_gain)
        rbend_gain = params_dict.get('rbendGain', rbend_gain)
        encodec_gain = params_dict.get('encodecGain', encodec_gain)
        pitch_fidelity_boost = params_dict.get('pitchFidelityBoost', pitch_fidelity_boost)
        onset_guidance_boost = params_dict.get('onsetGuidanceBoost', onset_guidance_boost)
        pitch_snap_strength = params_dict.get('pitchSnapStrength', pitch_snap_strength)
        monophonic_mode = params_dict.get('monophonicMode', monophonic_mode)
        fatten_mode = params_dict.get('fattenMode', False)
        fatten_type = params_dict.get('fattenType', 'fake')
        tape_speed = params_dict.get('tapeSpeed', 1.0)
        slowdown_method = params_dict.get('slowdownMethod', 'tape')
        use_overlap_decoder = params_dict.get('useOverlapDecoder', True)

        # Sample recreation enhancement features
        use_time_varying_noise = params_dict.get('useTimeVaryingNoise', use_time_varying_noise)
        onset_preservation = params_dict.get('onsetPreservation', onset_preservation)
        use_multiresolution_mixing = params_dict.get('useMultiresolutionMixing', use_multiresolution_mixing)
        use_onset_weighted_encodec = params_dict.get('useOnsetWeightedEncodec', use_onset_weighted_encodec)
        encodec_onset_boost = params_dict.get('encodecOnsetBoost', encodec_onset_boost)

        # Test-time adaptation
        use_test_time_adaptation = params_dict.get('useTestTimeAdaptation', use_test_time_adaptation)
        adaptation_steps = params_dict.get('adaptationSteps', adaptation_steps)
        adaptation_learning_rate = params_dict.get('adaptationLearningRate', adaptation_learning_rate)

        # Best-of-N sampling
        use_best_of_n = params_dict.get('useBestOfN', use_best_of_n)
        n_candidates = params_dict.get('nCandidates', n_candidates)

        # Self-consistency ensembling
        use_self_consistency = params_dict.get('useSelfConsistency', use_self_consistency)
        consistency_samples = params_dict.get('consistencySamples', consistency_samples)
        consistency_noise_scale = params_dict.get('consistencyNoiseScale', consistency_noise_scale)

        # Transpose input
        transpose_up_octave = params_dict.get('transposeUpOctave', transpose_up_octave)

        # DEBUG: Log voice separation parameter parsing
        print(f"\n🔍 DEBUG: Parsing enableVoiceSeparation from params JSON:")
        print(f"   Raw params_dict keys: {list(params_dict.keys())}")
        print(f"   'enableVoiceSeparation' in params_dict: {'enableVoiceSeparation' in params_dict}")
        print(f"   Value in params_dict: {params_dict.get('enableVoiceSeparation', 'KEY_NOT_FOUND')}")
        print(f"   Type: {type(params_dict.get('enableVoiceSeparation'))}")
        print(f"   Default (before parsing): {enable_voice_separation}")

        enable_voice_separation = params_dict.get('enableVoiceSeparation', enable_voice_separation)

        print(f"   Value after parsing: {enable_voice_separation}")
        print(f"   Type after parsing: {type(enable_voice_separation)}\n")

        instrument_group = params_dict.get('instrumentGroup')
        instrument_subgroup = params_dict.get('instrumentSubgroup')
        # Also check for scene data in params JSON
        if 'sceneDurations' in params_dict and not scene_durations:
            scene_durations = json.dumps(params_dict['sceneDurations'])
        if 'automationData' in params_dict and not automation_data:
            automation_data = json.dumps(params_dict['automationData'])

    # Parse scene_durations from JSON string to list
    scene_durations_list = None
    print(f"\n📥 FASTAPI ENDPOINT - SCENE DATA PARSING:")
    print(f"   Raw scene_durations param: {scene_durations}")
    print(f"   Type: {type(scene_durations)}")
    if scene_durations:
        try:
            scene_durations_list = json.loads(scene_durations) if isinstance(scene_durations, str) else scene_durations
            print(f"   ✅ Parsed scene_durations_list: {scene_durations_list}")
            print(f"   ✅ Number of scenes: {len(scene_durations_list)}")
            print(f"   ✅ Total duration: {sum(scene_durations_list):.2f}s")
            for i, dur in enumerate(scene_durations_list):
                print(f"      Scene {i}: {dur:.2f}s")
        except Exception as e:
            print(f"   ❌ Failed to parse scene_durations: {e}")
            scene_durations_list = None
    else:
        print(f"   ℹ️  No scene_durations received - will use simple generation")

    # Keep automation_data as string (will be parsed in task)
    if automation_data:
        print(f"🎚️ Automation data received: {len(automation_data)} chars")

    # Use whichever file was provided
    uploaded_file = audio_file or conditioningAudio or audioFile

    print(f"   Using file: {uploaded_file.filename if uploaded_file else 'None'}")
    print(f"   steps: {steps}, seed: {seed}, cfg: {cfg_weight}")
    print(f"   instrument: {instrument_group} / {instrument_subgroup}")

    # Save uploaded file (audio or video) to shared directory accessible by Celery
    audio_file_path = None
    if uploaded_file and uploaded_file.filename:
        file_extension = Path(uploaded_file.filename).suffix.lower()
        # Use shared directory instead of /tmp
        upload_dir = Path("/home/arlo/ScoreAI/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        temp_file_path = str(upload_dir / f"{uuid.uuid4()}{file_extension}")
        with open(temp_file_path, "wb") as f:
            content = await uploaded_file.read()
            f.write(content)

        # If it's a MIDI file, check if it's multi-track with voice separation disabled
        midi_extensions = ['.mid', '.midi']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']

        if file_extension in midi_extensions:
            # Check if it's multi-track MIDI with voice separation disabled
            is_multi, track_count, _ = is_multitrack_midi(temp_file_path)

            if is_multi and monophonic_mode and not enable_voice_separation:
                print(f"🎼 Multi-track MIDI detected ({track_count} tracks)")
                print(f"   Monophonic mode: {monophonic_mode}, Voice separation: {enable_voice_separation}")
                print(f"   Keeping as MIDI (will process tracks separately in task)")
                # Keep the MIDI file, don't render to audio
                audio_file_path = temp_file_path
            else:
                print(f"🎹 Rendering MIDI file to audio...")
                # Render MIDI to audio using the existing render_midi_to_audio function
                audio_file_path = render_midi_to_audio(
                    temp_file_path,
                    output_dir=str(upload_dir),
                    instrument_group=description  # Use description as instrument hint
                )
                # Remove temp MIDI file
                os.remove(temp_file_path)
                print(f"✅ MIDI rendered to audio: {audio_file_path}")

        elif file_extension in video_extensions:
            print(f"🎬 Extracting audio from video...")
            audio_file_path = str(upload_dir / f"{uuid.uuid4()}.wav")
            # Extract audio using ffmpeg
            result = subprocess.run([
                'ffmpeg', '-y', '-i', temp_file_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                audio_file_path
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ FFmpeg error: {result.stderr}")
                raise RuntimeError(f"Failed to extract audio from video: {result.stderr}")

            # Remove temp video file
            os.remove(temp_file_path)
            print(f"✅ Extracted audio from video to: {audio_file_path}")
        else:
            # It's already an audio file (.wav, .mp3, etc.)
            audio_file_path = temp_file_path
            print(f"✅ Using audio file directly: {audio_file_path}")

    # Calculate duration from audio file or scene data
    calculated_duration = None

    print(f"\n📏 DURATION CALCULATION:")
    print(f"   audio_file_path: {audio_file_path}")
    print(f"   scene_durations_list: {scene_durations_list}")
    print(f"   duration parameter: {duration}")

    if audio_file_path:
        # Priority 1: Get duration from audio file
        try:
            import torchaudio
            wav, sr = torchaudio.load(audio_file_path)
            calculated_duration = wav.shape[-1] / sr
            print(f"   ✅ Using audio file duration: {calculated_duration:.2f}s")
        except Exception as e:
            print(f"   ⚠️ Could not determine audio duration: {e}")

    if calculated_duration is None and scene_durations_list and len(scene_durations_list) > 0:
        # Priority 2: Get duration from scene durations
        calculated_duration = sum(scene_durations_list)
        print(f"   ✅ Using scene durations total: {calculated_duration:.2f}s")

    if calculated_duration is None and duration is not None:
        # Priority 3: Use duration parameter from frontend
        calculated_duration = duration
        print(f"   ✅ Using duration parameter: {calculated_duration:.2f}s")

    if calculated_duration is None:
        # Priority 4: Use default of 30 seconds (for simple MIDI generation without video)
        calculated_duration = 30.0
        print(f"   ℹ️  No duration provided - using default: {calculated_duration:.2f}s")

    print(f"   FINAL DURATION: {calculated_duration:.2f}s\n")

    print(f"🚀 Queueing Celery task with audio_file_path: {audio_file_path}")
    print(f"   Final duration: {calculated_duration:.2f}s")
    print(f"   Monophonic mode: {monophonic_mode}, Voice separation: {enable_voice_separation}")

    # Calculate expected voices for frontend placeholder creation
    expected_voices = 0
    if audio_file_path and monophonic_mode:
        # Check if it's multitrack MIDI
        midi_extensions = ['.mid', '.midi']
        file_ext = Path(audio_file_path).suffix.lower()
        if file_ext in midi_extensions:
            is_multi, track_count, _ = is_multitrack_midi(audio_file_path)
            if is_multi and not enable_voice_separation:
                # Multitrack MIDI - use track count
                expected_voices = track_count
                if fatten_mode:
                    # Double for fatten mode (both real and fake create 2x tracks)
                    expected_voices *= 2
                print(f"   📊 Expected voices: {expected_voices} (multitrack MIDI, fatten={fatten_mode})")
            else:
                # Single track or voice separation mode - default 4 voices
                expected_voices = 4
                if fatten_mode:
                    expected_voices *= 2
                print(f"   📊 Expected voices: {expected_voices} (single-track/separated, fatten={fatten_mode})")
        else:
            # Audio file - default 4 voices
            expected_voices = 4
            if fatten_mode:
                expected_voices *= 2
            print(f"   📊 Expected voices: {expected_voices} (audio file, fatten={fatten_mode})")

    # Enqueue the task
    task = generate_ace_step_task.delay(
        audio_file_path,
        description,
        calculated_duration,
        steps,
        seed,
        adapter_scale,
        cfg_weight,
        instrument_strength,
        noise_level,
        piano_roll_gain,
        amp_gain,
        rframe_gain,
        rbend_gain,
        encodec_gain,
        pitch_fidelity_boost,
        onset_guidance_boost,
        pitch_snap_strength,
        instrument_group,
        instrument_subgroup,
        monophonic_mode,
        fatten_mode,
        fatten_type,
        enable_voice_separation,
        scene_durations_list,
        automation_data,
        tape_speed,
        slowdown_method,
        use_overlap_decoder,
        use_time_varying_noise,
        onset_preservation,
        use_multiresolution_mixing,
        use_onset_weighted_encodec,
        encodec_onset_boost,
        use_test_time_adaptation,
        adaptation_steps,
        adaptation_learning_rate,
        use_best_of_n,
        n_candidates,
        use_self_consistency,
        consistency_samples,
        consistency_noise_scale,
        transpose_up_octave
    )

    return {
        "task_id": task.id,
        "expected_voices": expected_voices
    }

@app.post("/separate-stems")
async def separate_stems(
    audio_file: Optional[UploadFile] = File(None),
    audioFile: Optional[UploadFile] = File(None)
):
    """
    Separate audio into stems using Demucs htdemucs_6s model
    Returns: Dictionary with stem file URLs
    """
    import subprocess
    import uuid
    import shutil

    print(f"📥 Received /separate-stems request")

    # Use whichever file was provided
    uploaded_file = audio_file or audioFile

    if not uploaded_file or not uploaded_file.filename:
        raise HTTPException(400, "No audio file provided")

    print(f"   Processing file: {uploaded_file.filename}")

    # Save uploaded file to temporary location
    file_extension = Path(uploaded_file.filename).suffix.lower()
    upload_dir = Path("/home/arlo/ScoreAI/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    temp_input_path = str(upload_dir / f"{uuid.uuid4()}{file_extension}")
    with open(temp_input_path, "wb") as f:
        content = await uploaded_file.read()
        f.write(content)

    print(f"   Saved to: {temp_input_path}")

    # Create output directory for stems
    process_id = str(uuid.uuid4())
    output_base_dir = Path("/home/arlo/ScoreAI/audiofiles")
    output_dir = output_base_dir / f"stems_{process_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Output directory: {output_dir}")

    try:
        # Run demucs with htdemucs_6s model (6 stems: drums, bass, other, vocals, guitar, piano)
        print(f"🎵 Running Demucs htdemucs_6s separation...")

        # Demucs command
        cmd = [
            "demucs",
            "-n", "htdemucs_6s",
            "--two-stems", "vocals",  # This is ignored when using htdemucs_6s, it separates all 6
            "-o", str(output_dir),
            temp_input_path
        ]

        # Remove --two-stems since htdemucs_6s does 6-stem separation automatically
        cmd = [
            "demucs",
            "-n", "htdemucs_6s",
            "-o", str(output_dir),
            temp_input_path
        ]

        print(f"   Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"❌ Demucs failed:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            raise HTTPException(500, f"Stem separation failed: {result.stderr}")

        print(f"✅ Demucs completed successfully")
        print(f"   stdout: {result.stdout}")

        # Find the separated stems
        # Demucs creates: output_dir/htdemucs_6s/{filename_without_ext}/{stem}.wav
        input_filename = Path(temp_input_path).stem
        stems_dir = output_dir / "htdemucs_6s" / input_filename

        print(f"   Looking for stems in: {stems_dir}")

        if not stems_dir.exists():
            raise HTTPException(500, f"Stems directory not found: {stems_dir}")

        # Expected stems for htdemucs_6s: drums, bass, other, vocals, guitar, piano
        expected_stems = ["drums", "bass", "other", "vocals", "guitar", "piano"]
        stem_files = {}

        for stem_name in expected_stems:
            stem_file = stems_dir / f"{stem_name}.wav"
            if stem_file.exists():
                # Copy to root output directory with cleaner name
                output_stem_path = output_dir / f"{stem_name}.wav"
                shutil.copy(stem_file, output_stem_path)

                # Create download URL
                download_url = f"/download-stem/{process_id}/{stem_name}.wav"
                stem_files[stem_name] = download_url
                print(f"   ✅ {stem_name}: {output_stem_path}")
            else:
                print(f"   ⚠️ {stem_name}.wav not found")

        # Clean up temp input file
        os.remove(temp_input_path)

        print(f"✅ Stem separation complete: {len(stem_files)} stems")

        return {
            "process_id": process_id,
            "stems": stem_files,
            "stem_count": len(stem_files)
        }

    except subprocess.TimeoutExpired:
        print(f"❌ Demucs timeout")
        raise HTTPException(500, "Stem separation timed out")
    except Exception as e:
        print(f"❌ Error during stem separation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Stem separation error: {str(e)}")

@app.get("/download-stem/{process_id}/{filename}")
async def download_stem(process_id: str, filename: str):
    """Download separated stem file"""
    file_path = Path("/home/arlo/ScoreAI/audiofiles") / f"stems_{process_id}" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Stem file not found: {filename}")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)

@app.post("/generate-risers")
async def generate_risers(
    scene_durations: str = Form(...)
):
    """
    Select random riser samples for scene transitions
    Returns: List of riser file URLs
    """
    import glob
    import random
    import subprocess
    import uuid

    # Parse scene durations
    try:
        scene_durations_list = json.loads(scene_durations) if isinstance(scene_durations, str) else scene_durations
    except Exception as e:
        raise HTTPException(400, f"Invalid scene_durations: {e}")

    # Check riser directory
    riser_dir = "/home/arlo/Risers/"
    if not os.path.isdir(riser_dir):
        raise HTTPException(404, f"Riser folder not found: {riser_dir}")

    # Find all .wav riser files
    riser_files = [f for f in os.listdir(riser_dir) if f.endswith(".wav")]
    if not riser_files:
        raise HTTPException(404, "No riser files found.")

    # Calculate number of risers needed (skip first scene)
    scene_count = max(0, len(scene_durations_list) - 1)

    print(f"🎚️ Generating {scene_count} random risers from {len(riser_files)} available files")

    # Create output directory matching the download endpoint path
    process_id = str(uuid.uuid4())
    output_directory = Path("/home/arlo/ScoreAI/audiofiles") / f"ace_step_output_{process_id}"
    os.makedirs(output_directory, exist_ok=True)

    # Randomly select and copy risers
    file_paths = []
    for i in range(scene_count):
        chosen_file = random.choice(riser_files)
        src_path = os.path.join(riser_dir, chosen_file)
        dest_path = str(output_directory / f"{i}.wav")
        subprocess.run(["cp", src_path, dest_path], check=True)
        file_paths.append(f"/download/{process_id}/{i}.wav")
        print(f"   Riser {i+1}: {chosen_file} -> {dest_path}")

    print(f"🎈 Returned {scene_count} random risers")

    return {
        "file_paths": file_paths,
        "count": len(file_paths)
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Check task status"""
    task = generate_ace_step_task.AsyncResult(task_id)
    print(f"📊 Task {task_id} state: {task.state}")

    if task.state == "SUCCESS":
        result = task.result
        print(f"📊 Task result type: {type(result)}, value: {result}")

        # Handle different result formats
        if isinstance(result, dict) and "file_paths" in result:
            file_paths = result["file_paths"]
        elif isinstance(result, dict):
            file_paths = result.get("file_paths", [])
        else:
            print(f"⚠️ Unexpected result format: {result}")
            file_paths = []

        print(f"📊 Returning file_paths: {file_paths}")
        return {
            "status": "completed",
            "result": file_paths
        }
    elif task.state == "FAILURE":
        print(f"❌ Task failed: {task.info}")
        return {"status": "failed", "error": str(task.info)}
    elif task.state == "PROGRESS":
        # Return partial results for incremental display
        info = task.info or {}
        completed_voices = info.get("completed_voices", [])
        total_voices = info.get("total_voices", 0)
        progress = info.get("progress", 0.0)

        print(f"📊 Progress: {len(completed_voices)}/{total_voices} voices completed ({progress*100:.1f}%)")
        return {
            "status": "processing",
            "completed_voices": completed_voices,
            "total_voices": total_voices,
            "progress": progress
        }
    else:
        return {"status": task.state}

@app.get("/download/{process_id}/{filename}")
async def download_audio(process_id: str, filename: str):
    """Download generated audio file"""
    file_path = Path("/home/arlo/ScoreAI/audiofiles") / f"ace_step_output_{process_id}" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/api/list-midi-files")
async def list_midi_files():
    """List all MIDI files from the free-midi-chords directory"""
    midi_dir = Path("/home/arlo/free-midi-chords/MIDIS")

    if not midi_dir.exists():
        return {"files": [], "error": "MIDI directory not found"}

    try:
        # Get all .mid and .midi files
        midi_files = []
        for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
            midi_files.extend([f.name for f in midi_dir.glob(ext)])

        # Sort alphabetically
        midi_files.sort()

        print(f"📁 Found {len(midi_files)} MIDI files in {midi_dir}")
        return {"files": midi_files}
    except Exception as e:
        print(f"❌ Error listing MIDI files: {e}")
        return {"files": [], "error": str(e)}

@app.get("/api/get-midi-file/{filename}")
async def get_midi_file(filename: str):
    """Serve a specific MIDI file"""
    midi_dir = Path("/home/arlo/free-midi-chords/MIDIS")
    file_path = midi_dir / filename

    # Security check: ensure the file is within the MIDI directory
    try:
        file_path = file_path.resolve()
        midi_dir = midi_dir.resolve()
        if not str(file_path).startswith(str(midi_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        raise HTTPException(status_code=403, detail="Invalid file path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    print(f"📤 Serving MIDI file: {filename}")
    return FileResponse(file_path, media_type="audio/midi", filename=filename)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    global MODEL, GROUP_NAMES, SUBGROUP_NAMES, MANIFEST_PATHS, MANIFEST_DATA

    DEFAULT_CKPT = "/mnt/msdd/exps/logs_v2/lightning_logs/2025-09-06_16-12-31_all_groups_ft_v3_capivotpitch_ctrl/checkpoints/last.ckpt"
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    with open(args.manifest, "r") as f:
        MANIFEST_DATA = json.load(f)

    print("--- Initializing model ---")
    MODEL = load_model_any_ckpt(args.checkpoint, args.checkpoint_dir, args.manifest)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL.to(dev).eval()
    print(f"✅ Model on {dev}")

    GROUP_NAMES = list(APPROVED_GROUPS) if not isinstance(APPROVED_GROUPS, dict) else list(APPROVED_GROUPS.keys())
    SUBGROUP_NAMES = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
    MANIFEST_PATHS = [it["audio_path"] for it in MANIFEST_DATA if it.get("audio_path")]
    print(f"Groups: {len(GROUP_NAMES)} | Subgroups: {len(SUBGROUP_NAMES)} | Manifest files: {len(MANIFEST_PATHS)}")

    ui = create_ui()
    ui.launch(share=args.share, server_name="0.0.0.0", server_port=7860)

# FastAPI should be run with: uvicorn genfrominterface:app --host 0.0.0.0 --port 8000
# Celery worker should be run with: celery -A genfrominterface worker --loglevel=info
# Gradio UI (optional) can be run by uncommenting below:

# if __name__ == "__main__":
#     main()
 