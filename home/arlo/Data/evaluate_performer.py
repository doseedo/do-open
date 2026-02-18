#!/usr/bin/env python3
"""
evaluate_performer.py — Comprehensive evaluation for the Performer-AI paper.

Computes all metrics needed for a publishable paper:
  1. FAD  (Frechet Audio Distance)        — audio quality / realism
  2. Pitch Accuracy (F0 RMSE, Raw Pitch Acc, Raw Chroma Acc)  — conditioning adherence
  3. Onset F1                              — rhythmic/timing accuracy
  4. Timbre Similarity (Encodec cosine)    — timbral conditioning transfer
  5. Dynamics Correlation                  — loudness contour adherence
  6. Instrument Classification Accuracy    — group/subgroup recognition

Supports ablation modes:
  - full          : all conditioning
  - no_pitchbend  : zero out rbend
  - no_pianoroll  : zero out piano roll + amp
  - no_ctrlbranch : keep ctrl_enc tokens but disable ControlBranch residuals
  - unconditional : zero out everything (baseline)

Usage:
  conda run -n ace_step python3 evaluate_performer.py \\
      --ckpt /mnt/models/epoch=102-step=60000.ckpt \\
      --test_manifest /home/arlo/Data/test_manifest.json \\
      --checkpoint_dir /home/arlo/Data/ACE-Step/checkpoints \\
      --out_dir /home/arlo/Data/eval_output \\
      --n_samples 50 \\
      --steps 30 \\
      --mode full
"""

import sys
sys.path.insert(0, '/home/arlo/Data/dø')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data')

import argparse
import json
import os
import time
import warnings
from collections import defaultdict, Counter
from pathlib import Path
import random

import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ───────────────────────── constants ─────────────────────────
DCAE_SR  = 44100
DCAE_HOP = 4096
ENC_SR   = 24000
ENC_HOP  = 320
SLOW_HZ  = DCAE_SR / DCAE_HOP            # ~10.766 fps
FAST_PER_SLOW = (ENC_SR / ENC_HOP) / SLOW_HZ  # ~6.96

APPROVED_GROUPS    = ["piano", "guitar", "bass", "strings", "brass", "winds"]
APPROVED_SUBGROUPS = {
    "piano":   ["acoustic_piano", "keys", "undefined"],
    "guitar":  ["acoustic_guitar", "electric_guitar", "plucked", "undefined"],
    "bass":    ["electric_bass", "upright_bass", "undefined"],
    "strings": ["violin", "viola", "cello", "undefined"],
    "brass":   ["trumpet", "trombone", "french_horn", "tuba", "undefined"],
    "winds":   ["bassoon", "clarinet", "flute", "oboe", "sax"],
}
ALL_SUBS = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
GROUP2ID = {n: i for i, n in enumerate(APPROVED_GROUPS)}
SUB2ID   = {n: i for i, n in enumerate(ALL_SUBS)}

# Cache for Encodec model used in timbre similarity (loaded once)
_encodec_model = None


# ───────────────────────── MIDI → piano roll ─────────────────────────

def midi_to_piano_roll(midi_path: str, duration_sec: float) -> np.ndarray:
    """Convert BasicPitch MIDI to [128, T] piano roll at SLOW_HZ fps."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    num_frames = int(np.ceil(duration_sec * SLOW_HZ))
    roll = np.zeros((128, num_frames), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            s = int(note.start * SLOW_HZ)
            e = int(note.end * SLOW_HZ)
            if 0 <= note.pitch < 128:
                roll[note.pitch, s:e] = note.velocity / 127.0
    return roll


# ───────────────────────── model loading ─────────────────────────

def load_pipeline(ckpt_path: str, checkpoint_dir: str, device: str = "cuda"):
    """Load trained Pipeline from checkpoint."""
    from trainer_performerCN2 import Pipeline

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp   = ckpt.get("hyper_parameters", {})

    model = Pipeline(
        checkpoint_dir=checkpoint_dir,
        manifest_json=hp.get("manifest_json", "./final_training_manifest_final.json"),
        learning_rate=hp.get("learning_rate", 1e-4),
        T=hp.get("T", 1000),
        shift=hp.get("shift", 3.0),
        cond_cfg_drop_prob=0.0,
        max_steps=hp.get("max_steps", 200000),
        warmup_steps=hp.get("warmup_steps", 10),
        window_slow=hp.get("window_slow", 512),
        batch_size=1,
        encodec_drop_prob=0.0,
        encodec_channel_drop_prob=0.0,
        encodec_time_mask_prob=0.0,
        encodec_time_mask_max_frac=0.0,
        train_from_scratch=hp.get("train_from_scratch", False),
        inst_strength=hp.get("inst_strength", 2.5),
        film_strength=hp.get("film_strength", 0.8),
        channel_mod_strength=hp.get("channel_mod_strength", 0.8),
        pr_loss_weight=0.0,
        use_ctrl_branch=hp.get("use_ctrl_branch", True),
        freeze_base_for_ctrl=False,
        partial_mask_prob=0.0,
        control_scale=hp.get("control_scale", 1.0),
    )

    sd = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)}")

    model.eval()
    model.to(device)
    model.dcae.to(device)
    model.transformers.to(device)
    model.ctrl_enc.to(device)
    model.cond_adapter.to(device)
    model.token_summary.to(device)
    if hasattr(model, "ctrlnet"):
        model.ctrlnet.to(device)
    return model


# ───────────────────────── data loading ─────────────────────────

def load_test_item(entry: dict, window_slow: int = 512):
    """Load a single test item matching the dataset format."""
    stem     = entry["stem"]
    cond_dir = entry["cond_dir"]
    group    = entry["group"]
    subgroup = entry.get("subgroup", "undefined")

    # Load latent (dict with 'latents' key)
    lat_raw = torch.load(entry["latent_path"], map_location="cpu", weights_only=False)
    if isinstance(lat_raw, dict):
        latent = lat_raw["latents"]
    else:
        latent = lat_raw
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)  # [1,8,16,T]

    # Trim/pad to window_slow
    T = latent.shape[-1]
    if T > window_slow:
        latent = latent[..., :window_slow]
    elif T < window_slow:
        latent = F.pad(latent, (0, window_slow - T))
    T_slow = latent.shape[-1]

    # Load conditioning npy files
    def load_npy(name):
        p = os.path.join(cond_dir, f"{stem}.{name}.npy")
        if os.path.exists(p):
            return torch.from_numpy(np.load(p)).float()
        return None

    amp    = load_npy("amp")
    rbend  = load_npy("rbend")
    rframe = load_npy("rframe")

    if amp is None:
        return None

    # Piano roll from BasicPitch MIDI
    midi_path = entry.get("midi_path")
    if midi_path and os.path.exists(midi_path):
        duration_sec = T * DCAE_HOP / DCAE_SR
        pr_np = midi_to_piano_roll(midi_path, duration_sec)
        piano_roll = torch.from_numpy(pr_np).float()
    else:
        # Fallback: try pianoroll.npy
        piano_roll = load_npy("pianoroll")
        if piano_roll is None:
            return None

    # Resize conditioning to match T_slow
    def resize_1d(x, L):
        if x is None:
            return torch.zeros(L)
        if x.dim() == 1:
            return F.interpolate(x.unsqueeze(0).unsqueeze(0), size=L, mode="linear", align_corners=False).squeeze()
        return x

    def resize_2d(x, L):
        if x is None:
            return torch.zeros(128, L)
        if x.dim() == 2:
            return F.interpolate(x.unsqueeze(0), size=L, mode="linear", align_corners=False).squeeze(0)
        return x

    piano_roll = resize_2d(piano_roll, T_slow)
    amp        = resize_1d(amp, T_slow)
    rbend      = resize_1d(rbend, T_slow) if rbend is not None else torch.zeros(T_slow)
    rframe     = resize_1d(rframe, T_slow) if rframe is not None else torch.zeros(T_slow)
    rbend_mask = (rbend.abs() > 0).float()

    # Encodec tokens: zeros (not used by this model)
    T_fast = int(round(T_slow * FAST_PER_SLOW))
    enc_tok = torch.zeros(8, T_fast)

    gid  = GROUP2ID.get(group, 0)
    sgid = SUB2ID.get(subgroup, SUB2ID.get("undefined", 0))

    return {
        "latents": latent.squeeze(0),            # [8, 16, T_slow]
        "encodec_tokens": enc_tok,                # [8, T_fast]
        "conds": {
            "piano_roll": piano_roll,             # [128, T_slow]
            "amp":        amp,                    # [T_slow]
            "rbend":      rbend,                  # [T_slow]
            "rframe":     rframe,                 # [T_slow]
            "rbend_mask": rbend_mask,             # [T_slow]
        },
        "instrument": {
            "group_id":    torch.tensor(gid, dtype=torch.long),
            "subgroup_id": torch.tensor(sgid, dtype=torch.long),
        },
        "meta": entry,
    }


def collate_single(item):
    """Collate a single item into a batch dict."""
    return {
        "latents":        item["latents"].unsqueeze(0),
        "encodec_tokens": item["encodec_tokens"].unsqueeze(0),
        "conds": {k: v.unsqueeze(0) for k, v in item["conds"].items()},
        "instrument": {k: v.unsqueeze(0) for k, v in item["instrument"].items()},
    }


# ───────────────────────── ablation masking ─────────────────────────

def apply_ablation(batch: dict, mode: str) -> dict:
    """Zero out conditioning channels based on ablation mode."""
    if mode == "full":
        return batch
    if mode == "no_pitchbend":
        batch["conds"]["rbend"]      = batch["conds"]["rbend"].zero_()
        batch["conds"]["rbend_mask"] = batch["conds"]["rbend_mask"].zero_()
    elif mode == "no_pianoroll":
        batch["conds"]["piano_roll"] = batch["conds"]["piano_roll"].zero_()
        batch["conds"]["amp"]        = batch["conds"]["amp"].zero_()
    elif mode == "no_ctrlbranch":
        pass  # keep tokens but skip ControlBranch in generate_audio
    elif mode == "unconditional":
        batch["conds"]["piano_roll"] = batch["conds"]["piano_roll"].zero_()
        batch["conds"]["amp"]        = batch["conds"]["amp"].zero_()
        batch["conds"]["rbend"]      = batch["conds"]["rbend"].zero_()
        batch["conds"]["rbend_mask"] = batch["conds"]["rbend_mask"].zero_()
        batch["conds"]["rframe"]     = batch["conds"]["rframe"].zero_()
        batch["encodec_tokens"]      = batch["encodec_tokens"].zero_()
    return batch


# ───────────────────────── generation ─────────────────────────

@torch.no_grad()
def generate_audio(model, batch: dict, steps: int = 30, sr_out: int = 44100,
                   device: str = "cuda", disable_ctrlbranch: bool = False):
    """Generate audio from conditioning using Euler sampling."""
    model.transformers.eval()
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else
             ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v)
             for k, v in batch.items()}

    x0 = batch["latents"]
    B, C_lat, H, T_slow = x0.shape

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        tokens, mask = model.ctrl_enc(
            piano_roll=batch["conds"]["piano_roll"],
            amp=batch["conds"]["amp"],
            rframe=batch["conds"]["rframe"],
            rbend=batch["conds"]["rbend"],
            rbend_mask=batch["conds"]["rbend_mask"],
            encodec_tokens=batch["encodec_tokens"],
            group_id=batch["instrument"]["group_id"],
            subgroup_id=batch["instrument"]["subgroup_id"],
        )

    # Start from pure noise
    torch.manual_seed(42)
    x = torch.randn_like(x0)

    T_train = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
    dt = 1.0 / float(steps)

    tokens_adapt = model._match_mod_dtype(tokens, model.cond_adapter)

    # ControlBranch residuals (constant over loop)
    use_ctrl = model.use_ctrl_branch and hasattr(model, "ctrlnet") and not disable_ctrlbranch
    if use_ctrl:
        pr_128 = batch["conds"]["piano_roll"].to(x.device, dtype=x.dtype)
        amp_1t = batch["conds"]["amp"].to(x.device, dtype=x.dtype).unsqueeze(1)
        if amp_1t.shape[-1] != pr_128.shape[-1]:
            amp_1t = F.interpolate(amp_1t, size=pr_128.shape[-1], mode="nearest")
        ctrl_in = torch.cat([pr_128, amp_1t], dim=1)
        instrument_token = tokens_adapt[:, 0, :] if hasattr(model.ctrlnet, 'instrument_dim') and model.ctrlnet.instrument_dim else None
        res_list = model.ctrlnet(ctrl_in, T_out_list=[x.shape[-1]] * len(model.ctrlnet.to_blocks), instrument_token=instrument_token)
        scale = float(getattr(model.hparams, "control_scale", 1.0))
        model._ctrl_residuals = [r * scale for r in res_list]
    else:
        model._ctrl_residuals = None

    for i in range(steps, 0, -1):
        t_cont = torch.full((B,), i * dt, device=device)
        t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        cond_patch = model.cond_adapter(tokens_adapt, T_out=x.shape[-1], scale=1.0)
        cond_patch = cond_patch.to(device=x.device, dtype=x.dtype)

        # Pitch-height masking
        pr = batch["conds"]["piano_roll"].to(device=x.device, dtype=x.dtype)
        if pr.shape[-1] != T_slow:
            pr = F.interpolate(pr, size=T_slow, mode="nearest")
        W_hp = model._bank_softplus_resized(H, device=x.device, dtype=x.dtype)
        Hmap = torch.einsum('bpt,hp->bht', pr, W_hp)
        cond_patch = cond_patch * Hmap.unsqueeze(1)

        x_in = x + cond_patch
        v_pred = model._call_transformer_no_xattn(latents=x_in, t=t_idx)
        x = x - dt * v_pred

    model._ctrl_residuals = None

    # Decode
    audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
    x_for_dcae = model._match_mod_dtype(x[:1], model.dcae)
    audio_lengths = torch.tensor([audio_len], device=x_for_dcae.device, dtype=torch.long)
    sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

    return wav_pred[0].float().cpu(), sr_pred


def decode_gt_audio(model, batch: dict, sr_out: int = 44100, device: str = "cuda"):
    """Decode ground truth latents to audio for comparison."""
    x0 = batch["latents"].to(device)
    T_slow = x0.shape[-1]
    audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
    x_for_dcae = model._match_mod_dtype(x0[:1], model.dcae)
    audio_lengths = torch.tensor([audio_len], device=x_for_dcae.device, dtype=torch.long)
    sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)
    return wav_pred[0].float().cpu(), sr_pred


# ───────────────────────── METRIC FUNCTIONS ─────────────────────────

# ---------- 1. FAD (Frechet Audio Distance) ----------

def compute_fad(gen_dir: str, ref_dir: str):
    """Compute FAD between generated and reference directories using VGGish."""
    try:
        from frechet_audio_distance import FrechetAudioDistance
        fad = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )
        score = fad.score(ref_dir, gen_dir)
        return {"FAD_vggish": float(score)}
    except Exception as e:
        print(f"[FAD] Error: {e}")
        return {"FAD_vggish": None}


# ---------- 2. Pitch Accuracy (Mono + Poly) ----------

MONO_SUBGROUPS = {'violin', 'viola', 'cello', 'trumpet', 'trombone', 'french_horn',
                  'tuba', 'flute', 'clarinet', 'oboe', 'bassoon', 'sax',
                  'upright_bass', 'electric_bass'}


MONO_GROUPS = {'bass', 'strings', 'brass', 'winds'}

def _is_mono(subgroup: str, group: str, piano_roll: np.ndarray) -> bool:
    """Decide mono vs poly from subgroup, group, or piano roll polyphony."""
    if subgroup in MONO_SUBGROUPS:
        return True
    if group in MONO_GROUPS:
        return True
    # Heuristic: if avg simultaneous notes > 1.3, it's polyphonic
    active_per_frame = (piano_roll > 0.3).sum(axis=0)
    active_frames = active_per_frame[active_per_frame > 0]
    if len(active_frames) == 0:
        return True
    return float(active_frames.mean()) < 1.3


def compute_pitch_metrics_mono(gen_wav: np.ndarray, piano_roll: np.ndarray, sr: int = 44100):
    """
    Monophonic pitch evaluation using crepe.
    Uses relaxed tolerances since the model is supposed to sound natural, not robotic:
    - semitone_acc: within 100 cents (1 semitone) — correct note
    - chroma_acc: within 100 cents mod octave — correct pitch class
    - contour_corr: pitch contour correlation — does F0 move in the right direction?
    - voiced_recall: fraction of reference-voiced frames where gen is also voiced
    """
    import librosa
    from scipy.stats import pearsonr

    if gen_wav.ndim > 1:
        gen_wav = gen_wav.mean(axis=0)
    gen_wav_16k = librosa.resample(gen_wav, orig_sr=sr, target_sr=16000)

    try:
        import torchcrepe
        audio_tensor = torch.from_numpy(gen_wav_16k).float().unsqueeze(0)
        f0_gen, confidence = torchcrepe.predict(
            audio_tensor, 16000, hop_length=160,
            fmin=50, fmax=2000, model='tiny',
            return_periodicity=True, batch_size=1024, device='cpu'
        )
        f0_gen = f0_gen.squeeze().numpy()
        confidence = confidence.squeeze().numpy()
        voiced_mask = confidence > 0.5
    except Exception:
        f0_gen, _, voiced_prob = librosa.pyin(
            gen_wav_16k, fmin=50, fmax=2000, sr=16000,
            hop_length=160, fill_na=0.0
        )
        if f0_gen is None:
            return _empty_pitch_result()
        voiced_mask = f0_gen > 0

    if piano_roll.ndim < 2:
        return _empty_pitch_result()

    pr = piano_roll
    T_f0 = len(f0_gen)
    if pr.shape[-1] != T_f0:
        pr = F.interpolate(torch.from_numpy(pr).float().unsqueeze(0),
                           size=T_f0, mode="nearest").squeeze(0).numpy()

    # Get dominant pitch per frame from piano roll
    ref_midi = np.zeros(T_f0)
    ref_voiced = np.zeros(T_f0, dtype=bool)
    for t in range(T_f0):
        active = np.where(pr[:, t] > 0.3)[0]
        if len(active) > 0:
            weights = pr[active, t]
            ref_midi[t] = np.average(active, weights=weights)
            ref_voiced[t] = True

    ref_f0 = 440.0 * (2.0 ** ((ref_midi - 69) / 12.0))
    ref_f0[~ref_voiced] = 0

    # Voiced recall: when ref is voiced, is gen also producing sound?
    voiced_recall = float(voiced_mask[ref_voiced].mean()) if ref_voiced.sum() > 0 else 0.0

    both_voiced = voiced_mask & ref_voiced
    n_both = both_voiced.sum()
    if n_both < 10:
        return {**_empty_pitch_result(), "voiced_recall": voiced_recall, "eval_type": "mono"}

    gen_cents = 1200 * np.log2(np.clip(f0_gen[both_voiced], 1, None) / 440.0)
    ref_cents = 1200 * np.log2(np.clip(ref_f0[both_voiced], 1, None) / 440.0)
    diff = np.abs(gen_cents - ref_cents)

    # Semitone accuracy (within 100 cents = 1 semitone — correct note, allows vibrato/slides)
    semitone_acc = float(np.mean(diff < 100))

    # Chroma accuracy (within 100 cents mod octave — right pitch class)
    chroma_diff = np.abs((gen_cents - ref_cents) % 1200)
    chroma_diff = np.minimum(chroma_diff, 1200 - chroma_diff)
    chroma_acc = float(np.mean(chroma_diff < 100))

    # Pitch contour correlation — does the F0 move in the right direction?
    # Smooth both to ~4 Hz to ignore vibrato, then correlate
    from scipy.ndimage import uniform_filter1d
    smooth_k = max(1, int(100 * 0.01 / 0.01))  # ~100 frames = ~1s at 100fps
    gen_smooth = uniform_filter1d(gen_cents, smooth_k)
    ref_smooth = uniform_filter1d(ref_cents, smooth_k)
    if len(gen_smooth) > 2:
        contour_corr, _ = pearsonr(gen_smooth, ref_smooth)
        contour_corr = float(contour_corr)
    else:
        contour_corr = 0.0

    # Median absolute deviation in cents (robust to outliers)
    median_dev = float(np.median(diff))

    return {
        "semitone_acc": semitone_acc,
        "chroma_acc": chroma_acc,
        "contour_corr": contour_corr,
        "median_cents_dev": median_dev,
        "voiced_recall": voiced_recall,
        "eval_type": "mono",
    }


def compute_pitch_metrics_poly(gen_wav: np.ndarray, piano_roll: np.ndarray, sr: int = 44100):
    """
    Polyphonic pitch evaluation using basic_pitch.
    Extracts a piano roll from the generated audio, then compares
    against the reference piano roll using mir_eval.multipitch.
    """
    import librosa
    import mir_eval.multipitch
    import tempfile, soundfile as sf

    if gen_wav.ndim > 1:
        gen_wav = gen_wav.mean(axis=0)

    # Write to temp file for basic_pitch
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        sf.write(tmp.name, gen_wav, sr)
        tmp_path = tmp.name

    try:
        from basic_pitch.inference import predict as bp_predict
        _, gen_midi, _ = bp_predict(tmp_path, onset_threshold=0.5, frame_threshold=0.3)
    except Exception as e:
        print(f"  [poly-pitch] basic_pitch failed: {e}")
        os.unlink(tmp_path)
        return _empty_pitch_result()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Convert gen_midi to frame-level frequency sets
    ref_pr = piano_roll  # [128, T]
    T = ref_pr.shape[-1]
    duration = T / SLOW_HZ

    # Build reference frequency sets per frame
    n_eval_frames = int(duration * 100)  # evaluate at 100 Hz
    ref_freqs = []
    for i in range(n_eval_frames):
        t_sec = i / 100.0
        t_pr = int(t_sec * SLOW_HZ)
        if t_pr >= T:
            t_pr = T - 1
        active = np.where(ref_pr[:, t_pr] > 0.3)[0]
        freqs = 440.0 * (2.0 ** ((active - 69) / 12.0))
        ref_freqs.append(freqs)

    # Build generated frequency sets from MIDI
    gen_freqs = [np.array([]) for _ in range(n_eval_frames)]
    for inst in gen_midi.instruments:
        for note in inst.notes:
            s_frame = int(note.start * 100)
            e_frame = int(note.end * 100)
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            for f in range(max(0, s_frame), min(n_eval_frames, e_frame)):
                gen_freqs[f] = np.append(gen_freqs[f], freq)

    # mir_eval multipitch metrics
    ref_time = np.arange(n_eval_frames) / 100.0
    gen_time = ref_time.copy()

    metrics = mir_eval.multipitch.metrics(ref_time, ref_freqs, gen_time, gen_freqs)
    # Returns 14 values: (prec, rec, acc, e_sub, e_miss, e_fa, e_tot) x2 (with/without chroma)
    precision, recall, accuracy = metrics[0], metrics[1], metrics[2]

    # Also compute a note-level F1 via simple overlap
    # Count ref and gen notes, and matches (within 1 semitone + 50ms)
    ref_notes = []
    for pitch in range(128):
        row = (ref_pr[pitch] > 0.3).astype(float)
        diff = np.diff(row, prepend=0)
        onsets = np.where(diff > 0)[0]
        offsets = np.where(diff < 0)[0]
        for on in onsets:
            off = offsets[offsets > on]
            off = off[0] if len(off) > 0 else T
            ref_notes.append((pitch, on / SLOW_HZ, off / SLOW_HZ))

    gen_notes = []
    for inst in gen_midi.instruments:
        for note in inst.notes:
            gen_notes.append((note.pitch, note.start, note.end))

    # Match: same pitch (within 1 semitone) and onset within 100ms
    matched = 0
    used = set()
    for rp, rs, re in ref_notes:
        for gi, (gp, gs, ge) in enumerate(gen_notes):
            if gi in used:
                continue
            if abs(rp - gp) <= 1 and abs(rs - gs) < 0.1:
                matched += 1
                used.add(gi)
                break

    note_precision = matched / max(len(gen_notes), 1)
    note_recall = matched / max(len(ref_notes), 1)
    note_f1 = 2 * note_precision * note_recall / max(note_precision + note_recall, 1e-8)

    return {
        "multipitch_precision": float(precision),
        "multipitch_recall": float(recall),
        "multipitch_accuracy": float(accuracy),
        "note_f1": float(note_f1),
        "note_precision": float(note_precision),
        "note_recall": float(note_recall),
        "eval_type": "poly",
    }


def _empty_pitch_result():
    return {
        "semitone_acc": None, "chroma_acc": None, "contour_corr": None,
        "median_cents_dev": None, "voiced_recall": None,
        "multipitch_precision": None, "multipitch_recall": None,
        "multipitch_accuracy": None, "note_f1": None,
        "note_precision": None, "note_recall": None,
        "eval_type": None,
    }


def compute_pitch_metrics(gen_wav: np.ndarray, piano_roll: np.ndarray,
                          sr: int = 44100, subgroup: str = "undefined",
                          group: str = "undefined"):
    """Route to mono or poly pitch evaluation based on instrument type."""
    if piano_roll.ndim < 2:
        return _empty_pitch_result()
    mono = _is_mono(subgroup, group, piano_roll)
    if mono:
        return compute_pitch_metrics_mono(gen_wav, piano_roll, sr)
    else:
        return compute_pitch_metrics_poly(gen_wav, piano_roll, sr)


# ---------- 3. Onset F1 ----------

def compute_onset_f1(gen_wav: np.ndarray, piano_roll: np.ndarray, sr: int = 44100):
    """Compare detected onsets in generated audio vs piano roll onsets."""
    import librosa
    import mir_eval

    if gen_wav.ndim > 1:
        gen_wav = gen_wav.mean(axis=0)

    onset_env = librosa.onset.onset_strength(y=gen_wav, sr=sr, hop_length=512)
    gen_onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512, units='time'
    )

    # Reference onsets from piano roll: any note onset
    if piano_roll.ndim < 2:
        return {"onset_f1": 0.0, "onset_precision": 0.0, "onset_recall": 0.0}

    # Per-pitch onset detection (more accurate than just "any active")
    onset_times = []
    for pitch in range(128):
        row = (piano_roll[pitch] > 0.3).astype(float)
        diff = np.diff(row, prepend=0)
        onset_frames = np.where(diff > 0)[0]
        for f in onset_frames:
            onset_times.append(f / SLOW_HZ)

    ref_onsets = np.unique(np.array(sorted(onset_times)))
    if len(ref_onsets) == 0 or len(gen_onsets) == 0:
        return {"onset_f1": 0.0, "onset_precision": 0.0, "onset_recall": 0.0}

    f1, precision, recall = mir_eval.onset.f_measure(
        ref_onsets, np.array(gen_onsets), window=0.05
    )
    return {"onset_f1": float(f1), "onset_precision": float(precision), "onset_recall": float(recall)}


# ---------- 4. Timbre Similarity ----------

def get_encodec_model():
    global _encodec_model
    if _encodec_model is None:
        from encodec import EncodecModel
        _encodec_model = EncodecModel.encodec_model_24khz()
        _encodec_model.set_target_bandwidth(6.0)
        _encodec_model.eval()
    return _encodec_model

def compute_timbre_similarity(gen_wav: np.ndarray, ref_wav: np.ndarray, sr: int = 44100):
    """Compute Encodec embedding cosine similarity between generated and reference."""
    try:
        from encodec.utils import convert_audio
        model = get_encodec_model()

        def get_embedding(wav, sr_in):
            if wav.ndim == 1:
                wav = wav[np.newaxis, np.newaxis, :]
            elif wav.ndim == 2:
                wav = wav[np.newaxis, :]
            wav_t = torch.from_numpy(wav).float()
            wav_t = convert_audio(wav_t, sr_in, 24000, 1)
            with torch.no_grad():
                encoded = model.encode(wav_t)
            codes = encoded[0][0]  # [B, n_codebooks, T]
            return codes.float().mean(dim=-1).squeeze()

        emb_gen = get_embedding(gen_wav, sr)
        emb_ref = get_embedding(ref_wav, sr)
        cos_sim = float(F.cosine_similarity(emb_gen.unsqueeze(0), emb_ref.unsqueeze(0)).item())
        return {"timbre_cosine": cos_sim}
    except Exception as e:
        print(f"[timbre] Error: {e}")
        return {"timbre_cosine": None}


# ---------- 5. Dynamics Correlation ----------

def compute_dynamics_correlation(gen_wav: np.ndarray, ref_amp: np.ndarray, sr: int = 44100):
    """Correlate RMS envelope of generated audio with conditioning amplitude."""
    import librosa
    from scipy.stats import pearsonr

    if gen_wav.ndim > 1:
        gen_wav = gen_wav.mean(axis=0)

    hop = int(sr / SLOW_HZ)
    rms = librosa.feature.rms(y=gen_wav, hop_length=hop, frame_length=hop * 2)[0]

    L = min(len(rms), len(ref_amp))
    if L < 5:
        return {"dynamics_pearson": 0.0, "dynamics_rmse": 1.0}

    rms = rms[:L]
    ref = ref_amp[:L]

    rms_n = rms / (rms.max() + 1e-8)
    ref_n = ref / (ref.max() + 1e-8)

    corr, _ = pearsonr(rms_n, ref_n)
    rmse = float(np.sqrt(np.mean((rms_n - ref_n) ** 2)))
    return {"dynamics_pearson": float(corr), "dynamics_rmse": rmse}


# ---------- 6. Instrument Classification Accuracy ----------

def compute_classification_acc(model, batch: dict, group_gt: int, subgroup_gt: int, device: str = "cuda"):
    """Use the model's own classification heads to check instrument recognition."""
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else
                 ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v)
                 for k, v in batch.items()}

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        tokens, mask = model.ctrl_enc(
            piano_roll=batch_dev["conds"]["piano_roll"],
            amp=batch_dev["conds"]["amp"],
            rframe=batch_dev["conds"]["rframe"],
            rbend=batch_dev["conds"]["rbend"],
            rbend_mask=batch_dev["conds"]["rbend_mask"],
            encodec_tokens=batch_dev["encodec_tokens"],
            group_id=batch_dev["instrument"]["group_id"],
            subgroup_id=batch_dev["instrument"]["subgroup_id"],
        )

    inst_tok = tokens[:, 0, :]
    gp = model.group_head(inst_tok).argmax(-1).item()
    sp = model.sub_head(inst_tok).argmax(-1).item()
    return {"group_correct": int(gp == group_gt), "subgroup_correct": int(sp == subgroup_gt)}


# ---------- 7. CLAP Score ----------

_clap_model = None
_clap_processor = None

def compute_clap_score(gen_wav: np.ndarray, instrument_label: str, sr: int = 44100):
    """Compute CLAP similarity between generated audio and instrument text description."""
    global _clap_model, _clap_processor
    try:
        import librosa
        if _clap_model is None:
            from transformers import ClapModel, ClapProcessor
            _clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            _clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            _clap_model.eval()

        if gen_wav.ndim > 1:
            gen_wav = gen_wav.mean(axis=0)
        wav_48k = librosa.resample(gen_wav, orig_sr=sr, target_sr=48000)

        text = f"a recording of a {instrument_label} playing music"
        inputs = _clap_processor(
            text=[text], audios=[wav_48k], return_tensors="pt",
            sampling_rate=48000, padding=True
        )
        with torch.no_grad():
            outputs = _clap_model(**inputs)
        sim = float(outputs.logits_per_audio.item())
        return {"clap_score": sim}
    except Exception as e:
        print(f"[CLAP] Error: {e}")
        return {"clap_score": None}


# ───────────────────────── MAIN EVALUATION LOOP ─────────────────────────

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    gen_dir = out_dir / "generated"
    ref_dir = out_dir / "reference"
    gen_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.ckpt}...")
    model = load_pipeline(args.ckpt, args.checkpoint_dir, device)
    print("Model loaded.")

    with open(args.test_manifest) as f:
        test_entries = json.load(f)

    random.seed(42)
    random.shuffle(test_entries)

    group_counts = Counter(e["group"] for e in test_entries)
    n_total = min(args.n_samples, len(test_entries))
    selected = []

    if args.group:
        test_entries = [e for e in test_entries if e["group"] == args.group]
        selected = test_entries[:n_total]
    else:
        # Stratified: at least 3 per group, rest proportional
        min_per = 3
        remaining = n_total - min_per * len(group_counts)
        by_group = defaultdict(list)
        for e in test_entries:
            by_group[e["group"]].append(e)
        for g, items in by_group.items():
            quota = min_per + max(0, int(remaining * len(items) / len(test_entries)))
            selected.extend(items[:quota])
        selected = selected[:n_total]

    print(f"Evaluating {len(selected)} samples, mode={args.mode}")
    group_dist = Counter(e["group"] for e in selected)
    print(f"Distribution: {dict(group_dist)}")

    all_metrics = []
    times = []
    skipped = 0

    for idx, entry in enumerate(selected):
        t0 = time.time()
        stem_display = os.path.basename(entry['audio_path'])[:60]
        print(f"[{idx+1}/{len(selected)}] {entry['group']}/{entry.get('subgroup','?')}: {stem_display}")

        item = load_test_item(entry, window_slow=args.window_slow)
        if item is None:
            print("  -> skipped (missing conditioning or midi)")
            skipped += 1
            continue

        batch = collate_single(item)
        batch = apply_ablation(batch, args.mode)

        disable_ctrl = (args.mode == "no_ctrlbranch")

        # Generate
        try:
            gen_wav, sr_gen = generate_audio(model, batch, steps=args.steps, sr_out=44100,
                                             device=device, disable_ctrlbranch=disable_ctrl)
        except Exception as e:
            print(f"  -> generation failed: {e}")
            import traceback; traceback.print_exc()
            skipped += 1
            continue

        # Decode GT
        try:
            gt_wav, sr_gt = decode_gt_audio(model, collate_single(item), sr_out=44100, device=device)
        except Exception as e:
            print(f"  -> GT decode failed: {e}")
            skipped += 1
            continue

        gen_np = gen_wav.numpy()
        gt_np  = gt_wav.numpy()
        gen_mono = gen_np.mean(axis=0) if gen_np.ndim > 1 else gen_np
        gt_mono  = gt_np.mean(axis=0) if gt_np.ndim > 1 else gt_np

        # Save wavs for FAD computation
        tag = f"{idx:04d}_{entry['group']}_{entry['stem'][:40]}"
        torchaudio.save(str(gen_dir / f"{tag}.wav"),
                        gen_wav.unsqueeze(0) if gen_wav.dim() == 1 else gen_wav, sr_gen)
        torchaudio.save(str(ref_dir / f"{tag}.wav"),
                        gt_wav.unsqueeze(0) if gt_wav.dim() == 1 else gt_wav, sr_gt)

        # Compute per-sample metrics
        pr_np  = item["conds"]["piano_roll"].numpy()
        amp_np = item["conds"]["amp"].numpy()

        m = {}
        m.update(compute_pitch_metrics(gen_mono, pr_np, sr=44100,
                                        subgroup=entry.get("subgroup", "undefined"),
                                        group=entry.get("group", "undefined")))
        m.update(compute_onset_f1(gen_mono, pr_np, sr=44100))
        m.update(compute_dynamics_correlation(gen_mono, amp_np, sr=44100))
        m.update(compute_timbre_similarity(gen_mono, gt_mono, sr=44100))
        m.update(compute_classification_acc(
            model, batch,
            group_gt=item["instrument"]["group_id"].item(),
            subgroup_gt=item["instrument"]["subgroup_id"].item(),
            device=device,
        ))

        # CLAP is slow, only compute every 5th sample
        if idx % 5 == 0:
            m.update(compute_clap_score(gen_mono, entry["group"], sr=44100))

        m["group"]    = entry["group"]
        m["subgroup"] = entry.get("subgroup", "?")
        m["audio"]    = os.path.basename(entry["audio_path"])

        elapsed = time.time() - t0
        times.append(elapsed)
        m["gen_time_s"] = elapsed

        all_metrics.append(m)
        etype = m.get('eval_type', '?')
        if etype == 'mono':
            pa = m.get('semitone_acc', 0) or 0
            cc = m.get('contour_corr', 0) or 0
            print(f"  [{etype}] semitone={pa:.3f} contour={cc:.3f}  "
                  f"onset_f1={m.get('onset_f1',0):.3f}  dyn={m.get('dynamics_pearson',0):.3f}  "
                  f"timbre={m.get('timbre_cosine',0):.3f}  [{elapsed:.1f}s]")
        elif etype == 'poly':
            nf = m.get('note_f1', 0) or 0
            mp = m.get('multipitch_accuracy', 0) or 0
            print(f"  [{etype}] note_f1={nf:.3f} mp_acc={mp:.3f}  "
                  f"onset_f1={m.get('onset_f1',0):.3f}  dyn={m.get('dynamics_pearson',0):.3f}  "
                  f"timbre={m.get('timbre_cosine',0):.3f}  [{elapsed:.1f}s]")
        else:
            print(f"  [?] onset_f1={m.get('onset_f1',0):.3f}  "
                  f"dyn={m.get('dynamics_pearson',0):.3f}  timbre={m.get('timbre_cosine',0):.3f}  "
                  f"[{elapsed:.1f}s]")

    # ─── Compute FAD over all generated vs reference ───
    print(f"\nComputing FAD over {len(all_metrics)} samples...")
    fad_result = compute_fad(str(gen_dir), str(ref_dir))
    print(f"FAD (VGGish): {fad_result.get('FAD_vggish', 'N/A')}")

    # ─── Aggregate results ───
    print("\n" + "=" * 70)
    print(f"RESULTS — mode={args.mode}, n={len(all_metrics)}, skipped={skipped}")
    print("=" * 70)

    metric_keys = [
        # Mono pitch metrics
        "semitone_acc", "chroma_acc", "contour_corr", "median_cents_dev", "voiced_recall",
        # Poly pitch metrics
        "multipitch_precision", "multipitch_recall", "multipitch_accuracy",
        "note_f1", "note_precision", "note_recall",
        # Onset / dynamics / timbre / classification
        "onset_f1", "onset_precision", "onset_recall",
        "dynamics_pearson", "dynamics_rmse",
        "timbre_cosine", "group_correct", "subgroup_correct", "clap_score",
    ]
    summary = {}
    for k in metric_keys:
        vals = [m[k] for m in all_metrics if m.get(k) is not None]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
            print(f"  {k:25s}: {summary[k]['mean']:.4f} +/- {summary[k]['std']:.4f}  (n={len(vals)})")

    summary["FAD_vggish"] = fad_result.get("FAD_vggish")
    print(f"  {'FAD_vggish':25s}: {summary['FAD_vggish']}")

    # Per-group breakdown
    per_group = defaultdict(lambda: defaultdict(list))
    for m in all_metrics:
        g = m["group"]
        for k in metric_keys:
            if m.get(k) is not None:
                per_group[g][k].append(m[k])

    print("\n--- Per-Group Breakdown ---")
    group_summary = {}
    for g in sorted(per_group.keys()):
        n_g = len(per_group[g].get('pitch_acc', []))
        print(f"\n  [{g}] (n={n_g})")
        group_summary[g] = {}
        for k in metric_keys:
            vals = per_group[g].get(k, [])
            if vals:
                group_summary[g][k] = float(np.mean(vals))
                print(f"    {k:25s}: {np.mean(vals):.4f}")

    # Save full results
    results = {
        "mode": args.mode,
        "n_samples": len(all_metrics),
        "skipped": skipped,
        "steps": args.steps,
        "window_slow": args.window_slow,
        "summary": summary,
        "per_group": group_summary,
        "fad": fad_result,
        "per_sample": all_metrics,
        "avg_gen_time_s": float(np.mean(times)) if times else 0,
    }

    results_path = out_dir / f"results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # LaTeX table row
    print("\n--- LaTeX Table Row ---")
    s = summary
    def sv(k): return s.get(k, {}).get('mean', 0) if isinstance(s.get(k), dict) else (s.get(k) or 0)
    row = (f"{args.mode:15s} & "
           f"{sv('semitone_acc'):.3f} & "
           f"{sv('contour_corr'):.3f} & "
           f"{sv('note_f1'):.3f} & "
           f"{sv('multipitch_accuracy'):.3f} & "
           f"{sv('onset_f1'):.3f} & "
           f"{sv('dynamics_pearson'):.3f} & "
           f"{sv('timbre_cosine'):.3f} & "
           f"{fad_result.get('FAD_vggish', 'N/A')} \\\\")
    print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="/mnt/models/epoch=102-step=60000.ckpt")
    ap.add_argument("--test_manifest", type=str, default="/home/arlo/Data/test_manifest.json")
    ap.add_argument("--checkpoint_dir", type=str, default="/home/arlo/Data/ACE-Step/checkpoints")
    ap.add_argument("--out_dir", type=str, default="/home/arlo/Data/eval_output")
    ap.add_argument("--n_samples", type=int, default=50, help="Total samples to evaluate")
    ap.add_argument("--steps", type=int, default=30, help="Euler denoising steps")
    ap.add_argument("--mode", type=str, default="full",
                    choices=["full", "no_pitchbend", "no_pianoroll", "no_ctrlbranch", "unconditional"],
                    help="Ablation mode")
    ap.add_argument("--window_slow", type=int, default=512)
    ap.add_argument("--group", type=str, default=None, help="Evaluate single instrument group only")
    args = ap.parse_args()
    evaluate(args)
