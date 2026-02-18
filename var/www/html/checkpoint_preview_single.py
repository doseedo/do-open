\#!/usr/bin/env python3
"""
Inference from one audio file using one specific checkpoint.
- Extracts conditioning from the audio.
- Starts from pure noise (no GT latents needed).
- Samples latents and decodes with DCAE to preview.wav.

Usage:
  python infer_from_audio_single_ckpt.py \
    --input /path/to/audio_or_folder \
    --checkpoint /path/to/lightning.ckpt \
    --components_dir /home/arlo/Data/ACE-Step/checkpoints \
    --output_dir ./out_single \
    --group guitar --subgroup electric_guitar \
    --steps 30 --sr_out 48000
"""

import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
# Keep feature extraction on CPU to avoid GPU OOM:
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse, random, json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
import pretty_midi

# your modules
from conditioning_encoder import PerformanceConditionEncoder  # imported by Pipeline
from trainer_performer import Pipeline                        # provides ctrl_enc, cond_adapter, dcae, _call_transformer_no_xattn

# ---- constants matching your training grid ----
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000,  320
SLOW_HZ           = DCAE_SR / DCAE_HOP  # ~10.7666 Hz

# audio/feature params
SAMPLE_RATE = 44100
N_FFT       = 8192
HOP_LENGTH  = 4096
FMIN        = librosa.note_to_hz("C2")
FMAX        = librosa.note_to_hz("C7")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aiff", ".aif", ".ogg"}

APPROVED_GROUPS = ["piano","guitar","bass","strings","brass","winds"]
APPROVED_SUBGROUPS = {
    "piano":   ["acoustic_piano","keys","undefined"],
    "guitar":  ["acoustic_guitar","electric_guitar","plucked","undefined"],
    "bass":    ["electric_bass","upright_bass","undefined"],
    "strings": ["violin","viola","cello","undefined"],
    "brass":   ["trumpet","trombone","french_horn","tuba","undefined"],
    "winds":   ["bassoon","clarinet","flute","oboe","sax"],
}

def pick_random_file(folder: Path) -> Path:
    cands: List[Path] = []
    for ext in AUDIO_EXTS:
        cands.extend(folder.rglob(f"*{ext}"))
    if not cands:
        raise FileNotFoundError(f"No audio files with {AUDIO_EXTS} in {folder}")
    return random.choice(cands)

def safe_stem(name: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-","_")) else "_" for c in name)[:128] or "audio"

# ---------- conditioning extraction ----------

def extract_basicpitch(audio_path: Path, out_mid: Path, out_pr: Path) -> Dict[str, Any]:
    from basic_pitch.inference import predict as basicpitch_predict
    import basic_pitch
    onnx_model = Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"
    _, midi_pm, _ = basicpitch_predict(str(audio_path), model_or_model_path=str(onnx_model))
    midi_pm.write(str(out_mid))
    pm = pretty_midi.PrettyMIDI(str(out_mid))
    pr = pm.get_piano_roll(fs=SAMPLE_RATE / HOP_LENGTH)  # 128 x T_slow
    pr[pr > 0] = 1
    np.save(out_pr, pr.astype(np.uint8))
    return {"piano_roll_path": str(out_pr), "shape": list(pr.shape)}

def extract_signal_feats(audio_path: Path, out_amp: Path, out_rframe: Path, out_rbend: Path) -> Dict[str, Any]:
    y, sr = sf.read(str(audio_path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # RMS amplitude
    amp = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    if amp.max() > 0:
        amp = amp / float(amp.max())

    # f0 + voiced
    f0, vflag, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr,
                                frame_length=N_FFT, hop_length=HOP_LENGTH)
    vmask = np.where(vflag, 1.0, 0.0).astype(np.float32)
    f0 = np.nan_to_num(f0, nan=0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        rbend = 12.0 * np.log2(np.where(f0 > 0, f0 / 440.0, 1.0))
    rbend = np.where(np.isfinite(rbend), rbend, 0.0).astype(np.float32)
    rbend = rbend * vmask

    # unify lengths (defensive)
    T = min(len(amp), len(vmask), len(rbend))
    amp, rframe, rbend = amp[:T].astype(np.float32), vmask[:T], rbend[:T]

    np.save(out_amp,   amp)
    np.save(out_rframe, rframe)
    np.save(out_rbend,  rbend)

    return {"num_frames": int(T), "frame_rate_hz": float(SAMPLE_RATE / HOP_LENGTH)}

def make_rbend_mask_from_pr_rframe_amp(
    piano_roll: Optional[np.ndarray],
    rframe: np.ndarray,
    amp: np.ndarray,
    amp_thr: float = 0.06,
    smooth_k: int = 5
) -> np.ndarray:
    # Re-implements your dataloader’s mask logic (simplified).
    T = int(len(rframe))
    mask = np.ones(T, dtype=bool)

    if piano_roll is not None and piano_roll.size > 0:
        pr = (piano_roll > 0)
        active = pr.sum(axis=0)
        mono = active <= 1
        # allow power intervals (octave/fifth) when exactly 2 notes
        two = np.where(active == 2)[0]
        for t in two:
            notes = np.where(pr[:, t])[0]
            if len(notes) == 2:
                d = abs(int(notes[0]) - int(notes[1]))
                if min(abs(d - 12), abs(d - 7)) <= 1:
                    mono[t] = True
        # simple box smooth
        if smooth_k > 1:
            pad = smooth_k // 2
            x = mono.astype(np.float32)
            x = np.pad(x, (pad, pad), mode="edge")
            filt = np.ones(smooth_k, dtype=np.float32) / smooth_k
            x = np.convolve(x, filt, mode="valid")
            mono = x > 0.5
        mask &= mono

    mask &= (rframe > 0)
    bleed = (amp > amp_thr) & (rframe == 0)
    mask &= (~bleed)
    mask &= (amp > 0.01)

    if smooth_k > 1:
        pad = smooth_k // 2
        x = mask.astype(np.float32)
        x = np.pad(x, (pad, pad), mode="edge")
        filt = np.ones(smooth_k, dtype=np.float32) / smooth_k
        x = np.convolve(x, filt, mode="valid")
        mask = x > 0.5

    return mask.astype(np.float32)

# ---- timbre (encodec-like) features ----
def extract_timbre_features(audio_path: Path, target_sr=ENC_SR, n_ch=8) -> torch.Tensor:
    """
    We'll try Encodec codes → [C,T]; if that fails, fallback to 8-band log-mel as [C,T].
    Returns float tensor [C, T_fast].
    """
    # try encodec
    try:
        from encodec import EncodecModel
        from encodec.utils import convert_audio
        wav, sr = torchaudio.load(str(audio_path))
        wav = wav.float()
        wav24 = convert_audio(wav, sr, target_sr, 1)  # mono
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        model.to("cpu").eval()
        with torch.no_grad():
            enc = model.encode(wav24.unsqueeze(0))  # library returns nested structure
        # robustly hunt for a 3D tensor [B, Q, T] or [Q, T]
        def first_3d_tensor(x):
            if isinstance(x, torch.Tensor) and x.dim() >= 2:
                return x
            if isinstance(x, (list, tuple)):
                for it in x:
                    got = first_3d_tensor(it)
                    if got is not None:
                        return got
            if isinstance(x, dict):
                for v in x.values():
                    got = first_3d_tensor(v)
                    if got is not None:
                        return got
            return None
        cand = first_3d_tensor(enc)
        if cand is not None:
            t = cand.detach().cpu()
            if t.dim() == 3:
                # assume [B,Q,T] or [*,Q,T]; take first batch
                if t.shape[0] > 1:
                    t = t[0]
            # Now t is [Q,T] or [T,Q] – make it [Q,T]
            if t.shape[0] < t.shape[1]:
                # likely already [Q,T]; keep
                pass
            else:
                # transpose if looks like [T,Q]
                if t.shape[1] < t.shape[0]:
                    t = t.transpose(0, 1)
            # compress/expand to 8 "channels"
            Q, T = t.shape
            if Q > n_ch:
                t = t[:n_ch]
            elif Q < n_ch:
                pad = torch.zeros(n_ch - Q, T)
                t = torch.cat([t, pad], dim=0)
            return t.float()
    except Exception:
        pass

    # fallback: 8-band log-mel at ~ENC hop rate
    y, sr = sf.read(str(audio_path))
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=ENC_HOP,
                                       n_mels=n_ch, fmin=60, fmax=sr//2)
    m = librosa.power_to_db(m + 1e-8)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return torch.from_numpy(m.astype(np.float32))  # [C,T_fast]

# ---------- sampling (no GT latents needed) ----------

@torch.no_grad()
def sample_preview_from_noise(model: Pipeline,
                              piano_roll: torch.Tensor, amp: torch.Tensor,
                              rframe: torch.Tensor, rbend: torch.Tensor, rbend_mask: torch.Tensor,
                              encodec_tokens: torch.Tensor,
                              group_id: int, subgroup_id: int,
                              steps: int = 30, sr_out: int = 48000,
                              seed: int = 0) -> (torch.Tensor, int):
    """
    Builds control tokens on the model device, starts from pure noise with shape [1,8,16,T_slow],
    runs Euler-like updates via _call_transformer_no_xattn, decodes with DCAE.
    """
    device = next(model.parameters()).device

    # T_slow from cond length
    T_slow = int(piano_roll.shape[-1])

    # shapes
    B, C_lat, H_lat = 1, 8, 16

    # move conds
    pr  = piano_roll.to(device)
    a   = amp.to(device)
    rf  = rframe.to(device)
    rb  = rbend.to(device)
    rbm = rbend_mask.to(device)

    # encodec/timbre to [B,C_fast,T_fast] float
    enc = encodec_tokens.to(device)
    if enc.dim() == 2:        # [C_fast,T_fast] -> [1,C_fast,T_fast]
        enc = enc.unsqueeze(0)
    enc = enc.float()

    # ids
    gid = torch.tensor([group_id], device=device, dtype=torch.long)
    sgid = torch.tensor([subgroup_id], device=device, dtype=torch.long)

    # control tokens
    tokens, mask = model.ctrl_enc(
        piano_roll=pr.unsqueeze(0),       # [1,128,T]
        amp=a.unsqueeze(0),               # [1,T]
        rframe=rf.unsqueeze(0),           # [1,T]
        rbend=rb.unsqueeze(0),            # [1,T]
        rbend_mask=rbm.unsqueeze(0),      # [1,T]
        encodec_tokens=enc,               # [1,C_fast,T_fast]
        group_id=gid, subgroup_id=sgid
    )

    # latent init (pure noise)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    x = torch.randn(1, C_lat, H_lat, T_slow, device=device)

    T_train = 1000
    steps = max(1, int(steps))
    dt = 1.0 / float(steps)

    for i in range(steps, 0, -1):
        t_cont = torch.full((1,), i * dt, device=device)
        t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)

        tokens_adapt = tokens.to(dtype=x.dtype, device=device)
        cond_patch = model.cond_adapter(tokens_adapt, T_out=x.shape[-1], scale=model._adapter_gain_scale())
        cond_patch = cond_patch.to(device=x.device, dtype=x.dtype)

        x_in = x + cond_patch
        v_pred = model._call_transformer_no_xattn(latents=x_in, t=t_idx)
        x = x - dt * v_pred

    # decode
    model.dcae.to("cpu")
    audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
    x_cpu = x[:1].float().cpu()
    audio_lengths = torch.tensor([audio_len], dtype=torch.long)
    sr_pred, wav_pred = model.dcae.decode(x_cpu, audio_lengths=audio_lengths, sr=sr_out)
    model.dcae.to(device)
    return wav_pred[0].cpu(), int(sr_pred)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Audio file or folder (if folder, a random file is chosen)")
    ap.add_argument("--checkpoint", required=True, help="Path to Lightning .ckpt")
    ap.add_argument("--components_dir", required=True, help="ACEStep components dir for DCAE/transformer init")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--group", default="guitar", help=f"One of {APPROVED_GROUPS}")
    ap.add_argument("--subgroup", default="electric_guitar", help="Subgroup string")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--sr_out", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input)
    audio_path = pick_random_file(in_path) if in_path.is_dir() else in_path
    stem = safe_stem(audio_path.stem)

    out_dir = Path(args.output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_mid    = out_dir / f"{stem}.mid"
    out_pr     = out_dir / f"{stem}.pianoroll.npy"
    out_amp    = out_dir / f"{stem}.amp.npy"
    out_rframe = out_dir / f"{stem}.rframe.npy"
    out_rbend  = out_dir / f"{stem}.rbend.npy"
    out_prev   = out_dir / f"{stem}.preview.wav"

    print(f"🎧 Using audio: {audio_path}")

    # 1) PR via BasicPitch ONNX
    bp = extract_basicpitch(audio_path, out_mid, out_pr)
    pr_np = np.load(out_pr)  # [128,T]
    T_slow = pr_np.shape[1]

    # 2) amp/rframe/rbend via librosa/pyin
    sig = extract_signal_feats(audio_path, out_amp, out_rframe, out_rbend)
    amp_np    = np.load(out_amp)
    rframe_np = np.load(out_rframe)
    rbend_np  = np.load(out_rbend)

    # resample PR time to match the longest of {PR, amp, rframe, rbend}
    T_target = max(pr_np.shape[1], len(amp_np), len(rframe_np), len(rbend_np))
    if pr_np.shape[1] != T_target:
        pr_np = librosa.resample(pr_np.astype(float), orig_sr=pr_np.shape[1], target_sr=T_target, axis=1, res_type="nearest")
        pr_np = (pr_np > 0.5).astype(np.uint8)
    if len(amp_np) != T_target:
        amp_np = np.interp(np.linspace(0,1,T_target), np.linspace(0,1,len(amp_np)), amp_np)
    if len(rframe_np) != T_target:
        rframe_np = (np.interp(np.linspace(0,1,T_target), np.linspace(0,1,len(rframe_np)), rframe_np) > 0.5).astype(np.float32)
    if len(rbend_np) != T_target:
        rbend_np = np.interp(np.linspace(0,1,T_target), np.linspace(0,1,len(rbend_np)), rbend_np)

    # 3) rbend mask
    rb_mask_np = make_rbend_mask_from_pr_rframe_amp(pr_np, rframe_np, amp_np, amp_thr=0.06, smooth_k=5)

    # 4) timbre features (encodec codes or mel fallback), shape [C_fast,T_fast]
    enc_ct = extract_timbre_features(audio_path)  # [C,Tfast]

    # 5) load model
    # NB: Pipeline.load_from_checkpoint requires hparams it expects; we provide components_dir + dummy manifest.
    dummy_manifest = out_dir / "_dummy_manifest.json"
    dummy_manifest.write_text("[]")
    print(f"🧠 Loading checkpoint: {args.checkpoint}")
    model: Pipeline = Pipeline.load_from_checkpoint(
        args.checkpoint,
        checkpoint_dir=args.components_dir,
        manifest_json=str(dummy_manifest),
        preview_steps=args.steps,
        batch_size=1,
    )
    model.eval().freeze()
    device = next(model.parameters()).device
    print(f"Model device: {device}")

    # 6) pack tensors and run sampling from pure noise (shape only needs T_slow)
    # choose T_slow = T_target
    T_slow = int(T_target)
    pr_t   = torch.from_numpy(pr_np.astype(np.float32))          # [128,T]
    amp_t  = torch.from_numpy(amp_np.astype(np.float32))         # [T]
    rfr_t  = torch.from_numpy(rframe_np.astype(np.float32))      # [T]
    rbd_t  = torch.from_numpy(rbend_np.astype(np.float32))       # [T]
    rbm_t  = torch.from_numpy(rb_mask_np.astype(np.float32))     # [T]
    enc_t  = enc_ct                                              # [C_fast,T_fast] float

    # instrument ids
    group = args.group.lower()
    subgroup = args.subgroup.lower()
    g2id = {g:i for i,g in enumerate(APPROVED_GROUPS)}
    sub_flat = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
    sg2id = {sg:i for i,sg in enumerate(sub_flat)}
    gid  = g2id.get(group, g2id["guitar"])
    sgid = sg2id.get(subgroup, sg2id.get("undefined", 0))

    wav, sr = sample_preview_from_noise(
        model,
        pr_t, amp_t, rfr_t, rbd_t, rbm_t,
        enc_t,
        group_id=gid, subgroup_id=sgid,
        steps=args.steps, sr_out=args.sr_out, seed=args.seed
    )
    torchaudio.save(str(out_prev), wav, sr)

    # summary
    print("\n==== RESULT ====")
    print(json.dumps({
        "input_audio": str(audio_path),
        "output_dir": str(out_dir.resolve()),
        "preview_wav": str(out_prev),
        "piano_roll_shape": list(pr_np.shape),
        "amp_frames": int(len(amp_np)),
        "rframe_frames": int(len(rframe_np)),
        "rbend_frames": int(len(rbend_np)),
        "encodec_like_shape": list(enc_ct.shape),
        "steps": int(args.steps),
        "sr_out": int(args.sr_out),
        "group": group, "subgroup": subgroup
    }, indent=2))
    print(f"✅ Wrote {out_prev}")

if __name__ == "__main__":
    main()
