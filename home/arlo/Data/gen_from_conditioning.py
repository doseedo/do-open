#!/usr/bin/env python3
"""
Generate audio from conditioning ONLY (no GT latents) using a trained ACE-Step Pipeline.

Usage:
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python gen_from_conditioning.py \
    --audio /path/to/audio.wav \
    --checkpoint /mnt/msdd/exps/.../checkpoints/last.ckpt \
    --checkpoint_dir /home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/<hash> \
    --manifest ./final_training_manifest_final.json \
    --steps 40 \
    --sr_out 32000 \
    --out ./generated/out.wav \
    --group brass \
    --subgroup trumpet
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torchaudio

# so we can import your project modules
sys.path.append('/home/arlo/Data')
from trainer_performer import Pipeline  # (ctrl_enc, cond_adapter, _call_transformer_no_xattn)

# --- DCAE grid (matches trainer_performer.py) ---
DCAE_SR, DCAE_HOP = 44100, 4096
ENC_SR,  ENC_HOP  = 24000,  320
SLOW_HZ = DCAE_SR / DCAE_HOP  # ~10.766 Hz

# put this near the top (after imports)
def _resize_like(param_from_ckpt: torch.Tensor, target_param: torch.Tensor) -> torch.Tensor:
    """Return a tensor shaped like target_param, filled from param_from_ckpt (copy overlap)."""
    src = param_from_ckpt.detach().cpu()
    tgt = target_param.detach().cpu().clone()
    if src.shape == tgt.shape:
        return src
    # copy overlapping slice
    # works for embeddings/linears/conv weights (row-major assumption)
    common = tuple(min(a, b) for a, b in zip(src.shape, tgt.shape))
    if src.ndim == 1:
        tgt[:common[0]] = src[:common[0]]
    elif src.ndim == 2:
        tgt[:common[0], :common[1]] = src[:common[0], :common[1]]
    elif src.ndim == 3:
        tgt[:common[0], :common[1], :common[2]] = src[:common[0], :common[1], :common[2]]
    elif src.ndim == 4:
        tgt[:common[0], :common[1], :common[2], :common[3]] = src[:common[0], :common[1], :common[2], :common[3]]
    else:
        # fallback: flatten copy
        n = min(src.numel(), tgt.numel())
        tgt.view(-1)[:n] = src.view(-1)[:n]
    return tgt

def load_model_any_ckpt(checkpoint_path: str, checkpoint_dir: str, manifest_json: str) -> "Pipeline":
    """
    Instantiate Pipeline with current code, then load a legacy checkpoint:
    - Resizes known variable-size tensors (group/subgroup heads, embeddings).
    - Loads remaining weights with strict=False.
    """
    # 1) Instantiate a fresh model with current vocab sizes
    model = Pipeline(
        checkpoint_dir=checkpoint_dir,
        manifest_json=manifest_json,
    )
    model.eval()

    # 2) Load ckpt state dict
    blob = torch.load(checkpoint_path, map_location="cpu")
    sd = blob.get("state_dict", blob)

    # 3) Resize/patch known variable-size parts to current shapes
    patch_keys = [
        # ctrl_enc instrument vocab
        ("ctrl_enc.subgroup_emb.weight", model.ctrl_enc.subgroup_emb.weight),
        ("ctrl_enc.group_emb.weight",    model.ctrl_enc.group_emb.weight),
        # aux classification heads (if present in ckpt)
        ("group_head.weight",            model.group_head.weight),
        ("group_head.bias",              model.group_head.bias),
        ("sub_head.weight",              model.sub_head.weight),
        ("sub_head.bias",                model.sub_head.bias),
    ]
    for k, target in patch_keys:
        if k in sd:
            if tuple(sd[k].shape) != tuple(target.shape):
                sd[k] = _resize_like(sd[k], target)
                print(f"[compat] resized {k}: ckpt {tuple(sd[k].shape)} → model {tuple(target.shape)}")
        else:
            # If the ckpt didn’t have the head, leave model init values
            pass

    # 4) Now load with strict=False (size-mismatch handled, missing/extra ok)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        print(f"[compat] missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[compat] unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

    return model




# ------------------------------- Conditioning I/O -------------------------------

def extract_conditioning_from_audio(audio_path: str, output_dir: str = "./extracted_conditioning") -> dict:
    """Extract (or reuse cached) conditioning files for the given audio."""
    p = Path(audio_path)
    stem = p.stem
    safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in stem)[:128] or "audio"
    out_dir = Path(output_dir) / safe

    req = [
        out_dir / f"{safe}.pianoroll.npy",
        out_dir / f"{safe}.amp.npy",
        out_dir / f"{safe}.rframe.npy",
        out_dir / f"{safe}.rbend.npy",
        out_dir / f"{safe}.encodec.pt",
    ]
    if all(x.exists() for x in req):
        print("✅ Using existing extracted conditioning:", out_dir)
        return {"dir": str(out_dir), "stem": safe}

    cmd = ["python", "test_extract_local.py", "--input", str(p), "--output", output_dir]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"Extraction failed with code {res.returncode}")
    print("✅ Conditioning extracted:", out_dir)
    return {"dir": str(out_dir), "stem": safe}


def _pad1(x: np.ndarray, L: int) -> np.ndarray:
    if x.shape[0] >= L:
        return x[:L]
    return np.pad(x, (0, L - x.shape[0]), mode="constant")


def _pad2(x: np.ndarray, L: int) -> np.ndarray:
    if x.shape[1] >= L:
        return x[:, :L]
    return np.pad(x, ((0, 0), (0, L - x.shape[1])), mode="constant")


def load_conditioning(extraction: dict, window_slow: int):
    """Load numpy + encodec tensors and force all to length == window_slow."""
    out_dir = Path(extraction["dir"])
    stem = extraction["stem"]

    pr_path   = out_dir / f"{stem}.pianoroll.npy"
    amp_path  = out_dir / f"{stem}.amp.npy"
    rf_path   = out_dir / f"{stem}.rframe.npy"
    rb_path   = out_dir / f"{stem}.rbend.npy"
    enc_path  = out_dir / f"{stem}.encodec.pt"

    for p in (pr_path, amp_path, rf_path, rb_path, enc_path):
        if not p.exists():
            raise FileNotFoundError(p)

    piano_roll = np.load(pr_path)  # (128, T)
    amp        = np.load(amp_path) # (T,)
    rframe     = np.load(rf_path)  # (T,)
    rbend      = np.load(rb_path)  # (T,)
    enc_data = torch.load(enc_path, map_location="cpu")
    # normalize -> tensor [B, C_fast, T_fast]
    if isinstance(enc_data, list) and len(enc_data) > 0:
        x = enc_data[0]
        if hasattr(x, "codes"):
            enc = x.codes
        elif isinstance(x, tuple):
            enc = x[0] if x[0] is not None else x[1]
        else:
            enc = x
    elif isinstance(enc_data, tuple):
        enc = enc_data[0]
    elif hasattr(enc_data, "codes"):
        enc = enc_data.codes
    else:
        enc = enc_data

    if not isinstance(enc, torch.Tensor):
        enc = torch.tensor(enc)
    if enc.ndim == 2:  # (C_fast, T_fast)
        enc = enc.unsqueeze(0)  # -> [1, C_fast, T_fast]
    assert enc.ndim == 3, f"encodec_tokens shape unexpected: {tuple(enc.shape)}"



    
    # Still align the *slow* streams to window_slow:
    piano_roll = _pad2(piano_roll, window_slow)
    amp        = _pad1(amp, window_slow)
    rframe     = _pad1(rframe, window_slow)
    rbend      = _pad1(rbend, window_slow)

    print("✅ Conditioning shapes:",
          "piano_roll", piano_roll.shape,
          "amp", amp.shape,
          "rframe", rframe.shape,
          "rbend", rbend.shape,
          "encodec_tokens", tuple(enc.shape))
    # Optional: sanity ratio
    ratio = enc.shape[-1] / float(window_slow)
    print(f"[check] fast/slow ratio ≈ {ratio:.3f} (expect ~6.96)")

    return piano_roll, amp, rframe, rbend, enc.long()

# ------------------------------- Vocab helpers -------------------------------

def _get_vocab_from_model_or_ckpt(model: Pipeline, ckpt_hp: dict):
    """Return (group_names, subgroup_names) from model.hparams if present, else from ckpt hyper_parameters."""
    mh = getattr(model, "hparams", None)
    g = getattr(mh, "group_names", None) if mh is not None else None
    s = getattr(mh, "subgroup_names", None) if mh is not None else None
    if (not g) or (not s):
        g = ckpt_hp.get("group_names", g)
        s = ckpt_hp.get("subgroup_names", s)
    return g, s


def _resolve_ids(group_names, subgroup_names, group: str | None, subgroup: str | None):
    """Map strings to integer IDs using the checkpoint vocab. Fallback to 0/undefined when needed."""
    if not group_names or not subgroup_names:
        print("[vocab] WARNING: no group/subgroup names in checkpoint; defaulting to ids (0, 0)")
        return 0, 0

    # Group
    if group in group_names:
        g_id = group_names.index(group)
    else:
        print(f"[vocab] WARNING: '{group}' not in group_names; defaulting to id 0 ({group_names[0]})")
        g_id = 0

    # Subgroup (global list saved during training)
    if subgroup in subgroup_names:
        s_id = subgroup_names.index(subgroup)
    else:
        if "undefined" in subgroup_names:
            s_id = subgroup_names.index("undefined")
            print(f"[vocab] WARNING: '{subgroup}' not in subgroup_names; using 'undefined' (id {s_id})")
        else:
            s_id = 0
            print(f"[vocab] WARNING: '{subgroup}' not in subgroup_names; defaulting to id 0 ({subgroup_names[0]})")
    return g_id, s_id


# ------------------------------- Generation Core -------------------------------

@torch.no_grad()
def generate_from_conditioning(
    model: Pipeline,
    piano_roll: np.ndarray,
    amp: np.ndarray,
    rframe: np.ndarray,
    rbend: np.ndarray,
    encodec_tokens: torch.Tensor,
    steps: int = 40,
    sr_out: int = 32000,
    seed: int = 0,
    out_path: str = "generated.wav",
    group_id: int = 0,
    subgroup_id: int = 0,
    force_T_slow: int | None = None,
):
    assert hasattr(model, "ctrl_enc"), "Loaded Pipeline must have ctrl_enc"
    assert hasattr(model, "cond_adapter"), "Loaded Pipeline must have cond_adapter"
    assert hasattr(model, "_call_transformer_no_xattn"), "Pipeline needs _call_transformer_no_xattn"




    device = model.device
    model.eval()

    if hasattr(model, "transformers"):
        model.transformers.to(device)



    if hasattr(model, "ctrl_enc"):
        model.ctrl_enc.to(device).eval()
    if hasattr(model, "dcae"):
        model.dcae.to(device).eval()

    T_slow = int(force_T_slow) if force_T_slow is not None else int(getattr(getattr(model, "hparams", {}), "window_slow", 2584))


    # Build cond tensors (CPU first; move on call)
    pr_t   = torch.from_numpy(piano_roll).float().unsqueeze(0)   # [1,128,T]
    amp_t  = torch.from_numpy(amp).float().unsqueeze(0)          # [1,T]
    rf_t   = torch.from_numpy(rframe).float().unsqueeze(0)       # [1,T]
    rb_t   = torch.from_numpy(rbend).float().unsqueeze(0)        # [1,T]
    rbm_t  = torch.from_numpy((rframe > 0.5)).bool().unsqueeze(0)# [1,T] bool

    # --- Use ids passed from main(), validate & make tensors ---
    g_vocab = int(model.ctrl_enc.group_emb.num_embeddings)
    s_vocab = int(model.ctrl_enc.subgroup_emb.num_embeddings)
    if not (0 <= int(group_id) < g_vocab):
        print(f"[vocab] WARNING: group_id {group_id} out of range [0,{g_vocab-1}], clamping.")
    if not (0 <= int(subgroup_id) < s_vocab):
        print(f"[vocab] WARNING: subgroup_id {subgroup_id} out of range [0,{s_vocab-1}], clamping.")
    g_id = max(0, min(int(group_id),    g_vocab - 1))
    s_id = max(0, min(int(subgroup_id), s_vocab - 1))
    group_id_t    = torch.tensor([g_id], dtype=torch.long, device=device)
    subgroup_id_t = torch.tensor([s_id], dtype=torch.long, device=device)
    print(f"[vocab] resolved ids → group_id={g_id} (vocab={g_vocab}), subgroup_id={s_id} (vocab={s_vocab})")

    # Encode control tokens on model device
    tokens, mask = model.ctrl_enc(
        piano_roll=pr_t.to(device),
        amp=amp_t.to(device),
        rframe=rf_t.to(device),
        rbend=rb_t.to(device),
        rbend_mask=rbm_t.to(device),
        encodec_tokens=encodec_tokens.to(device),  # [1,C,T]
        group_id=group_id_t,
        subgroup_id=subgroup_id_t,
    )

    # === build adapter patch once to discover the latent grid ===
    ca_param = next(model.cond_adapter.parameters(), None)
    tokens_adapt = tokens.to(
        device=ca_param.device if ca_param is not None else tokens.device,
        dtype=ca_param.dtype if ca_param is not None else tokens.dtype,
    )

    # Inference: fixed adapter scaling (env overrides)
    adapter_scale = float(os.environ.get("ADAPTER_SCALE", 1.0))

    cond_patch_base = model.cond_adapter(tokens_adapt, T_out=T_slow, scale=adapter_scale)
    cond_patch_base = cond_patch_base.to(device=device, dtype=tokens.dtype)

    # Seed latents to EXACTLY the adapter's latent grid (B,C,H,T)
    torch.manual_seed(int(seed))
    x = torch.randn_like(cond_patch_base)

    print(f"Adapter latent grid: {tuple(cond_patch_base.shape)}")
    print(f"Init x grid:         {tuple(x.shape)}")

    # Timetable
    T_train = int(getattr(getattr(model, "scheduler", None), "config", {}).get("num_train_timesteps", 1000))
    steps = max(1, int(steps))
    dt = 1.0 / float(steps)
    print(f"Sampling {steps} steps → T_slow={T_slow}, T_train={T_train}, sr_out={sr_out}")

    # Denoise
    for i in range(steps, 0, -1):
        cond_patch = cond_patch_base  # static adapter injection
        if cond_patch.dtype != x.dtype or cond_patch.device != x.device:
            cond_patch = cond_patch.to(device=x.device, dtype=x.dtype)

        x_in = x + cond_patch
        t_cont = torch.full((1,), i * dt, device=x.device, dtype=torch.float32)
        t_idx  = (t_cont * (T_train - 1)).long().clamp(0, T_train - 1)  # [1]

        v_pred = model._call_transformer_no_xattn(latents=x_in, t=t_idx)
        x = x - dt * v_pred  # RF-style update

        if i % max(1, steps // 5) == 0:
            print(f"  step {i:3d}/{steps}: |x|={x.float().pow(2).mean().sqrt().item():.4f}")

    # --- Decode (GPU, float32) ---
    audio_len  = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
    audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=device)

    if hasattr(model, "dcae"):
        model.dcae.to(device).float()

    x_for_dcae = x[:1].to(device=device, dtype=torch.float32)
    sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

    # Stats + near-silence guard
    wav = wav_pred[0].float().cpu()
    maxabs = float(wav.abs().max().item())
    print(f"[decode/gpu-f32] wav stats: mean={wav.mean().item():.3e}, std={wav.std().item():.3e}, maxabs={maxabs:.3e}, sr={sr_pred}")

    if maxabs < 1e-5:
        print("[decode] near-silence detected; retrying decode on CPU/float32…")
        if hasattr(model, "dcae"):
            model.dcae.to("cpu").float()
        x_cpu = x[:1].to(device="cpu", dtype=torch.float32)
        sr_pred, wav_pred = model.dcae.decode(x_cpu, audio_lengths=audio_lengths.cpu(), sr=sr_out)
        wav = wav_pred[0].float().cpu()
        print(f"[decode/cpu-f32] wav stats: mean={wav.mean().item():.3e}, std={wav.std().item():.3e}, maxabs={wav.abs().max().item():.3e}, sr={sr_pred}")

    # Save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav, sr_pred)
    print(f"✅ Wrote: {out_path}")
    
    return out_path


# ------------------------------- Entrypoint -------------------------------

def main():
    ap = argparse.ArgumentParser(description="From-noise generation from conditioning (no GT latents).")
    ap.add_argument("--audio", required=True, help="Path to the source audio used for conditioning extraction.")
    ap.add_argument("--checkpoint", required=True, help="Lightning .ckpt path (e.g., last.ckpt).")
    ap.add_argument("--checkpoint_dir", required=True, help="ACEStep snapshots dir for ACEStepTrainComponents.")
    ap.add_argument("--manifest", required=True, help="Training manifest json path (needed by Pipeline).")
    ap.add_argument("--steps", type=int, default=40, help="Denoising steps.")
    ap.add_argument("--sr_out", type=int, default=32000, help="Output audio sample rate.")
    ap.add_argument("--seed", type=int, default=0, help="Noise seed.")
    ap.add_argument("--out", type=str, default="./generated/generated.wav", help="Output WAV path.")
    ap.add_argument("--extract_dir", type=str, default="./extracted_conditioning", help="Where to write/read conditioning.")
    ap.add_argument("--group", type=str, default="piano", help="Instrument group (string, resolved via ckpt vocab).")
    ap.add_argument("--subgroup", type=str, default="acoustic_piano", help="Instrument subgroup (resolved via ckpt vocab).")
    ap.add_argument("--drop_encodec", action="store_true",
                help="If set, zeros the EnCodec token stream (no timbre conditioning).")
    ap.add_argument("--encodec_dropout_p", type=float, default=0.0,
                    help="Probability to drop each EnCodec token (0..1). Overrides to 1.0 if --drop_encodec.")

    args = ap.parse_args()

    # Load ckpt hparams so we can resolve vocab names
    try:
        ckpt_blob = torch.load(args.checkpoint, map_location="cpu")
        ckpt_hp = ckpt_blob.get("hyper_parameters", {})
    except Exception:
        ckpt_hp = {}

    # 1) Load model (allow minor key drift if code added heads later)
    print("Loading model:", args.checkpoint)
    model = load_model_any_ckpt(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        manifest_json=args.manifest,
    )
    print("Loaded Pipeline.")
    model.eval()
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move the LightningModule itself (and anything registered under it)
    model.to(target_device).eval()

    # move nested bits that may not be registered or were constructed lazily
    if hasattr(model, "transformers") and model.transformers is not None:
        model.transformers.to(target_device).eval()
        for name in ("speaker_embedder", "text_embedder", "lyric_embedder"):
            m = getattr(model.transformers, name, None)
            if m is not None:
                m.to(target_device).eval()
    if hasattr(model, "ctrl_enc") and model.ctrl_enc is not None:
        model.ctrl_enc.to(target_device).eval()
    if hasattr(model, "dcae") and model.dcae is not None:
        model.dcae.to(target_device).eval()

    print(f"[device] using {target_device}")
    
    win_slow = int(getattr(getattr(model, "hparams", {}), "window_slow", 2584))
    print("window_slow =", win_slow)

    # Embedding sizes (debug)
    try:
        print("[emb] group_vocab:", model.ctrl_enc.group_emb.num_embeddings,
              "subgroup_vocab:", model.ctrl_enc.subgroup_emb.num_embeddings)
    except Exception:
        pass

    # Resolve vocab from model or ckpt
    group_names, subgroup_names = _get_vocab_from_model_or_ckpt(model, ckpt_hp)
    if group_names:
        print(f"[vocab] groups ({len(group_names)}): {group_names}")
    if subgroup_names:
        preview = ", ".join(subgroup_names[:10]) + (" ..." if len(subgroup_names) > 10 else "")
        print(f"[vocab] subgroups ({len(subgroup_names)}): {preview}")

    from dataloader import APPROVED_GROUPS, APPROVED_SUBGROUPS

    if not group_names:
        group_names = APPROVED_GROUPS[:]                 # preserves training order
    if not subgroup_names:
        subgroup_names = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})


    g_id, s_id = _resolve_ids(group_names, subgroup_names, args.group, args.subgroup)
    print(f"Using group={args.group} (id={g_id}), subgroup={args.subgroup} (id={s_id})")

    # 2) Get conditioning
    ext = extract_conditioning_from_audio(args.audio, output_dir=args.extract_dir)
    pr, amp, rfr, rbd, enc = load_conditioning(ext, window_slow=win_slow)


    # ---- Optionally drop/zero EnCodec tokens ----
    drop_p = 1.0 if args.drop_encodec else float(max(0.0, min(1.0, args.encodec_dropout_p)))
    if drop_p > 0.0:
        # enc is Long (codebook indices). Multiply by a {0,1} mask to zero-out dropped positions.
        with torch.no_grad():
            keep_prob = 1.0 - drop_p
            # elementwise Bernoulli mask over the whole (B, C_fast, T_fast)
            keep = torch.bernoulli(torch.full(enc.shape, keep_prob)).to(dtype=torch.long)
            enc = enc * keep
        print(f"[encodec] dropout applied: p={drop_p:.3f} "
            f"({'FULL DROP' if drop_p >= 1.0 - 1e-9 else 'partial'})")



    T_cond  = min(pr.shape[1], len(amp), len(rfr), len(rbd))
    T_model = int(getattr(getattr(model, "hparams", {}), "window_slow", T_cond))
    T_slow  = min(T_cond, T_model)

    # Crop slow streams
    pr  = pr[:, :T_slow]
    amp = amp[:T_slow]
    rfr = rfr[:T_slow]
    rbd = rbd[:T_slow]

    # --- NOW align EnCodec to the final slow length ---
    FAST_PER_SLOW = (ENC_SR / ENC_HOP) / (DCAE_SR / DCAE_HOP)  # ≈ 6.96
    expect_fast = int(round(T_slow * FAST_PER_SLOW))
    T_fast = enc.shape[-1]
    if T_fast < expect_fast:
        enc = torch.nn.functional.pad(enc, (0, expect_fast - T_fast))
    elif T_fast > expect_fast:
        enc = enc[..., :expect_fast]

    print(f"[ratio] fast/slow = {enc.shape[-1]}/{T_slow} = {enc.shape[-1]/float(T_slow):.3f} (expect ~6.96)")
    print(f"[dur] target seconds ≈ {T_slow * (DCAE_HOP / DCAE_SR):.3f}s")


    # 3) Run generation
    output_path = generate_from_conditioning(
        model,
        pr, amp, rfr, rbd, enc,
        steps=args.steps,
        sr_out=args.sr_out,
        seed=args.seed,
        out_path=args.out,
        group_id=g_id,
        subgroup_id=s_id,
        force_T_slow=T_slow,
    )
    
    # 4) Copy the conditioning MIDI file to the output directory
    from shutil import copy2
    midi_source = Path(ext["dir"]) / f"{ext['stem']}.mid"
    if midi_source.exists():
        midi_dest = output_path.parent / f"{output_path.stem}_conditioning.mid"
        copy2(midi_source, midi_dest)
        print(f"✅ Copied conditioning MIDI: {midi_dest}")
    else:
        print(f"⚠️  Conditioning MIDI not found: {midi_source}")


if __name__ == "__main__":
    main()
