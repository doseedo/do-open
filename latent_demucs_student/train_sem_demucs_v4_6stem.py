#!/usr/bin/env python3
"""Train SemDemucs v4 6-stem: STFT masks + per-stem sem embeddings.

Same as train_sem_demucs_v4.py but for 6 stems using htdemucs_6s teacher
and DistillDataset6 for training data.
"""
import argparse
import importlib.util
import math
import os
import random
import re
import glob as _glob
import sys
import time

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sem_demucs import SemDemucs
from distill_dataset_6 import DistillDataset6, STEMS_6

STEMS = STEMS_6
SR = 48000

SEM_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_V1_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"


def load_sem_encoder(device="cuda"):
    spec = importlib.util.spec_from_file_location(
        "sem", os.path.join(SEM_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sd = torch.load(SEM_V1_CKPT, map_location="cpu", weights_only=False)
    m = mod.SemanticEncoderWithHeads().to(device).eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters(): p.requires_grad = False
    return m


def load_htdemucs_6s(device="cuda"):
    from demucs.pretrained import get_model
    m = get_model('htdemucs_6s').to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def compute_stft_masks_6s(teacher, mix, n_fft=2048, hop=512):
    from demucs.apply import apply_model
    with torch.no_grad():
        stems = apply_model(teacher, mix.float(), device=str(mix.device))
        stem_mono = stems.mean(dim=2)
        window = torch.hann_window(n_fft, device=mix.device)
        mags = []
        for i in range(6):
            spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                              return_complex=True).abs()
            mags.append(spec)
        mags = torch.stack(mags, dim=1)
        return mags / (mags.sum(dim=1, keepdim=True) + 1e-8)


# Frozen pitch/visual teachers
PITCH_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_pitch")
PITCH_CKPT = "/scratch/latent_pitch_ckpts/pitch_final.pt"
VISUAL_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_visual")
VISUAL_CKPT = "/scratch/latent_visual_ckpts/latent_visual_final.pt"


def _load_teacher(dir_path, ckpt_path, model_file, class_name, **kwargs):
    spec = importlib.util.spec_from_file_location("mod", os.path.join(dir_path, model_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pos = sd["model"].get("pos")
    if pos is not None and "max_len" not in kwargs:
        kwargs["max_len"] = pos.shape[1]
    m = cls(**kwargs).cuda().eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters(): p.requires_grad = False
    return m


def collate_6(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/sem_demucs_v4_6stem_ckpts")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop-sec", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lambda-mask", type=float, default=2.0)
    ap.add_argument("--lambda-emb", type=float, default=1.0)
    ap.add_argument("--lambda-pitch", type=float, default=1.0)
    ap.add_argument("--lambda-rms", type=float, default=1.0)
    ap.add_argument("--lambda-vocal", type=float, default=0.3)
    ap.add_argument("--channels", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    crop_frames = int(args.crop_sec * 25)
    ds = DistillDataset6(crop_frames=crop_frames)
    if len(ds) == 0: print("ERROR: empty"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate_6, persistent_workers=args.workers > 0)

    print("[v4-6] loading frozen teachers...")
    sem_enc = load_sem_encoder()
    teacher = load_htdemucs_6s()
    pitch_t = _load_teacher(PITCH_DIR, PITCH_CKPT, "model.py", "LatentBasicPitchStudent")
    visual_t = _load_teacher(VISUAL_DIR, VISUAL_CKPT, "infer.py", "LatentToPeakEnvelope")
    print("[v4-6] all teachers loaded")

    model = SemDemucs(n_stems=6, channels=args.channels).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[v4-6] SemDemucs: {n:.1f}M params (channels={args.channels}, 6 stems)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    def lr_lambda(step):
        if step < args.warmup: return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(prog * math.pi))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    step = 0
    _ckpts = [f for f in _glob.glob(os.path.join(args.out, "*_step*.pt"))
              if re.search(r'step(\d+)', f)]
    if _ckpts:
        _latest = max(_ckpts, key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
        _sd = torch.load(_latest, map_location="cuda", weights_only=False)
        model.load_state_dict(_sd["model"])
        step = _sd.get("step", 0)
        for _ in range(step): sched.step()
        print(f"[resume] from step {step} ({os.path.basename(_latest)})")

    hist = {k: [] for k in ("total", "mask", "emb", "pitch", "rms", "vocal")}
    t0 = time.time()

    while step < args.steps:
        for audio, stem_lats, presence in loader:
            audio = audio.cuda(non_blocking=True)
            stem_lats = stem_lats.cuda(non_blocking=True)  # [B, 6, 64, T]
            presence = presence.cuda(non_blocking=True)  # [B, 6]
            B = audio.shape[0]

            # STFT mask targets from htdemucs_6s
            tgt_masks = compute_stft_masks_6s(teacher, audio)

            with torch.no_grad():
                tgt_embs, tgt_pitch, tgt_rms = [], [], []
                for si in range(6):
                    lat = stem_lats[:, si]  # [B, 64, T]
                    lat_t = lat.transpose(1, 2)  # [B, T, 64]
                    tgt_embs.append(sem_enc(lat_t)["embedding"])
                    p_out = pitch_t(lat_t)
                    tgt_pitch.append(0.5 * (torch.sigmoid(p_out["onset_logits"]) +
                                            torch.sigmoid(p_out["frame_logits"])))
                    rms_out = visual_t(lat_t.transpose(1, 2)).transpose(1, 2)
                    tgt_rms.append(rms_out)
                tgt_emb = torch.stack(tgt_embs, dim=1)
                tgt_pitch_t = torch.stack(tgt_pitch, dim=1)
                tgt_rms_t = torch.stack(tgt_rms, dim=1)

            opt.zero_grad(set_to_none=True)
            out = model(audio.float())

            # Mask loss
            pred_masks = out["stft_masks"]
            F_m = min(pred_masks.shape[2], tgt_masks.shape[2])
            T_m = min(pred_masks.shape[3], tgt_masks.shape[3])
            l_mask = F.mse_loss(pred_masks[:, :, :F_m, :T_m], tgt_masks[:, :, :F_m, :T_m])

            # Embedding loss (masked by presence)
            diff_e = (out["embedding"] - tgt_emb).abs()
            mask_e = presence[:, :, None].expand_as(diff_e)
            l_emb = (diff_e * mask_e).sum() / mask_e.sum().clamp(min=1)

            # Pitch loss (masked)
            T_p = min(out["pitch_logits"].shape[2], tgt_pitch_t.shape[2])
            l_pitch = F.binary_cross_entropy_with_logits(
                out["pitch_logits"][:, :, :T_p], tgt_pitch_t[:, :, :T_p], reduction='none')
            mask_p = presence[:, :, None, None].expand_as(l_pitch)
            l_pitch = (l_pitch * mask_p).sum() / mask_p.sum().clamp(min=1)

            # RMS loss (masked)
            T_r = min(out["rms"].shape[2], tgt_rms_t.shape[2])
            diff_r = (out["rms"][:, :, :T_r] - tgt_rms_t[:, :, :T_r]).abs()
            mask_r = presence[:, :, None, None].expand_as(diff_r)
            l_rms = (diff_r * mask_r).sum() / mask_r.sum().clamp(min=1)

            # Vocal loss
            vocal_labels = torch.zeros(B, 6, device='cuda')
            vocal_labels[:, 3] = 1.0  # vocals is index 3
            l_vocal = F.binary_cross_entropy_with_logits(out["vocal"], vocal_labels)

            loss = (args.lambda_mask * l_mask + args.lambda_emb * l_emb +
                    args.lambda_pitch * l_pitch + args.lambda_rms * l_rms +
                    args.lambda_vocal * l_vocal)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            hist["total"].append(loss.item())
            hist["mask"].append(l_mask.item())
            hist["emb"].append(l_emb.item())
            hist["pitch"].append(l_pitch.item())
            hist["rms"].append(l_rms.item())
            hist["vocal"].append(l_vocal.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                print(f"[step {step:5d}] total={avg(hist['total']):.4f} "
                      f"mask={avg(hist['mask']):.4f} "
                      f"emb={avg(hist['emb']):.4f} "
                      f"pitch={avg(hist['pitch']):.4f} "
                      f"rms={avg(hist['rms']):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"sem_demucs_v4_6s_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps: break

    final = os.path.join(args.out, "sem_demucs_v4_6s_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
