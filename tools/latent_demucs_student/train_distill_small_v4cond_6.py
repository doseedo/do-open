#!/usr/bin/env python3
"""Train v4cond 6-stem: oracle per-stem sem + STFT masks from htdemucs_6s.

Same architecture as v4cond 4-stem but with 6 stems:
  drums, bass, other, vocals, guitar, piano

Uses:
  - htdemucs_6s teacher for STFT mask targets
  - SemanticEncoder v1 on GT stem latents for sem conditioning
  - DistillDataset6 for 6-stem training data
"""
import argparse
import importlib.util
import math
import os
import re
import glob as _glob
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_dataset_6 import DistillDataset6, STEMS_6
from train_distill_small_v4cond import SmallAdditiveDemucsV4, MaskEncoder

STEMS = STEMS_6  # ["drums", "bass", "other", "vocals", "guitar", "piano"]

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
        stem_mono = stems.mean(dim=2)  # [B, 6, N]
        window = torch.hann_window(n_fft, device=mix.device)
        mags = []
        for i in range(6):
            spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                              return_complex=True).abs()
            mags.append(spec)
        mags = torch.stack(mags, dim=1)
        return mags / (mags.sum(dim=1, keepdim=True) + 1e-8)


def collate_6(batch):
    audios, stems, masks = [], [], []
    for a, s, m in batch:
        audios.append(a)
        stems.append(s)
        masks.append(m)
    return torch.stack(audios), torch.stack(stems), torch.stack(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/distill_small_v4cond_6stem_ckpts")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop_frames", type=int, default=200)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = DistillDataset6(crop_frames=args.crop_frames)
    if len(ds) == 0:
        print("ERROR: empty dataset")
        return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate_6, persistent_workers=args.workers > 0)

    print("[v4cond-6] loading frozen helpers...")
    sem_enc = load_sem_encoder()
    htdemucs = load_htdemucs_6s()
    print("[v4cond-6] all helpers loaded")

    model = SmallAdditiveDemucsV4(n_stems=6, hidden=args.hidden).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[v4cond-6] {n:.1f}M params (hidden={args.hidden}, 6 stems)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0

    # Auto-resume
    _ckpts = [f for f in _glob.glob(os.path.join(args.out, "*_step*.pt"))
              if re.search(r'step(\d+)', f)]
    if _ckpts:
        _latest = max(_ckpts, key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
        _sd = torch.load(_latest, map_location="cuda", weights_only=False)
        model.load_state_dict(_sd["model"])
        step = _sd.get("step", 0)
        for _ in range(step):
            sched.step()
        print(f"[resume] from step {step} ({os.path.basename(_latest)})")

    losses = []
    per_class = {c: [] for c in STEMS}
    t0 = time.time()

    while step < args.steps:
        for audio, stems, mask in loader:
            audio = audio.cuda(non_blocking=True).to(torch.bfloat16)
            stems = stems.cuda(non_blocking=True)  # [B, 6, 64, T] fp32
            mask = mask.cuda(non_blocking=True)     # [B, 6] presence mask
            B = audio.shape[0]

            with torch.no_grad():
                # Per-stem sem embeddings from GT stem latents
                stem_embs = []
                for si in range(6):
                    stem_lat = stems[:, si]  # [B, 64, T]
                    sem_out = sem_enc(stem_lat.float().transpose(1, 2))
                    stem_embs.append(sem_out["embedding"])  # [B, 128]
                sem_cond = torch.stack(stem_embs, dim=1)  # [B, 6, 128]

                # STFT masks from htdemucs_6s
                stft_masks = compute_stft_masks_6s(htdemucs, audio)  # [B, 6, F, T_stft]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(audio, sem_cond, stft_masks)  # [B, 6, 64, T]

            T = min(pred.shape[-1], stems.shape[-1])
            diff = (pred[..., :T].float() - stems[..., :T].float()).abs()

            # Apply presence mask: only compute loss on stems that are present
            # mask is [B, 6], expand to [B, 6, 64, T]
            mask_expanded = mask[:, :, None, None].expand_as(diff[..., :T])
            masked_diff = diff * mask_expanded

            # Per-stem L1 (averaged over present stems)
            per_ch = masked_diff.sum(dim=(0, 2, 3)) / (mask.sum(dim=0).clamp(min=1) *
                                                         diff.shape[2] * T)
            loss = per_ch.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            losses.append(loss.item())
            with torch.no_grad():
                for ci, c in enumerate(STEMS):
                    per_class[c].append(per_ch[ci].item())
            step += 1

            if step % args.log_every == 0:
                avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
                pc_str = " ".join(
                    f"{c[:3]}={sum(per_class[c][-30:])/max(1,len(per_class[c][-30:])):.3f}"
                    for c in STEMS)
                el = time.time() - t0
                print(f"[step {step:6d}] l1={avg:.4f} {pc_str} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed={el:.0f}s",
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"v4cond6_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args),
                            "loss": sum(losses[-100:]) / max(1, len(losses[-100:]))}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "v4cond6_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, final l1={sum(losses[-100:])/100:.4f}, saved {final}")


if __name__ == "__main__":
    main()
