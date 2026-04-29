#!/usr/bin/env python3
"""Train the semantic-conditioned waveform → 4-stem-latents model.

Same as train_distill.py but the model is conditioned on a frozen
semantic encoder embedding. The mix waveform is VAE-encoded to get
the latent, which is then run through the semantic encoder to get
a 128-dim conditioning vector.

Pipeline per batch:
  1. waveform [B, 2, N] → frozen VAE encoder → mix_latent [B, 64, T]
  2. mix_latent → frozen sem encoder → sem_emb [B, 128]
  3. waveform + sem_emb → student model → pred_stems [B, 4, 64, T]
  4. L1 loss vs teacher stem latents
"""
import argparse
import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_model import WaveformToStemLatentsSemCond
from distill_dataset import DistillDataset, collate, STEMS

SEM_ENCODER_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_V1_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"


def load_vae_encoder(device="cuda"):
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae").to(device).eval().to(torch.bfloat16)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_sem_encoder(device="cuda"):
    spec = importlib.util.spec_from_file_location(
        "sem_model_enc", os.path.join(SEM_ENCODER_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sd = torch.load(SEM_V1_CKPT, map_location="cpu", weights_only=False)
    m = mod.SemanticEncoderWithHeads().to(device).eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters():
        p.requires_grad = False
    print(f"[sem-distill] loaded sem encoder v1 (step {sd['step']})")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/distill_sem_ckpts")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop_frames", type=int, default=200)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = DistillDataset(crop_frames=args.crop_frames)
    if len(ds) == 0:
        print("ERROR: empty dataset")
        return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[sem-distill] loading frozen VAE encoder...")
    vae = load_vae_encoder()

    print("[sem-distill] loading frozen sem encoder v1...")
    sem_enc = load_sem_encoder()

    print("[sem-distill] building model...")
    model = WaveformToStemLatentsSemCond(n_stems=4).cuda()
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[sem-distill] {n_total:.1f}M total, {n_train:.1f}M trainable")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"[sem-distill] resumed from {args.resume}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0
    losses = []
    per_class = {c: [] for c in STEMS}
    t0 = time.time()

    while step < args.steps:
        for audio, stems in loader:
            audio = audio.cuda(non_blocking=True).to(torch.bfloat16)
            stems = stems.cuda(non_blocking=True)  # fp32 target

            # Compute semantic embedding from mix
            with torch.no_grad():
                mix_latent = vae.encode(audio).latent_dist.sample()  # [B, 64, T]
                # sem encoder expects [B, T, 64]
                sem_out = sem_enc(mix_latent.float().transpose(1, 2))
                sem_emb = sem_out["embedding"]  # [B, 128]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(audio, sem_emb)  # [B, 4, 64, T]

            # Align T
            T = min(pred.shape[-1], stems.shape[-1])
            pred = pred[..., :T]
            stems = stems[..., :T]
            diff = (pred.float() - stems.float()).abs()
            per_ch = diff.mean(dim=(0, 2, 3))
            loss = per_ch.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
                    f"{c}={sum(per_class[c][-30:])/max(1,len(per_class[c][-30:])):.3f}"
                    for c in STEMS
                )
                el = time.time() - t0
                print(f"[step {step:6d}] l1={avg:.4f} {pc_str} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed={el:.0f}s",
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"distill_sem_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args),
                            "loss": sum(losses[-100:])/max(1, len(losses[-100:]))}, p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "distill_sem_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, final l1={sum(losses[-100:])/100:.4f}, saved {final}")


if __name__ == "__main__":
    main()
