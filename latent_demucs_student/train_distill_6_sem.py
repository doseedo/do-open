#!/usr/bin/env python3
"""Train sem-conditioned 6-stem distillation: mix waveform + sem_emb → 6 stem latents.

Same as train_distill_6.py but model is conditioned on frozen sem encoder v1.
Pipeline per batch:
  1. waveform → frozen VAE encoder → mix latent
  2. mix latent → frozen sem encoder → 128-dim embedding
  3. waveform + sem_emb → student → 6 stem latents
  4. Masked L1 loss vs GT stem latents
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
from distill_dataset_6 import DistillDataset6, collate, STEMS_6

SEM_DEMUCS_CKPT = "/scratch/latent_demucs_student/sem_demucs_v2_ckpts/sem_demucs_v2_final.pt"

# Fallback: v1 sem encoder via VAE round-trip (used until SemDemucs v2 is trained)
SEM_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"


def load_sem_demucs(device="cuda"):
    """Load frozen SemDemucs v2 for direct waveform → per-stem conditioning."""
    from sem_demucs import SemDemucs
    if not os.path.exists(SEM_DEMUCS_CKPT):
        return None
    sd = torch.load(SEM_DEMUCS_CKPT, map_location="cpu", weights_only=False)
    m = SemDemucs().to(device).eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters():
        p.requires_grad = False
    print(f"[6sem] loaded SemDemucs v2 (step {sd['step']})")
    return m


def load_vae_encoder(device="cuda"):
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae").to(device).eval().to(torch.bfloat16)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_sem_encoder(device="cuda"):
    spec = importlib.util.spec_from_file_location(
        "sem_model", os.path.join(SEM_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sd = torch.load(SEM_CKPT, map_location="cpu", weights_only=False)
    m = mod.SemanticEncoderWithHeads().to(device).eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters():
        p.requires_grad = False
    print(f"[6sem] loaded sem encoder v1 (step {sd['step']})")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/distill6_sem_ckpts")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop_frames", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = DistillDataset6(crop_frames=args.crop_frames)
    if len(ds) == 0:
        print("ERROR: empty dataset"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    # Try SemDemucs v2 first (direct waveform → conditioning, no round-trip)
    sem_demucs = load_sem_demucs()
    vae = None
    sem_enc = None
    if sem_demucs is None:
        print("[6sem] SemDemucs v2 not ready, using VAE round-trip fallback")
        print("[6sem] loading frozen VAE encoder...")
        vae = load_vae_encoder()
        print("[6sem] loading frozen sem encoder...")
        sem_enc = load_sem_encoder()
    else:
        print("[6sem] using SemDemucs v2 for conditioning (no VAE round-trip)")

    print("[6sem] building sem-conditioned model (n_stems=6)...")
    model = WaveformToStemLatentsSemCond(n_stems=6).cuda()
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[6sem] {n_total:.1f}M total, {n_train:.1f}M trainable")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0
    losses = []
    per_stem = {s: [] for s in STEMS_6}
    t0 = time.time()

    while step < args.steps:
        for audio, stems, mask in loader:
            audio = audio.cuda(non_blocking=True).to(torch.bfloat16)
            stems = stems.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            # Compute sem embedding from mix
            with torch.no_grad():
                if sem_demucs is not None:
                    # Direct path: waveform → SemDemucs v2 → per-stem embeddings
                    sd_out = sem_demucs(audio.float())
                    # Average the 4 (or 6) stem embeddings into one mix embedding
                    # The separator FiLM uses a single 128-dim vector
                    sem_emb = sd_out["embedding"].mean(dim=1)  # [B, 128]
                else:
                    # Fallback: VAE encode → sem encoder round-trip
                    mix_latent = vae.encode(audio).latent_dist.sample()
                    sem_out = sem_enc(mix_latent.float().transpose(1, 2))
                    sem_emb = sem_out["embedding"]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(audio, sem_emb)

            T = min(pred.shape[-1], stems.shape[-1])
            pred = pred[..., :T]
            stems = stems[..., :T]
            diff = (pred.float() - stems.float()).abs()
            per_ch = diff.mean(dim=(2, 3))  # [B, 6]
            denom = mask.sum().clamp_min(1.0)
            loss = (per_ch * mask).sum() / denom

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()

            losses.append(loss.item())
            with torch.no_grad():
                for ci, c in enumerate(STEMS_6):
                    m = mask[:, ci]
                    if m.sum() > 0:
                        per_stem[c].append((per_ch[:, ci] * m).sum().item() / m.sum().item())
            step += 1

            if step % args.log_every == 0:
                avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
                ps = " ".join(
                    f"{c}={sum(per_stem[c][-30:])/max(1,len(per_stem[c][-30:])):.3f}"
                    for c in STEMS_6
                )
                el = time.time() - t0
                print(f"[step {step:6d}] l1={avg:.4f} {ps} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed={el:.0f}s",
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"distill6_sem_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "distill6_sem_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
