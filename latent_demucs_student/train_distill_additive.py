#!/usr/bin/env python3
"""Train additive-conditioned 4-stem distillation.

Same as train_distill_sem.py but uses additive conditioning instead of FiLM.
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
from distill_model_additive import WaveformToStemLatentsAdditive
from distill_dataset import DistillDataset, collate, STEMS

SEM_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"


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
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/distill_additive_ckpts")
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

    ds = DistillDataset(crop_frames=args.crop_frames)
    if len(ds) == 0:
        print("ERROR: empty dataset"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[additive] loading frozen VAE encoder...")
    vae = load_vae_encoder()
    print("[additive] loading frozen sem encoder v1...")
    sem_enc = load_sem_encoder()

    print("[additive] building additive-conditioned model...")
    model = WaveformToStemLatentsAdditive(n_stems=4).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[additive] {n:.1f}M params")

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
            stems = stems.cuda(non_blocking=True)

            # Compute per-stem conditioning: VAE encode → sem encoder
            # Then broadcast the same mix embedding to all 4 stems
            with torch.no_grad():
                mix_latent = vae.encode(audio).latent_dist.sample()
                sem_out = sem_enc(mix_latent.float().transpose(1, 2))
                sem_emb = sem_out["embedding"]  # [B, 128]
                # Expand to [B, 4, 128] — same embedding for all stems
                cond = sem_emb.unsqueeze(1).expand(-1, 4, -1)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(audio, cond)  # [B, 4, 64, T]

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
                p = os.path.join(args.out, f"additive_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "additive_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, final l1={sum(losses[-100:])/100:.4f}, saved {final}")


if __name__ == "__main__":
    main()
