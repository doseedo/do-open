#!/usr/bin/env python3
"""Train small additive demucs with per-stem SemanticEncoder v1 conditioning.

Same as small_add4 (hidden=96, 43M) but with TRUE per-stem [B, 4, 128]
conditioning from SemanticEncoder v1 on GT stem latents.

small_add4 got L1=0.50 with broadcast mix embedding.
perstem (full 103M) is at 0.369 eval at step 2.5K.
Question: can the small model close the gap with per-stem conditioning?
"""
import argparse
import importlib.util
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers.models.autoencoders.autoencoder_oobleck import OobleckEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_dataset import DistillDataset, collate, STEMS

SEM_ENCODER_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_V1_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"

LATENT_DIM = 64


class SmallAdditiveDemucs(nn.Module):
    def __init__(self, n_stems=4, hidden=96, mults=(1,2,4,8,16), cond_dim=128):
        super().__init__()
        self.n_stems = n_stems
        self.encoder = OobleckEncoder(
            encoder_hidden_size=hidden, audio_channels=2,
            downsampling_ratios=[2,4,4,4,4], channel_multiples=list(mults))
        in_ch = hidden * mults[-1]
        self.encoder.conv2 = nn.Identity()
        self.stem_bias = nn.ModuleList([
            nn.Sequential(nn.Linear(cond_dim, in_ch), nn.GELU(), nn.Linear(in_ch, in_ch))
            for _ in range(n_stems)])
        for b in self.stem_bias:
            nn.init.zeros_(b[-1].weight); nn.init.zeros_(b[-1].bias)
        self.stem_projs = nn.ModuleList([
            nn.Conv1d(in_ch, LATENT_DIM, 3, padding=1) for _ in range(n_stems)])

    def forward(self, waveform, cond):
        """waveform: [B, 2, N], cond: [B, n_stems, 128] → [B, n_stems, 64, T]"""
        h = self.encoder(waveform)
        B, C, T = h.shape
        stems = []
        for i in range(self.n_stems):
            c = cond[:, i]  # [B, 128]
            bias = self.stem_bias[i](c).unsqueeze(-1)
            stems.append(self.stem_projs[i](h + bias))
        return torch.stack(stems, dim=1)


def load_vae_encoder(device="cuda"):
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae").to(device).eval().to(torch.bfloat16)
    for p in vae.parameters(): p.requires_grad = False
    return vae


def load_sem_encoder(device="cuda"):
    spec = importlib.util.spec_from_file_location(
        "sem", os.path.join(SEM_ENCODER_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sd = torch.load(SEM_V1_CKPT, map_location="cpu", weights_only=False)
    m = mod.SemanticEncoderWithHeads().to(device).eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters(): p.requires_grad = False
    print(f"[small-perstem] loaded sem encoder v1 (step {sd['step']})")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/distill_small_perstem_ckpts")
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

    ds = DistillDataset(crop_frames=args.crop_frames)
    if len(ds) == 0:
        print("ERROR: empty dataset"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[small-perstem] loading frozen VAE encoder + sem encoder...")
    vae = load_vae_encoder()
    sem_enc = load_sem_encoder()

    model = SmallAdditiveDemucs(n_stems=4, hidden=args.hidden).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[small-perstem] {n:.1f}M params (hidden={args.hidden})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0; losses = []; per_class = {c: [] for c in STEMS}; t0 = time.time()

    while step < args.steps:
        for audio, stems in loader:
            audio = audio.cuda(non_blocking=True).to(torch.bfloat16)
            stems = stems.cuda(non_blocking=True)

            with torch.no_grad():
                stem_embs = []
                for si in range(4):
                    stem_lat = stems[:, si]  # [B, 64, T]
                    emb = sem_enc(stem_lat.float().transpose(1, 2))["embedding"]
                    stem_embs.append(emb)
                cond = torch.stack(stem_embs, dim=1)  # [B, 4, 128]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(audio, cond)

            T = min(pred.shape[-1], stems.shape[-1])
            diff = (pred[..., :T].float() - stems[..., :T].float()).abs()
            per_ch = diff.mean(dim=(0, 2, 3))
            loss = per_ch.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            losses.append(loss.item())
            with torch.no_grad():
                for ci, c in enumerate(STEMS):
                    per_class[c].append(per_ch[ci].item())
            step += 1

            if step % args.log_every == 0:
                avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
                ps = ' '.join(f'{c}={sum(per_class[c][-30:])/max(1,len(per_class[c][-30:])):.3f}'
                              for c in STEMS)
                print(f'[step {step:6d}] l1={avg:.4f} {ps} '
                      f'lr={sched.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.0f}s',
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f'small_perstem_step{step}.pt')
                torch.save({'step': step, 'model': model.state_dict(),
                            'hidden': args.hidden, 'n_stems': 4,
                            'loss': sum(losses[-100:])/max(1, len(losses[-100:]))}, p)
                print(f'  -> saved {p}', flush=True)

            if step >= args.steps: break

    final = os.path.join(args.out, 'small_perstem_final.pt')
    torch.save({'step': step, 'model': model.state_dict(),
                'hidden': args.hidden, 'n_stems': 4}, final)
    print(f'[done] {step} steps, final l1={sum(losses[-100:])/100:.4f}')


if __name__ == "__main__":
    main()
