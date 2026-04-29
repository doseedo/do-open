#!/usr/bin/env python3
"""Train small additive demucs with oracle sem + STFT mask conditioning.

Conditioning computed from GT directly:
  1. Per-stem sem: SemanticEncoder v1 on GT stem latents → [B, 4, 128]
  2. STFT masks: ratio masks from htdemucs on mix → [B, 4, F, T]
     compressed to [B, 4, T, 32] via MaskEncoder

Combined: [B, 4, T, 160] frame-level conditioning.

This is the quality ceiling for the v4cond architecture — at inference,
SemDemucs v4 approximates these oracle signals from the mix waveform.
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
MASK_FEAT_DIM = 32
SEM_DIM = 128
COND_DIM = SEM_DIM + MASK_FEAT_DIM  # 160


class MaskEncoder(nn.Module):
    """Compress STFT masks [B*S, F, T] → [B*S, T, 32]."""
    def __init__(self, n_freqs=1025, out_dim=32):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv1d(n_freqs, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, out_dim, 1),
        )

    def forward(self, masks):
        B, S, Fr, T = masks.shape
        m = masks.reshape(B * S, Fr, T)
        h = self.compress(m)
        return h.transpose(1, 2).reshape(B, S, T, -1)


class SmallAdditiveDemucsV4(nn.Module):
    def __init__(self, n_stems=4, hidden=96, mults=(1,2,4,8,16),
                 cond_dim=COND_DIM, n_freqs=1025):
        super().__init__()
        self.n_stems = n_stems
        self.mask_encoder = MaskEncoder(n_freqs=n_freqs, out_dim=MASK_FEAT_DIM)

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

    def forward(self, waveform, sem_emb, stft_masks):
        h = self.encoder(waveform)
        B, C, T = h.shape

        mask_feat = self.mask_encoder(stft_masks)  # [B, 4, T_mask, 32]

        stems = []
        for i in range(self.n_stems):
            sem_i = sem_emb[:, i]  # [B, 128]
            mf_i = mask_feat[:, i]  # [B, T_mask, 32]

            mf_i_t = mf_i.transpose(1, 2)
            if mf_i_t.shape[-1] != T:
                mf_i_t = F.interpolate(mf_i_t, size=T, mode='linear', align_corners=False)
            mf_i = mf_i_t.transpose(1, 2)

            sem_broadcast = sem_i.unsqueeze(1).expand(-1, T, -1)
            cond_i = torch.cat([sem_broadcast, mf_i], dim=-1)

            bias = self.stem_bias[i](cond_i)
            h_mod = h + bias.transpose(1, 2)
            stems.append(self.stem_projs[i](h_mod))

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
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sd = torch.load(SEM_V1_CKPT, map_location="cpu", weights_only=False)
    m = mod.SemanticEncoderWithHeads().to(device).eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters(): p.requires_grad = False
    return m


def load_htdemucs(device="cuda"):
    from demucs.pretrained import get_model
    m = get_model('htdemucs').to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def compute_stft_masks(teacher, mix, n_fft=2048, hop=512):
    from demucs.apply import apply_model
    with torch.no_grad():
        stems = apply_model(teacher, mix.float(), device=str(mix.device))
        stem_mono = stems.mean(dim=2)
        window = torch.hann_window(n_fft, device=mix.device)
        mags = []
        for i in range(4):
            spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                              return_complex=True).abs()
            mags.append(spec)
        mags = torch.stack(mags, dim=1)
        return mags / (mags.sum(dim=1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/distill_small_v4cond_ckpts")
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
    if len(ds) == 0: print("ERROR: empty dataset"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[v4cond] loading frozen helpers...")
    vae = load_vae_encoder()
    sem_enc = load_sem_encoder()
    htdemucs = load_htdemucs()
    print("[v4cond] all helpers loaded")

    model = SmallAdditiveDemucsV4(n_stems=4, hidden=args.hidden).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[v4cond] {n:.1f}M params (hidden={args.hidden})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0
    # Auto-resume from latest checkpoint
    import re, glob as _glob
    _ckpts = [f for f in _glob.glob(os.path.join(args.out, "*_step*.pt"))
              if re.search(r'step(\d+)', f)]
    if _ckpts:
        _latest = max(_ckpts, key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
        _sd = torch.load(_latest, map_location="cuda", weights_only=False)
        model.load_state_dict(_sd["model"])
        step = _sd.get("step", 0)
        for _ in range(step): sched.step()
        print(f"[resume] from step {step} ({os.path.basename(_latest)})")
    losses = []; per_class = {c: [] for c in STEMS}; t0 = time.time()

    while step < args.steps:
        for audio, stems in loader:
            audio = audio.cuda(non_blocking=True).to(torch.bfloat16)
            stems = stems.cuda(non_blocking=True)

            with torch.no_grad():
                # Per-stem sem from GT stem latents
                stem_embs = []
                for si in range(4):
                    stem_lat = stems[:, si]  # [B, 64, T]
                    emb = sem_enc(stem_lat.float().transpose(1, 2))["embedding"]
                    stem_embs.append(emb)
                sem_cond = torch.stack(stem_embs, dim=1)  # [B, 4, 128]

                # STFT masks from htdemucs
                stft_masks = compute_stft_masks(htdemucs, audio)  # [B, 4, F, T_stft]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(audio, sem_cond, stft_masks)

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
                p = os.path.join(args.out, f'v4cond_step{step}.pt')
                torch.save({'step': step, 'model': model.state_dict(),
                            'hidden': args.hidden}, p)
                print(f'  -> saved {p}', flush=True)

            if step >= args.steps: break

    final = os.path.join(args.out, 'v4cond_final.pt')
    torch.save({'step': step, 'model': model.state_dict(), 'hidden': args.hidden}, final)
    print(f'[done] {step} steps, final l1={sum(losses[-100:])/100:.4f}')


if __name__ == "__main__":
    main()
