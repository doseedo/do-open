#!/usr/bin/env python3
"""Train mask + embedding heads on frozen htdemucs encoder+transformer.

Targets:
  - Masks: STFT ratio masks from full htdemucs output (run once, cached)
  - Embeddings: SemanticEncoder v1 on GT stem latents + mix latent

The full htdemucs decoder runs only to generate mask targets. At inference,
only the encoder+transformer+heads run — no decoder, no iSTFT needed from
the model (browser does its own iSTFT on mix × mask).
"""
import argparse
import importlib.util
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from htdemucs_sem import HTDemucsHeads

STEMS = ["drums", "bass", "other", "vocals"]  # htdemucs order
SR = 48000
SPF = 1920

SEM_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_V1_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"


def load_vae_encoder(device="cuda"):
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae").to(device).eval().to(torch.bfloat16)
    for p in vae.parameters(): p.requires_grad = False
    return vae


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


def load_htdemucs_teacher(device="cuda"):
    """Full htdemucs for generating mask targets."""
    from demucs.pretrained import get_model
    m = get_model('htdemucs').models[0].to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def compute_stft_masks(teacher, mix, n_fft=4096, hop=1024):
    """Run full htdemucs, compute STFT ratio masks from output stems."""
    with torch.no_grad():
        stems = teacher(mix)  # [B, 4, 2, N]
        stem_mono = stems.mean(dim=2)  # [B, 4, N]
        window = torch.hann_window(n_fft, device=mix.device)
        mags = []
        for i in range(4):
            spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                              return_complex=True).abs()
            mags.append(spec)
        mags = torch.stack(mags, dim=1)  # [B, 4, F, T_stft]
        masks = mags / (mags.sum(dim=1, keepdim=True) + 1e-8)
    return masks  # [B, 4, F, T_stft]


def _load_stem_wav(path):
    audio, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(audio.T if audio.ndim > 1 else audio[None]).float()
    if t.shape[0] == 1: t = t.repeat(2, 1)
    elif t.shape[0] > 2: t = t[:2]
    return t


class MaskEmbDataset(Dataset):
    def __init__(self, crop_sec=6.0, seed=0):
        self.crop_samples = int(crop_sec * SR)
        self.rng = random.Random(seed)
        self.items = []

        musdb_wav = Path("/scratch/musdb18_wavs")
        musdb_lat = Path("/scratch/musdb18_latents")
        for split in ["train", "test"]:
            for td in sorted((musdb_lat / split).iterdir()):
                if not td.is_dir(): continue
                mix_wav = musdb_wav / split / td.name / "mixture.wav"
                if not mix_wav.exists(): continue
                stem_lats = {}
                for s in STEMS:
                    p = td / f"{s}.vae.pt"
                    if p.exists(): stem_lats[s] = str(p)
                if len(stem_lats) != 4: continue
                self.items.append({"mix_wav": str(mix_wav), "stem_lats": stem_lats})
        print(f"[ds] {len(self.items)} tracks")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            audio = _load_stem_wav(it["mix_wav"])
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        N = audio.shape[1]
        if N > self.crop_samples:
            start = self.rng.randint(0, N - self.crop_samples)
            audio = audio[:, start:start + self.crop_samples]
        elif N < self.crop_samples:
            audio = F.pad(audio, (0, self.crop_samples - N))
            start = 0
        else:
            start = 0

        crop_frames = self.crop_samples // SPF
        start_frame = start // SPF
        stem_lats = []
        for s in STEMS:
            raw = torch.load(it["stem_lats"][s], map_location="cpu", weights_only=False)
            z = raw["latents"] if isinstance(raw, dict) else raw
            if z.dim() == 2 and z.shape[0] == 64: z = z.t()
            z = z[start_frame:start_frame + crop_frames].float()
            if z.shape[0] < crop_frames:
                z = F.pad(z, (0, 0, 0, crop_frames - z.shape[0]))
            stem_lats.append(z)

        return audio, torch.stack(stem_lats)


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/htdemucs_sem_ckpts")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop-sec", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lambda-mask", type=float, default=1.0)
    ap.add_argument("--lambda-stem-emb", type=float, default=1.0)
    ap.add_argument("--lambda-mix-emb", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = MaskEmbDataset(crop_sec=args.crop_sec)
    if len(ds) == 0: print("ERROR: no data"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[train] loading frozen helpers...")
    vae = load_vae_encoder()
    sem_enc = load_sem_encoder()
    teacher = load_htdemucs_teacher()
    print("[train] all frozen models loaded")

    model = HTDemucsHeads().cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable) / 1e6
    print(f"[train] trainable: {n_train:.2f}M")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    def lr_lambda(step):
        if step < args.warmup: return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(prog * math.pi))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    step = 0
    hist = {k: [] for k in ("total", "mask", "stem_emb", "mix_emb")}
    t0 = time.time()

    while step < args.steps:
        for audio, stem_lats in loader:
            audio = audio.cuda(non_blocking=True)
            stem_lats = stem_lats.cuda(non_blocking=True)
            B = audio.shape[0]

            # Target masks from full htdemucs teacher
            tgt_masks = compute_stft_masks(teacher, audio)  # [B, 4, F, T_stft]

            # Target embeddings from sem encoder on stems + mix
            with torch.no_grad():
                tgt_stem_embs = []
                for si in range(4):
                    emb = sem_enc(stem_lats[:, si])["embedding"]
                    tgt_stem_embs.append(emb)
                tgt_stem = torch.stack(tgt_stem_embs, dim=1)

                mix_lat = vae.encode(audio.to(torch.bfloat16)).latent_dist.sample()
                tgt_mix = sem_enc(mix_lat.float().transpose(1, 2))["embedding"]

            # Student forward
            out = model(audio)

            # Mask loss: align shapes then MSE
            pred_masks = out["stft_masks"]
            T_m = min(pred_masks.shape[-1], tgt_masks.shape[-1])
            F_m = min(pred_masks.shape[-2], tgt_masks.shape[-2])
            l_mask = F.mse_loss(pred_masks[:, :, :F_m, :T_m], tgt_masks[:, :, :F_m, :T_m])

            # Embedding losses
            l_stem = F.l1_loss(out["embedding"], tgt_stem)
            l_mix = F.l1_loss(out["mix_embedding"], tgt_mix)

            loss = (args.lambda_mask * l_mask +
                    args.lambda_stem_emb * l_stem +
                    args.lambda_mix_emb * l_mix)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()

            hist["total"].append(loss.item())
            hist["mask"].append(l_mask.item())
            hist["stem_emb"].append(l_stem.item())
            hist["mix_emb"].append(l_mix.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                print(f"[step {step:5d}] total={avg(hist['total']):.4f} "
                      f"mask={avg(hist['mask']):.4f} "
                      f"stem_emb={avg(hist['stem_emb']):.4f} "
                      f"mix_emb={avg(hist['mix_emb']):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

            if step % args.save_every == 0:
                trainable_sd = {k: v for k, v in model.state_dict().items()
                                if any(k.startswith(p) for p in
                                       ["mask_head", "stem_queries", "stem_embed_head",
                                        "mix_query", "mix_embed_head"])}
                p = os.path.join(args.out, f"htdemucs_heads_step{step}.pt")
                torch.save({"step": step, "trainable_sd": trainable_sd,
                            "args": vars(args)}, p)
                print(f"  -> saved {p} ({os.path.getsize(p)/1e6:.1f}MB)", flush=True)

            if step >= args.steps: break

    trainable_sd = {k: v for k, v in model.state_dict().items()
                    if any(k.startswith(p) for p in
                           ["mask_head", "stem_queries", "stem_embed_head",
                            "mix_query", "mix_embed_head"])}
    final = os.path.join(args.out, "htdemucs_heads_final.pt")
    torch.save({"step": step, "trainable_sd": trainable_sd, "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
