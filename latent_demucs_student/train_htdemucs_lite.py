#!/usr/bin/env python3
"""Train HTDemucs-Lite: htdemucs with attention replaced by conv fusion.

Uses pretrained htdemucs encoder/decoder weights (9.6M convs).
Trains only the conv fusion (~1.2M) and embedding head (~0.1M).
Teacher: frozen htdemucs for stem separation targets + SemanticEncoder for embeddings.

Phase 1: freeze encoder/decoder, train fusion + embed only
Phase 2: unfreeze everything, fine-tune end-to-end at lower LR
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
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from htdemucs_lite import HTDemucsLite

STEMS = ["drums", "bass", "other", "vocals"]  # htdemucs order
SR = 44100  # htdemucs native sample rate

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


def load_htdemucs_teacher(device="cuda"):
    from demucs.pretrained import get_model
    m = get_model('htdemucs').to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def load_vae_encoder(device="cuda"):
    """Load frozen oobleck VAE encoder for computing latents from waveforms."""
    from audiocraft.models.builders import get_audiocraft_model
    vae_path = "/scratch/ACE-Step-1.5/checkpoints/vae"
    if os.path.exists(vae_path):
        from audiocraft.models.loaders import load_compression_model
        vae = load_compression_model(vae_path).to(device).eval().to(torch.bfloat16)
    else:
        # Fallback: try the oobleck directly
        raise FileNotFoundError(f"VAE not found at {vae_path}")
    for p in vae.parameters(): p.requires_grad = False
    return vae


class StemDataset(Dataset):
    """Load mix waveforms + per-stem VAE latents from MUSDB."""
    def __init__(self, crop_sec=6.0, seed=0):
        self.crop_samples = int(crop_sec * SR)
        self.rng = random.Random(seed)
        self.items = []

        musdb_wav = Path("/scratch/musdb18_wavs")
        musdb_lat = Path("/scratch/musdb18_latents")
        for split in ["train", "test"]:
            for td in sorted((musdb_lat / split).iterdir()):
                if not td.is_dir():
                    continue
                mix_wav = musdb_wav / split / td.name / "mixture.wav"
                if not mix_wav.exists():
                    continue
                stem_lats = {}
                stems_musdb = ["drums", "bass", "vocals", "other"]
                for s in stems_musdb:
                    p = td / f"{s}.vae.pt"
                    if p.exists():
                        stem_lats[s] = str(p)
                if len(stem_lats) != 4:
                    continue
                self.items.append({"mix_wav": str(mix_wav), "stem_lats": stem_lats})
        print(f"[lite-ds] {len(self.items)} tracks")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            audio, sr = sf.read(it["mix_wav"], dtype="float32")
            audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio[None]).float()
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]
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

        # Load stem latents in htdemucs order: drums, bass, other, vocals
        SPF = 1920
        crop_frames = self.crop_samples // SPF
        start_frame = start // SPF
        stem_lats = []
        for s in STEMS:  # htdemucs order
            raw = torch.load(it["stem_lats"][s], map_location="cpu", weights_only=False)
            z = raw["latents"] if isinstance(raw, dict) else raw
            if z.dim() == 2 and z.shape[0] == 64:
                z = z.t()
            z = z.float()
            z = z[start_frame:start_frame + crop_frames]
            if z.shape[0] < crop_frames:
                z = F.pad(z, (0, 0, 0, crop_frames - z.shape[0]))
            stem_lats.append(z)

        return audio, torch.stack(stem_lats)


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/htdemucs_lite_ckpts")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop-sec", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-finetune", type=float, default=3e-5,
                    help="LR after unfreezing encoder/decoder")
    ap.add_argument("--unfreeze-step", type=int, default=3000,
                    help="Step to unfreeze encoder/decoder for fine-tuning")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lambda-stems", type=float, default=1.0)
    ap.add_argument("--lambda-emb", type=float, default=1.0)
    ap.add_argument("--lambda-masks", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = StemDataset(crop_sec=args.crop_sec)
    if len(ds) == 0:
        print("ERROR: empty dataset")
        return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[lite] loading frozen teachers...")
    teacher = load_htdemucs_teacher()
    sem_enc = load_sem_encoder()
    print("[lite] teachers loaded")

    print("[lite] building HTDemucsLite...")
    model = HTDemucsLite(embed_dim=128).cuda()
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    fusion_p = sum(p.numel() for p in model.conv_fusion.parameters()) / 1e6
    embed_p = (sum(p.numel() for p in model.embed_proj.parameters()) +
               model.stem_queries.numel()) / 1e6
    print(f"[lite] {total_p:.1f}M total ({fusion_p:.1f}M fusion, {embed_p:.1f}M embed)")

    # Phase 1: freeze encoder/decoder, train only fusion + embed
    frozen_modules = [model.encoder, model.decoder, model.tencoder, model.tdecoder,
                      model.freq_emb]
    for mod in frozen_modules:
        for p in mod.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable) / 1e6
    print(f"[lite] phase 1: training {n_train:.1f}M params (fusion + embed), "
          f"encoder/decoder frozen")

    opt = torch.optim.AdamW(trainable, lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(prog * math.pi))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

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

    unfrozen = step >= args.unfreeze_step
    hist = {k: [] for k in ("total", "stems", "emb", "masks")}
    t0 = time.time()

    while step < args.steps:
        for audio, stem_lats in loader:
            audio = audio.cuda(non_blocking=True)
            stem_lats = stem_lats.cuda(non_blocking=True)  # [B, 4, T_lat, 64]
            B = audio.shape[0]

            # Phase 2: unfreeze encoder/decoder at unfreeze_step
            if not unfrozen and step >= args.unfreeze_step:
                print(f"[lite] step {step}: unfreezing encoder/decoder, "
                      f"LR → {args.lr_finetune}")
                for mod in frozen_modules:
                    for p in mod.parameters():
                        p.requires_grad = True
                # Rebuild optimizer with all params at lower LR
                all_params = list(model.parameters())
                opt = torch.optim.AdamW(all_params, lr=args.lr_finetune,
                                        betas=(0.9, 0.95), weight_decay=0.01)
                # Reset scheduler for fine-tune phase
                remaining = args.steps - step
                def lr_lambda_ft(s):
                    if s < 200:
                        return (s + 1) / 200
                    prog = (s - 200) / max(1, remaining - 200)
                    return 0.5 * (1 + math.cos(prog * math.pi))
                sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_ft)
                unfrozen = True
                n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
                print(f"[lite] phase 2: training {n_train:.1f}M params (all)")

            # Teacher targets
            with torch.no_grad():
                # Stem separation targets from htdemucs teacher
                from demucs.apply import apply_model
                tgt_stems = apply_model(teacher, audio.float(),
                                        device=str(audio.device))  # [B, 4, 2, N]

                # STFT mask targets from teacher stems
                stem_mono = tgt_stems.mean(dim=2)  # [B, 4, N]
                n_fft, hop = 2048, 512
                window = torch.hann_window(n_fft, device=audio.device)
                mags = []
                for i in range(4):
                    spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                                      return_complex=True).abs()
                    mags.append(spec)
                mags_t = torch.stack(mags, dim=1)
                tgt_masks = mags_t / (mags_t.sum(dim=1, keepdim=True) + 1e-8)

                # Embedding targets from SemanticEncoder on GT stem latents
                tgt_embs = []
                for si in range(4):
                    lat = stem_lats[:, si]  # [B, T_lat, 64]
                    tgt_embs.append(sem_enc(lat)["embedding"])  # [B, 128]
                tgt_emb = torch.stack(tgt_embs, dim=1)  # [B, 4, 128]

            # Student forward
            opt.zero_grad(set_to_none=True)
            out = model(audio.float())

            # Stem separation loss (L1 on waveforms)
            pred_stems = out["stems"]
            T_s = min(pred_stems.shape[-1], tgt_stems.shape[-1])
            l_stems = F.l1_loss(pred_stems[..., :T_s], tgt_stems[..., :T_s])

            # Embedding loss
            l_emb = F.l1_loss(out["embedding"], tgt_emb)

            # STFT mask loss
            pred_masks = out["stft_masks"]
            F_m = min(pred_masks.shape[2], tgt_masks.shape[2])
            T_m = min(pred_masks.shape[3], tgt_masks.shape[3])
            l_masks = F.mse_loss(pred_masks[:, :, :F_m, :T_m],
                                 tgt_masks[:, :, :F_m, :T_m])

            loss = (args.lambda_stems * l_stems +
                    args.lambda_emb * l_emb +
                    args.lambda_masks * l_masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            hist["total"].append(loss.item())
            hist["stems"].append(l_stems.item())
            hist["emb"].append(l_emb.item())
            hist["masks"].append(l_masks.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                phase = "ft" if unfrozen else "frozen"
                print(f"[step {step:5d}] total={avg(hist['total']):.4f} "
                      f"stems={avg(hist['stems']):.4f} "
                      f"emb={avg(hist['emb']):.4f} "
                      f"masks={avg(hist['masks']):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"({phase}) elapsed={time.time()-t0:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"lite_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "lite_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
