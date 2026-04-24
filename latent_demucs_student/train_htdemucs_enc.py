#!/usr/bin/env python3
"""Train HTDemucs-Enc: encoder-only, masks + embeddings from bottleneck.

No decoder. Pretrained htdemucs encoder (2.9M) + conv fusion (1.2M) +
mask head (1.0M) + embed head (0.1M) = 5.2M total.

Phase 1: freeze encoder, train fusion + heads
Phase 2: unfreeze encoder, fine-tune everything
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

STEMS = ["drums", "bass", "other", "vocals"]
SR = 44100

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


def load_htdemucs_teacher(model_name='htdemucs', device="cuda"):
    from demucs.pretrained import get_model
    m = get_model(model_name).to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def compute_teacher_masks(teacher, mix, n_stems=4, n_fft=2048, hop=512):
    from demucs.apply import apply_model
    with torch.no_grad():
        stems = apply_model(teacher, mix.float(), device=str(mix.device))
        stem_mono = stems.mean(dim=2)
        window = torch.hann_window(n_fft, device=mix.device)
        mags = []
        for i in range(n_stems):
            spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                              return_complex=True).abs()
            mags.append(spec)
        mags = torch.stack(mags, dim=1)
        return mags / (mags.sum(dim=1, keepdim=True) + 1e-8)


class StemDataset(Dataset):
    def __init__(self, crop_sec=6.0, seed=0):
        self.crop_samples = int(crop_sec * SR)
        self.rng = random.Random(seed)
        self.items = []
        musdb_wav = Path("/scratch/musdb18_wavs")
        musdb_lat = Path("/scratch/musdb18_latents")
        SPF = 1920
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
        print(f"[enc-ds] {len(self.items)} tracks")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            audio, sr = sf.read(it["mix_wav"], dtype="float32")
            audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio[None]).float()
            if audio.shape[0] == 1: audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2: audio = audio[:2]
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
        SPF = 1920
        crop_frames = self.crop_samples // SPF
        start_frame = start // SPF
        stem_lats = []
        for s in STEMS:
            raw = torch.load(it["stem_lats"][s], map_location="cpu", weights_only=False)
            z = raw["latents"] if isinstance(raw, dict) else raw
            if z.dim() == 2 and z.shape[0] == 64: z = z.t()
            z = z.float()[start_frame:start_frame + crop_frames]
            if z.shape[0] < crop_frames:
                z = F.pad(z, (0, 0, 0, crop_frames - z.shape[0]))
            stem_lats.append(z)
        return audio, torch.stack(stem_lats)


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/htdemucs_enc_ckpts")
    ap.add_argument("--n-stems", type=int, default=4)
    ap.add_argument("--model-name", default="htdemucs",
                    help="htdemucs or htdemucs_6s")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop-sec", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-finetune", type=float, default=3e-5)
    ap.add_argument("--unfreeze-step", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lambda-masks", type=float, default=2.0)
    ap.add_argument("--lambda-emb", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    if args.n_stems == 6:
        from distill_dataset_6 import DistillDataset6
        ds = DistillDataset6(crop_frames=int(args.crop_sec * 25))
        def collate_6(batch):
            return (torch.stack([b[0] for b in batch]),
                    torch.stack([b[1] for b in batch]),
                    torch.stack([b[2] for b in batch]))
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, drop_last=True,
                            collate_fn=collate_6, persistent_workers=args.workers > 0)
    else:
        ds = StemDataset(crop_sec=args.crop_sec)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=args.workers, drop_last=True,
                            collate_fn=collate, persistent_workers=args.workers > 0)
    if len(ds) == 0: print("ERROR: empty"); return

    print(f"[enc] loading frozen teachers (teacher={args.model_name})...")
    teacher = load_htdemucs_teacher(args.model_name)
    sem_enc = load_sem_encoder()
    print("[enc] teachers loaded")

    from htdemucs_enc import HTDemucsEnc
    model = HTDemucsEnc(n_stems=args.n_stems, model_name=args.model_name).cuda()
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[enc] {total_p:.1f}M total params")

    # Phase 1: freeze encoder
    frozen = [model.encoder, model.tencoder, model.freq_emb]
    for mod in frozen:
        for p in mod.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable) / 1e6
    print(f"[enc] phase 1: training {n_train:.1f}M params (fusion + heads), encoder frozen")

    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
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

    unfrozen = step >= args.unfreeze_step
    hist = {k: [] for k in ("total", "masks", "emb")}
    t0 = time.time()

    while step < args.steps:
        for batch in loader:
            if args.n_stems == 6:
                audio, stem_lats, presence = batch
                presence = presence.cuda(non_blocking=True)
            else:
                audio, stem_lats = batch
                presence = None
            audio = audio.cuda(non_blocking=True)
            stem_lats = stem_lats.cuda(non_blocking=True)
            B = audio.shape[0]

            if not unfrozen and step >= args.unfreeze_step:
                print(f"[enc] step {step}: unfreezing encoder, LR → {args.lr_finetune}")
                for mod in frozen:
                    for p in mod.parameters(): p.requires_grad = True
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr_finetune,
                                        betas=(0.9, 0.95), weight_decay=0.01)
                remaining = args.steps - step
                def lr_ft(s):
                    if s < 200: return (s+1)/200
                    return 0.5 * (1 + math.cos((s-200)/max(1,remaining-200) * math.pi))
                sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_ft)
                unfrozen = True

            with torch.no_grad():
                tgt_masks = compute_teacher_masks(teacher, audio, n_stems=args.n_stems)
                tgt_embs = []
                for si in range(args.n_stems):
                    lat = stem_lats[:, si]  # [B, 64, T] or [B, T, 64]
                    # sem_enc expects [B, T, 64]
                    if lat.shape[1] == 64:
                        lat = lat.transpose(1, 2)
                    tgt_embs.append(sem_enc(lat)["embedding"])
                tgt_emb = torch.stack(tgt_embs, dim=1)

            opt.zero_grad(set_to_none=True)
            out = model(audio.float())

            pred_masks = out["stft_masks"]
            F_m = min(pred_masks.shape[2], tgt_masks.shape[2])
            T_m = min(pred_masks.shape[3], tgt_masks.shape[3])
            l_masks = F.mse_loss(pred_masks[:, :, :F_m, :T_m],
                                 tgt_masks[:, :, :F_m, :T_m])
            if presence is not None:
                diff_e = (out["embedding"] - tgt_emb).abs()
                mask_e = presence[:, :, None].expand_as(diff_e)
                l_emb = (diff_e * mask_e).sum() / mask_e.sum().clamp(min=1)
            else:
                l_emb = F.l1_loss(out["embedding"], tgt_emb)

            loss = args.lambda_masks * l_masks + args.lambda_emb * l_emb
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            hist["total"].append(loss.item())
            hist["masks"].append(l_masks.item())
            hist["emb"].append(l_emb.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                phase = "ft" if unfrozen else "frozen"
                print(f"[step {step:5d}] total={avg(hist['total']):.4f} "
                      f"masks={avg(hist['masks']):.4f} "
                      f"emb={avg(hist['emb']):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"({phase}) elapsed={time.time()-t0:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"enc_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps: break

    final = os.path.join(args.out, "enc_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
