#!/usr/bin/env python3
"""Train LatentMaskRefiner: (latent + noisy mask) → refined mask.

Uses:
  - Frozen v4-small (provides noisy masks)
  - Frozen v4cond (provides clean stem latents from predicted conditioning)
  - GT stem audio → STFT ratio mask (target)

Target mask: for each stem, compute its STFT magnitude and divide by
the sum over all stems → per-bin ratio mask in [0, 1]. Same as what
v4cond/htdemucs_teacher uses.
"""
import argparse
import glob
import importlib.util
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sem_demucs import SemDemucs
from latent_mask_refiner import LatentMaskRefiner

STEMS = ["drums", "bass", "other", "vocals"]
STEMS_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]
SR = 48000
SPF = 1920  # samples per latent frame @ 25Hz
N_FFT = 2048
HOP = 512


def find_latest_ckpt(pattern):
    ckpts = [f for f in glob.glob(pattern) if re.search(r'step(\d+)', f)]
    if not ckpts:
        return None
    return max(ckpts, key=lambda x: int(re.search(r'step(\d+)', x).group(1)))


class MixStemDataset(Dataset):
    """Load mix audio + per-stem audio (for computing true stem masks)."""
    def __init__(self, crop_frames=200, seed=0, n_stems=4):
        self.crop_samples = crop_frames * SPF
        self.rng = random.Random(seed)
        self.n_stems = n_stems
        self.stems = STEMS_6 if n_stems == 6 else STEMS
        self.items = []

        musdb_wav = Path("/scratch/musdb18_wavs")
        for split in ["train", "test"]:
            for td in sorted((musdb_wav / split).iterdir()):
                if not td.is_dir():
                    continue
                mix = td / "mixture.wav"
                if not mix.exists():
                    continue
                stem_wavs = {}
                for s in self.stems:
                    p = td / f"{s}.wav"
                    if p.exists():
                        stem_wavs[s] = str(p)
                if len(stem_wavs) < 2:
                    continue
                self.items.append({"mix": str(mix), "stems": stem_wavs})
        print(f"[refiner-ds] {len(self.items)} tracks ({n_stems} stems)")

    def __len__(self):
        return len(self.items)

    def _load_audio(self, path):
        audio, sr = sf.read(path, dtype="float32")
        audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio[None]).float()
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]
        return audio

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            mix = self._load_audio(it["mix"])
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        N = mix.shape[1]
        if N > self.crop_samples:
            start = self.rng.randint(0, N - self.crop_samples)
        else:
            start = 0
        mix = mix[:, start:start + self.crop_samples] if N > self.crop_samples else \
              F.pad(mix, (0, self.crop_samples - N))

        # Load per-stem audio, fill zero if missing
        stem_audios = []
        presence = []
        for s in self.stems:
            if s in it["stems"]:
                try:
                    sa = self._load_audio(it["stems"][s])
                    sa = sa[:, start:start + self.crop_samples] if sa.shape[1] > self.crop_samples else \
                         F.pad(sa, (0, self.crop_samples - sa.shape[1]))
                    stem_audios.append(sa)
                    presence.append(1.0)
                except Exception:
                    stem_audios.append(torch.zeros(2, self.crop_samples))
                    presence.append(0.0)
            else:
                stem_audios.append(torch.zeros(2, self.crop_samples))
                presence.append(0.0)

        stems_t = torch.stack(stem_audios)  # [S, 2, N]
        return mix, stems_t, torch.tensor(presence)


def collate(batch):
    mix = torch.stack([b[0] for b in batch])
    stems = torch.stack([b[1] for b in batch])
    presence = torch.stack([b[2] for b in batch])
    return mix, stems, presence


def compute_gt_masks(stem_audios, n_fft=N_FFT, hop=HOP):
    """stem_audios: [B, S, 2, N] → [B, S, F, T_stft] ratio masks."""
    B, S, C, N = stem_audios.shape
    mono = stem_audios.mean(dim=2)  # [B, S, N]
    window = torch.hann_window(n_fft, device=stem_audios.device)
    specs = []
    for i in range(S):
        sp = torch.stft(mono[:, i], n_fft, hop, window=window,
                        return_complex=True).abs()
        specs.append(sp)
    specs = torch.stack(specs, dim=1)  # [B, S, F, T_stft]
    return specs / (specs.sum(dim=1, keepdim=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/latent_mask_refiner_ckpts")
    ap.add_argument("--n-stems", type=int, default=4)
    ap.add_argument("--v4small-ckpt-dir",
                    default="/scratch/latent_demucs_student/sem_demucs_v4_small_ckpts")
    ap.add_argument("--v4cond-ckpt-dir",
                    default="/scratch/latent_demucs_student/v4cond_predicted_4stem_ckpts")
    ap.add_argument("--v4small-channels", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop_frames", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = MixStemDataset(crop_frames=args.crop_frames, n_stems=args.n_stems)
    if len(ds) == 0:
        print("ERROR: empty"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    # Load frozen v4-small
    print("[refiner] loading frozen v4-small...")
    v4small = SemDemucs(n_stems=args.n_stems, channels=args.v4small_channels).cuda().eval()
    v4s_path = find_latest_ckpt(os.path.join(args.v4small_ckpt_dir, "*_step*.pt"))
    if v4s_path is None:
        print(f"ERROR: no v4-small ckpt in {args.v4small_ckpt_dir}"); return
    v4s_sd = torch.load(v4s_path, map_location="cuda", weights_only=False)
    v4small.load_state_dict(v4s_sd["model"])
    for p in v4small.parameters(): p.requires_grad = False
    print(f"  loaded {os.path.basename(v4s_path)} (step {v4s_sd.get('step','?')})")

    # Load frozen v4cond
    print("[refiner] loading frozen v4cond...")
    from train_distill_small_v4cond import SmallAdditiveDemucsV4
    v4cond = SmallAdditiveDemucsV4(n_stems=args.n_stems, hidden=96).cuda().eval()
    v4c_path = find_latest_ckpt(os.path.join(args.v4cond_ckpt_dir, "*_step*.pt"))
    if v4c_path is None:
        print(f"ERROR: no v4cond ckpt in {args.v4cond_ckpt_dir}"); return
    v4c_sd = torch.load(v4c_path, map_location="cuda", weights_only=False)
    v4cond.load_state_dict(v4c_sd["model"])
    for p in v4cond.parameters(): p.requires_grad = False
    print(f"  loaded {os.path.basename(v4c_path)} (step {v4c_sd.get('step','?')})")

    # Build refiner
    model = LatentMaskRefiner(latent_dim=64, n_freqs=N_FFT // 2 + 1,
                              hidden=args.hidden).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[refiner] model: {n:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    def lr_lambda(step):
        if step < args.warmup: return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(prog * math.pi))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    step = 0
    _ckpts = [f for f in glob.glob(os.path.join(args.out, "*_step*.pt"))
              if re.search(r'step(\d+)', f)]
    if _ckpts:
        _latest = max(_ckpts, key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
        _sd = torch.load(_latest, map_location="cuda", weights_only=False)
        model.load_state_dict(_sd["model"])
        step = _sd.get("step", 0)
        for _ in range(step): sched.step()
        print(f"[resume] from step {step} ({os.path.basename(_latest)})")

    hist = {"total": [], "noisy_baseline": []}
    t0 = time.time()

    while step < args.steps:
        for mix, stem_audios, presence in loader:
            mix = mix.cuda(non_blocking=True)
            stem_audios = stem_audios.cuda(non_blocking=True)
            presence = presence.cuda(non_blocking=True)
            B = mix.shape[0]

            # Target: GT stem masks (from actual stem audio)
            with torch.no_grad():
                tgt_masks = compute_gt_masks(stem_audios)  # [B, S, F, T_stft]

                # Noisy masks + embeddings from v4-small
                v4s_out = v4small(mix.float())
                noisy_masks = v4s_out["stft_masks"]  # [B, S, F, T_stft]
                pred_emb = v4s_out["embedding"]

                # Clean per-stem latents from v4cond
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_latents = v4cond(mix.to(torch.bfloat16), pred_emb, noisy_masks)
                pred_latents = pred_latents.float()  # [B, S, 64, T]

            # Refine mask per stem
            opt.zero_grad(set_to_none=True)
            refined_list = []
            for si in range(args.n_stems):
                refined = model(pred_latents[:, si], noisy_masks[:, si])
                refined_list.append(refined)
            refined = torch.stack(refined_list, dim=1)  # [B, S, F, T_stft]

            # Loss: MSE against GT masks, weighted by presence
            F_m = min(refined.shape[2], tgt_masks.shape[2])
            T_m = min(refined.shape[3], tgt_masks.shape[3])
            diff = (refined[:, :, :F_m, :T_m] - tgt_masks[:, :, :F_m, :T_m]).pow(2)
            mask_w = presence[:, :, None, None].expand_as(diff)
            loss = (diff * mask_w).sum() / mask_w.sum().clamp(min=1)

            # Baseline: how good is v4-small's mask vs GT? (for comparison)
            with torch.no_grad():
                noisy_diff = (noisy_masks[:, :, :F_m, :T_m] - tgt_masks[:, :, :F_m, :T_m]).pow(2)
                noisy_baseline = (noisy_diff * mask_w).sum() / mask_w.sum().clamp(min=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            hist["total"].append(loss.item())
            hist["noisy_baseline"].append(noisy_baseline.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-100:]) / max(1, len(xs[-100:]))
                r = avg(hist["total"])
                n_base = avg(hist["noisy_baseline"])
                improvement = (n_base - r) / n_base * 100 if n_base > 0 else 0
                print(f"[step {step:5d}] refined_mse={r:.5f} "
                      f"noisy_baseline={n_base:.5f} "
                      f"improvement={improvement:+.1f}% "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"refiner_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "refiner_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
