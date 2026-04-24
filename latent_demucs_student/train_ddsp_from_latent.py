#!/usr/bin/env python3
"""Train DDSP-from-latent with BasicPitch supervision.

Data:
  - z latents from Latents2/
  - BasicPitch posteriograms from MultiF0/
  - Source audio from Mixes_v5/

Losses:
  - pitch_bce: BCE(predicted_activation, basicpitch_activation) — cheap pitch signal
  - recon: multi-res STFT loss(synthesized_audio, target_audio) — quality

Target audio: the ORIGINAL mix audio (not DCAE decoded), since DDSP should
match real audio, not DCAE's degraded output.
"""
import argparse
import glob
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ddsp_from_latent import DDSPSynth, BP_N_PITCHES


class BasicPitchLatentDataset(Dataset):
    """Pair (z_latent, basic_pitch_posteriogram, source_audio) triples."""
    def __init__(self, crop_frames=60, seed=0):
        self.crop_frames = crop_frames
        self.crop_samples = crop_frames * 1920  # 48kHz / 25fps
        self.rng = random.Random(seed)
        self.items = []

        latents_root = Path("/home/arlo/gcs-bucket/Latents2/protools")
        mf0_root = Path("/home/arlo/gcs-bucket/MultiF0/protools")
        audio_root = Path("/home/arlo/gcs-bucket/protools")  # audio stored alongside latent paths

        # Mirror directory structure: iterate MultiF0, construct paired paths directly
        # Walk date-by-date for progress visibility on slow GCS mount
        n_scanned = 0
        dates = sorted([d for d in mf0_root.iterdir() if d.is_dir()])
        print(f"[bp-ds] scanning {len(dates)} dates under {mf0_root}", flush=True)
        for date_dir in dates:
            files_in_date = list(date_dir.rglob("*.npy"))
            n_scanned += len(files_in_date)
            print(f"  {date_dir.name}: {len(files_in_date)} files (total scanned: {n_scanned}, matched: {len(self.items)})", flush=True)
            for mf0_path in files_in_date:
                rel = mf0_path.relative_to(mf0_root)
                stem = mf0_path.name[:-4]  # strip .npy
                lat_path = latents_root / rel.parent / f"{stem}.vae.pt"
                if not lat_path.exists():
                    continue
                audio_path = None
                for ext in ['.wav', '.flac']:
                    p = audio_root / rel.parent / f"{stem}{ext}"
                    if p.exists():
                        audio_path = p
                        break
                if audio_path is None:
                    continue
                self.items.append({
                    "latent": str(lat_path),
                    "mf0": str(mf0_path),
                    "audio": str(audio_path),
                })
                if len(self.items) >= 5000:
                    break
            if len(self.items) >= 5000:
                break

        print(f"[bp-ds] {len(self.items)} items paired (scanned {n_scanned})", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            # Load latent
            raw = torch.load(it["latent"], map_location="cpu", weights_only=False)
            z = raw["latents"] if isinstance(raw, dict) else raw
            if z.dim() == 2 and z.shape[0] == 64: pass
            else: z = z.t() if z.dim() == 2 else z
            z = z.float()

            # Load BasicPitch (could be 88 or 264 wide, normalize to 88)
            bp = np.load(it["mf0"])
            bp = torch.from_numpy(bp).float()  # [T_bp, 88 or 264]
            if bp.shape[1] == 264:
                # Downsample 264 → 88 by averaging every 3 bins
                bp = bp.unsqueeze(0).unsqueeze(0)  # [1, 1, T, 264]
                bp = F.avg_pool2d(bp, kernel_size=(1, 3), stride=(1, 3)).squeeze(0).squeeze(0)
            elif bp.shape[1] != 88:
                return self.__getitem__((idx + 1) % len(self))

            # Load audio
            audio, sr = sf.read(it["audio"], dtype="float32")
            audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio[None]).float()
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)  # mono
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        # Align: BP is at ~86Hz, z at ~25Hz, audio at 48kHz
        T_z = z.shape[-1]
        if T_z < self.crop_frames:
            return self.__getitem__((idx + 1) % len(self))

        # Random crop in z space
        start_z = self.rng.randint(0, T_z - self.crop_frames)
        z = z[:, start_z:start_z + self.crop_frames]

        # Compute corresponding BP range: BP is ~3.44x z rate
        bp_ratio = bp.shape[0] / T_z
        start_bp = int(start_z * bp_ratio)
        end_bp = int((start_z + self.crop_frames) * bp_ratio)
        bp = bp[start_bp:end_bp]
        # Downsample BP to match z's T
        bp = bp.t().unsqueeze(0)  # [1, 264, T_bp]
        bp = F.interpolate(bp, size=self.crop_frames, mode='linear',
                           align_corners=False).squeeze(0)  # [264, crop_frames]

        # Audio crop
        start_samples = start_z * 1920
        audio = audio[:, start_samples:start_samples + self.crop_samples]
        if audio.shape[1] < self.crop_samples:
            audio = F.pad(audio, (0, self.crop_samples - audio.shape[1]))
        audio = audio[0]  # [N]

        return z, bp, audio


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


def multi_res_stft_loss(pred, target, fft_sizes=(512, 1024, 2048)):
    """Multi-resolution STFT loss."""
    sc_total, lm_total = 0.0, 0.0
    for n_fft in fft_sizes:
        window = torch.hann_window(n_fft, device=pred.device)
        hop = n_fft // 4
        p = torch.stft(pred, n_fft, hop, window=window, return_complex=True).abs()
        t = torch.stft(target, n_fft, hop, window=window, return_complex=True).abs()
        sc_total += (p - t).norm() / (t.norm() + 1e-8)
        lm_total += F.l1_loss((p + 1e-5).log(), (t + 1e-5).log())
    return sc_total / len(fft_sizes), lm_total / len(fft_sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/ddsp_from_latent_ckpts")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop-frames", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--lambda-bce", type=float, default=1.0)
    ap.add_argument("--lambda-recon", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = BasicPitchLatentDataset(crop_frames=args.crop_frames)
    if len(ds) == 0:
        print("ERROR: empty"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    model = DDSPSynth().cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[ddsp] {n:.2f}M params")

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
        print(f"[resume] from step {step}")

    hist = {"total": [], "bce": [], "sc": [], "lm": []}
    t0 = time.time()

    while step < args.steps:
        for z, bp_gt, audio_gt in loader:
            z = z.cuda(non_blocking=True)
            bp_gt = bp_gt.cuda(non_blocking=True)
            audio_gt = audio_gt.cuda(non_blocking=True)

            opt.zero_grad(set_to_none=True)
            audio_pred, params = model(z, n_audio_samples=audio_gt.shape[-1])

            # BCE on pitch activation
            l_bce = F.binary_cross_entropy_with_logits(
                params["pitch_logits"], (bp_gt > 0.3).float())

            # Multi-res STFT reconstruction
            l_sc, l_lm = multi_res_stft_loss(audio_pred, audio_gt)

            loss = args.lambda_bce * l_bce + args.lambda_recon * (l_sc + l_lm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            hist["total"].append(loss.item())
            hist["bce"].append(l_bce.item())
            hist["sc"].append(l_sc.item())
            hist["lm"].append(l_lm.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                print(f"[step {step:5d}] total={avg(hist['total']):.4f} "
                      f"bce={avg(hist['bce']):.4f} "
                      f"sc={avg(hist['sc']):.4f} "
                      f"lm={avg(hist['lm']):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"ddsp_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "ddsp_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
