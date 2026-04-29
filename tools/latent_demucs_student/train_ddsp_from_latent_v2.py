#!/usr/bin/env python3
"""Train DDSPv2: bigger model + transient events + HPSS-based timbre supervision.

Improvements over v1 training:
  - Larger model (4.7M)
  - Transient gate BCE supervision (from spectral flux on target audio)
  - HPSS split: harmonic_target vs percussive_target via median filtering
    Harmonic synth output trained against harmonic_target only
    Noise + transient trained against percussive_target
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
from ddsp_from_latent_v2 import (DDSPSynthV2, BP_N_PITCHES,
                                  render_harmonic_v2, render_filtered_noise,
                                  render_transients, bp_pitch_freqs,
                                  detect_transient_targets)
from train_ddsp_from_latent import BasicPitchLatentDataset, collate, multi_res_stft_loss


def hpss_split(audio, n_fft=2048, hop=512, kernel_h=17, kernel_p=17):
    """Median-filter HPSS (Fitzgerald 2010).

    Returns: (harmonic_audio, percussive_audio) — both [B, N]
    """
    window = torch.hann_window(n_fft, device=audio.device)
    spec = torch.stft(audio, n_fft, hop, window=window, return_complex=True)
    mag = spec.abs()  # [B, F, T]
    phase = spec / (mag + 1e-8)

    # Median filter along time → harmonic estimate
    # Median filter along freq → percussive estimate
    # Implement with avg filter as approximation (true median is slow)
    # Pad reflectively
    pad_h = kernel_h // 2
    pad_p = kernel_p // 2
    # Harmonic: smooth in time
    mag_h = F.avg_pool2d(mag.unsqueeze(1), kernel_size=(1, kernel_h),
                         stride=1, padding=(0, pad_h)).squeeze(1)
    # Percussive: smooth in freq
    mag_p = F.avg_pool2d(mag.unsqueeze(1), kernel_size=(kernel_p, 1),
                         stride=1, padding=(pad_p, 0)).squeeze(1)
    # Soft mask
    mask_h = mag_h ** 2 / (mag_h ** 2 + mag_p ** 2 + 1e-8)
    mask_p = 1 - mask_h
    spec_h = phase * mag * mask_h
    spec_p = phase * mag * mask_p
    h_audio = torch.istft(spec_h, n_fft, hop, window=window, length=audio.shape[-1])
    p_audio = torch.istft(spec_p, n_fft, hop, window=window, length=audio.shape[-1])
    return h_audio, p_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/ddsp_from_latent_v2_ckpts")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop-frames", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--lambda-bce", type=float, default=1.0)
    ap.add_argument("--lambda-trans-bce", type=float, default=0.5)
    ap.add_argument("--lambda-recon", type=float, default=1.0)
    ap.add_argument("--lambda-harm", type=float, default=1.0,
                    help="Weight on harmonic-only reconstruction loss")
    ap.add_argument("--lambda-perc", type=float, default=1.0,
                    help="Weight on percussive-only reconstruction loss")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = BasicPitchLatentDataset(crop_frames=args.crop_frames)
    if len(ds) == 0: print("ERROR empty"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    model = DDSPSynthV2().cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[ddsp-v2] {n:.2f}M params", flush=True)

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
        print(f"[resume] from step {step}", flush=True)

    hist = {k: [] for k in ("total", "bce", "trans_bce", "sc_full", "lm_full",
                              "sc_harm", "sc_perc")}
    t0 = time.time()

    while step < args.steps:
        for z, bp_gt, audio_gt in loader:
            z = z.cuda(non_blocking=True)
            bp_gt = bp_gt.cuda(non_blocking=True)
            audio_gt = audio_gt.cuda(non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # Render full audio + access individual components
            B, _, T = z.shape
            n_audio = audio_gt.shape[-1]
            params = model.net(z)
            harm = render_harmonic_v2(
                params["pitch_activation"], params["partial_amps"],
                model.pitch_freqs_hz, n_audio, sr=model.sr, top_k=48)
            noise = render_filtered_noise(params["noise_filter"], n_audio)
            trans = render_transients(params["transient_gate"], params["transient_spec"],
                                      n_audio, sr=model.sr, fps=model.fps)
            audio_pred = harm + noise + trans
            audio_perc_pred = noise + trans  # everything non-harmonic

            # Pitch BCE
            l_bce = F.binary_cross_entropy_with_logits(
                params["pitch_logits"], (bp_gt > 0.3).float())

            # Transient BCE
            with torch.no_grad():
                trans_targets = detect_transient_targets(audio_gt, sr=model.sr, fps=model.fps)
                # Threshold to binary onset
                trans_targets_bin = (trans_targets > 0.3).float()
            l_trans_bce = F.binary_cross_entropy_with_logits(
                params["transient_gate_logits"], trans_targets_bin)

            # HPSS split target
            with torch.no_grad():
                audio_harm_gt, audio_perc_gt = hpss_split(audio_gt)

            # Multi-res STFT losses
            l_sc_full, l_lm_full = multi_res_stft_loss(audio_pred, audio_gt)
            l_sc_harm, l_lm_harm = multi_res_stft_loss(harm, audio_harm_gt)
            l_sc_perc, l_lm_perc = multi_res_stft_loss(audio_perc_pred, audio_perc_gt)

            loss = (args.lambda_bce * l_bce
                    + args.lambda_trans_bce * l_trans_bce
                    + args.lambda_recon * (l_sc_full + l_lm_full)
                    + args.lambda_harm * (l_sc_harm + l_lm_harm)
                    + args.lambda_perc * (l_sc_perc + l_lm_perc))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            hist["total"].append(loss.item())
            hist["bce"].append(l_bce.item())
            hist["trans_bce"].append(l_trans_bce.item())
            hist["sc_full"].append(l_sc_full.item())
            hist["lm_full"].append(l_lm_full.item())
            hist["sc_harm"].append(l_sc_harm.item())
            hist["sc_perc"].append(l_sc_perc.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                print(f"[step {step:5d}] total={avg(hist['total']):.3f} "
                      f"bce={avg(hist['bce']):.3f} "
                      f"trans={avg(hist['trans_bce']):.3f} "
                      f"sc={avg(hist['sc_full']):.3f} "
                      f"lm={avg(hist['lm_full']):.3f} "
                      f"sc_h={avg(hist['sc_harm']):.3f} "
                      f"sc_p={avg(hist['sc_perc']):.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"el={time.time()-t0:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"ddspv2_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "ddspv2_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}", flush=True)


if __name__ == "__main__":
    main()
