#!/usr/bin/env python3
"""Train v4cond with PREDICTED conditioning from frozen v4-small.

Same architecture as v4cond, but instead of oracle sem embeddings + htdemucs masks,
it gets conditioning from frozen v4-small (the real inference pipeline).
This teaches v4cond to handle imperfect/noisy predicted inputs.

Supports both 4-stem and 6-stem via --n-stems flag.
"""
import argparse
import importlib.util
import glob
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sem_demucs import SemDemucs
from train_distill_small_v4cond import SmallAdditiveDemucsV4

STEMS_4 = ["drums", "bass", "other", "vocals"]
STEMS_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]
SEM_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_V1_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"
VAE_CKPT = "/scratch/ACE-Step-1.5/checkpoints/vae"


def load_vae_encoder(device="cuda"):
    from audiocraft.models.loaders import load_compression_model
    vae = load_compression_model(VAE_CKPT).to(device).eval().to(torch.bfloat16)
    for p in vae.parameters(): p.requires_grad = False
    return vae


def find_latest_ckpt(pattern):
    ckpts = [f for f in glob.glob(pattern) if re.search(r'step(\d+)', f)]
    if not ckpts:
        return None
    return max(ckpts, key=lambda x: int(re.search(r'step(\d+)', x).group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stems", type=int, default=4)
    ap.add_argument("--v4small-ckpt-dir", required=True,
                    help="Dir with v4-small checkpoints to load frozen")
    ap.add_argument("--v4small-channels", type=int, default=64)
    ap.add_argument("--out", required=True)
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
    STEMS = STEMS_6 if args.n_stems == 6 else STEMS_4

    # Load dataset
    if args.n_stems == 6:
        from distill_dataset_6 import DistillDataset6
        ds = DistillDataset6(crop_frames=args.crop_frames)
        def collate_fn(batch):
            return (torch.stack([b[0] for b in batch]),
                    torch.stack([b[1] for b in batch]),
                    torch.stack([b[2] for b in batch]))
    else:
        from distill_dataset import DistillDataset, collate
        ds = DistillDataset(crop_frames=args.crop_frames)
        collate_fn = collate

    if len(ds) == 0:
        print("ERROR: empty dataset"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate_fn, persistent_workers=args.workers > 0)

    # Load frozen v4-small (the predictor)
    print(f"[v4cond-pred] loading frozen v4-small (n_stems={args.n_stems})...")
    v4small = SemDemucs(n_stems=args.n_stems, channels=args.v4small_channels).cuda().eval()
    v4s_path = find_latest_ckpt(os.path.join(args.v4small_ckpt_dir, "*_step*.pt"))
    if v4s_path is None:
        print(f"ERROR: no v4-small checkpoint in {args.v4small_ckpt_dir}"); return
    v4s_sd = torch.load(v4s_path, map_location="cuda", weights_only=False)
    v4small.load_state_dict(v4s_sd["model"])
    for p in v4small.parameters():
        p.requires_grad = False
    print(f"  loaded {os.path.basename(v4s_path)} (step {v4s_sd.get('step','?')})")

    # Build v4cond (the student being trained)
    print(f"[v4cond-pred] building v4cond (n_stems={args.n_stems}, hidden={args.hidden})...")
    model = SmallAdditiveDemucsV4(n_stems=args.n_stems, hidden=args.hidden).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0

    # Auto-resume
    _ckpts = [f for f in glob.glob(os.path.join(args.out, "*_step*.pt"))
              if re.search(r'step(\d+)', f)]
    if _ckpts:
        _latest = max(_ckpts, key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
        _sd = torch.load(_latest, map_location="cuda", weights_only=False)
        model.load_state_dict(_sd["model"])
        step = _sd.get("step", 0)
        for _ in range(step):
            sched.step()
        print(f"[resume] from step {step} ({os.path.basename(_latest)})")

    losses = []
    per_class = {c: [] for c in STEMS}
    t0 = time.time()

    while step < args.steps:
        for batch in loader:
            if args.n_stems == 6:
                audio, stems, mask = batch
                mask = mask.cuda(non_blocking=True)
            else:
                audio, stems = batch
                mask = None
            audio = audio.cuda(non_blocking=True).to(torch.bfloat16)
            stems = stems.cuda(non_blocking=True)
            B = audio.shape[0]

            # Get PREDICTED conditioning from frozen v4-small (NOT oracle)
            with torch.no_grad():
                v4s_out = v4small(audio.float())
                pred_emb = v4s_out["embedding"]      # [B, S, 128]
                pred_masks = v4s_out["stft_masks"]    # [B, S, F, T]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(audio, pred_emb, pred_masks)

            T = min(pred.shape[-1], stems.shape[-1])
            diff = (pred[..., :T].float() - stems[..., :T].float()).abs()

            if mask is not None:
                mask_expanded = mask[:, :, None, None].expand_as(diff[..., :T])
                masked_diff = diff * mask_expanded
                per_ch = masked_diff.sum(dim=(0, 2, 3)) / (
                    mask.sum(dim=0).clamp(min=1) * diff.shape[2] * T)
            else:
                per_ch = diff.mean(dim=(0, 2, 3))

            loss = per_ch.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            losses.append(loss.item())
            with torch.no_grad():
                for ci, c in enumerate(STEMS):
                    per_class[c].append(per_ch[ci].item())
            step += 1

            if step % args.log_every == 0:
                avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
                if args.n_stems == 6:
                    pc_str = " ".join(
                        f"{c[:3]}={sum(per_class[c][-30:])/max(1,len(per_class[c][-30:])):.3f}"
                        for c in STEMS)
                else:
                    pc_str = " ".join(
                        f"{c}={sum(per_class[c][-30:])/max(1,len(per_class[c][-30:])):.3f}"
                        for c in STEMS)
                print(f"[step {step:6d}] l1={avg:.4f} {pc_str} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed={time.time()-t0:.0f}s",
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"v4cond_pred_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "v4cond_pred_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
