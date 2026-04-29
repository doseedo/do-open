#!/usr/bin/env python3
"""Train the latent-lyric student (encoder-decoder, ACE-Step tokens).

Loss = cross-entropy over next-token predictions, with PAD tokens masked out.

Usage:
    python train_vocal.py --pairs /scratch/latent_whisper_student/training_pairs.json
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_model import LatentLyricStudent, configure, PAD_ID, SOS_ID, EOS_ID
from vocal_dataset import VocalLyricDataset, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index",
                    default="/scratch/latent_whisper_student/cache_index.json",
                    help="precached index (from scripts/precache_pairs.py)")
    ap.add_argument("--out",
                    default="/scratch/latent_whisper_student/ckpts_vocal")
    ap.add_argument("--size", default="base", choices=["tiny", "base", "small"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=448)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every",  type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    # -- data ---------------------------------------------------------
    ds = VocalLyricDataset(index_json=args.index, max_tokens=args.max_tokens)
    if len(ds) == 0:
        print("ERROR: empty dataset.")
        return

    loader = DataLoader(ds,
                        batch_size=args.batch,
                        shuffle=True,
                        num_workers=args.workers,
                        drop_last=True,
                        collate_fn=collate,
                        persistent_workers=args.workers > 0)

    # -- model --------------------------------------------------------
    cfg = configure(args.size)
    model = LatentLyricStudent(**cfg).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train-vocal] {args.size}: {n:.1f}M  d={cfg['d_model']} "
          f"enc={cfg['n_enc_layers']} dec={cfg['n_dec_layers']} "
          f"vocab={model.vocab}")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"[train-vocal] resumed from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + torch.tensor(prog * 3.1415926535).cos().item())

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # -- train --------------------------------------------------------
    model.train()
    step = 0
    hist = []
    hist_acc = []
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            mix = batch["mix_lat"].cuda(non_blocking=True)   # [B, 64, 750]
            tok = batch["tokens"].cuda(non_blocking=True)    # [B, T_tok]

            dec_in  = tok[:, :-1]
            dec_lab = tok[:, 1:]
            tgt_pad = (dec_in == PAD_ID)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mix, dec_in,
                               tgt_key_padding_mask=tgt_pad)

            logits = logits.float()
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, V),
                dec_lab.reshape(-1),
                ignore_index=PAD_ID,
                label_smoothing=args.label_smoothing,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            hist.append(loss.item())

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                mask = (dec_lab != PAD_ID)
                if mask.any():
                    acc = ((pred == dec_lab) & mask).float().sum() / mask.float().sum()
                    hist_acc.append(acc.item())

            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                el = time.time() - t0
                print(f"[step {step:6d}] loss={avg(hist):.4f} "
                      f"acc={avg(hist_acc):.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={el:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"student_step{step}.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "args": vars(args),
                    "size": args.size,
                }, p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "student_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "cfg": cfg,
                "args": vars(args), "size": args.size}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
