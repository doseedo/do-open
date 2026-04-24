#!/usr/bin/env python3
"""Train LatentPANNsStudent by distilling from PANNs CNN14 clipwise scores.

Loss: binary cross-entropy with teacher probabilities as soft targets
      (PANNs scores are already sigmoid outputs in [0, 1]).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_model import LatentPANNsStudent
from dataset import LatentPANNsDataset, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index",
                    default="/scratch/latent_panns_student/cache_index.json")
    ap.add_argument("--out",
                    default="/scratch/latent_panns_student/ckpts")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every",  type=int, default=20)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = LatentPANNsDataset(index_json=args.index)
    if len(ds) == 0:
        print("ERROR: empty dataset. Run gen_teacher.py first.")
        return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate,
                        persistent_workers=args.workers > 0)

    model = LatentPANNsStudent().cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[panns-train] {n:.1f}M params")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"[panns-train] resumed from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + torch.tensor(prog * 3.1415926535).cos().item())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    step = 0
    hist, hist_top1 = [], []
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            lat = batch["latent"].cuda(non_blocking=True)   # [B, 750, 64]
            tgt = batch["scores"].cuda(non_blocking=True)   # [B, 527]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(lat)                          # [B, 527]
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            hist.append(loss.item())
            with torch.no_grad():
                pred_top = logits.argmax(-1)
                tgt_top  = tgt.argmax(-1)
                hist_top1.append((pred_top == tgt_top).float().mean().item())

            step += 1
            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                el = time.time() - t0
                print(f"[step {step:6d}] loss={avg(hist):.4f} "
                      f"top1={avg(hist_top1):.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={el:.0f}s", flush=True)
            if step % args.save_every == 0:
                p = os.path.join(args.out, f"panns_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  → saved {p}", flush=True)
            if step >= args.steps:
                break

    final = os.path.join(args.out, "panns_final.pt")
    torch.save({"step": step, "model": model.state_dict(),
                "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
