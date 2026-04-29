#!/usr/bin/env python3
"""Train the latent-demucs student on combined teacher + GT pairs.

Loss: per-channel L1 with [B, 4] mask. Teacher items supervise all 4
channels; GT items supervise only the matching class slot.
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_model import LatentDemucsStudent
from dataset_combined import CombinedSeparationDataset, collate, CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/ckpts_v2")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=400)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_teacher", action="store_true")
    ap.add_argument("--no_gt", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = CombinedSeparationDataset(
        crop_frames=args.crop,
        use_teacher=not args.no_teacher,
        use_gt=not args.no_gt,
    )
    if len(ds) == 0:
        print("ERROR: empty dataset"); return

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    model = LatentDemucsStudent().cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[student] {n:.1f}M params")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"[student] resumed from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step = 0
    losses = []
    per_class = {c: [] for c in CLASSES}
    t0 = time.time()
    while step < args.steps:
        for mix, stems, mask in loader:
            mix = mix.cuda(non_blocking=True)
            stems = stems.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)               # [B, 4]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(mix)                              # [B, 4, 64, T]
            # masked L1 in fp32
            diff = (pred.float() - stems.float()).abs()        # [B, 4, 64, T]
            per_ch = diff.mean(dim=(2, 3))                     # [B, 4]
            denom = mask.sum().clamp_min(1.0)
            loss = (per_ch * mask).sum() / denom

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            losses.append(loss.item())
            with torch.no_grad():
                for ci, c in enumerate(CLASSES):
                    m = mask[:, ci]
                    if m.sum() > 0:
                        per_class[c].append((per_ch[:, ci] * m).sum().item() / m.sum().item())
            step += 1

            if step % args.log_every == 0:
                avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
                el = time.time() - t0
                pc_str = " ".join(
                    f"{c}={sum(per_class[c][-30:])/max(1,len(per_class[c][-30:])):.3f}"
                    for c in CLASSES
                )
                print(f"[step {step:6d}] l1={avg:.4f} {pc_str} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed={el:.0f}s",
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"student_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args),
                            "loss": sum(losses[-100:])/max(1,len(losses[-100:]))}, p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "student_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, final l1={sum(losses[-100:])/100:.4f}, saved {final}")


if __name__ == "__main__":
    main()
