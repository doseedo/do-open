#!/usr/bin/env python3
"""Train the 6-stem distilled waveform → 6-stem-latents model.

Loss = masked L1 in latent space, normalized by the mask sum so MUSDB
items (4 active channels) and v7 items (6 active) contribute fairly.

Stem order matches htdemucs_6s: [drums, bass, other, vocals, guitar, piano]
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_model import WaveformToFourStemLatents
from distill_dataset_6 import DistillDataset6, collate, STEMS_6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/distill6_ckpts")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop_frames", type=int, default=200)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no_musdb", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = DistillDataset6(crop_frames=args.crop_frames,
                         use_musdb=not args.no_musdb)
    if len(ds) == 0:
        print("ERROR: empty dataset (need stem6_*.vae.pt under /scratch/mixesV7_latents")
        print("       and/or MoisesDB latents under /scratch/moisesdb_latents)")
        print("       Run build_stem6_targets.py and/or prep_moisesdb.py first.")
        return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[distill6] building model from Oobleck encoder weights (n_stems=6)...")
    model = WaveformToFourStemLatents(n_stems=6).cuda()
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[distill6] {n_total:.1f}M params")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"[distill6] resumed from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    step, t0 = 0, time.time()
    losses = []
    per_class = {c: [] for c in STEMS_6}
    while step < args.steps:
        for audio, stems, mask in loader:
            audio = audio.cuda(non_blocking=True).to(torch.bfloat16)
            stems = stems.cuda(non_blocking=True)
            mask  = mask.cuda(non_blocking=True)             # [B, 6]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(audio)                           # [B, 6, 64, T]
            T = min(pred.shape[-1], stems.shape[-1])
            pred = pred[..., :T]
            stems = stems[..., :T]
            diff = (pred.float() - stems.float()).abs()       # [B, 6, 64, T]
            per_ch = diff.mean(dim=(2, 3))                    # [B, 6]
            denom = mask.sum().clamp_min(1.0)
            loss = (per_ch * mask).sum() / denom

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            losses.append(loss.item())
            with torch.no_grad():
                for ci, c in enumerate(STEMS_6):
                    m = mask[:, ci]
                    if m.sum() > 0:
                        per_class[c].append((per_ch[:, ci] * m).sum().item() / m.sum().item())
            step += 1

            if step % args.log_every == 0:
                avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
                pc_str = " ".join(
                    f"{c}={sum(per_class[c][-30:])/max(1,len(per_class[c][-30:])):.3f}"
                    for c in STEMS_6
                )
                el = time.time() - t0
                print(f"[step {step:6d}] l1={avg:.4f} {pc_str} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed={el:.0f}s",
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"distill6_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args),
                            "loss": sum(losses[-100:])/max(1,len(losses[-100:]))}, p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "distill6_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, final l1={sum(losses[-100:])/100:.4f}, saved {final}")


if __name__ == "__main__":
    main()
