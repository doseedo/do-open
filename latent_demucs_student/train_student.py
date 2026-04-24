#!/usr/bin/env python3
"""Train the latent demucs student.

Dataset: walks /scratch/mixesV7_latents for sessions that have
full_mix.vae.pt + teacher_{drums,bass,vocals,other}.vae.pt and pairs
them. Random-crops to crop_frames during training.
"""
import os, sys, time, json, random, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student_model import LatentDemucsStudent

LATENT_ROOT = Path("/scratch/mixesV7_latents")
STEMS = ["drums", "bass", "vocals", "other"]


def _load_latents(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64:
        z = z.t()                   # → [T, 64]
    return z.float()


class TeacherDataset(Dataset):
    def __init__(self, crop_frames=400, seed=0):
        self.crop = crop_frames
        self.rng = random.Random(seed)
        self.items = []
        for full in LATENT_ROOT.rglob("full_mix.vae.pt"):
            if all((full.parent / f"teacher_{s}.vae.pt").exists() for s in STEMS):
                self.items.append(full.parent)
        print(f"[student dataset] {len(self.items)} sessions")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        try:
            mix = _load_latents(d / "full_mix.vae.pt")           # [T, 64]
            stems = [_load_latents(d / f"teacher_{s}.vae.pt") for s in STEMS]
        except Exception:
            return self.__getitem__((idx + 1) % len(self.items))

        T = min([mix.shape[0]] + [s.shape[0] for s in stems])
        mix = mix[:T]; stems = [s[:T] for s in stems]
        if T <= self.crop:
            pad = self.crop - T
            mix = F.pad(mix, (0, 0, 0, pad))
            stems = [F.pad(s, (0, 0, 0, pad)) for s in stems]
            T = self.crop
        start = self.rng.randint(0, T - self.crop)
        mix = mix[start:start + self.crop]                          # [C, 64]
        stems = torch.stack([s[start:start + self.crop] for s in stems])  # [4, C, 64]
        return mix.transpose(0, 1).contiguous(), stems.transpose(1, 2).contiguous()
        # → mix [64, T], stems [4, 64, T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/ckpts")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=400)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = TeacherDataset(crop_frames=args.crop)
    if len(ds) == 0:
        print("ERROR: no teacher sessions found yet. Wait for make_teacher_data.py.")
        return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)

    model = LatentDemucsStudent().cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[student] {n:.1f}M params")

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cuda", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"[student] resumed from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    step = 0
    losses = []
    t0 = time.time()
    while step < args.steps:
        for mix, stems in loader:
            mix = mix.cuda(non_blocking=True)        # [B, 64, T]
            stems = stems.cuda(non_blocking=True)    # [B, 4, 64, T]

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(mix)                    # [B, 4, 64, T]
                loss = F.l1_loss(pred.float(), stems.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
                el = time.time() - t0
                print(f"[step {step:6d}] l1={avg:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed={el:.0f}s",
                      flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"student_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args),
                            "loss": sum(losses[-100:]) / max(1, len(losses[-100:]))},
                           p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "student_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, final l1={sum(losses[-100:])/100:.4f}, saved {final}")


if __name__ == "__main__":
    main()
