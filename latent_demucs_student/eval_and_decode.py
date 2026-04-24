#!/usr/bin/env python3
"""Sanity-check the student:
  1. Baseline L1s (zero / mix-copied / model) on a held-out batch
  2. Decode predictions for one teacher session → 4 wavs

Usage:
  python eval_and_decode.py --ckpt /scratch/latent_demucs_student/ckpts_v2/student_step10000.pt
"""
import os, sys, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/scratch/ACE-Step-1.5")
from student_model import LatentDemucsStudent
from dataset_combined import CombinedSeparationDataset, collate, CLASSES


def load_vae():
    from diffusers.models import AutoencoderOobleck
    return AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae"
    ).cuda().eval().to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="/scratch/latent_demucs_student/eval_out")
    ap.add_argument("--n_batches", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("[eval] loading dataset...")
    ds = CombinedSeparationDataset(crop_frames=400, seed=123)

    print("[eval] loading model...")
    model = LatentDemucsStudent().cuda()
    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()

    # ── Part 1: baselines on n_batches batches ─────────────────────────
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True,
                        num_workers=2, collate_fn=collate, drop_last=True)

    sums = {"zero": 0.0, "mix": 0.0, "model": 0.0}
    counts = 0
    pc_sums = {k: {c: 0.0 for c in CLASSES} for k in sums}
    pc_counts = {c: 0 for c in CLASSES}

    it = iter(loader)
    for b in range(args.n_batches):
        try:
            mix, stems, mask = next(it)
        except StopIteration:
            break
        mix, stems, mask = mix.cuda(), stems.cuda(), mask.cuda()

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(mix).float()                  # [B, 4, 64, T]

        zero_pred = torch.zeros_like(stems)
        mix_pred  = mix.unsqueeze(1).expand(-1, 4, -1, -1) # broadcast input mix to 4 channels

        for name, p in (("zero", zero_pred), ("mix", mix_pred), ("model", pred)):
            diff = (p - stems).abs()                       # [B, 4, 64, T]
            per_ch = diff.mean(dim=(2, 3))                 # [B, 4]
            denom = mask.sum().clamp_min(1.0)
            sums[name] += ((per_ch * mask).sum() / denom).item()
            for ci, c in enumerate(CLASSES):
                m = mask[:, ci]
                if m.sum() > 0:
                    pc_sums[name][c] += ((per_ch[:, ci] * m).sum() / m.sum()).item()
        for ci, c in enumerate(CLASSES):
            if mask[:, ci].sum() > 0:
                pc_counts[c] += 1
        counts += 1

    print("\n=== Baseline L1 (lower = better) ===")
    print(f"{'baseline':<8} {'overall':>9} " + " ".join(f"{c:>9}" for c in CLASSES))
    for name in ("zero", "mix", "model"):
        overall = sums[name] / max(counts, 1)
        pcs = [pc_sums[name][c] / max(pc_counts[c], 1) for c in CLASSES]
        print(f"{name:<8} {overall:>9.4f} " + " ".join(f"{v:>9.4f}" for v in pcs))

    # ── Part 2: decode one teacher session to wavs ─────────────────────
    print("\n[decode] picking a teacher item...")
    teacher_items = [it for it in ds.items if it["kind"] == "teacher"]
    if not teacher_items:
        print("no teacher items, skipping decode"); return
    chosen = teacher_items[0]
    d = chosen["dir"]
    print(f"[decode] {d}")

    from dataset_combined import _load
    mix_full = _load(d / "full_mix.vae.pt")                # [T, 64]
    T = min(mix_full.shape[0], 400)
    mix_full = mix_full[:T]
    mix_in = mix_full.transpose(0, 1).unsqueeze(0).cuda()  # [1, 64, T]

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(mix_in).float()                   # [1, 4, 64, T]

    print("[decode] loading VAE...")
    vae = load_vae()

    def decode_to_wav(z, name):
        # z: [1, 64, T] float on cuda → wav
        z_bf = z.to(torch.bfloat16).cuda()
        with torch.no_grad():
            audio = vae.decode(z_bf).sample.float().cpu()  # [1, 2, N]
        audio = audio.squeeze(0).transpose(0, 1).numpy()   # [N, 2]
        out = os.path.join(args.out_dir, f"{name}.wav")
        sf.write(out, audio, 48000)
        print(f"  → {out}  ({audio.shape[0]/48000:.1f}s)")

    decode_to_wav(mix_in, "input_mix")
    for ci, c in enumerate(CLASSES):
        decode_to_wav(pred[:, ci], f"pred_{c}")
    # also decode the teacher targets for reference
    for c in CLASSES:
        teacher_z = _load(d / f"teacher_{c}.vae.pt")[:T]   # [T, 64]
        teacher_z = teacher_z.transpose(0, 1).unsqueeze(0).cuda()
        decode_to_wav(teacher_z, f"target_{c}")

    print(f"\n[done] wavs in {args.out_dir}")


if __name__ == "__main__":
    main()
