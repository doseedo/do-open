"""
Pre-build a static training cache for the stretch cleaner.

Usage:
    python -m latent_editor.build_stretch_cache \
        --latent-roots /mnt/data2/Latents2/protools \
        --out /scratch/latent_stretch_cache \
        --num 30000 --win-frames 64
"""
from __future__ import annotations
import argparse, os, sys, time
import torch

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa
from latent_editor.stretch_dataset import LatentStretchDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=30_000)
    ap.add_argument("--win-frames", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("loading frozen VAE...")
    h = AceStepHandler()
    h.initialize_service(
        project_root="/scratch/ACE-Step-1.5",
        config_path="acestep-v15-sft", device="cuda",
    )
    vae = h.vae
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    ds = LatentStretchDataset(
        roots=args.latent_roots, vae=vae, win_frames=args.win_frames,
        device="cuda", seed=args.seed,
    )
    print(f"  files indexed: {len(ds.files)}")

    shard, sidx, written = [], 0, 0
    t0 = time.time()
    for i in range(args.num):
        shard.append(ds[i])
        if len(shard) >= args.shard_size:
            p = os.path.join(args.out, f"shard_{sidx:05d}.pt")
            torch.save(shard, p); written += len(shard)
            print(f"  wrote {p}  ({written}/{args.num}, {written/(time.time()-t0):.2f} ex/s)")
            shard, sidx = [], sidx + 1

    if shard:
        p = os.path.join(args.out, f"shard_{sidx:05d}.pt")
        torch.save(shard, p); written += len(shard)
        print(f"  wrote {p}  ({written}/{args.num})")

    print(f"done. {written} examples in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
