"""
Pre-build a static training cache for the latent editor.

Why: LatentSpliceDataset calls the GPU VAE (decode + encode) per item, which
caps throughput at ~1-2 it/s. This script runs the data pipeline once, in
batches, and writes (L_naive, L_target, mask, phase, cut_frame) tensors to
disk. Training then loads from disk at full IO speed.

We deliberately do NOT cache the wav_target tensor (1 MB/example). The
training loop decodes the boundary slice of L_target on-the-fly via the
frozen VAE -- a single small forward, batched.

Usage:
    python -m latent_editor.build_cache \
        --latent-roots /mnt/data2/Latents2/protools \
        --out /scratch/latent_editor_cache \
        --num 20000 --win-frames 64
"""
from __future__ import annotations
import argparse, os, sys, time
import torch

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa

from latent_editor.dataset import LatentSpliceDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=20_000)
    ap.add_argument("--win-frames", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("loading frozen VAE...")
    h = AceStepHandler()
    h.initialize_service(
        project_root="/scratch/ACE-Step-1.5",
        config_path="acestep-v15-sft",
        device="cuda",
    )
    vae = h.vae
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    ds = LatentSpliceDataset(
        roots=args.latent_roots, vae=vae, win_frames=args.win_frames,
        device="cuda", seed=args.seed,
    )
    print(f"  files indexed: {len(ds.files)}")

    shard, shard_idx, written = [], 0, 0
    t0 = time.time()
    for i in range(args.num):
        item = ds[i]
        # drop wav (recomputed on the fly during training)
        item.pop("wav_target", None)
        shard.append(item)

        if len(shard) >= args.shard_size:
            path = os.path.join(args.out, f"shard_{shard_idx:05d}.pt")
            torch.save(shard, path)
            written += len(shard)
            dt = time.time() - t0
            print(f"  wrote {path}  ({written}/{args.num}, {written/dt:.2f} ex/s)")
            shard = []
            shard_idx += 1

    if shard:
        path = os.path.join(args.out, f"shard_{shard_idx:05d}.pt")
        torch.save(shard, path)
        written += len(shard)
        print(f"  wrote {path}  ({written}/{args.num})")

    print(f"done. {written} examples in {time.time()-t0:.0f}s "
          f"(avg {written/(time.time()-t0):.2f} ex/s)")


if __name__ == "__main__":
    main()
