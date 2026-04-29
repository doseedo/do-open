"""
Build a stretch-cleaner cache where the dirty distribution is
LINEAR-INTERP round-trip in latent space (matches what
stemphonic_server.py:_stretch_latent does in production), instead of
phase-vocoder round-trip on decoded audio.

Round trip = F.interpolate(L, size=round(T*r)) -> F.interpolate(..., size=T).
The artifact is the smoothing/aliasing introduced by linear interpolation.

No VAE required: this runs entirely on CPU latent tensors and is ~100x
faster than the audio-domain builder.

Usage:
    python -m latent_editor.build_stretch_cache_linear \
        --latent-roots /scratch/Latents2/protoolsA \
        --out /scratch/latent_stretch_cache_linear \
        --num 30000 --win-frames 64 --shard-size 1000
"""
from __future__ import annotations
import argparse, os, random, time
import torch
import torch.nn.functional as F

from latent_editor.dataset import _list_latent_files, _load_latent


def linear_round_trip(L: torch.Tensor, r: float) -> torch.Tensor:
    """L: [T,64].  Linear-interp to round(T*r) frames, then back to T."""
    T = L.shape[0]
    Tmid = max(2, int(round(T * r)))
    x = L.t().unsqueeze(0)  # [1, 64, T]
    x = F.interpolate(x, size=Tmid, mode="linear", align_corners=False)
    x = F.interpolate(x, size=T,    mode="linear", align_corners=False)
    return x.squeeze(0).t().contiguous()  # [T, 64]


def crop(L: torch.Tensor, n: int, rng: random.Random) -> torch.Tensor:
    T = L.shape[0]
    if T <= n:
        return torch.cat([L, torch.zeros(n - T, 64)], 0)
    s = rng.randint(0, T - n)
    return L[s : s + n].clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=30_000)
    ap.add_argument("--win-frames", type=int, default=64)
    ap.add_argument("--r-min", type=float, default=0.6)
    ap.add_argument("--r-max", type=float, default=1.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    files = _list_latent_files(args.latent_roots)
    if not files:
        raise SystemExit(f"no latent files in {args.latent_roots}")
    print(f"  files indexed: {len(files)}")

    shard, sidx, written = [], 0, 0
    t0 = time.time()
    i = 0
    while written + len(shard) < args.num:
        i += 1
        try:
            L = _load_latent(rng.choice(files))
        except Exception:
            continue
        if L.dim() != 2 or L.shape[1] != 64 or L.shape[0] < 2:
            continue
        L = crop(L, args.win_frames, rng)
        r = rng.uniform(args.r_min, args.r_max)
        L_in = linear_round_trip(L, r)
        shard.append({
            "L_input":  L_in,
            "L_target": L,
            "stretch_r": torch.tensor(r, dtype=torch.float32),
        })
        if len(shard) >= args.shard_size:
            p = os.path.join(args.out, f"shard_{sidx:05d}.pt")
            torch.save(shard, p); written += len(shard)
            print(f"  wrote {p}  ({written}/{args.num}, {written/(time.time()-t0):.1f} ex/s)")
            shard, sidx = [], sidx + 1

    if shard:
        p = os.path.join(args.out, f"shard_{sidx:05d}.pt")
        torch.save(shard, p); written += len(shard)
        print(f"  wrote {p}  ({written}/{args.num})")

    print(f"done. {written} examples in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
