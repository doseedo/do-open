"""Build drum-subsep teacher data starting from drum LATENTS we already
have on disk (musdb drums.vae.pt + protools drum directories).

Pipeline per file:
  1. Load drum latent  → L_mix [T, 64]   (no encode needed)
  2. VAE-decode        → drum wav [2, S]
  3. MDX23C-DrumSep    → 6 sub-stem wavs (kick/snare/toms/hh/ride/crash)
  4. VAE-encode each   → L_stems [6, T, 64]
  5. Random crops + save shards.

Usage:
    python -m latent_drumsep.build_cache_from_latents \
        --latent-paths "/scratch/musdb18_latents/train/*/drums.vae.pt" \
                       "/scratch/musdb18_latents/test/*/drums.vae.pt" \
        --out /scratch/latent_drumsep_cache \
        --num 200 --crops-per-file 4 --shard-size 100
"""
from __future__ import annotations
import argparse, glob, os, random, sys, time
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa: E402

from latent_editor.dataset import _load_latent  # noqa: E402
from latent_drumsep import STEMS  # noqa: E402
from latent_drumsep.build_cache import (  # reuse helpers
    get_drumsep, encode, load_stereo, crop_aligned, SR,
)


def expand(patterns):
    out = []
    for p in patterns:
        out.extend(sorted(glob.glob(p, recursive=True)))
    return out


@torch.no_grad()
def decode(vae, L_TC: torch.Tensor) -> np.ndarray:
    x = L_TC.transpose(0, 1).unsqueeze(0).to("cuda", torch.bfloat16)
    return vae.decode(x).sample[0].float().cpu().numpy()  # [2, S]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-paths", nargs="+", required=True,
                    help="glob patterns matching drum .vae.pt files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--win-frames", type=int, default=64)
    ap.add_argument("--crops-per-file", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("loading frozen ACE VAE...")
    h = AceStepHandler()
    h.initialize_service(project_root="/scratch/ACE-Step-1.5",
                         config_path="acestep-v15-sft", device="cuda")
    vae = h.vae
    for p in vae.parameters(): p.requires_grad = False
    vae.eval()

    print("loading MDX23C-DrumSep...")
    sep, sep_out_dir = get_drumsep()

    files = expand(args.latent_paths)
    print(f"  drum latent files: {len(files)}")
    if not files:
        raise SystemExit("no input files")
    random.shuffle(files)

    shard, sidx, written = [], 0, 0
    t0 = time.time()
    for fp in files:
        if written + len(shard) >= args.num:
            break
        try:
            L_mix = _load_latent(fp)  # [T, 64]
        except Exception as e:
            print(f"  load fail {fp}: {e}"); continue
        if L_mix.dim() != 2 or L_mix.shape[1] != 64 or L_mix.shape[0] < args.win_frames:
            continue

        # decode → wav for drumsep
        try:
            wav_mix = decode(vae, L_mix)  # [2, S]
        except Exception as e:
            print(f"  decode fail {fp}: {e}"); continue

        tmp = os.path.join(sep_out_dir, f"_in_{os.getpid()}.wav")
        sf.write(tmp, wav_mix.T, SR)
        try:
            stem_files = sep.separate(tmp)
        except Exception as e:
            print(f"  drumsep fail {fp}: {e}")
            try: os.remove(tmp)
            except Exception: pass
            continue

        stem_paths = {s: None for s in STEMS}
        for fn in stem_files:
            full = fn if os.path.isabs(fn) else os.path.join(sep_out_dir, fn)
            base = os.path.basename(full).lower()
            for s in STEMS:
                if s in base:
                    stem_paths[s] = full
                    break
        if any(v is None for v in stem_paths.values()):
            print(f"  missing stems for {fp}: { {k: bool(v) for k,v in stem_paths.items()} }")
            try: os.remove(tmp)
            except Exception: pass
            continue

        try:
            L_stems = []
            for s in STEMS:
                w = load_stereo(stem_paths[s])
                if w is None: raise RuntimeError(f"reload fail {s}")
                L_stems.append(encode(vae, w))
        except Exception as e:
            print(f"  encode fail {fp}: {e}"); continue
        finally:
            for p in [tmp] + [v for v in stem_paths.values() if v]:
                try: os.remove(p)
                except Exception: pass

        n_added = 0
        for _ in range(args.crops_per_file):
            cm, cs = crop_aligned(L_mix, L_stems, args.win_frames)
            shard.append({
                "L_mix":   cm,
                "L_stems": torch.stack(cs, dim=0),
                "src":     os.path.basename(os.path.dirname(fp)) + "/" + os.path.basename(fp),
            })
            n_added += 1
            if len(shard) >= args.shard_size:
                p = os.path.join(args.out, f"shard_{sidx:05d}.pt")
                torch.save(shard, p); written += len(shard)
                rate = written / (time.time() - t0)
                print(f"  wrote {p}  ({written}/{args.num}, {rate:.2f} ex/s)")
                shard, sidx = [], sidx + 1
            if written + len(shard) >= args.num:
                break
        print(f"  + {n_added} crops from {os.path.basename(os.path.dirname(fp))}")

    if shard:
        p = os.path.join(args.out, f"shard_{sidx:05d}.pt")
        torch.save(shard, p); written += len(shard)
        print(f"  wrote {p}  ({written}/{args.num})")
    print(f"done. {written} examples in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
