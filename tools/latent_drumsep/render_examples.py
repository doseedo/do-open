"""Render audio from the latent drum sub-separator student.

For a few cached examples, decode:
  - the input drum mix
  - the 6 teacher (cached) stems
  - the 6 student-predicted stems

Saves .wav files in a flat directory for easy A/B listening.

Usage:
    python -m latent_drumsep.render_examples \
        --ckpt /scratch/latent_drumsep_ckpts/drumsep_014000.pt \
        --num 3 --out /scratch/drumsep_render
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa: E402

from latent_drumsep.model import LatentDrumSubsep
from latent_drumsep.dataset import CachedDrumsepDataset
from latent_drumsep import STEMS

SR = 48000


@torch.no_grad()
def decode_one(vae, L_TC: torch.Tensor) -> np.ndarray:
    x = L_TC.t().unsqueeze(0).to("cuda", torch.bfloat16)
    return vae.decode(x).sample[0].float().cpu().numpy()  # [2, S]


def safe(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(",", "")[:60]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/scratch/latent_drumsep_cache")
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--out", default="/scratch/drumsep_render")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print("loading VAE...")
    h = AceStepHandler()
    h.initialize_service(project_root="/scratch/ACE-Step-1.5",
                         config_path="acestep-v15-sft", device=device)
    vae = h.vae
    for p in vae.parameters(): p.requires_grad = False
    vae.eval()

    print(f"loading student: {args.ckpt}")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = sd.get("model", sd.get("state_dict", sd))
    pos = state.get("pos")
    max_len = pos.shape[1] if pos is not None else 64
    student = LatentDrumSubsep(max_len=max_len).to(device).eval()
    student.load_state_dict(state)

    print("loading cache...")
    ds = CachedDrumsepDataset(args.cache_dir)
    print(f"  total items: {len(ds)}")

    rng = np.random.default_rng(args.seed)
    # Pick items deterministically from across the cache, biased toward late
    # indices (least likely to have been trained on heavily) so we get a fair
    # listening sample.
    candidates = list(range(len(ds) - min(500, len(ds)), len(ds)))
    rng.shuffle(candidates)
    chosen = candidates[: args.num]

    for slot, i in enumerate(chosen):
        ex = ds[i]
        L_mix = ex["L_mix"].to(device)            # [T, 64]
        L_stems_tgt = ex["L_stems"].to(device)    # [S, T, 64]
        src = ex.get("src", f"item{i}")
        base = f"{slot:02d}_{safe(str(src)).replace('.vae.pt','')}"

        print(f"\n[{slot+1}/{len(chosen)}] {src}")

        # Decode mix
        wav_mix = decode_one(vae, L_mix)
        sf.write(os.path.join(args.out, f"{base}__00_input_drum_mix.wav"),
                 wav_mix.T, SR)
        print(f"  mix: {wav_mix.shape[1]/SR:.2f}s")

        # Run student
        L_stems_pred = student(L_mix.unsqueeze(0))[0]   # [S, T, 64]

        # Decode each stem (teacher and student)
        for s_idx, stem in enumerate(STEMS):
            wav_tgt = decode_one(vae, L_stems_tgt[s_idx])
            wav_st = decode_one(vae, L_stems_pred[s_idx])
            sf.write(os.path.join(args.out, f"{base}__{s_idx+1}_{stem}_TEACHER.wav"),
                     wav_tgt.T, SR)
            sf.write(os.path.join(args.out, f"{base}__{s_idx+1}_{stem}_STUDENT.wav"),
                     wav_st.T, SR)

        # Build a "stems sum" — sum of all student stems, should ≈ mix
        L_sum = L_stems_pred.sum(dim=0)  # [T, 64]
        wav_sum = decode_one(vae, L_sum)
        sf.write(os.path.join(args.out, f"{base}__99_student_sum.wav"),
                 wav_sum.T, SR)

    print(f"\ndone. wavs in {args.out}")
    # list what we saved
    print("\nfiles:")
    for f in sorted(os.listdir(args.out)):
        size_kb = os.path.getsize(os.path.join(args.out, f)) // 1024
        print(f"  {f}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
