"""Render a LONG drum track through the latent student via 64-frame chunking.

The student's pos embedding is fixed at 64 frames, so for longer audio we
chunk the latent into 64-frame windows, run the student per chunk, and
concatenate the per-stem outputs along the time axis. Each chunk runs
independently — there's no overlap-add since the student is residual-around-input
(no boundary discontinuity introduced beyond what the input already had).

If chunking works for drums (transient, locally interpretable signal), the
64-frame cap is just a deployment quirk we can paper over with chunking.

Usage:
    python -m latent_drumsep.render_long \
        --ckpt /scratch/latent_drumsep_ckpts/drumsep_014000.pt \
        --drums-latent "/scratch/musdb18_latents/test/Al James - Schoolboy Facination/drums.vae.pt" \
        --seconds 30 --start-sec 30 --out /scratch/drumsep_long
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa: E402

from latent_editor.dataset import _load_latent  # noqa: E402
from latent_drumsep.model import LatentDrumSubsep
from latent_drumsep import STEMS

SR = 48000
FRAMES_PER_SEC = SR / 1920  # = 25 latent frames per second


@torch.no_grad()
def decode_one(vae, L_TC: torch.Tensor) -> np.ndarray:
    x = L_TC.t().unsqueeze(0).to("cuda", torch.bfloat16)
    return vae.decode(x).sample[0].float().cpu().numpy()  # [2, S]


def safe(s):
    return s.replace("/", "_").replace(" ", "_").replace(",", "")[:60]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--drums-latent", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--start-sec", type=float, default=30.0)
    ap.add_argument("--out", default="/scratch/drumsep_long")
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
    chunk = pos.shape[1] if pos is not None else 64
    student = LatentDrumSubsep(max_len=chunk).to(device).eval()
    student.load_state_dict(state)
    print(f"  student chunk size: {chunk} frames = {chunk/FRAMES_PER_SEC:.2f}s")

    print(f"loading drums latent: {args.drums_latent}")
    L_full = _load_latent(args.drums_latent).to(device)
    print(f"  full duration: {L_full.shape[0]/FRAMES_PER_SEC:.1f}s ({L_full.shape[0]} frames)")

    start_f = int(round(args.start_sec * FRAMES_PER_SEC))
    n_f = int(round(args.seconds * FRAMES_PER_SEC))
    end_f = min(start_f + n_f, L_full.shape[0])
    L = L_full[start_f:end_f].contiguous()
    n_f = L.shape[0]
    print(f"  using {start_f}–{end_f} frames ({n_f} frames = {n_f/FRAMES_PER_SEC:.2f}s)")

    # Round n_f down to a multiple of chunk to avoid the last partial window
    n_chunks = n_f // chunk
    n_used = n_chunks * chunk
    L = L[:n_used]
    print(f"  chunked into {n_chunks} × {chunk}-frame windows = {n_used/FRAMES_PER_SEC:.2f}s used")

    # ---- Run student on each chunk, gather per-stem latents ----
    pred_chunks = []  # list of [S, chunk, 64]
    for ci in range(n_chunks):
        L_chunk = L[ci * chunk : (ci + 1) * chunk].unsqueeze(0)   # [1, chunk, 64]
        L_pred = student(L_chunk)[0]                              # [S, chunk, 64]
        pred_chunks.append(L_pred)
    L_stems_full = torch.cat(pred_chunks, dim=1)                  # [S, n_used, 64]
    print(f"  student output: {tuple(L_stems_full.shape)}")

    # ---- Decode mix and each stem (decode in long form, no chunking needed for VAE) ----
    base = safe(os.path.basename(os.path.dirname(args.drums_latent))) + "_long"
    print("decoding mix...")
    wav_mix = decode_one(vae, L)
    sf.write(os.path.join(args.out, f"{base}__00_input_mix.wav"), wav_mix.T, SR)

    print("decoding stems...")
    for s_idx, stem in enumerate(STEMS):
        wav = decode_one(vae, L_stems_full[s_idx])
        sf.write(os.path.join(args.out, f"{base}__{s_idx+1}_{stem}_STUDENT.wav"),
                 wav.T, SR)
        print(f"  {stem}: ok")

    # Sum-of-stems for sanity
    wav_sum = decode_one(vae, L_stems_full.sum(dim=0))
    sf.write(os.path.join(args.out, f"{base}__99_student_sum.wav"), wav_sum.T, SR)

    print(f"\ndone. files in {args.out}")
    for f in sorted(os.listdir(args.out)):
        size_kb = os.path.getsize(os.path.join(args.out, f)) // 1024
        print(f"  {f}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
