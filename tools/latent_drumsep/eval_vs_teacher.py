"""Eval the latent drum sub-separator student against the cached teacher.

Loads the latest student ckpt and runs it on a deterministic slice of the
training cache (last shard, last 200 items), comparing predictions to the
cached teacher latents.

Reports:
  - per-stem latent L1 (student vs teacher)
  - identity baseline L1 (mix repeated 6× → "no separation")
  - per-stem decoded STFT log-mag L1
  - identity vs student improvement %

"Is it learning?" → student should be meaningfully below identity by every
metric, and improving across recent ckpts.

Usage:
    python -m latent_drumsep.eval_vs_teacher \
        --cache-dir /scratch/latent_drumsep_cache \
        --ckpts /scratch/latent_drumsep_ckpts/drumsep_002000.pt \
                /scratch/latent_drumsep_ckpts/drumsep_006000.pt \
                /scratch/latent_drumsep_ckpts/drumsep_012000.pt \
        --num-eval 200
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa: E402

from latent_drumsep.model import LatentDrumSubsep
from latent_drumsep.dataset import CachedDrumsepDataset
from latent_drumsep import STEMS
from latent_editor.test_stretch_methods import stft_log_l1  # noqa: E402


def load_student(ckpt_path, device="cuda"):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model", sd.get("state_dict", sd))
    pos = state.get("pos")
    max_len = pos.shape[1] if pos is not None else 64
    m = LatentDrumSubsep(max_len=max_len).to(device).eval()
    m.load_state_dict(state)
    return m, max_len


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/scratch/latent_drumsep_cache")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--num-eval", type=int, default=200)
    ap.add_argument("--decode-n", type=int, default=24,
                    help="how many examples to decode for STFT eval (decode is slow)")
    args = ap.parse_args()

    device = "cuda"

    print("loading VAE...")
    h = AceStepHandler()
    h.initialize_service(project_root="/scratch/ACE-Step-1.5",
                         config_path="acestep-v15-sft", device=device)
    vae = h.vae
    for p in vae.parameters(): p.requires_grad = False
    vae.eval()

    print("loading cache...")
    ds = CachedDrumsepDataset(args.cache_dir)
    print(f"  total items: {len(ds)}")
    # Take last N items deterministically
    n = min(args.num_eval, len(ds))
    eval_idxs = list(range(len(ds) - n, len(ds)))
    print(f"  eval slice: last {n} items")

    # Stack into a single tensor for fast batched eval
    L_mix_list = []
    L_stems_list = []
    for i in eval_idxs:
        ex = ds[i]
        L_mix_list.append(ex["L_mix"])
        L_stems_list.append(ex["L_stems"])
    L_mix_all = torch.stack(L_mix_list, dim=0).to(device)        # [N, T, 64]
    L_stems_all = torch.stack(L_stems_list, dim=0).to(device)    # [N, S, T, 64]
    N, T, _ = L_mix_all.shape
    S = L_stems_all.shape[1]
    print(f"  N={N}, T={T}, S={S}, stems={STEMS}")

    # ----- Identity baseline: predict 6 copies of L_mix -----
    L_id = L_mix_all.unsqueeze(1).expand(-1, S, -1, -1).contiguous()  # [N,S,T,64]
    id_l1_per_stem = (L_id - L_stems_all).abs().mean(dim=(0, 2, 3))   # [S]
    id_l1 = id_l1_per_stem.mean().item()
    print(f"\nidentity baseline (predict 6× mix):")
    print(f"  overall L1: {id_l1:.5f}")
    for s, n in zip(STEMS, id_l1_per_stem.tolist()):
        print(f"    {s:7s}: {n:.5f}")

    # ----- For each ckpt -----
    print("\n" + "=" * 80)
    print(f"{'ckpt':<30} {'lat L1':>10} {'vs id':>10} {'%cut':>8}   stem breakdown")
    print("-" * 80)
    rows = []
    for ckpt in args.ckpts:
        student, _ = load_student(ckpt, device=device)

        # Batch in chunks to avoid OOM
        preds = []
        bs = 32
        for i in range(0, N, bs):
            preds.append(student(L_mix_all[i : i + bs]))
        pred = torch.cat(preds, dim=0)                                 # [N,S,T,64]
        l1_per_stem = (pred - L_stems_all).abs().mean(dim=(0, 2, 3))   # [S]
        l1 = l1_per_stem.mean().item()
        cut = (id_l1 - l1) / id_l1 * 100
        rows.append((ckpt, l1, l1_per_stem.tolist()))

        name = os.path.basename(ckpt)
        stems_str = " ".join(f"{n:.3f}" for n in l1_per_stem.tolist())
        print(f"{name:<30} {l1:>10.5f} {id_l1 - l1:>+10.5f} {cut:>7.1f}%  {stems_str}")

    print("-" * 80)
    print(f"identity                       {id_l1:>10.5f}")

    # ----- Audio-domain (STFT) eval on the latest ckpt -----
    print("\n" + "=" * 80)
    print(f"Decoded STFT log-mag eval on last ckpt ({os.path.basename(args.ckpts[-1])}):")
    print("=" * 80)
    student, _ = load_student(args.ckpts[-1], device=device)

    nd = min(args.decode_n, N)
    stft_id_per_stem = np.zeros(S)
    stft_st_per_stem = np.zeros(S)
    counted = 0
    for i in range(nd):
        L_mix_i = L_mix_all[i : i + 1]                       # [1,T,64]
        L_tgt_i = L_stems_all[i]                             # [S,T,64]
        L_id_i = L_mix_i.expand(S, -1, -1)                   # [S,T,64]
        L_st_i = student(L_mix_i)[0]                         # [S,T,64]

        # Decode each stem (S decodes per item)
        def dec(x_STC):
            x = x_STC.transpose(1, 2).to(torch.bfloat16)
            return vae.decode(x).sample.float().cpu().numpy()  # [S,2,Saudio]

        wav_tgt = dec(L_tgt_i)
        wav_id = dec(L_id_i)
        wav_st = dec(L_st_i)
        S_min = min(wav_tgt.shape[-1], wav_id.shape[-1], wav_st.shape[-1])
        for s in range(S):
            stft_id_per_stem[s] += stft_log_l1(wav_id[s, :, :S_min], wav_tgt[s, :, :S_min])
            stft_st_per_stem[s] += stft_log_l1(wav_st[s, :, :S_min], wav_tgt[s, :, :S_min])
        counted += 1
        if (i + 1) % 4 == 0:
            print(f"  decoded {i+1}/{nd}")

    stft_id_per_stem /= max(1, counted)
    stft_st_per_stem /= max(1, counted)
    print(f"\n{'stem':<8} {'identity':>10} {'student':>10} {'delta':>10} {'%cut':>8}")
    for s, name in enumerate(STEMS):
        d = stft_id_per_stem[s] - stft_st_per_stem[s]
        cut = d / stft_id_per_stem[s] * 100
        print(f"{name:<8} {stft_id_per_stem[s]:>10.4f} {stft_st_per_stem[s]:>10.4f} {d:>+10.4f} {cut:>7.1f}%")
    overall_id = stft_id_per_stem.mean()
    overall_st = stft_st_per_stem.mean()
    overall_cut = (overall_id - overall_st) / overall_id * 100
    print(f"{'OVERALL':<8} {overall_id:>10.4f} {overall_st:>10.4f} {overall_id - overall_st:>+10.4f} {overall_cut:>7.1f}%")


if __name__ == "__main__":
    main()
