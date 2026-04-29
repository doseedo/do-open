"""Train the latent drum sub-separator.

Loss = per-stem latent L1 + multi-res STFT on the *decoded* sum of all
stems vs. the decoded mix (cycle consistency) and per-stem decoded vs
target. Same recipe as latent_editor.train_stretch.

Usage:
    python -m latent_drumsep.train \
        --cache-dir /scratch/latent_drumsep_cache \
        --out /scratch/latent_drumsep_ckpts \
        --steps 60000 --batch 8 --workers 0 \
        --latent-weight 1.0 --stft-weight 5.0
"""
from __future__ import annotations
import argparse, os, sys, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa: E402

from latent_editor.train import multi_res_stft_loss  # noqa: E402
from latent_drumsep.dataset import CachedDrumsepDataset, collate
from latent_drumsep.model import LatentDrumSubsep
from latent_drumsep import STEMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--win-frames", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--latent-weight", type=float, default=1.0)
    ap.add_argument("--stft-weight", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print("loading frozen ACE VAE (for STFT loss)...")
    handler = AceStepHandler()
    handler.initialize_service(project_root="/scratch/ACE-Step-1.5",
                               config_path="acestep-v15-sft", device=device)
    vae = handler.vae
    for p in vae.parameters(): p.requires_grad = False
    vae.eval()

    ds = CachedDrumsepDataset(args.cache_dir)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                    collate_fn=collate, shuffle=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    print(f"cached dataset: {len(ds)} examples")

    model = LatentDrumSubsep(n_stems=len(STEMS), max_len=args.win_frames).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    def infinite(loader):
        while True:
            for b in loader: yield b

    step = 0
    t0 = time.time()
    for batch in infinite(dl):
        if step >= args.steps: break
        L_mix   = batch["L_mix"].to(device)            # [B, T, 64]
        L_stems = batch["L_stems"].to(device)          # [B, S, T, 64]
        B, S, T, D = L_stems.shape

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(L_mix)                         # [B, S, T, 64]
            l_lat = F.l1_loss(pred, L_stems)

            # STFT loss: decode each stem (flatten S into batch)
            pred_flat = pred.reshape(B * S, T, D).transpose(1, 2)   # [B*S, 64, T]
            tgt_flat  = L_stems.reshape(B * S, T, D).transpose(1, 2)
            wav_pred = vae.decode(pred_flat.to(torch.bfloat16)).sample.float()
            with torch.no_grad():
                wav_tgt = vae.decode(tgt_flat.to(torch.bfloat16)).sample.float()
            n = min(wav_pred.shape[-1], wav_tgt.shape[-1])
            l_stft = multi_res_stft_loss(wav_pred[..., :n], wav_tgt[..., :n])

            loss = args.latent_weight * l_lat + args.stft_weight * l_stft

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d}  loss {loss.item():.4f}  "
                  f"lat {l_lat.item():.4f}  stft {l_stft.item():.4f}  "
                  f"({dt/(step+1):.2f}s/it)")
        if step > 0 and step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "step": step,
                        "args": vars(args), "stems": list(STEMS)},
                       os.path.join(args.out, f"drumsep_{step:06d}.pt"))
            print(f"  saved drumsep_{step:06d}.pt")
        step += 1

    torch.save({"model": model.state_dict(), "step": step,
                "args": vars(args), "stems": list(STEMS)},
               os.path.join(args.out, "drumsep_final.pt"))
    print("done.")


if __name__ == "__main__":
    main()
