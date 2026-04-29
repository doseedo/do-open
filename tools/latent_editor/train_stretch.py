"""
Train the latent stretch-artifact cleaner.

Loss: latent L1 + multi-resolution STFT on the *full* decoded window.
Unlike the boundary editor, the artifact is global, so we decode the whole
window (still cheap at win=64 frames ≈ 2.56 s).

Usage with cache (preferred):
    python -m latent_editor.train_stretch \
        --cache-dir /scratch/latent_stretch_cache \
        --out /scratch/latent_stretch_ckpts \
        --steps 80000 --batch 8 --workers 4
"""
from __future__ import annotations
import argparse, os, sys, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa

from latent_editor.stretch_dataset import (
    LatentStretchDataset, CachedStretchDataset, stretch_collate,
)
from latent_editor.stretch_model import LatentStretchCleaner
from latent_editor.train import multi_res_stft_loss
from latent_editor.dataset import SAMPLES_PER_FRAME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-roots", nargs="+", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=80_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--win-frames", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--latent-weight", type=float, default=1.0)
    ap.add_argument("--stft-weight", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    print("loading frozen VAE...")
    handler = AceStepHandler()
    handler.initialize_service(
        project_root="/scratch/ACE-Step-1.5",
        config_path="acestep-v15-sft",
        device=device,
    )
    vae = handler.vae
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    if args.cache_dir:
        ds = CachedStretchDataset(args.cache_dir)
        dl = DataLoader(
            ds, batch_size=args.batch, num_workers=args.workers,
            collate_fn=stretch_collate, shuffle=True, drop_last=True,
            persistent_workers=args.workers > 0,
        )
        print(f"cached dataset: {len(ds)} examples")
    else:
        if not args.latent_roots:
            raise SystemExit("must pass --cache-dir or --latent-roots")
        ds = LatentStretchDataset(
            roots=args.latent_roots, vae=vae, win_frames=args.win_frames, device=device,
        )
        dl = DataLoader(ds, batch_size=args.batch, num_workers=0, collate_fn=stretch_collate)

    model = LatentStretchCleaner(max_len=args.win_frames).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    def infinite(loader):
        while True:
            for b in loader:
                yield b

    step = 0
    t0 = time.time()
    for batch in infinite(dl):
        if step >= args.steps:
            break
        L_in  = batch["L_input"].to(device)
        L_tgt = batch["L_target"].to(device)
        r     = batch["stretch_r"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(L_in, r)
            l_lat = F.l1_loss(pred, L_tgt)

            wav_pred = vae.decode(pred.transpose(1, 2).to(torch.bfloat16)).sample.float()
            with torch.no_grad():
                wav_tgt = vae.decode(L_tgt.transpose(1, 2).to(torch.bfloat16)).sample.float()
            n = min(wav_pred.shape[-1], wav_tgt.shape[-1])
            l_stft = multi_res_stft_loss(wav_pred[..., :n], wav_tgt[..., :n])

            loss = args.latent_weight * l_lat + args.stft_weight * l_stft

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(
                f"step {step:6d}  loss {loss.item():.4f}  "
                f"lat {l_lat.item():.4f}  stft {l_stft.item():.4f}  "
                f"r∈[{r.min().item():.2f},{r.max().item():.2f}]  "
                f"({dt/(step+1):.2f}s/it)"
            )
        if step > 0 and step % args.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step, "args": vars(args)},
                os.path.join(args.out, f"stretch_{step:06d}.pt"),
            )
            print(f"  saved stretch_{step:06d}.pt")
        step += 1

    torch.save(
        {"model": model.state_dict(), "step": step, "args": vars(args)},
        os.path.join(args.out, "stretch_final.pt"),
    )
    print("done.")


if __name__ == "__main__":
    main()
