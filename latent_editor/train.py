"""
Train the boundary-repair latent editor.

Loss = latent L1 (full window, weighted toward boundary)
     + multi-resolution STFT on the decoded waveform around the cut.

The Oobleck VAE is loaded from the ACE-Step handler and frozen. We only
backprop through the decoder for the small windowed STFT loss to keep
VRAM bounded.

Usage:
    python -m latent_editor.train \
        --latent-roots /scratch/Latents2 /scratch/stemphonic/data/ossl_latents \
        --out /scratch/latent_editor_ckpts \
        --steps 50000 --batch 4
"""
from __future__ import annotations
import argparse, os, sys, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa

from latent_editor.dataset import (
    LatentSpliceDataset, CachedSpliceDataset, collate, SAMPLES_PER_FRAME, EditorBatch,
)
from latent_editor.model import LatentEditor


# ---------- losses ----------

def multi_res_stft_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x, y: [B, 2, S] in [-1,1]
    loss = 0.0
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=x.device)
        Xs = torch.stft(
            x.reshape(-1, x.shape[-1]), n_fft=n_fft, hop_length=hop,
            win_length=n_fft, window=win, return_complex=True,
        ).abs()
        Ys = torch.stft(
            y.reshape(-1, y.shape[-1]), n_fft=n_fft, hop_length=hop,
            win_length=n_fft, window=win, return_complex=True,
        ).abs()
        loss = loss + F.l1_loss(torch.log1p(Xs), torch.log1p(Ys))
    return loss / 3.0


def latent_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred,target: [B,T,64], mask: [B,T]
    diff = (pred - target).abs().mean(-1)  # [B,T]
    w = 1.0 + 4.0 * mask                   # 5x weight inside boundary band
    return (diff * w).mean()


# ---------- training ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-roots", nargs="+", default=None,
                    help="Directories of *.vae.pt files (used when --cache-dir not given)")
    ap.add_argument("--cache-dir", default=None,
                    help="Pre-built shard cache from build_cache.py (preferred)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--win-frames", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--stft-radius-frames", type=int, default=8)
    ap.add_argument("--stft-weight", type=float, default=1.0)
    ap.add_argument("--latent-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    # ---- frozen VAE from ACE-Step handler ----
    print("Loading ACE-Step handler (for frozen Oobleck VAE)...")
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

    # ---- data ----
    if args.cache_dir:
        ds = CachedSpliceDataset(args.cache_dir)
        dl = DataLoader(
            ds, batch_size=args.batch, num_workers=args.workers,
            collate_fn=collate, shuffle=True, drop_last=True, persistent_workers=args.workers > 0,
        )
        print(f"using cached dataset: {len(ds)} examples from {args.cache_dir}")
    else:
        if not args.latent_roots:
            raise SystemExit("must pass --cache-dir or --latent-roots")
        ds = LatentSpliceDataset(
            roots=args.latent_roots, vae=vae, win_frames=args.win_frames, device=device,
        )
        # workers=0 because the dataset calls the GPU VAE
        dl = DataLoader(ds, batch_size=args.batch, num_workers=0, collate_fn=collate)

    # ---- model ----
    model = LatentEditor(max_len=args.win_frames).to(device)
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
        b: EditorBatch = batch
        L_naive = b.L_naive.to(device)
        L_target = b.L_target.to(device)
        mask = b.mask.to(device)
        phase = b.phase.to(device)
        cut_frame = b.cut_frame.to(device)
        wav_target = b.wav_target.to(device) if b.wav_target is not None else None

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(L_naive, mask, phase)            # [B,T,64]
            l_lat = latent_loss(pred, L_target, mask)

            # decode a *small* window around the cut for STFT loss
            r = args.stft_radius_frames
            T = pred.shape[1]
            # assume cut_frame is identical across the batch (it is, = T//2)
            cf = int(cut_frame[0].item())
            lo = max(0, cf - r); hi = min(T, cf + r + 1)
            sub_pred = pred[:, lo:hi].transpose(1, 2)     # [B,64,w]
            wav_pred = vae.decode(sub_pred.to(torch.bfloat16)).sample.float()
            if wav_target is not None:
                s_lo = lo * SAMPLES_PER_FRAME
                s_hi = hi * SAMPLES_PER_FRAME
                wav_tgt = wav_target[:, :, s_lo:s_hi]
            else:
                # decode the matching slice of L_target on the fly
                with torch.no_grad():
                    sub_tgt = L_target[:, lo:hi].transpose(1, 2).to(torch.bfloat16)
                    wav_tgt = vae.decode(sub_tgt).sample.float()
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
                f"({dt/(step+1):.2f}s/it)"
            )
        if step > 0 and step % args.save_every == 0:
            ckpt = {"model": model.state_dict(), "step": step, "args": vars(args)}
            torch.save(ckpt, os.path.join(args.out, f"editor_{step:06d}.pt"))
            print(f"  saved editor_{step:06d}.pt")
        step += 1

    torch.save(
        {"model": model.state_dict(), "step": step, "args": vars(args)},
        os.path.join(args.out, "editor_final.pt"),
    )
    print("done.")


if __name__ == "__main__":
    main()
