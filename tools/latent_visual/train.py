"""Train a tiny latent → peak-envelope model.

Goal: given a [T, 64] latent, predict a [T, 2] (min, max) amplitude
envelope so the studio can render the entire track's waveform display
INSTANTLY without waiting for the WebGPU decoder. Output runs in
~1 ms in the browser via ONNX Runtime Web.

Architecture: 4-layer 1D conv on the latent time axis. ~100K params.
Trained against decoded ground truth from the Oobleck VAE.

Usage:
    python -m latent_visual.train \\
        --vae /scratch/ACE-Step-1.5/checkpoints/vae \\
        --latent-roots /scratch/Latents2/protoolsA \\
        --out /scratch/latent_visual_ckpts \\
        --steps 5000 --batch 8 --win 64
"""
from __future__ import annotations
import argparse, os, glob, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck

SR = 48000
SAMPLES_PER_FRAME = 1920


class LatentToPeakEnvelope(nn.Module):
    """Maps [B, 64, T] latent → [B, 2, T] (min, max) amplitude envelope.
    Tiny 4-layer causal-ish 1D conv, ~100K params."""

    def __init__(self, in_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 2, kernel_size=1),
        )

    def forward(self, latent):
        # latent: [B, 64, T]
        return self.net(latent)  # [B, 2, T]


def compute_peak_envelope(audio_2chan, frame_size=SAMPLES_PER_FRAME):
    """audio_2chan: [2, S] tensor → [2, T] (min, max) per frame.
    Min and max are signed-amplitude bounds; the visual renderer draws
    a vertical line from min to max at each pixel."""
    n = audio_2chan.shape[1]
    T = n // frame_size
    a = audio_2chan[:, :T * frame_size].reshape(2, T, frame_size).mean(dim=0)  # mono [T, fs]
    mx = a.max(dim=-1).values
    mn = a.min(dim=-1).values
    return torch.stack([mn, mx], dim=0)  # [2, T]


def make_pairs(vae, latent_paths, win, n_pairs):
    """Build (latent_window, envelope_window) pairs by decoding random
    latent windows through the VAE and computing the peak envelope of
    the decoded audio."""
    pairs = []
    for _ in range(n_pairs):
        path = random.choice(latent_paths)
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        L = blob.get("latents") if isinstance(blob, dict) else blob
        if L is None or L.dim() != 2:
            continue
        # Latents in /scratch/Latents2 are stored as [64, T] (channels-first)
        if L.shape[0] == 64 and L.shape[1] != 64:
            L = L.transpose(0, 1)            # → [T, 64]
        if L.shape[1] != 64 or L.shape[0] < win:
            continue
        s = random.randint(0, L.shape[0] - win)
        L_win = L[s:s + win].float()  # [win, 64]
        # Decode this window through the VAE
        with torch.no_grad():
            z = L_win.transpose(0, 1).unsqueeze(0).cuda().to(torch.bfloat16)
            audio = vae.decode(z).sample.squeeze(0).float().cpu()  # [2, win*1920]
        env = compute_peak_envelope(audio)  # [2, win]
        if env.shape[1] != win:
            # Decoder may produce slightly off length — pad/trim
            if env.shape[1] > win:
                env = env[:, :win]
            else:
                env = torch.cat([env, env[:, -1:].expand(2, win - env.shape[1])], dim=1)
        pairs.append((L_win, env))
    return pairs


def train(args):
    torch.manual_seed(0)
    random.seed(0)
    print(f"[init] loading vae from {args.vae}")
    vae = AutoencoderOobleck.from_pretrained(args.vae).to("cuda").to(torch.bfloat16).eval()

    # Find every .vae.pt across the latent roots
    latent_paths = []
    for root in args.latent_roots.split(","):
        latent_paths.extend(glob.glob(os.path.join(root, "**", "*.vae.pt"), recursive=True))
    print(f"[init] found {len(latent_paths)} latent files")
    if not latent_paths:
        raise RuntimeError("no latent files found")

    model = LatentToPeakEnvelope().cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.log")
    log = open(log_path, "w")

    # Pre-build a fixed-size pool of training pairs
    print(f"[init] building {args.pool_size} training pairs (this takes a bit) …")
    pool = make_pairs(vae, latent_paths, args.win, args.pool_size)
    print(f"[init] pool ready: {len(pool)} pairs")
    if not pool:
        raise RuntimeError("no training pairs built")

    t0 = time.perf_counter()
    for step in range(args.steps):
        idxs = random.sample(range(len(pool)), min(args.batch, len(pool)))
        L_batch = torch.stack([pool[i][0] for i in idxs]).transpose(1, 2).cuda()  # [B, 64, T]
        env_batch = torch.stack([pool[i][1] for i in idxs]).cuda()                # [B, 2, T]

        pred = model(L_batch)
        loss = F.l1_loss(pred, env_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            msg = f"step {step:5d}  loss={loss.item():.5f}  ({elapsed:.0f}s)"
            print(msg); log.write(msg + "\n"); log.flush()

        if step > 0 and step % args.refresh_pool == 0:
            new_n = max(64, args.pool_size // 4)
            print(f"[pool refresh @ {step}] adding {new_n} new pairs")
            pool.extend(make_pairs(vae, latent_paths, args.win, new_n))

    ckpt_path = os.path.join(args.out, "latent_visual_final.pt")
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"[done] {ckpt_path}  (loss={loss.item():.5f})")

    # Export to ONNX (single file, packed)
    print("[export] tracing to ONNX …")
    model.eval()
    dummy = torch.randn(1, 64, args.win, device="cuda")
    onnx_path = os.path.join(args.out, "latent_visual.onnx")
    torch.onnx.export(
        model.cpu(), (dummy.cpu(),),
        onnx_path,
        input_names=["latent"],
        output_names=["envelope"],
        dynamic_axes={"latent": {0: "batch", 2: "frames"},
                      "envelope": {0: "batch", 2: "frames"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"[done] {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="/scratch/ACE-Step-1.5/checkpoints/vae")
    ap.add_argument("--latent-roots", default="/scratch/Latents2/protoolsA")
    ap.add_argument("--out", default="/scratch/latent_visual_ckpts")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--win", type=int, default=128)
    ap.add_argument("--pool-size", type=int, default=512)
    ap.add_argument("--refresh-pool", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    train(args)
