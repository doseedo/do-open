#!/usr/bin/env python3
"""Train HTDemucs-Lite 6-stem: htdemucs_6s with attention replaced by conv fusion.

Uses pretrained htdemucs_6s encoder/decoder weights.
Replaces CrossTransformer with conv fusion.
Outputs per-stem embeddings [B, 6, 128] and STFT masks [B, 6, F, T].

6 stems: drums, bass, other, vocals, guitar, piano
"""
import argparse
import importlib.util
import math
import os
import random
import re
import glob as _glob
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from htdemucs_lite import ConvFusion

STEMS_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]
SR = 44100

SEM_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_V1_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"


class HTDemucsLite6(nn.Module):
    """HTDemucs-Lite for 6 stems, based on htdemucs_6s."""

    def __init__(self, embed_dim=128):
        super().__init__()
        from demucs.pretrained import get_model
        teacher = get_model('htdemucs_6s').models[0]

        self.encoder = teacher.encoder
        self.decoder = teacher.decoder
        self.tencoder = teacher.tencoder
        self.tdecoder = teacher.tdecoder
        self.freq_emb = teacher.freq_emb
        self.freq_emb_scale = teacher.freq_emb_scale

        self.samplerate = teacher.samplerate
        self.audio_channels = teacher.audio_channels
        self.sources = teacher.sources  # 6 sources
        self.nfft = teacher.nfft
        self.depth = len(self.encoder)
        self.use_train_segment = False

        self._spec = teacher._spec
        self._magnitude = teacher._magnitude
        self._ispec = teacher._ispec
        self._mask = teacher._mask

        # htdemucs_6s has no channel_upsampler (bottom_channels=0)
        self.conv_fusion = ConvFusion(channels=384, n_freq=8)

        del teacher

        # Per-stem embedding head
        self.embed_dim = embed_dim
        self.n_stems = len(self.sources)
        self.stem_queries = nn.Parameter(torch.randn(self.n_stems, 1, 384) * 0.02)
        self.embed_proj = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, mix):
        length = mix.shape[-1]

        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved = []
        saved_t = []
        lengths = []
        lengths_t = []

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        # Conv fusion instead of crosstransformer
        x, xt = self.conv_fusion(x, xt)

        emb_features = xt  # [B, 384, T_time]

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        zout = self._mask(z, x)
        x = self._ispec(zout, length)

        xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        stems = xt + x  # [B, 6, 2, N]

        # Per-stem embeddings via query-based pooling
        embeddings = []
        for i in range(S):
            q = self.stem_queries[i].expand(B, -1, -1)
            kv = emb_features.transpose(1, 2)
            attn = torch.bmm(q, kv.transpose(1, 2)).softmax(dim=-1)
            pooled = torch.bmm(attn, kv).squeeze(1)
            embeddings.append(self.embed_proj(pooled))
        embeddings = torch.stack(embeddings, dim=1)  # [B, 6, 128]

        # STFT masks from separated stems
        with torch.no_grad():
            stem_mono = stems.mean(dim=2)
            n_fft, hop = 2048, 512
            window = torch.hann_window(n_fft, device=mix.device)
            mags = []
            for i in range(S):
                spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                                  return_complex=True).abs()
                mags.append(spec)
            mags = torch.stack(mags, dim=1)
            stft_masks = mags / (mags.sum(dim=1, keepdim=True) + 1e-8)

        return {
            "stems": stems,             # [B, 6, 2, N]
            "embedding": embeddings,    # [B, 6, 128]
            "stft_masks": stft_masks,   # [B, 6, F, T]
        }


def load_sem_encoder(device="cuda"):
    spec = importlib.util.spec_from_file_location(
        "sem", os.path.join(SEM_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sd = torch.load(SEM_V1_CKPT, map_location="cpu", weights_only=False)
    m = mod.SemanticEncoderWithHeads().to(device).eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters(): p.requires_grad = False
    return m


def load_htdemucs_6s_teacher(device="cuda"):
    from demucs.pretrained import get_model
    m = get_model('htdemucs_6s').to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


class StemDataset6(Dataset):
    """Load mix wavs + per-stem VAE latents for 6-stem training."""
    def __init__(self, crop_sec=6.0, seed=0):
        self.crop_samples = int(crop_sec * SR)
        self.rng = random.Random(seed)
        self.items = []
        SPF = 1920  # samples per frame at 48kHz, but htdemucs is 44.1kHz

        musdb_wav = Path("/scratch/musdb18_wavs")
        musdb_lat = Path("/scratch/musdb18_latents")

        STEM_FILES = {
            "drums": "drums.vae.pt", "bass": "bass.vae.pt",
            "other": "other.vae.pt", "vocals": "vocals.vae.pt",
            "guitar": "guitar.vae.pt", "piano": "piano.vae.pt",
        }
        for split in ["train", "test"]:
            for td in sorted((musdb_lat / split).iterdir()):
                if not td.is_dir():
                    continue
                mix_wav = musdb_wav / split / td.name / "mixture.wav"
                if not mix_wav.exists():
                    continue
                stem_paths = {}
                mask = []
                for s in STEMS_6:
                    p = td / STEM_FILES[s]
                    if p.exists():
                        stem_paths[s] = str(p)
                        mask.append(1.0)
                    else:
                        stem_paths[s] = None
                        mask.append(0.0)
                if sum(mask) < 2:
                    continue
                self.items.append({
                    "mix_wav": str(mix_wav),
                    "stem_paths": stem_paths,
                    "mask": mask,
                })
        print(f"[lite6-ds] {len(self.items)} tracks")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            audio, sr = sf.read(it["mix_wav"], dtype="float32")
            audio = torch.from_numpy(audio.T if audio.ndim > 1 else audio[None]).float()
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        N = audio.shape[1]
        if N > self.crop_samples:
            start = self.rng.randint(0, N - self.crop_samples)
            audio = audio[:, start:start + self.crop_samples]
        elif N < self.crop_samples:
            audio = F.pad(audio, (0, self.crop_samples - N))
            start = 0
        else:
            start = 0

        SPF = 1920
        crop_frames = self.crop_samples // SPF
        start_frame = start // SPF
        stem_lats = []
        for s in STEMS_6:
            if it["stem_paths"][s] is not None:
                raw = torch.load(it["stem_paths"][s], map_location="cpu",
                                 weights_only=False)
                z = raw["latents"] if isinstance(raw, dict) else raw
                if z.dim() == 2 and z.shape[0] == 64:
                    z = z.t()
                z = z.float()
                z = z[start_frame:start_frame + crop_frames]
                if z.shape[0] < crop_frames:
                    z = F.pad(z, (0, 0, 0, crop_frames - z.shape[0]))
                stem_lats.append(z)
            else:
                stem_lats.append(torch.zeros(crop_frames, 64))

        return audio, torch.stack(stem_lats), torch.tensor(it["mask"])


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/htdemucs_lite_6stem_ckpts")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop-sec", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-finetune", type=float, default=3e-5)
    ap.add_argument("--unfreeze-step", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lambda-stems", type=float, default=1.0)
    ap.add_argument("--lambda-emb", type=float, default=1.0)
    ap.add_argument("--lambda-masks", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = StemDataset6(crop_sec=args.crop_sec)
    if len(ds) == 0:
        print("ERROR: empty dataset")
        return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[lite6] loading frozen teachers...")
    teacher = load_htdemucs_6s_teacher()
    sem_enc = load_sem_encoder()
    print("[lite6] teachers loaded")

    print("[lite6] building HTDemucsLite6...")
    model = HTDemucsLite6(embed_dim=128).cuda()
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    fusion_p = sum(p.numel() for p in model.conv_fusion.parameters()) / 1e6
    print(f"[lite6] {total_p:.1f}M total ({fusion_p:.1f}M fusion, 6 stems)")

    # Phase 1: freeze encoder/decoder
    frozen_modules = [model.encoder, model.decoder, model.tencoder, model.tdecoder,
                      model.freq_emb]
    for mod in frozen_modules:
        for p in mod.parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable) / 1e6
    print(f"[lite6] phase 1: training {n_train:.1f}M params, encoder/decoder frozen")

    opt = torch.optim.AdamW(trainable, lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(prog * math.pi))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    step = 0

    # Auto-resume
    _ckpts = [f for f in _glob.glob(os.path.join(args.out, "*_step*.pt"))
              if re.search(r'step(\d+)', f)]
    if _ckpts:
        _latest = max(_ckpts, key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
        _sd = torch.load(_latest, map_location="cuda", weights_only=False)
        model.load_state_dict(_sd["model"])
        step = _sd.get("step", 0)
        for _ in range(step):
            sched.step()
        print(f"[resume] from step {step} ({os.path.basename(_latest)})")

    unfrozen = step >= args.unfreeze_step
    hist = {k: [] for k in ("total", "stems", "emb", "masks")}
    t0 = time.time()

    while step < args.steps:
        for audio, stem_lats, mask in loader:
            audio = audio.cuda(non_blocking=True)
            stem_lats = stem_lats.cuda(non_blocking=True)  # [B, 6, T_lat, 64]
            mask = mask.cuda(non_blocking=True)  # [B, 6]
            B = audio.shape[0]

            # Phase 2: unfreeze
            if not unfrozen and step >= args.unfreeze_step:
                print(f"[lite6] step {step}: unfreezing, LR → {args.lr_finetune}")
                for mod in frozen_modules:
                    for p in mod.parameters():
                        p.requires_grad = True
                all_params = list(model.parameters())
                opt = torch.optim.AdamW(all_params, lr=args.lr_finetune,
                                        betas=(0.9, 0.95), weight_decay=0.01)
                remaining = args.steps - step
                def lr_lambda_ft(s):
                    if s < 200:
                        return (s + 1) / 200
                    prog = (s - 200) / max(1, remaining - 200)
                    return 0.5 * (1 + math.cos(prog * math.pi))
                sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_ft)
                unfrozen = True

            with torch.no_grad():
                from demucs.apply import apply_model
                tgt_stems = apply_model(teacher, audio.float(),
                                        device=str(audio.device))  # [B, 6, 2, N]

                stem_mono = tgt_stems.mean(dim=2)
                n_fft, hop = 2048, 512
                window = torch.hann_window(n_fft, device=audio.device)
                mags = []
                for i in range(6):
                    spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                                      return_complex=True).abs()
                    mags.append(spec)
                mags_t = torch.stack(mags, dim=1)
                tgt_masks = mags_t / (mags_t.sum(dim=1, keepdim=True) + 1e-8)

                tgt_embs = []
                for si in range(6):
                    lat = stem_lats[:, si]
                    tgt_embs.append(sem_enc(lat)["embedding"])
                tgt_emb = torch.stack(tgt_embs, dim=1)  # [B, 6, 128]

            opt.zero_grad(set_to_none=True)
            out = model(audio.float())

            # Stem loss with presence mask
            pred_stems = out["stems"]
            T_s = min(pred_stems.shape[-1], tgt_stems.shape[-1])
            diff_s = (pred_stems[..., :T_s] - tgt_stems[..., :T_s]).abs()
            mask_s = mask[:, :, None, None].expand_as(diff_s)
            l_stems = (diff_s * mask_s).sum() / mask_s.sum().clamp(min=1)

            # Embedding loss with mask
            diff_e = (out["embedding"] - tgt_emb).abs()
            mask_e = mask[:, :, None].expand_as(diff_e)
            l_emb = (diff_e * mask_e).sum() / mask_e.sum().clamp(min=1)

            # STFT mask loss
            pred_masks = out["stft_masks"]
            F_m = min(pred_masks.shape[2], tgt_masks.shape[2])
            T_m = min(pred_masks.shape[3], tgt_masks.shape[3])
            l_masks = F.mse_loss(pred_masks[:, :, :F_m, :T_m],
                                 tgt_masks[:, :, :F_m, :T_m])

            loss = (args.lambda_stems * l_stems +
                    args.lambda_emb * l_emb +
                    args.lambda_masks * l_masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            hist["total"].append(loss.item())
            hist["stems"].append(l_stems.item())
            hist["emb"].append(l_emb.item())
            hist["masks"].append(l_masks.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                phase = "ft" if unfrozen else "frozen"
                print(f"[step {step:5d}] total={avg(hist['total']):.4f} "
                      f"stems={avg(hist['stems']):.4f} "
                      f"emb={avg(hist['emb']):.4f} "
                      f"masks={avg(hist['masks']):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"({phase}) elapsed={time.time()-t0:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"lite6_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "lite6_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
