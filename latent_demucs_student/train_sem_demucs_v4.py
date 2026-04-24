#!/usr/bin/env python3
"""Train SemDemucs v4: STFT masks + per-stem sem embeddings.

Mask targets: STFT ratio masks from frozen htdemucs teacher.
Embedding targets: SemanticEncoder v1 on GT stem latents.

Initializes from v3 checkpoint (pitch/rms/embedding/vocal heads).
Mask head is new (STFT frequency-domain, replaces v3's time-only mask).
"""
import argparse
import importlib.util
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sem_demucs import SemDemucs

STEMS = ["drums", "bass", "other", "vocals"]  # htdemucs order
STEMS_MUSDB = ["drums", "bass", "vocals", "other"]  # MUSDB order
# Map: MUSDB index → htdemucs index
MUSDB_TO_HTD = [0, 1, 3, 2]  # drums→0, bass→1, vocals→3, other→2
SR = 48000
SPF = 1920

SEM_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_V1_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"


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


def load_htdemucs_teacher(device="cuda"):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    m = get_model('htdemucs').to(device).eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def compute_stft_masks_from_teacher(teacher, mix, n_fft=2048, hop=512):
    """Run htdemucs, compute STFT ratio masks from separated stems."""
    from demucs.apply import apply_model
    with torch.no_grad():
        stems = apply_model(teacher, mix, device=str(mix.device))  # [B, 4, 2, N]
        stem_mono = stems.mean(dim=2)  # [B, 4, N]
        window = torch.hann_window(n_fft, device=mix.device)
        mags = []
        for i in range(4):
            spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                              return_complex=True).abs()
            mags.append(spec)
        mags = torch.stack(mags, dim=1)  # [B, 4, F, T_stft]
        masks = mags / (mags.sum(dim=1, keepdim=True) + 1e-8)
    return masks  # [B, 4, F, T_stft] in htdemucs stem order


def _load_stem_wav(path):
    audio, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(audio.T if audio.ndim > 1 else audio[None]).float()
    if t.shape[0] == 1: t = t.repeat(2, 1)
    elif t.shape[0] > 2: t = t[:2]
    return t


def _load_latent(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64: z = z.t()
    return z.float()


class V4Dataset(Dataset):
    def __init__(self, crop_sec=6.0, seed=0):
        self.crop_samples = int(crop_sec * SR)
        self.rng = random.Random(seed)
        self.items = []

        musdb_wav = Path("/scratch/musdb18_wavs")
        musdb_lat = Path("/scratch/musdb18_latents")
        for split in ["train", "test"]:
            for td in sorted((musdb_lat / split).iterdir()):
                if not td.is_dir(): continue
                mix_wav = musdb_wav / split / td.name / "mixture.wav"
                if not mix_wav.exists(): continue
                stem_lats = {}
                for s in STEMS_MUSDB:
                    p = td / f"{s}.vae.pt"
                    if p.exists(): stem_lats[s] = str(p)
                if len(stem_lats) != 4: continue
                self.items.append({"mix_wav": str(mix_wav), "stem_lats": stem_lats})
        print(f"[v4-ds] {len(self.items)} tracks")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            audio = _load_stem_wav(it["mix_wav"])
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

        crop_frames = self.crop_samples // SPF
        start_frame = start // SPF
        # Load stem latents in htdemucs order
        stem_lats = []
        for s in STEMS:  # htdemucs order: drums, bass, other, vocals
            lat = _load_latent(it["stem_lats"][s])
            lat = lat[start_frame:start_frame + crop_frames]
            if lat.shape[0] < crop_frames:
                lat = F.pad(lat, (0, 0, 0, crop_frames - lat.shape[0]))
            stem_lats.append(lat)

        return audio, torch.stack(stem_lats)


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/sem_demucs_v4_ckpts")
    ap.add_argument("--resume-v3", default="",
                    help="v3 checkpoint to initialize pitch/rms/emb/vocal heads from")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop-sec", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lambda-mask", type=float, default=2.0)
    ap.add_argument("--lambda-emb", type=float, default=1.0)
    ap.add_argument("--lambda-pitch", type=float, default=1.0)
    ap.add_argument("--lambda-rms", type=float, default=1.0)
    ap.add_argument("--lambda-vocal", type=float, default=0.3)
    ap.add_argument("--channels", type=int, default=64,
                    help="encoder channels (64→2.2M, 128→8.3M, 192→18.2M, 256→31.8M)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = V4Dataset(crop_sec=args.crop_sec)
    if len(ds) == 0: print("ERROR: no data"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate, persistent_workers=args.workers > 0)

    print("[v4] loading frozen teachers...")
    sem_enc = load_sem_encoder()
    teacher = load_htdemucs_teacher()

    # Load pitch + visual teachers
    PITCH_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_pitch")
    PITCH_CKPT = "/scratch/latent_pitch_ckpts/pitch_final.pt"
    VISUAL_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_visual")
    VISUAL_CKPT = "/scratch/latent_visual_ckpts/latent_visual_final.pt"

    def _load_teacher(dir_path, ckpt_path, model_file, class_name, **kwargs):
        spec = importlib.util.spec_from_file_location("mod", os.path.join(dir_path, model_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, class_name)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pos = sd["model"].get("pos")
        if pos is not None and "max_len" not in kwargs:
            kwargs["max_len"] = pos.shape[1]
        m = cls(**kwargs).cuda().eval()
        m.load_state_dict(sd["model"])
        for p in m.parameters(): p.requires_grad = False
        return m

    pitch_t = _load_teacher(PITCH_DIR, PITCH_CKPT, "model.py", "LatentBasicPitchStudent")
    visual_t = _load_teacher(VISUAL_DIR, VISUAL_CKPT, "infer.py", "LatentToPeakEnvelope")
    print("[v4] all teachers loaded")

    model = SemDemucs(channels=args.channels).cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[v4] SemDemucs: {n:.1f}M params (channels={args.channels})")

    # Initialize from v3 checkpoint if available
    if args.resume_v3 and os.path.exists(args.resume_v3):
        v3_sd = torch.load(args.resume_v3, map_location="cpu", weights_only=False)
        result = model.load_state_dict(v3_sd["model"], strict=False)
        print(f"[v4] initialized from v3: missing={len(result.missing_keys)}, "
              f"unexpected={len(result.unexpected_keys)}")
        if result.missing_keys:
            print(f"  missing: {result.missing_keys[:10]}")
    else:
        print("[v4] training from scratch")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    def lr_lambda(step):
        if step < args.warmup: return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(prog * math.pi))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    step = 0
    # Auto-resume from latest checkpoint
    import re, glob as _glob
    _ckpts = [f for f in _glob.glob(os.path.join(args.out, "*_step*.pt"))
              if re.search(r'step(\d+)', f)]
    if _ckpts:
        _latest = max(_ckpts, key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
        _sd = torch.load(_latest, map_location="cuda", weights_only=False)
        model.load_state_dict(_sd["model"])
        step = _sd.get("step", 0)
        for _ in range(step): sched.step()
        print(f"[resume] from step {step} ({os.path.basename(_latest)})")
    hist = {k: [] for k in ("total", "mask", "emb", "pitch", "rms", "vocal")}
    t0 = time.time()

    while step < args.steps:
        for audio, stem_lats in loader:
            audio = audio.cuda(non_blocking=True)
            stem_lats = stem_lats.cuda(non_blocking=True)  # [B, 4, T, 64] htdemucs order
            B = audio.shape[0]

            # STFT mask targets from htdemucs teacher
            tgt_masks = compute_stft_masks_from_teacher(teacher, audio)  # [B, 4, F, T_stft]

            # Embedding + pitch + rms targets from frozen teachers
            with torch.no_grad():
                tgt_embs = []
                tgt_pitch_o, tgt_pitch_f, tgt_rms = [], [], []
                for si in range(4):
                    lat = stem_lats[:, si]
                    tgt_embs.append(sem_enc(lat)["embedding"])
                    p_out = pitch_t(lat)
                    tgt_pitch_o.append(torch.sigmoid(p_out["onset_logits"]))
                    tgt_pitch_f.append(torch.sigmoid(p_out["frame_logits"]))
                    rms_out = visual_t(lat.transpose(1, 2)).transpose(1, 2)
                    tgt_rms.append(rms_out)
                tgt_emb = torch.stack(tgt_embs, dim=1)
                tgt_pitch = torch.stack(
                    [0.5 * (o + f) for o, f in zip(tgt_pitch_o, tgt_pitch_f)], dim=1)
                tgt_rms_t = torch.stack(tgt_rms, dim=1)

            # Student forward
            opt.zero_grad(set_to_none=True)
            out = model(audio.float())

            # Mask loss: align shapes then MSE
            pred_masks = out["stft_masks"]
            F_m = min(pred_masks.shape[2], tgt_masks.shape[2])
            T_m = min(pred_masks.shape[3], tgt_masks.shape[3])
            l_mask = F.mse_loss(pred_masks[:, :, :F_m, :T_m], tgt_masks[:, :, :F_m, :T_m])

            # Embedding loss
            l_emb = F.l1_loss(out["embedding"], tgt_emb)

            # Pitch loss
            T_p = min(out["pitch_logits"].shape[2], tgt_pitch.shape[2])
            l_pitch = F.binary_cross_entropy_with_logits(
                out["pitch_logits"][:, :, :T_p], tgt_pitch[:, :, :T_p])

            # RMS loss
            T_r = min(out["rms"].shape[2], tgt_rms_t.shape[2])
            l_rms = F.l1_loss(out["rms"][:, :, :T_r], tgt_rms_t[:, :, :T_r])

            # Vocal loss
            vocal_labels = torch.tensor([[0, 0, 0, 1]] * B, device='cuda', dtype=torch.float32)  # htdemucs order
            l_vocal = F.binary_cross_entropy_with_logits(out["vocal"], vocal_labels)

            loss = (args.lambda_mask * l_mask +
                    args.lambda_emb * l_emb +
                    args.lambda_pitch * l_pitch +
                    args.lambda_rms * l_rms +
                    args.lambda_vocal * l_vocal)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            hist["total"].append(loss.item())
            hist["mask"].append(l_mask.item())
            hist["emb"].append(l_emb.item())
            hist["pitch"].append(l_pitch.item())
            hist["rms"].append(l_rms.item())
            hist["vocal"].append(l_vocal.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                print(f"[step {step:5d}] total={avg(hist['total']):.4f} "
                      f"mask={avg(hist['mask']):.4f} "
                      f"emb={avg(hist['emb']):.4f} "
                      f"pitch={avg(hist['pitch']):.4f} "
                      f"rms={avg(hist['rms']):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"sem_demucs_v4_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps: break

    final = os.path.join(args.out, "sem_demucs_v4_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
