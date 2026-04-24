#!/usr/bin/env python3
"""Train SemDemucs v2: mix waveform → per-stem frame-level signals.

Targets from frozen teachers on GT stem latents:
  - pitch: latent_pitch teacher onset/frame logits [B, T, 128]
  - rms: latent_visual teacher envelope [B, T, 2]
  - embedding: sem encoder v1 global embedding [B, 128]
  - vocal: manifest voice label (binary)

Data: MUSDB18-HQ (150 tracks, real GT stems).
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
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sem_demucs import SemDemucs

STEMS = ["drums", "bass", "vocals", "other"]
SR = 48000
LATENT_FPS = 25
SPF = SR // LATENT_FPS

# Teacher paths
SEM_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")
SEM_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"
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
    # Handle max_len from pos shape
    pos = sd["model"].get("pos")
    if pos is not None and "max_len" not in kwargs:
        kwargs["max_len"] = pos.shape[1]
    m = cls(**kwargs).cuda().eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_teachers():
    print("[v2] loading frozen teachers...")
    sem = _load_teacher(SEM_DIR, SEM_CKPT, "model.py", "SemanticEncoderWithHeads")
    pitch = _load_teacher(PITCH_DIR, PITCH_CKPT, "model.py", "LatentBasicPitchStudent")
    visual = _load_teacher(VISUAL_DIR, VISUAL_CKPT, "infer.py", "LatentToPeakEnvelope")
    print("[v2] all teachers loaded")
    return sem, pitch, visual


def _load_latent(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64:
        z = z.t()
    return z.float()


class SemDemucsV2Dataset(Dataset):
    def __init__(self, crop_sec=6.0, seed=0):
        self.crop_samples = int(crop_sec * SR)
        self.crop_frames = int(crop_sec * LATENT_FPS)
        self.rng = random.Random(seed)
        self.items = []

        from pathlib import Path
        musdb_lat = Path("/scratch/musdb18_latents")
        musdb_wav = Path("/scratch/musdb18_wavs")
        for split in ["train", "test"]:
            for td in sorted((musdb_lat / split).iterdir()):
                if not td.is_dir():
                    continue
                mix_wav = musdb_wav / split / td.name / "mixture.wav"
                if not mix_wav.exists():
                    continue
                stem_lats = {}
                for s in STEMS:
                    p = td / f"{s}.vae.pt"
                    if p.exists():
                        stem_lats[s] = str(p)
                if len(stem_lats) == 4:
                    self.items.append({
                        "audio": str(mix_wav),
                        "stem_lats": stem_lats,
                    })
        print(f"[v2-ds] {len(self.items)} tracks")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            audio_np, sr = sf.read(it["audio"], dtype="float32")
            audio = torch.from_numpy(
                audio_np.T if audio_np.ndim > 1 else audio_np[None]).float()
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]
            if sr != SR:
                audio = torchaudio.functional.resample(audio, sr, SR)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        stem_latents = []
        for s in STEMS:
            try:
                stem_latents.append(_load_latent(it["stem_lats"][s]))
            except Exception:
                return self.__getitem__((idx + 1) % len(self))

        # Random crop
        T_aud = audio.shape[1]
        T_lat = min(z.shape[0] for z in stem_latents)
        T_avail = min(T_aud // SPF, T_lat)
        cf = min(self.crop_frames, T_avail)
        if T_avail > cf:
            start_f = self.rng.randint(0, T_avail - cf)
        else:
            start_f = 0

        start_s = start_f * SPF
        audio_crop = audio[:, start_s:start_s + cf * SPF]
        if audio_crop.shape[1] < cf * SPF:
            audio_crop = F.pad(audio_crop, (0, cf * SPF - audio_crop.shape[1]))

        lat_crops = []
        for z in stem_latents:
            zc = z[start_f:start_f + cf]
            if zc.shape[0] < cf:
                zc = F.pad(zc, (0, 0, 0, cf - zc.shape[0]))
            lat_crops.append(zc)

        # Vocal label: stems[2] = vocals → 1.0, others → 0.0
        vocal_labels = torch.tensor([0.0, 0.0, 1.0, 0.0])

        return audio_crop, torch.stack(lat_crops), vocal_labels


def collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/sem_demucs_v2_ckpts")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop-sec", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--lambda-pitch", type=float, default=1.0)
    ap.add_argument("--lambda-rms", type=float, default=1.0)
    ap.add_argument("--lambda-emb", type=float, default=1.0)
    ap.add_argument("--lambda-vocal", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = SemDemucsV2Dataset(crop_sec=args.crop_sec)
    if len(ds) == 0:
        print("ERROR: no data"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate,
                        persistent_workers=args.workers > 0)

    sem_enc, pitch_t, visual_t = load_teachers()

    model = SemDemucs().cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[v2] SemDemucs: {n:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)
    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
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
    hist = {k: [] for k in ("total", "pitch", "rms", "emb", "vocal")}
    t0 = time.time()

    while step < args.steps:
        for audio, stem_lats, vocal_labels in loader:
            audio = audio.cuda(non_blocking=True)
            stem_lats = stem_lats.cuda(non_blocking=True)  # [B, 4, T, 64]
            vocal_labels = vocal_labels.cuda(non_blocking=True)  # [B, 4]

            B = audio.shape[0]

            # Get teacher targets for each stem
            with torch.no_grad():
                tgt_pitch_onset = []
                tgt_pitch_frame = []
                tgt_rms = []
                tgt_emb = []
                for si in range(4):
                    lat = stem_lats[:, si]  # [B, T, 64]

                    # Pitch teacher
                    p_out = pitch_t(lat)
                    tgt_pitch_onset.append(torch.sigmoid(p_out["onset_logits"]))
                    tgt_pitch_frame.append(torch.sigmoid(p_out["frame_logits"]))

                    # RMS teacher
                    rms_out = visual_t(lat.transpose(1, 2)).transpose(1, 2)
                    tgt_rms.append(rms_out)

                    # Sem encoder
                    sem_out = sem_enc(lat)
                    tgt_emb.append(sem_out["embedding"])

                tgt_pitch = torch.stack(
                    [0.5 * (o + f) for o, f in zip(tgt_pitch_onset, tgt_pitch_frame)],
                    dim=1)  # [B, 4, T, 128]
                tgt_rms = torch.stack(tgt_rms, dim=1)   # [B, 4, T, 2]
                tgt_emb = torch.stack(tgt_emb, dim=1)   # [B, 4, 128]

            # Student forward
            opt.zero_grad(set_to_none=True)
            out = model(audio)

            # Align frame counts (model output T' may differ from latent T)
            T_pred = out["pitch_logits"].shape[2]
            T_tgt = tgt_pitch.shape[2]
            T = min(T_pred, T_tgt)

            # Pitch loss (BCE on logits vs teacher sigmoid)
            l_pitch = F.binary_cross_entropy_with_logits(
                out["pitch_logits"][:, :, :T],
                tgt_pitch[:, :, :T])

            # RMS loss (L1)
            l_rms = F.l1_loss(out["rms"][:, :, :T], tgt_rms[:, :, :T])

            # Embedding loss (L1 on global)
            l_emb = F.l1_loss(out["embedding"], tgt_emb)

            # Vocal loss (BCE)
            l_vocal = F.binary_cross_entropy_with_logits(out["vocal"], vocal_labels)

            loss = (args.lambda_pitch * l_pitch +
                    args.lambda_rms * l_rms +
                    args.lambda_emb * l_emb +
                    args.lambda_vocal * l_vocal)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            hist["total"].append(loss.item())
            hist["pitch"].append(l_pitch.item())
            hist["rms"].append(l_rms.item())
            hist["emb"].append(l_emb.item())
            hist["vocal"].append(l_vocal.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                el = time.time() - t0
                print(f"[step {step:6d}] total={avg(hist['total']):.4f} "
                      f"pitch={avg(hist['pitch']):.4f} "
                      f"rms={avg(hist['rms']):.4f} "
                      f"emb={avg(hist['emb']):.4f} "
                      f"vocal={avg(hist['vocal']):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={el:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"sem_demucs_v2_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  → saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "sem_demucs_v2_final.pt")
    torch.save({"step": step, "model": model.state_dict(),
                "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
