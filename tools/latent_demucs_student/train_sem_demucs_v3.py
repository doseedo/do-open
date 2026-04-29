#!/usr/bin/env python3
"""Train SemDemucs v3: v2 + separation mask head.

Adds mask training to existing v2 targets (pitch, rms, embedding, vocal).
Mask target: ideal ratio mask (IRM) from GT stem waveforms at model frame rate.

Initializes from v2 checkpoint — existing heads keep their learned weights,
mask head starts from zero (no-op at step 0).

Data: MUSDB18-HQ (150 tracks, real GT stems + stem waveforms for IRM).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sem_demucs import SemDemucs

STEMS = ["drums", "bass", "vocals", "other"]
SR = 48000
LATENT_FPS = 25
SPF = SR // LATENT_FPS  # 1920

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
    pos = sd["model"].get("pos")
    if pos is not None and "max_len" not in kwargs:
        kwargs["max_len"] = pos.shape[1]
    m = cls(**kwargs).cuda().eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_vae_encoder(device="cuda"):
    """Frozen VAE encoder — needed to get mix latent for mix_embedding target."""
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae").to(device).eval().to(torch.bfloat16)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def load_teachers():
    print("[v3] loading frozen teachers...")
    sem = _load_teacher(SEM_DIR, SEM_CKPT, "model.py", "SemanticEncoderWithHeads")
    pitch = _load_teacher(PITCH_DIR, PITCH_CKPT, "model.py", "LatentBasicPitchStudent")
    visual = _load_teacher(VISUAL_DIR, VISUAL_CKPT, "infer.py", "LatentToPeakEnvelope")
    print("[v3] all teachers loaded")
    return sem, pitch, visual


def _load_latent(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64:
        z = z.t()
    return z.float()


def _load_stem_wav(path):
    """Load a stem wav, return [2, N] float32."""
    audio, sr = sf.read(path, dtype="float32")
    t = torch.from_numpy(audio.T if audio.ndim > 1 else audio[None]).float()
    if t.shape[0] == 1:
        t = t.repeat(2, 1)
    elif t.shape[0] > 2:
        t = t[:2]
    return t


def compute_irm(stem_wavs, frame_size):
    """Compute ideal ratio mask from GT stem waveforms.

    stem_wavs: list of 4 tensors [2, N] (drums, bass, vocals, other)
    frame_size: samples per frame (model stride, 1536)

    Returns: [4, T'] IRM where T' = N // frame_size, sums to 1 across stems.
    """
    N = min(w.shape[-1] for w in stem_wavs)
    # Truncate to frame-aligned length
    T = N // frame_size
    N_aligned = T * frame_size

    energies = []
    for w in stem_wavs:
        # [2, N] → [2, T, frame_size] → energy per frame
        frames = w[:, :N_aligned].reshape(2, T, frame_size)
        energy = frames.pow(2).mean(dim=(0, 2))  # [T] — mean over channels + samples
        energies.append(energy)

    energies = torch.stack(energies, dim=0)  # [4, T]
    irm = energies / (energies.sum(dim=0, keepdim=True) + 1e-8)
    return irm  # [4, T']


class SemDemucsV3Dataset(Dataset):
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
                # Check stem latents
                stem_lats = {}
                for s in STEMS:
                    p = td / f"{s}.vae.pt"
                    if p.exists():
                        stem_lats[s] = str(p)
                if len(stem_lats) != 4:
                    continue
                # Check stem wavs (other = mix - drums - bass - vocals)
                stem_wavs = {}
                has_wavs = True
                for s in ["drums", "bass", "vocals"]:
                    p = musdb_wav / split / td.name / f"{s}.wav"
                    if p.exists():
                        stem_wavs[s] = str(p)
                    else:
                        has_wavs = False
                if not has_wavs:
                    continue
                stem_wavs["other"] = None  # computed as residual
                self.items.append({
                    "audio": str(mix_wav),
                    "stem_lats": stem_lats,
                    "stem_wavs": stem_wavs,
                    "mix_wav": str(mix_wav),
                })
        print(f"[v3-ds] {len(self.items)} tracks")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            # Load mix audio
            audio = _load_stem_wav(it["audio"])

            # Load stem latents
            stem_latents = []
            for s in STEMS:
                stem_latents.append(_load_latent(it["stem_lats"][s]))

            # Load stem wavs for IRM
            stem_wavs = []
            for s in STEMS:
                if s == "other":
                    # other = mix - drums - bass - vocals
                    mix_full = _load_stem_wav(it["mix_wav"])
                    other = mix_full.clone()
                    for sw in stem_wavs:
                        N = min(other.shape[-1], sw.shape[-1])
                        other[:, :N] -= sw[:, :N]
                    stem_wavs.append(other)
                else:
                    stem_wavs.append(_load_stem_wav(it["stem_wavs"][s]))
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
        end_s = start_s + cf * SPF

        # Crop audio
        audio_crop = audio[:, start_s:end_s]
        if audio_crop.shape[1] < cf * SPF:
            audio_crop = F.pad(audio_crop, (0, cf * SPF - audio_crop.shape[1]))

        # Crop stem latents
        lat_crops = []
        for z in stem_latents:
            zc = z[start_f:start_f + cf]
            if zc.shape[0] < cf:
                zc = F.pad(zc, (0, 0, 0, cf - zc.shape[0]))
            lat_crops.append(zc)

        # Crop stem wavs and compute IRM
        sw_crops = []
        for sw in stem_wavs:
            swc = sw[:, start_s:end_s]
            if swc.shape[1] < cf * SPF:
                swc = F.pad(swc, (0, cf * SPF - swc.shape[1]))
            sw_crops.append(swc)

        # IRM at model frame rate (stride=1536)
        irm = compute_irm(sw_crops, frame_size=1536)  # [4, T_model]

        vocal_labels = torch.tensor([0.0, 0.0, 1.0, 0.0])

        return audio_crop, torch.stack(lat_crops), vocal_labels, irm


def collate(batch):
    # IRM may have different T due to rounding — pad to max
    max_irm_T = max(b[3].shape[-1] for b in batch)
    irms = []
    for b in batch:
        irm = b[3]
        if irm.shape[-1] < max_irm_T:
            irm = F.pad(irm, (0, max_irm_T - irm.shape[-1]))
        irms.append(irm)
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        torch.stack([b[2] for b in batch]),
        torch.stack(irms),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/latent_demucs_student/sem_demucs_v3_ckpts")
    ap.add_argument("--resume-v2",
                    default="/scratch/latent_demucs_student/sem_demucs_v2_ckpts/sem_demucs_v2_step10000.pt",
                    help="v2 checkpoint to initialize from (existing heads)")
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
    ap.add_argument("--lambda-mask", type=float, default=2.0,
                    help="Mask loss weight (higher = prioritize separation)")
    ap.add_argument("--lambda-mix-emb", type=float, default=1.0,
                    help="Mix-level embedding distillation weight")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    ds = SemDemucsV3Dataset(crop_sec=args.crop_sec)
    if len(ds) == 0:
        print("ERROR: no data"); return
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate,
                        persistent_workers=args.workers > 0)

    sem_enc, pitch_t, visual_t = load_teachers()

    print("[v3] loading frozen VAE encoder (for mix_embedding target)...")
    vae_enc = load_vae_encoder()

    model = SemDemucs().cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[v3] SemDemucs: {n:.1f}M params")

    # Initialize from v2 checkpoint (mask_head starts zero-init'd)
    if args.resume_v2 and os.path.exists(args.resume_v2):
        v2_sd = torch.load(args.resume_v2, map_location="cpu", weights_only=False)
        result = model.load_state_dict(v2_sd["model"], strict=False)
        print(f"[v3] initialized from v2: {len(v2_sd['model'])} keys loaded, "
              f"missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    else:
        print("[v3] training from scratch (no v2 checkpoint)")

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
    hist = {k: [] for k in ("total", "pitch", "rms", "emb", "mix_emb", "vocal", "mask")}
    t0 = time.time()

    while step < args.steps:
        for audio, stem_lats, vocal_labels, irm_target in loader:
            audio = audio.cuda(non_blocking=True)
            stem_lats = stem_lats.cuda(non_blocking=True)
            vocal_labels = vocal_labels.cuda(non_blocking=True)
            irm_target = irm_target.cuda(non_blocking=True)  # [B, 4, T_irm]

            B = audio.shape[0]

            # Get teacher targets for each stem
            with torch.no_grad():
                tgt_pitch_onset = []
                tgt_pitch_frame = []
                tgt_rms = []
                tgt_emb = []
                for si in range(4):
                    lat = stem_lats[:, si]

                    p_out = pitch_t(lat)
                    tgt_pitch_onset.append(torch.sigmoid(p_out["onset_logits"]))
                    tgt_pitch_frame.append(torch.sigmoid(p_out["frame_logits"]))

                    rms_out = visual_t(lat.transpose(1, 2)).transpose(1, 2)
                    tgt_rms.append(rms_out)

                    sem_out = sem_enc(lat)
                    tgt_emb.append(sem_out["embedding"])

                tgt_pitch = torch.stack(
                    [0.5 * (o + f) for o, f in zip(tgt_pitch_onset, tgt_pitch_frame)],
                    dim=1)
                tgt_rms = torch.stack(tgt_rms, dim=1)
                tgt_emb = torch.stack(tgt_emb, dim=1)

                # Mix-level embedding target: VAE encode mix → sem encoder
                # This is EXACTLY what distill_sem (L1=0.333) and the deployed
                # decoder were conditioned on
                mix_latent = vae_enc.encode(audio.to(torch.bfloat16)).latent_dist.sample()
                tgt_mix_emb = sem_enc(mix_latent.float().transpose(1, 2))["embedding"]  # [B, 128]

            # Student forward
            opt.zero_grad(set_to_none=True)
            out = model(audio)

            # Align frame counts
            T_pred = out["pitch_logits"].shape[2]
            T_tgt = tgt_pitch.shape[2]
            T = min(T_pred, T_tgt)

            # Pitch loss
            l_pitch = F.binary_cross_entropy_with_logits(
                out["pitch_logits"][:, :, :T], tgt_pitch[:, :, :T])

            # RMS loss
            l_rms = F.l1_loss(out["rms"][:, :, :T], tgt_rms[:, :, :T])

            # Per-stem embedding loss (distill from SemanticEncoder v1 on individual stems)
            l_emb = F.l1_loss(out["embedding"], tgt_emb)

            # Mix-level embedding loss (distill from SemanticEncoder v1 on MIX latent)
            # This is the critical one — must match deployed decoder's conditioning
            l_mix_emb = F.l1_loss(out["mix_embedding"], tgt_mix_emb)

            # Vocal loss
            l_vocal = F.binary_cross_entropy_with_logits(out["vocal"], vocal_labels)

            # Mask loss: MSE between predicted masks and IRM
            T_mask_pred = out["masks"].shape[-1]
            T_mask_tgt = irm_target.shape[-1]
            T_mask = min(T_mask_pred, T_mask_tgt)
            l_mask = F.mse_loss(out["masks"][:, :, :T_mask], irm_target[:, :, :T_mask])

            loss = (args.lambda_pitch * l_pitch +
                    args.lambda_rms * l_rms +
                    args.lambda_emb * l_emb +
                    args.lambda_mix_emb * l_mix_emb +
                    args.lambda_vocal * l_vocal +
                    args.lambda_mask * l_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            hist["total"].append(loss.item())
            hist["pitch"].append(l_pitch.item())
            hist["rms"].append(l_rms.item())
            hist["emb"].append(l_emb.item())
            hist["mix_emb"].append(l_mix_emb.item())
            hist["vocal"].append(l_vocal.item())
            hist["mask"].append(l_mask.item())
            step += 1

            if step % args.log_every == 0:
                avg = lambda xs: sum(xs[-50:]) / max(1, len(xs[-50:]))
                el = time.time() - t0
                print(f"[step {step:6d}] total={avg(hist['total']):.4f} "
                      f"pitch={avg(hist['pitch']):.4f} "
                      f"rms={avg(hist['rms']):.4f} "
                      f"emb={avg(hist['emb']):.4f} "
                      f"mix_emb={avg(hist['mix_emb']):.4f} "
                      f"mask={avg(hist['mask']):.4f} "
                      f"vocal={avg(hist['vocal']):.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={el:.0f}s", flush=True)

            if step % args.save_every == 0:
                p = os.path.join(args.out, f"sem_demucs_v3_step{step}.pt")
                torch.save({"step": step, "model": model.state_dict(),
                            "args": vars(args)}, p)
                print(f"  -> saved {p}", flush=True)

            if step >= args.steps:
                break

    final = os.path.join(args.out, "sem_demucs_v3_final.pt")
    torch.save({"step": step, "model": model.state_dict(),
                "args": vars(args)}, final)
    print(f"[done] {step} steps, saved {final}")


if __name__ == "__main__":
    main()
