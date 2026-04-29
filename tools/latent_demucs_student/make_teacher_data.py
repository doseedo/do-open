#!/usr/bin/env python3
"""Generate teacher (mix_latent → 4 stem latents) pairs by running
htdemucs on mixesV7 full_mix.flac files and VAE-encoding each stem.

For each session that already has /scratch/mixesV7_latents/<rel>/full_mix.vae.pt:
  1. download gs://ptxsessiondata/mixesV7/<rel>/full_mix.flac
  2. htdemucs → drums, bass, vocals, other
  3. VAE encode each → [T, 64]
  4. save teacher_{drums,bass,vocals,other}.vae.pt next to full_mix.vae.pt
  5. delete the temp flac

Skip sessions where all 4 teacher files already exist.
"""
import os, sys, time, argparse, subprocess, tempfile, traceback
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

sys.path.insert(0, "/scratch/ACE-Step-1.5")

LATENT_ROOT = Path("/scratch/mixesV7_latents")
FLAC_BUCKET = "gs://ptxsessiondata/mixesV7"

# Default = 4-stem htdemucs.  Pass --model htdemucs_6s for the 6-stem variant
# (drums/bass/other/vocals/guitar/piano).
_STEMS_4 = ["drums", "bass", "vocals", "other"]
_STEMS_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]


def find_sessions(stems, prefix, limit=None):
    sessions = []
    for full_mix in LATENT_ROOT.rglob("full_mix.vae.pt"):
        rel = full_mix.parent.relative_to(LATENT_ROOT)
        # already done?
        if all((full_mix.parent / f"{prefix}{s}.vae.pt").exists() for s in stems):
            continue
        sessions.append(rel)
        if limit and len(sessions) >= limit:
            break
    return sessions


def load_vae():
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae")
    return vae.cuda().eval().to(torch.bfloat16)


def load_demucs(name="htdemucs"):
    from demucs.pretrained import get_model
    m = get_model(name)
    m.cuda().eval()
    return m


def vae_encode(vae, audio_2ch_48k):
    """audio: torch.Tensor [2, N] float32 @ 48k → [T, 64] float32 cpu."""
    x = audio_2ch_48k.unsqueeze(0).cuda().to(torch.bfloat16)  # [1, 2, N]
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()                # [1, 64, T]
    return z.squeeze(0).permute(1, 0).cpu().float()           # [T, 64]


def run_demucs(model, audio_2ch_44k):
    """audio: [2, N] float32 @ 44.1k → dict {stem: [2, N] @ 44.1k}."""
    from demucs.apply import apply_model
    x = audio_2ch_44k.unsqueeze(0).cuda()                     # [1, 2, N]
    with torch.no_grad():
        sources = apply_model(model, x, device="cuda", split=True,
                              overlap=0.1, progress=False)[0] # [4, 2, N]
    return {name: sources[i].cpu() for i, name in enumerate(model.sources)}


def process_session(rel, vae, demucs_model, tmp_dir, log, stems, prefix):
    out_dir = LATENT_ROOT / rel
    flac_uri = f"{FLAC_BUCKET}/{rel}/full_mix.flac"
    local_flac = tmp_dir / (str(rel).replace("/", "_") + ".flac")

    # 1. download
    r = subprocess.run(
        ["gsutil", "-q", "cp", flac_uri, str(local_flac)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not local_flac.exists():
        log(f"  miss flac: {rel}")
        return False

    try:
        # 2. load audio
        audio_np, sr = sf.read(str(local_flac), dtype="float32")
        audio = torch.from_numpy(audio_np.T if audio_np.ndim > 1 else audio_np[None])
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        # 3. demucs wants 44.1k
        if sr != 44100:
            audio_44 = torchaudio.functional.resample(audio, sr, 44100)
        else:
            audio_44 = audio
        stems_44 = run_demucs(demucs_model, audio_44)

        # 4. VAE wants 48k → resample each stem and encode
        for name, wav_44 in stems_44.items():
            if name not in stems:
                continue
            wav_48 = torchaudio.functional.resample(wav_44, 44100, 48000)
            z = vae_encode(vae, wav_48)
            torch.save({"latents": z, "stem": name},
                       out_dir / f"{prefix}{name}.vae.pt")
        return True
    except Exception as e:
        log(f"  ERROR {rel}: {e}")
        traceback.print_exc()
        return False
    finally:
        if local_flac.exists():
            local_flac.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="max sessions (0 = all)")
    ap.add_argument("--model", default="htdemucs",
                    choices=["htdemucs", "htdemucs_6s"],
                    help="demucs variant: htdemucs (4-stem) or htdemucs_6s (6-stem)")
    ap.add_argument("--prefix", default=None,
                    help="output filename prefix (default teacher_ for 4s, teacher6_ for 6s)")
    args = ap.parse_args()

    if args.model == "htdemucs_6s":
        stems = _STEMS_6
        prefix = args.prefix or "teacher6_"
    else:
        stems = _STEMS_4
        prefix = args.prefix or "teacher_"

    sessions = find_sessions(stems, prefix, limit=args.limit or None)
    print(f"[teacher] model={args.model} prefix={prefix} stems={stems}")
    print(f"[teacher] {len(sessions)} sessions to process")
    if not sessions:
        return

    print("[teacher] loading VAE...")
    vae = load_vae()
    print(f"[teacher] loading {args.model}...")
    dem = load_demucs(args.model)
    print(f"[teacher] demucs sources: {dem.sources}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="teacher_", dir="/scratch/tmp"))
    print(f"[teacher] tmp: {tmp_dir}")

    ok, fail, t0 = 0, 0, time.time()
    def log(msg):
        print(msg, flush=True)
    for i, rel in enumerate(sessions):
        elapsed = time.time() - t0
        rate = (ok + fail) / max(elapsed, 1)
        print(f"[{i+1}/{len(sessions)}] {rel}  "
              f"(ok={ok} fail={fail} {rate:.2f}/s)", flush=True)
        if process_session(rel, vae, dem, tmp_dir, log, stems, prefix):
            ok += 1
        else:
            fail += 1
    print(f"[done] ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
