"""Speed benchmark: latent student vs MDX23C teacher.

Two relevant operating points:

  (A) "I'm already in latent land" — the use case the student was built for.
      Student input is L_mix [T, 64], output is 6 latent stems. No audio.
      Teacher equivalent: decode L_mix → audio file → sep.separate → encode each
      stem back. We measure the *equivalent* teacher cost to produce latent stems.

  (B) "I have audio, I want stem audio" — the apples-to-apples user-facing case.
      Student path = encode(audio) → student → decode each stem.
      Teacher path = sep.separate(audio_file).
      Here the student pays VAE round-trip cost.

We pick a real cached drum example, build the audio for it once, then time
each path repeatedly with warmup.

Usage:
    python -m latent_drumsep.bench_speed \
        --ckpt /scratch/latent_drumsep_ckpts/drumsep_014000.pt \
        --reps 5
"""
from __future__ import annotations
import argparse, os, sys, time, tempfile
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "/scratch/ACE-Step-1.5")
from acestep.handler import AceStepHandler  # noqa: E402

from latent_drumsep.model import LatentDrumSubsep
from latent_drumsep.dataset import CachedDrumsepDataset
from latent_drumsep import STEMS

SR = 48000


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/scratch/latent_drumsep_cache")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--clip-seconds", type=float, default=10.0,
                    help="length of synthesized test clip (looped from 2.5s window)")
    args = ap.parse_args()

    device = "cuda"

    print("loading VAE...")
    h = AceStepHandler()
    h.initialize_service(project_root="/scratch/ACE-Step-1.5",
                         config_path="acestep-v15-sft", device=device)
    vae = h.vae
    for p in vae.parameters(): p.requires_grad = False
    vae.eval()

    print("loading student...")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = sd.get("model", sd.get("state_dict", sd))
    pos = state.get("pos")
    max_len = pos.shape[1] if pos is not None else 64
    student = LatentDrumSubsep(max_len=max_len).to(device).eval()
    student.load_state_dict(state)

    print("loading teacher (audio_separator MDX23C)...")
    from audio_separator.separator import Separator
    sep_out = tempfile.mkdtemp(prefix="bench_sep_")
    sep = Separator(
        model_file_dir="/scratch/audio_separator_models",
        output_dir=sep_out,
        log_level=40,  # WARN
    )
    sep.load_model("MDX23C-DrumSep-aufr33-jarredou.ckpt")

    print("loading cache & picking a drum example...")
    ds = CachedDrumsepDataset(args.cache_dir)
    ex = ds[len(ds) - 50]  # late item
    L_mix_short = ex["L_mix"].to(device)             # [T=64, 64]
    T_short = L_mix_short.shape[0]

    # Build a longer clip by tiling, so the test isn't dominated by overhead
    target_T = max(T_short, int(round(args.clip_seconds * SR / 1920)))
    repeats = max(1, target_T // T_short)
    L_mix = L_mix_short.repeat(repeats, 1)[:target_T].contiguous()    # [T_clip, 64]
    T = L_mix.shape[0]
    clip_seconds = T * 1920 / SR
    print(f"  test clip: T={T} latent frames = {clip_seconds:.2f}s of audio")
    if T > max_len:
        print(f"  WARNING: T ({T}) > student max_len ({max_len}); clamping")
        L_mix = L_mix[:max_len]
        T = max_len
        clip_seconds = T * 1920 / SR

    # Decode mix to audio (used by all teacher paths)
    cuda_sync()
    wav_mix = vae.decode(L_mix.t().unsqueeze(0).to(torch.bfloat16)).sample[0].float().cpu().numpy()
    cuda_sync()
    wav_mix_path = os.path.join(sep_out, "bench_input_mix.wav")
    sf.write(wav_mix_path, wav_mix.T, SR)
    print(f"  wrote test wav: {wav_mix_path}  shape={wav_mix.shape}")

    # ---------- Timer helpers ----------
    def time_student_latent_only():
        cuda_sync()
        t0 = time.perf_counter()
        out = student(L_mix.unsqueeze(0))   # [1, S, T, 64]
        cuda_sync()
        return time.perf_counter() - t0, out

    def time_student_audio_to_audio():
        # encode mix → student → decode each stem
        cuda_sync()
        t0 = time.perf_counter()
        wav_t = torch.from_numpy(wav_mix).unsqueeze(0).to(device, torch.bfloat16)
        enc = vae.encode(wav_t)
        L_in = (enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc.sample)
        L_in = L_in[0].transpose(0, 1).float()    # [T, 64]  cast to fp32 for student
        if L_in.shape[0] > max_len:
            L_in = L_in[:max_len]
        L_out = student(L_in.unsqueeze(0))[0]     # [S, T, 64]
        # Decode each stem
        wavs = []
        for s in range(L_out.shape[0]):
            x = L_out[s].t().unsqueeze(0).to(torch.bfloat16)
            wavs.append(vae.decode(x).sample[0].float().cpu().numpy())
        cuda_sync()
        return time.perf_counter() - t0, wavs

    def time_teacher_audio():
        cuda_sync()
        t0 = time.perf_counter()
        stem_files = sep.separate(wav_mix_path)
        cuda_sync()
        return time.perf_counter() - t0, stem_files

    def time_teacher_latent_equivalent():
        # decode L_mix → file → separate → encode each stem
        cuda_sync()
        t0 = time.perf_counter()
        x = L_mix.t().unsqueeze(0).to(torch.bfloat16)
        wav = vae.decode(x).sample[0].float().cpu().numpy()
        path = os.path.join(sep_out, "bench_dec.wav")
        sf.write(path, wav.T, SR)
        stem_files = sep.separate(path)
        # encode each stem back to latent
        latents = []
        for sf_path in stem_files:
            full = os.path.join(sep_out, sf_path) if not os.path.isabs(sf_path) else sf_path
            stem_wav, _ = sf.read(full, dtype="float32", always_2d=True)
            if stem_wav.shape[1] == 1:
                stem_wav = np.concatenate([stem_wav, stem_wav], axis=1)
            stem_wav = stem_wav.T.astype(np.float32)        # [2, S]
            wt = torch.from_numpy(stem_wav).unsqueeze(0).to(device, torch.bfloat16)
            enc = vae.encode(wt)
            L = (enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc.sample)
            latents.append(L[0].transpose(0, 1))
        cuda_sync()
        return time.perf_counter() - t0, latents

    # ---------- Warmup ----------
    print("\nwarmup...")
    for _ in range(2):
        time_student_latent_only()
    for _ in range(1):
        time_student_audio_to_audio()
    for _ in range(1):
        time_teacher_audio()

    # ---------- Run reps ----------
    def run(name, fn):
        ts = []
        for _ in range(args.reps):
            t, _ = fn()
            ts.append(t)
        ts = np.array(ts)
        return name, ts.mean(), ts.std(), ts.min()

    print(f"\nrunning {args.reps} reps each ({clip_seconds:.1f}s of audio per call)...")
    rows = [
        run("student: latent in → latent stems out", time_student_latent_only),
        run("student: audio in → audio stems out  ", time_student_audio_to_audio),
        run("teacher: audio in → audio stems out  ", time_teacher_audio),
        run("teacher: latent in → latent stems out", time_teacher_latent_equivalent),
    ]

    print(f"\n{'pipeline':<46} {'mean (s)':>10} {'min (s)':>10} {'std':>8}  {'×realtime':>12}")
    print("-" * 92)
    for name, m, s, mn in rows:
        rt = clip_seconds / m if m > 0 else float("inf")
        print(f"{name:<46} {m:>10.4f} {mn:>10.4f} {s:>8.4f}  {rt:>10.1f}×")

    # speedup table
    print("\nSpeedups (relative — higher = student wins):")
    s_lat = rows[0][1]
    s_aud = rows[1][1]
    t_aud = rows[2][1]
    t_lat = rows[3][1]
    print(f"  student-latent vs teacher-latent (the use case):  {t_lat / s_lat:>8.1f}×")
    print(f"  student-audio  vs teacher-audio  (apples-apples):  {t_aud / s_aud:>8.1f}×")
    print(f"  student-latent vs teacher-audio  (best vs worst):  {t_aud / s_lat:>8.1f}×")


if __name__ == "__main__":
    main()
