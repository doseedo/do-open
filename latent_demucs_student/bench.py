#!/usr/bin/env python3
"""Benchmark inference time: htdemucs + VAE encode pipeline vs distill student."""
import os, sys, time, torch, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/scratch/ACE-Step-1.5")
from distill_model import WaveformToFourStemLatents
from distill_dataset import _load_audio_48k_stereo
import torchaudio

WAV = "/scratch/musdb18_wavs/train/Flags - 54/mixture.wav"
N_RUNS = 3

print("[bench] loading audio...")
audio = _load_audio_48k_stereo(WAV)[:, :48000 * 10]   # 10s clip
secs = audio.shape[1] / 48000
print(f"[bench] audio: {secs:.1f}s")

# ── Pipeline A: htdemucs + VAE encode ──────────────────────────────
print("\n[A] loading htdemucs + VAE...")
from demucs.pretrained import get_model
from demucs.apply import apply_model
from diffusers.models import AutoencoderOobleck
dem = get_model("htdemucs"); dem.cuda().eval()
vae = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae")
vae = vae.cuda().eval().to(torch.bfloat16)

# resample 48k → 44.1k for demucs
audio_44 = torchaudio.functional.resample(audio, 48000, 44100)

def run_pipeline_A():
    """htdemucs + 4× VAE encode + 4× VAE decode (full audio→stem-wavs path)."""
    x = audio_44.unsqueeze(0).cuda()
    with torch.no_grad():
        sources = apply_model(dem, x, device="cuda", split=True, overlap=0.1, progress=False)[0]  # [4, 2, N]
    out_wavs = []
    for i in range(4):
        s48 = torchaudio.functional.resample(sources[i], 44100, 48000)
        s = s48.unsqueeze(0).to(torch.bfloat16).cuda()
        with torch.no_grad():
            z = vae.encode(s).latent_dist.sample()        # encode
            w = vae.decode(z).sample                       # decode (so both paths end at audio)
        out_wavs.append(w)
    return out_wavs

# warmup
_ = run_pipeline_A()
torch.cuda.synchronize()
times_A = []
for _ in range(N_RUNS):
    t0 = time.time(); _ = run_pipeline_A(); torch.cuda.synchronize()
    times_A.append(time.time() - t0)
print(f"[A] htdemucs + VAE encode × 4: {sum(times_A)/len(times_A):.2f}s avg "
      f"({times_A}) → realtime ratio: {secs/(sum(times_A)/len(times_A)):.1f}x")

# free demucs to free GPU
del dem
torch.cuda.empty_cache()

# ── Pipeline B: distill student ─────────────────────────────────────
print("\n[B] loading distill student...")
ckpt = "/scratch/latent_demucs_student/distill_ckpts/distill_final.pt"
model = WaveformToFourStemLatents().cuda().eval()
sd = torch.load(ckpt, map_location="cuda", weights_only=False)
model.load_state_dict(sd["model"])

# need the decoder for B too — reload (vae was deleted)
print("[B] reloading VAE decoder...")
vae_b = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae")
vae_b = vae_b.cuda().eval().to(torch.bfloat16)

def run_pipeline_B():
    """distill student + 4× VAE decode (full audio→stem-wavs path)."""
    x = audio.unsqueeze(0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)                                # [1, 4, 64, T]
        out_wavs = []
        for i in range(4):
            z = pred[:, i].to(torch.bfloat16)              # [1, 64, T]
            w = vae_b.decode(z).sample
            out_wavs.append(w)
    return out_wavs

_ = run_pipeline_B()
torch.cuda.synchronize()
times_B = []
for _ in range(N_RUNS):
    t0 = time.time(); _ = run_pipeline_B(); torch.cuda.synchronize()
    times_B.append(time.time() - t0)
print(f"[B] distill student: {sum(times_B)/len(times_B):.3f}s avg "
      f"({times_B}) → realtime ratio: {secs/(sum(times_B)/len(times_B)):.1f}x")

print(f"\n[speedup] {sum(times_A)/sum(times_B):.1f}x")
