#!/usr/bin/env python3
"""Per-component benchmark: distill student vs each piece by itself."""
import os, sys, time, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/scratch/ACE-Step-1.5")
from distill_model import WaveformToFourStemLatents
from distill_dataset import _load_audio_48k_stereo

WAV = "/scratch/musdb18_wavs/train/Flags - 54/mixture.wav"
N = 5

audio = _load_audio_48k_stereo(WAV)[:, :48000 * 10]
secs = audio.shape[1] / 48000
print(f"audio: {secs:.0f}s @ 48k stereo\n")

def bench(name, fn):
    fn()  # warmup
    torch.cuda.synchronize()
    t = []
    for _ in range(N):
        t0 = time.time(); fn(); torch.cuda.synchronize()
        t.append(time.time() - t0)
    avg = sum(t) / N
    print(f"  {name:38s} {avg*1000:8.1f} ms   ({secs/avg:6.1f}x rt)")

# 1) VAE encode (just one pass, the mix)
from diffusers.models import AutoencoderOobleck
vae = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae")
vae = vae.cuda().eval().to(torch.bfloat16)

print("== single component times ==")
def vae_encode_once():
    x = audio.unsqueeze(0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        return vae.encode(x).latent_dist.sample()
bench("VAE encode (1× mix)", vae_encode_once)

def vae_decode_once():
    x = audio.unsqueeze(0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()
        return vae.decode(z).sample
bench("VAE encode + decode (1×)", vae_decode_once)

# 2) htdemucs alone (waveform separation)
from demucs.pretrained import get_model
from demucs.apply import apply_model
dem = get_model("htdemucs").cuda().eval()
audio_44 = torchaudio.functional.resample(audio, 48000, 44100)
def demucs_only():
    x = audio_44.unsqueeze(0).cuda()
    with torch.no_grad():
        return apply_model(dem, x, device="cuda", split=True, overlap=0.1, progress=False)
bench("htdemucs (waveform separation only)", demucs_only)

del dem; torch.cuda.empty_cache()

# 3) distill student
ckpt = "/scratch/latent_demucs_student/distill_ckpts/distill_final.pt"
m = WaveformToFourStemLatents().cuda().eval()
sd = torch.load(ckpt, map_location="cuda", weights_only=False)
m.load_state_dict(sd["model"])
def student_only():
    x = audio.unsqueeze(0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return m(x)
bench("distill student (waveform → 4 latents)", student_only)
