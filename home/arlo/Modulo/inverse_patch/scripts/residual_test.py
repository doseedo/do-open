#!/usr/bin/env python3
"""
Residual test: Can (DCAE_decoded - SMS_rendered) be used as a residual
that reconstructs the full audio when added back to SMS_rendered?

Pipeline:
  1. Load original audio
  2. Load pre-extracted SMS params → render via additive synth → sms_wav
  3. Load DCAE latent → decode → dcae_wav (DCAE's reconstruction of original)
  4. residual = dcae_wav - sms_wav
  5. reconstructed = sms_wav + residual  (should == dcae_wav exactly)
  6. Also compare original vs sms_wav vs dcae_wav to see what SMS captures

Saves all wavs for listening.
"""

import sys
import os
import numpy as np
import torch
import torchaudio
import orjson

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

SAMPLE_RATE = 44100
OUTPUT_DIR = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/residual_test"


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def get_noise_band_edges(n_bands=8, min_freq=80, max_freq=16000):
    min_mel = hz_to_mel(min_freq)
    max_mel = hz_to_mel(max_freq)
    mel_edges = np.linspace(min_mel, max_mel, n_bands + 1)
    return mel_to_hz(mel_edges)


def additive_synth(freqs, amps, noise_amps=None, sr=SAMPLE_RATE, n_audio_samples=None):
    """Render SMS params to audio via additive synthesis."""
    T, n_sines = freqs.shape

    if n_audio_samples is None:
        n_audio_samples = T * 512  # fallback
    hop = n_audio_samples / T

    audio = np.zeros(n_audio_samples, dtype=np.float64)
    freqs_np = freqs.numpy().astype(np.float64)
    amps_np = amps.numpy().astype(np.float64)

    t_samples = np.arange(n_audio_samples, dtype=np.float64) / sr
    t_frames = np.arange(T, dtype=np.float64) * hop / sr

    for s in range(n_sines):
        a_track = amps_np[:, s]
        if a_track.max() < 0.0005:
            continue
        f_track = freqs_np[:, s]
        f_interp = np.interp(t_samples, t_frames, f_track)
        a_interp = np.interp(t_samples, t_frames, a_track)
        phase = np.cumsum(2 * np.pi * f_interp / sr)
        audio += a_interp * np.sin(phase)

    # Noise bands
    if noise_amps is not None:
        noise_np = noise_amps.numpy().astype(np.float64)
        band_edges = get_noise_band_edges()
        rng = np.random.RandomState(42)
        white = rng.randn(n_audio_samples)
        fft_noise = np.fft.rfft(white)
        fft_freqs = np.fft.rfftfreq(n_audio_samples, 1.0 / sr)

        sine_energy = np.sqrt(np.sum(amps_np ** 2, axis=1))
        sine_energy_interp = np.interp(t_samples, t_frames, sine_energy)

        for b in range(noise_np.shape[1]):
            na_track = noise_np[:, b]
            if na_track.max() < 0.0005:
                continue
            lo, hi = band_edges[b], band_edges[b + 1]
            mask = (fft_freqs >= lo) & (fft_freqs <= hi)
            band_fft = fft_noise.copy()
            band_fft[~mask] = 0
            band_audio = np.fft.irfft(band_fft, n_audio_samples)
            band_rms = np.sqrt(np.mean(band_audio ** 2)) + 1e-10
            band_audio /= band_rms
            a_interp = np.interp(t_samples, t_frames, na_track)
            max_noise = np.maximum(sine_energy_interp * 2.0, 0.001)
            a_interp = np.minimum(a_interp, max_noise)
            audio += a_interp * band_audio

    return audio.astype(np.float32)


def rms(x):
    return np.sqrt(np.mean(x ** 2))

def snr_db(signal, noise):
    s = rms(signal)
    n = rms(noise)
    if n < 1e-10:
        return float('inf')
    return 20 * np.log10(s / n)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Load SMS entry ---
    base = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch'
    sms_path = f'{base}/data/sms_hybrid/sms_000004.pt'
    print(f"Loading SMS: {sms_path}")
    sms = torch.load(sms_path, weights_only=True, map_location='cpu')
    freqs = sms['freqs']       # [T, 64]
    amps = sms['amps']         # [T, 64]
    noise = sms['noise_amps']  # [T, 8]
    audio_path = sms['audio_path']
    latent_path = sms['latent_path']
    T_sms = freqs.shape[0]
    print(f"  SMS frames: {T_sms}, sines: {freqs.shape[1]}, noise bands: {noise.shape[1]}")
    print(f"  Audio: {audio_path}")
    print(f"  Latent: {latent_path}")

    # --- 2. Load original audio ---
    print(f"\nLoading original audio...")
    orig_audio, orig_sr = torchaudio.load(audio_path)
    if orig_sr != SAMPLE_RATE:
        orig_audio = torchaudio.functional.resample(orig_audio, orig_sr, SAMPLE_RATE)
    orig_audio = orig_audio.mean(dim=0).numpy()  # mono
    n_samples = len(orig_audio)
    print(f"  Original: {n_samples} samples, {n_samples/SAMPLE_RATE:.2f}s")

    # --- 3. Render SMS → audio ---
    print(f"\nRendering SMS via additive synth...")
    sms_wav = additive_synth(freqs, amps, noise, sr=SAMPLE_RATE, n_audio_samples=n_samples)
    print(f"  SMS wav: {len(sms_wav)} samples, rms={rms(sms_wav):.6f}")

    # --- 4. Decode DCAE latent → audio ---
    print(f"\nLoading DCAE...")
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH).to(device)
    dcae.eval()
    print(f"  DCAE loaded on {device}")

    print(f"\nLoading latent: {latent_path}")
    lat = torch.load(latent_path, weights_only=True, map_location='cpu')
    z = lat.get('latents', lat.get('latent'))
    if z is None:
        # Try loading as raw tensor
        z = lat if isinstance(lat, torch.Tensor) else None
    print(f"  Latent shape: {z.shape}")

    # Decode
    z_4d = z.unsqueeze(0).to(device) if z.dim() == 3 else z.to(device)
    audio_lengths = torch.tensor([n_samples], device=device)
    with torch.no_grad():
        sr_out, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    dcae_wav = wavs[0].mean(dim=0).cpu().numpy()  # mono
    print(f"  DCAE wav: {len(dcae_wav)} samples, rms={rms(dcae_wav):.6f}")

    # --- 5. Align lengths ---
    min_len = min(len(orig_audio), len(sms_wav), len(dcae_wav))
    orig_audio = orig_audio[:min_len]
    sms_wav = sms_wav[:min_len]
    dcae_wav = dcae_wav[:min_len]

    # --- 6. Compute residual ---
    residual = dcae_wav - sms_wav
    reconstructed = sms_wav + residual  # should exactly equal dcae_wav

    # --- 7. Metrics ---
    print(f"\n{'='*60}")
    print(f"METRICS")
    print(f"{'='*60}")
    print(f"  Original RMS:      {rms(orig_audio):.6f}")
    print(f"  DCAE decoded RMS:  {rms(dcae_wav):.6f}")
    print(f"  SMS rendered RMS:  {rms(sms_wav):.6f}")
    print(f"  Residual RMS:      {rms(residual):.6f}")
    print()

    # How much does SMS capture of the original?
    sms_error = orig_audio - sms_wav
    print(f"  Original vs SMS:   SNR = {snr_db(orig_audio, sms_error):.1f} dB")

    # How much does DCAE capture of the original?
    dcae_error = orig_audio - dcae_wav
    print(f"  Original vs DCAE:  SNR = {snr_db(orig_audio, dcae_error):.1f} dB")

    # How much of DCAE output does SMS capture?
    print(f"  DCAE vs SMS:       SNR = {snr_db(dcae_wav, residual):.1f} dB")

    # Reconstruction check (should be ~inf or very high)
    recon_error = dcae_wav - reconstructed
    print(f"  DCAE vs Recon:     SNR = {snr_db(dcae_wav, recon_error):.1f} dB  (should be inf)")

    # Energy breakdown
    sms_energy = np.sum(sms_wav ** 2)
    res_energy = np.sum(residual ** 2)
    total_energy = np.sum(dcae_wav ** 2)
    print(f"\n  Energy breakdown of DCAE output:")
    print(f"    SMS component:      {100*sms_energy/total_energy:.1f}%")
    print(f"    Residual component: {100*res_energy/total_energy:.1f}%")
    print(f"    Cross-term:         {100*(total_energy - sms_energy - res_energy)/total_energy:.1f}%")

    # --- 8. Save wavs ---
    def save(name, audio):
        p = os.path.join(OUTPUT_DIR, name)
        a = np.clip(audio, -1, 1)
        torchaudio.save(p, torch.from_numpy(a).unsqueeze(0).float(), SAMPLE_RATE)
        print(f"  Saved: {p}")

    print(f"\nSaving outputs to {OUTPUT_DIR}/")
    save("01_original.wav", orig_audio / (np.abs(orig_audio).max() + 1e-8))
    save("02_dcae_decoded.wav", dcae_wav / (np.abs(dcae_wav).max() + 1e-8))
    save("03_sms_rendered.wav", sms_wav / (np.abs(sms_wav).max() + 1e-8))
    save("04_residual.wav", residual / (np.abs(residual).max() + 1e-8))
    save("05_sms_plus_residual.wav", reconstructed / (np.abs(reconstructed).max() + 1e-8))

    # Also save un-normalized versions to compare levels
    peak = max(np.abs(orig_audio).max(), np.abs(dcae_wav).max(), np.abs(sms_wav).max(), 1e-8)
    save("06_orig_matched.wav", orig_audio / peak)
    save("07_dcae_matched.wav", dcae_wav / peak)
    save("08_sms_matched.wav", sms_wav / peak)
    save("09_residual_matched.wav", residual / peak)

    print(f"\nDone! Listen to 03 vs 04 to hear what SMS captures vs what it misses.")


if __name__ == "__main__":
    main()
