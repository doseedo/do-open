"""Quick diagnostic: what does the classifier see for square_stab?"""
import sys
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
import numpy as np

# Import the functions we need
from test_inverse_synth import (
    WF_FNS, make_filter_envelope, apply_tv_filter,
    make_adsr_envelope, apply_envelope, detect_pitch_yin,
    classify_waveform, SAMPLE_RATE
)

# Generate square_stab audio
audio = WF_FNS['square'](220)
cutoff_env = make_filter_envelope(400, 5000, 0.002, 0.05, 0.0, 0.1, 0.1)
audio = apply_tv_filter(audio, cutoff_env, 0.4)
amp_env = make_adsr_envelope(0.002, 0.1, 0.0, 0.1, 0.15)
audio = apply_envelope(audio, amp_env)

print(f"Audio shape: {audio.shape}, duration: {len(audio)/SAMPLE_RATE:.3f}s")
print(f"RMS: {np.sqrt(np.mean(audio**2)):.4f}, peak: {np.max(np.abs(audio)):.4f}")

# Detect pitch
pitch, pitch_conf = detect_pitch_yin(audio)
print(f"\nPitch: {pitch:.1f}Hz (conf={pitch_conf:.2f})")

# Now run classify_waveform with detailed output
f0 = pitch
sr = SAMPLE_RATE
x_full = audio.astype(np.float64)
frame_sz = sr // 100
n_frames = min(len(x_full) // frame_sz, 100)
rms = np.array([np.sqrt(np.mean(x_full[i*frame_sz:(i+1)*frame_sz]**2))
                for i in range(n_frames)])
peak_frame = np.argmax(rms)
print(f"\nPeak energy frame: {peak_frame} ({peak_frame * frame_sz / sr * 1000:.1f}ms)")

min_periods = max(int(5 * sr / f0), sr // 20)
center = peak_frame * frame_sz
start = max(0, center - min_periods // 2)
end = min(len(x_full), center + min_periods)
n = max(end - start, min_periods)
start = max(0, end - n)
x = x_full[start:start + n]
print(f"Analysis window: {start/sr*1000:.1f}-{(start+len(x))/sr*1000:.1f}ms ({len(x)} samples)")

# FFT
window = np.hanning(len(x))
x_windowed = x * window
spec = np.abs(np.fft.rfft(x_windowed))
freqs = np.fft.rfftfreq(len(x_windowed), 1.0 / sr)
freq_res = freqs[1] - freqs[0]
print(f"Freq resolution: {freq_res:.1f}Hz")

max_harmonic = min(20, int((sr / 2) / f0))
print(f"Max harmonic: {max_harmonic}")

harmonic_amps = []
half_win = max(1, int(f0 * 0.1 / freq_res))
for h in range(1, max_harmonic + 1):
    freq_h = h * f0
    bin_h = int(round(freq_h / freq_res))
    lo = max(0, bin_h - half_win)
    hi = min(len(spec), bin_h + half_win + 1)
    if lo >= hi or hi > len(spec):
        break
    harmonic_amps.append(np.max(spec[lo:hi]))

amps = np.array(harmonic_amps)
noise_floor = np.median(spec[spec > 0]) if np.any(spec > 0) else 1e-10

print(f"\nHarmonic amplitudes (noise_floor={noise_floor:.2f}):")
for h, a in enumerate(amps, 1):
    tag = "odd" if h % 2 == 1 else "EVEN"
    above = "above" if a > noise_floor * 2 else "below"
    print(f"  h{h} ({h*f0:.0f}Hz): amp={a:.2f} [{tag}] {above} noise")

odd_indices = list(range(0, len(amps), 2))
even_indices = list(range(1, len(amps), 2))
odd_energy = np.sum(amps[odd_indices] ** 2)
even_energy = np.sum(amps[even_indices] ** 2)
even_odd_ratio = even_energy / (odd_energy + 1e-12)
print(f"\nEven/odd energy ratio: {even_odd_ratio:.4f}")
print(f"  Odd energy: {odd_energy:.2f}, Even energy: {even_energy:.2f}")

# 1/n model check
if len(even_indices) >= 2:
    odd_h_arr = np.array([i + 1 for i in odd_indices], dtype=float)
    even_h_arr = np.array([i + 1 for i in even_indices], dtype=float)
    a_estimates = amps[odd_indices] * odd_h_arr
    a_val = np.median(a_estimates)
    print(f"\n1/n model: A_estimate = {a_val:.2f}")
    print(f"  A from each odd: {a_estimates}")

    predicted_even = a_val / even_h_arr
    even_amps_arr = amps[even_indices]
    print(f"\nEven harmonic comparison (actual vs 1/n predicted):")
    for i, (ei, pred, act) in enumerate(zip(even_h_arr, predicted_even, even_amps_arr)):
        ratio = act / (pred + 1e-12)
        print(f"  h{int(ei)} ({int(ei*f0)}Hz): actual={act:.2f}, predicted={pred:.2f}, ratio={ratio:.3f}")

    valid_pred = predicted_even > noise_floor * 3
    if np.sum(valid_pred) >= 2:
        ratios = even_amps_arr[valid_pred] / (predicted_even[valid_pred] + 1e-12)
        median_ratio = float(np.median(ratios))
        print(f"\nMedian ratio (valid only): {median_ratio:.4f}")
        print(f"  is_saw = {median_ratio > 0.3}")

# Rolloff
odd_amps_vals = amps[odd_indices]
above_noise = amps > noise_floor * 2
valid = odd_amps_vals > noise_floor * 2
if np.sum(valid) >= 2:
    odd_h_numbers = [i + 1 for i in odd_indices]
    log_h = np.log(np.array(odd_h_numbers)[valid])
    log_a = np.log(odd_amps_vals[valid])
    slope, intercept = np.polyfit(log_h, log_a, 1)
    print(f"\nRolloff slope (odd harmonics): {slope:.3f}")
    print(f"  slope > -1.5 → square, slope ≤ -1.5 → triangle")

# Final classification
wf, conf = classify_waveform(audio, pitch)
print(f"\nFinal classification: {wf} (conf={conf:.2f})")

# Also test on other patches for comparison
print("\n" + "="*60)
print("Comparison with other patches:")
for name, wf_type, pitch_val, fbase, fpeak, fa, fd, fs, fr, fnoff, res, aa, ad, a_s, ar, anoff in [
    ('acid_bass', 'saw', 110, 200, 6000, 0.005, 0.2, 0.3, 0.3, 1.2, 0.7, 0.005, 0.2, 0.3, 0.3, 1.2),
    ('bright_lead', 'square', 440, 500, 8000, 0.01, 0.15, 0.3, 0.2, 1.2, 0.0, 0.01, 0.15, 0.5, 0.2, 1.2),
    ('dark_bass', 'saw', 110, 200, 800, 0.005, 0.2, 0.3, 0.3, 1.2, 0.5, 0.005, 0.2, 0.3, 0.3, 1.2),
]:
    a = WF_FNS[wf_type](pitch_val)
    ce = make_filter_envelope(fbase, fpeak, fa, fd, fs, fr, fnoff)
    a = apply_tv_filter(a, ce, res)
    ae = make_adsr_envelope(aa, ad, a_s, ar, anoff)
    a = apply_envelope(a, ae)
    p, pc = detect_pitch_yin(a)
    w, wc = classify_waveform(a, p)
    print(f"  {name} ({wf_type}@{pitch_val}): classified as {w} (conf={wc:.2f}), pitch={p:.1f}Hz")
