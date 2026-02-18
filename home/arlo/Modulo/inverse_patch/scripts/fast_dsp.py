#!/usr/bin/env python3
"""Fast DSP rendering with numba for inverse synthesis optimization."""

import numpy as np
import numba
import math

SAMPLE_RATE = 44100
N_SAMPLES = int(SAMPLE_RATE * 2.0)
DURATION = 2.0


# ============================================================
# Waveform Generators
# ============================================================

def generate_saw(freq, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    t = np.arange(n_samples) / sr
    phase = (t * freq) % 1.0
    return (2.0 * phase - 1.0).astype(np.float32) * 0.8


def generate_square(freq, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    t = np.arange(n_samples) / sr
    phase = (t * freq) % 1.0
    return np.where(phase < 0.5, 0.8, -0.8).astype(np.float32)


def generate_triangle(freq, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    t = np.arange(n_samples) / sr
    phase = (t * freq) % 1.0
    return (4.0 * np.abs(phase - 0.5) - 1.0).astype(np.float32) * 0.8


def generate_sine(freq, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    """Pure sine — sub-bass, flute-like, FM carrier."""
    t = np.arange(n_samples) / sr
    return (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)


def generate_pulse(freq, duty=0.25, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    """Pulse wave with variable duty cycle — nasal, PWM character."""
    t = np.arange(n_samples) / sr
    phase = (t * freq) % 1.0
    return np.where(phase < duty, 0.8, -0.8).astype(np.float32)


def generate_noise(freq=None, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    """White noise — deterministic seed for optimizer reproducibility."""
    rng = np.random.RandomState(12345)
    return (rng.randn(n_samples) * 0.8).astype(np.float32)


def generate_supersaw(freq, n_voices=7, detune_cents=15, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    """7-voice detuned saw — classic supersaw / unison pad."""
    t = np.arange(n_samples, dtype=np.float64) / sr
    audio = np.zeros(n_samples, dtype=np.float64)
    detune_ratio = 2 ** (detune_cents / 1200)
    spread = np.linspace(-1, 1, n_voices)
    for s in spread:
        f = freq * (detune_ratio ** s)
        phase = (t * f) % 1.0
        audio += 2.0 * phase - 1.0
    audio /= n_voices
    return (audio * 0.8).astype(np.float32)


ALL_WF_FNS = {
    'saw': generate_saw,
    'square': generate_square,
    'triangle': generate_triangle,
    'sine': generate_sine,
    'pulse': generate_pulse,
    'noise': generate_noise,
    'supersaw': generate_supersaw,
}


# ============================================================
# Filter
# ============================================================

@numba.njit(cache=True)
def fast_tv_biquad(audio, cutoff_curve, Q, block_size=64, sr=44100):
    """Fast time-varying biquad lowpass with numba. Matches DSP exactly."""
    N = len(audio)
    out = np.zeros(N, dtype=np.float32)
    w1 = 0.0
    w2 = 0.0
    pi2 = 2.0 * math.pi

    n_blocks = (N + block_size - 1) // block_size
    for block in range(n_blocks):
        start = block * block_size
        end = min(start + block_size, N)
        mid = min(start + block_size // 2, N - 1)

        fc = max(20.0, min(float(cutoff_curve[mid]), sr / 2.0 - 100.0))
        w0 = pi2 * fc / sr
        sin_w0 = math.sin(w0)
        cos_w0 = math.cos(w0)
        alpha = sin_w0 / (2.0 * Q)

        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        b0n = b0 / a0; b1n = b1 / a0; b2n = b2 / a0
        a1n = a1 / a0; a2n = a2 / a0

        for n in range(start, end):
            x = audio[n]
            y = b0n * x + w1
            w1 = b1n * x - a1n * y + w2
            w2 = b2n * x - a2n * y
            out[n] = y

    return out


@numba.njit(cache=True)
def fast_tv_biquad_highpass(audio, cutoff_curve, Q, block_size=64, sr=44100):
    """Fast time-varying biquad highpass with numba. Cookbook HPF coefficients."""
    N = len(audio)
    out = np.zeros(N, dtype=np.float32)
    w1 = 0.0
    w2 = 0.0
    pi2 = 2.0 * math.pi

    n_blocks = (N + block_size - 1) // block_size
    for block in range(n_blocks):
        start = block * block_size
        end = min(start + block_size, N)
        mid = min(start + block_size // 2, N - 1)

        fc = max(20.0, min(float(cutoff_curve[mid]), sr / 2.0 - 100.0))
        w0 = pi2 * fc / sr
        sin_w0 = math.sin(w0)
        cos_w0 = math.cos(w0)
        alpha = sin_w0 / (2.0 * Q)

        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        b0n = b0 / a0; b1n = b1 / a0; b2n = b2 / a0
        a1n = a1 / a0; a2n = a2 / a0

        for n in range(start, end):
            x = audio[n]
            y = b0n * x + w1
            w1 = b1n * x - a1n * y + w2
            w2 = b2n * x - a2n * y
            out[n] = y

    return out


@numba.njit(cache=True)
def fast_tv_biquad_bandpass(audio, cutoff_curve, Q, block_size=64, sr=44100):
    """Fast time-varying biquad bandpass with numba. Cookbook BPF (constant skirt gain)."""
    N = len(audio)
    out = np.zeros(N, dtype=np.float32)
    w1 = 0.0
    w2 = 0.0
    pi2 = 2.0 * math.pi

    n_blocks = (N + block_size - 1) // block_size
    for block in range(n_blocks):
        start = block * block_size
        end = min(start + block_size, N)
        mid = min(start + block_size // 2, N - 1)

        fc = max(20.0, min(float(cutoff_curve[mid]), sr / 2.0 - 100.0))
        w0 = pi2 * fc / sr
        sin_w0 = math.sin(w0)
        cos_w0 = math.cos(w0)
        alpha = sin_w0 / (2.0 * Q)

        b0 = sin_w0 / 2.0
        b1 = 0.0
        b2 = -sin_w0 / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        b0n = b0 / a0; b1n = b1 / a0; b2n = b2 / a0
        a1n = a1 / a0; a2n = a2 / a0

        for n in range(start, end):
            x = audio[n]
            y = b0n * x + w1
            w1 = b1n * x - a1n * y + w2
            w2 = b2n * x - a2n * y
            out[n] = y

    return out


def fast_normalize(audio):
    peak = np.abs(audio).max()
    if peak > 1e-6:
        return (audio / peak * 0.8).astype(np.float32)
    return audio.astype(np.float32)


FILTER_TYPES = ['lowpass', 'highpass', 'bandpass']


def fast_apply_tv_filter(audio, cutoff_curve, resonance, filter_type='lowpass'):
    """Apply time-varying filter. Two passes for 24dB/oct (LPF/HPF) or single pass (BPF)."""
    Q = 0.707 + resonance * 15.0
    if filter_type == 'highpass':
        f1 = fast_normalize(fast_tv_biquad_highpass(audio, cutoff_curve, Q))
        f2 = fast_normalize(fast_tv_biquad_highpass(f1, cutoff_curve, Q))
        return fast_normalize(f2)
    elif filter_type == 'bandpass':
        # Single pass for BPF — two passes would be too narrow
        return fast_normalize(fast_tv_biquad_bandpass(audio, cutoff_curve, Q))
    else:  # lowpass (default)
        f1 = fast_normalize(fast_tv_biquad(audio, cutoff_curve, Q))
        f2 = fast_normalize(fast_tv_biquad(f1, cutoff_curve, Q))
        return fast_normalize(f2)


@numba.njit(cache=True)
def fast_make_filter_envelope(base_hz, peak_hz, attack_s, decay_s,
                               sustain_ratio, release_s, note_off_s,
                               n_samples=88200, sr=44100):
    """Generate filter cutoff envelope in Hz at audio sample rate."""
    sustain_hz = base_hz + sustain_ratio * (peak_hz - base_hz)
    env = np.full(n_samples, base_hz, dtype=np.float32)

    attack_n = int(attack_s * sr)
    decay_n = int(decay_s * sr)
    release_n = int(release_s * sr)
    note_off_n = int(note_off_s * sr)

    t = 0
    end_a = min(t + attack_n, n_samples)
    if end_a > t:
        for i in range(t, end_a):
            env[i] = base_hz + (peak_hz - base_hz) * (i - t) / max(end_a - t - 1, 1)
    t = end_a

    end_d = min(t + decay_n, n_samples)
    if end_d > t:
        for i in range(t, end_d):
            env[i] = peak_hz + (sustain_hz - peak_hz) * (i - t) / max(end_d - t - 1, 1)
    t = end_d

    end_s = min(note_off_n, n_samples)
    if end_s > t:
        for i in range(t, end_s):
            env[i] = sustain_hz
    t = end_s

    end_r = min(t + release_n, n_samples)
    if end_r > t:
        for i in range(t, end_r):
            env[i] = sustain_hz + (base_hz - sustain_hz) * (i - t) / max(end_r - t - 1, 1)
    t = end_r

    for i in range(t, n_samples):
        env[i] = base_hz

    return env


@numba.njit(cache=True)
def fast_make_amp_envelope(attack_s, decay_s, sustain, release_s, note_off_s,
                           n_samples=88200, sr=44100):
    """Generate amplitude ADSR envelope."""
    env = np.zeros(n_samples, dtype=np.float32)
    an = max(int(attack_s * sr), 1)
    dn = max(int(decay_s * sr), 1)
    noff = int(note_off_s * sr)
    rn = max(int(release_s * sr), 1)

    # Attack
    end_a = min(an, n_samples)
    for i in range(end_a):
        env[i] = float(i) / float(max(an - 1, 1))

    # Decay
    end_d = min(an + dn, n_samples)
    for i in range(an, end_d):
        env[i] = 1.0 + (sustain - 1.0) * float(i - an) / float(max(dn - 1, 1))

    # Sustain
    end_s = min(noff, n_samples)
    for i in range(end_d, end_s):
        env[i] = sustain

    # Release
    end_r = min(noff + rn, n_samples)
    for i in range(noff, end_r):
        env[i] = sustain * (1.0 - float(i - noff) / float(max(rn - 1, 1)))

    return env


@numba.njit(cache=True)
def fast_filter_lfo(cutoff_env, rate_hz, depth_hz, sr=44100):
    """Add sine LFO modulation to filter cutoff envelope.

    Args:
        cutoff_env: [N] float32 — base cutoff envelope in Hz
        rate_hz: float — LFO rate in Hz [0.5, 10]
        depth_hz: float — LFO depth in Hz [0, 3000]
    Returns:
        modulated cutoff envelope (clamped to [20, sr/2-100])
    """
    N = len(cutoff_env)
    out = np.empty(N, dtype=np.float32)
    pi2 = 2.0 * math.pi
    max_hz = sr / 2.0 - 100.0
    for i in range(N):
        lfo = math.sin(pi2 * rate_hz * float(i) / sr) * depth_hz
        val = cutoff_env[i] + lfo
        if val < 20.0:
            val = 20.0
        elif val > max_hz:
            val = max_hz
        out[i] = val
    return out


def full_render(params, waveform, sr=SAMPLE_RATE, n_samples=N_SAMPLES,
                filter_type='lowpass'):
    """Render full subtractive synth patch from params.
    params: [fb_hz, fp_hz, res, fa, fd, fs, fr, fnoff, aa, ad, a_s, ar, anoff]
           optionally: [..., lfo_rate, lfo_depth] (15 params)
    filter_type: 'lowpass', 'highpass', or 'bandpass'
    """
    fb_hz, fp_hz, res, fa, fd, fs, fr, fnoff, aa, ad, a_s, ar, anoff = params[:13]

    cutoff = fast_make_filter_envelope(fb_hz, fp_hz, fa, fd, fs, fr, fnoff, n_samples, sr)

    # Optional LFO modulation (params 13-14)
    if len(params) >= 15:
        lfo_rate = params[13]
        lfo_depth = params[14]
        if lfo_depth > 1.0:  # Only apply if depth is meaningful
            cutoff = fast_filter_lfo(cutoff, lfo_rate, lfo_depth, sr)

    filtered = fast_apply_tv_filter(waveform, cutoff, res, filter_type)

    amp = fast_make_amp_envelope(aa, ad, a_s, ar, anoff, n_samples, sr)
    return fast_normalize(filtered * amp)


def spectral_similarity(a, b, sr=SAMPLE_RATE, nperseg=1024):
    """Spectral cosine similarity between two signals."""
    from scipy.signal import stft as sp_stft
    _, _, Z1 = sp_stft(a, fs=sr, nperseg=nperseg)
    _, _, Z2 = sp_stft(b, fs=sr, nperseg=nperseg)
    S1, S2 = np.abs(Z1), np.abs(Z2)
    t = min(S1.shape[1], S2.shape[1])
    S1, S2 = S1[:, :t], S2[:, :t]
    return float(np.sum(S1 * S2) / (np.sqrt(np.sum(S1**2) * np.sum(S2**2)) + 1e-10))


def time_correlation(a, b, sr=SAMPLE_RATE, frame_ms=10):
    """Envelope time correlation."""
    fs = sr * frame_ms // 1000
    nf = min(len(a), len(b)) // fs
    e1 = np.abs(a[:nf * fs].reshape(nf, fs)).mean(-1)
    e2 = np.abs(b[:nf * fs].reshape(nf, fs)).mean(-1)
    if e1.std() < 1e-8 or e2.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(e1, e2)[0, 1])


def mel_distance(a, b, sr=SAMPLE_RATE, n_fft=2048, n_mels=128):
    """Mel spectrogram L1 distance."""
    import librosa
    M1 = librosa.feature.melspectrogram(y=a.astype(np.float32), sr=sr, n_fft=n_fft,
                                         n_mels=n_mels, power=1.0)
    M2 = librosa.feature.melspectrogram(y=b.astype(np.float32), sr=sr, n_fft=n_fft,
                                         n_mels=n_mels, power=1.0)
    t = min(M1.shape[1], M2.shape[1])
    return float(np.mean(np.abs(M1[:, :t] - M2[:, :t])))


# ============================================================
# Effects (numpy, for audio-domain optimization)
# ============================================================

def fast_distortion(audio, drive, tone_hz, sr=SAMPLE_RATE):
    """Soft-clip distortion with tone control.

    Args:
        audio: [N] float32
        drive: float, [1, 20] — gain before clipping
        tone_hz: float, [1000, 10000] — lowpass cutoff after clipping
    """
    driven = np.tanh(audio * drive).astype(np.float32)

    # Simple 2nd-order lowpass via FFT
    n_fft = 2048
    hop = 512
    from scipy.signal import stft as sp_stft, istft as sp_istft
    f, t_arr, Z = sp_stft(driven, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    H = 1.0 / (1.0 + (f / tone_hz) ** 2)
    Z_filt = Z * H[:, np.newaxis]
    _, out = sp_istft(Z_filt, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    out = out[:len(audio)].astype(np.float32)
    return fast_normalize(out)


def fast_delay(audio, time_s, feedback, mix, sr=SAMPLE_RATE):
    """Feedback delay line.

    Args:
        audio: [N] float32
        time_s: float, [0.05, 0.5] — delay time in seconds
        feedback: float, [0, 0.8] — feedback amount
        mix: float, [0, 1] — wet/dry mix
    """
    delay_samples = int(time_s * sr)
    N = len(audio)
    wet = np.zeros(N, dtype=np.float32)

    for tap in range(1, 6):  # up to 5 taps
        offset = tap * delay_samples
        if offset >= N:
            break
        gain = feedback ** tap
        if gain < 0.01:
            break
        n_copy = N - offset
        wet[offset:offset + n_copy] += audio[:n_copy] * gain

    out = audio * (1.0 - mix) + wet * mix
    return fast_normalize(out)


def fast_reverb(audio, size, decay, mix, sr=SAMPLE_RATE):
    """Convolution reverb with noise IR.

    Args:
        audio: [N] float32
        size: float, [0.2, 1.0] — room size (IR length fraction)
        decay: float, [0.2, 0.9] — decay rate
        mix: float, [0, 1] — wet/dry mix
    """
    max_ir = sr  # 1 second max IR
    ir_len = int(size * max_ir)

    # Exponential decay noise IR
    rng = np.random.RandomState(42)
    noise = rng.randn(ir_len).astype(np.float32) * 0.01
    decay_rate = max(decay * sr / 3.0, 100.0)
    t = np.arange(ir_len, dtype=np.float32)
    decay_env = np.exp(-t / decay_rate)
    ir = noise * decay_env

    # FFT convolution
    from scipy.signal import fftconvolve
    wet = fftconvolve(audio, ir, mode='full')[:len(audio)].astype(np.float32)

    # Normalize wet to match dry level
    wet_peak = np.abs(wet).max()
    dry_peak = np.abs(audio).max()
    if wet_peak > 1e-6:
        wet = wet / wet_peak * dry_peak

    out = audio * (1.0 - mix) + wet * mix
    return fast_normalize(out)


def fast_chorus(audio, rate, depth_ms, mix, sr=SAMPLE_RATE):
    """Chorus via modulated delay with linear interpolation.

    Args:
        audio: [N] float32
        rate: float, [0.5, 5] — LFO rate in Hz
        depth_ms: float, [1, 10] — modulation depth in ms
        mix: float, [0, 1] — dry/wet mix
    """
    N = len(audio)
    t = np.arange(N, dtype=np.float32)
    base_delay_ms = 10.0
    mod = depth_ms * np.sin(2 * np.pi * rate * t / sr)
    delay_samples = (base_delay_ms + mod) * sr / 1000.0

    wet = np.zeros(N, dtype=np.float32)
    for n in range(N):
        d = delay_samples[n]
        d_int = int(d)
        d_frac = d - d_int
        if n - d_int - 1 >= 0:
            wet[n] = audio[n - d_int] * (1 - d_frac) + audio[n - d_int - 1] * d_frac
        elif n - d_int >= 0:
            wet[n] = audio[n - d_int]

    out = audio * (1 - mix) + wet * mix
    return fast_normalize(out)


def fast_wavefold(audio, fold_amount):
    """Wavefolder: fold waveform peaks back.

    Args:
        audio: [N] float32
        fold_amount: float, [1, 8] — folding intensity
    """
    a = audio * fold_amount
    folded = np.arcsin(np.clip(np.sin(a * np.pi / 2), -1, 1)) * 2 / np.pi
    return fast_normalize(folded.astype(np.float32))


def full_render_with_effects(params, waveform, effects=None,
                              sr=SAMPLE_RATE, n_samples=N_SAMPLES,
                              n_synth_params=13, filter_type='lowpass'):
    """Render subtractive synth with optional effects.

    params: [fb_hz, fp_hz, res, fa, fd, fs, fr, fnoff, aa, ad, a_s, ar, anoff,
             (optional: lfo_rate, lfo_depth),
             *effect_params]
    effects: list of effect names, e.g. ['distortion', 'delay', 'reverb']
    n_synth_params: 13 (no LFO) or 15 (with LFO)
    filter_type: 'lowpass', 'highpass', or 'bandpass'

    Effect params come after the synth params, in the order of effects list:
      distortion: [drive, tone_hz]
      delay: [time_s, feedback, mix]
      reverb: [size, decay, mix]
    """
    # Base synth (13 or 15 params)
    audio = full_render(params[:n_synth_params], waveform, sr, n_samples,
                        filter_type=filter_type)

    if not effects:
        return audio

    idx = n_synth_params
    for fx in effects:
        if fx == 'distortion':
            drive = max(params[idx], 1.0)
            tone_hz = np.clip(params[idx + 1], 1000, 10000)
            audio = fast_distortion(audio, drive, tone_hz, sr)
            idx += 2
        elif fx == 'delay':
            time_s = np.clip(params[idx], 0.05, 0.5)
            feedback = np.clip(params[idx + 1], 0.0, 0.8)
            fx_mix = np.clip(params[idx + 2], 0.0, 1.0)
            audio = fast_delay(audio, time_s, feedback, fx_mix, sr)
            idx += 3
        elif fx == 'reverb':
            size = np.clip(params[idx], 0.2, 1.0)
            decay = np.clip(params[idx + 1], 0.2, 0.9)
            fx_mix = np.clip(params[idx + 2], 0.0, 1.0)
            audio = fast_reverb(audio, size, decay, fx_mix, sr)
            idx += 3
        elif fx == 'chorus':
            rate = np.clip(params[idx], 0.5, 5.0)
            depth_ms = np.clip(params[idx + 1], 1.0, 10.0)
            fx_mix = np.clip(params[idx + 2], 0.0, 1.0)
            audio = fast_chorus(audio, rate, depth_ms, fx_mix, sr)
            idx += 3
        elif fx == 'wavefold':
            fold_amount = np.clip(params[idx], 1.0, 8.0)
            audio = fast_wavefold(audio, fold_amount)
            idx += 1

    return audio


# Effect parameter bounds for scipy.optimize
EFFECT_BOUNDS = {
    'distortion': [(1.0, 20.0), (1000, 10000)],
    'delay': [(0.05, 0.5), (0.0, 0.8), (0.0, 1.0)],
    'reverb': [(0.2, 1.0), (0.2, 0.9), (0.0, 1.0)],
    'chorus': [(0.5, 5.0), (1.0, 10.0), (0.0, 1.0)],
    'wavefold': [(1.0, 8.0)],
}

EFFECT_DEFAULTS = {
    'distortion': [2.0, 5000],
    'delay': [0.2, 0.3, 0.3],
    'reverb': [0.5, 0.5, 0.3],
    'chorus': [1.5, 5.0, 0.4],
    'wavefold': [3.0],
}

# LFO parameter bounds (appended to synth params, making 15 total)
LFO_BOUNDS = [(0.5, 10.0), (0.0, 3000.0)]   # [rate_hz, depth_hz]
LFO_DEFAULTS = [2.0, 500.0]


# ============================================================
# FM Synthesis
# ============================================================

MAX_FM_INDEX = 8.0

# Common modulator ratios for FM: (ratio, description)
FM_RATIOS = [1.0, 2.0, 3.0, 1.414, 2.758, 0.5, 1.5]

FM_BOUNDS = [
    # mod_ratio, fm_index_peak, fm_a, fm_d, fm_s, fm_r, fm_noff,
    # amp_a, amp_d, amp_s, amp_r, amp_noff
    (0.25, 8.0),      # mod_ratio
    (0.0, 8.0),       # fm_index_peak
    (0.001, 2.0),     # fm_attack
    (0.001, 2.0),     # fm_decay
    (0.0, 1.0),       # fm_sustain (ratio of peak)
    (0.001, 2.0),     # fm_release
    (0.1, 3.0),       # fm_noteoff
    (0.001, 2.0),     # amp_attack
    (0.001, 2.0),     # amp_decay
    (0.0, 1.0),       # amp_sustain
    (0.001, 2.0),     # amp_release
    (0.1, 3.0),       # amp_noteoff
]


def fm_render(params, carrier_freq, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    """Render FM synthesis patch.

    Classic Chowning FM: output(t) = sin(2pi*fc*t + I(t) * sin(2pi*fm*t))

    params: [mod_ratio, fm_index_peak, fm_a, fm_d, fm_s, fm_r, fm_noff,
             amp_a, amp_d, amp_s, amp_r, amp_noff]  (12 params)
    carrier_freq: carrier frequency in Hz (not optimized — inferred from pitch)
    """
    mod_ratio = max(params[0], 0.25)
    fm_index_peak = max(params[1], 0.0)
    fm_a, fm_d, fm_s, fm_r, fm_noff = params[2], params[3], params[4], params[5], params[6]
    amp_a, amp_d, amp_s, amp_r, amp_noff = params[7], params[8], params[9], params[10], params[11]

    mod_freq = carrier_freq * mod_ratio

    # FM index envelope (ADSR on fm_index)
    fm_env = fast_make_amp_envelope(
        max(fm_a, 0.001), max(fm_d, 0.001),
        np.clip(fm_s, 0, 1), max(fm_r, 0.001),
        max(fm_noff, 0.1), n_samples, sr
    )
    fm_index_curve = fm_env * fm_index_peak

    # Generate FM audio
    t = np.arange(n_samples, dtype=np.float64) / sr
    modulator = np.sin(2 * np.pi * mod_freq * t)
    phase = 2 * np.pi * carrier_freq * t + fm_index_curve * modulator
    output = np.sin(phase).astype(np.float32)

    # Amplitude envelope
    amp_env = fast_make_amp_envelope(
        max(amp_a, 0.001), max(amp_d, 0.001),
        np.clip(amp_s, 0, 1), max(amp_r, 0.001),
        max(amp_noff, 0.1), n_samples, sr
    )

    return fast_normalize(output * amp_env)
