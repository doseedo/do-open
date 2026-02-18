#!/usr/bin/env python3
"""
Differentiable DSP for audio-domain inverse synthesis.

All components are fully differentiable through torch autograd:
- Waveform generators (bandlimited additive synthesis)
- Time-varying lowpass filter (frequency-domain)
- ADSR envelopes (soft-gated)
- DifferentiablePatch: waveform → filter → envelope → audio
- AudioDomainLoss: multi-resolution STFT + mel + envelope correlation
- Differentiable effects: distortion, delay, reverb
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 44100
DURATION = 2.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# Filter range (wider than the latent VCF — full audible range)
LOG_CUTOFF_MIN = math.log10(20)
LOG_CUTOFF_MAX = math.log10(20000)


# ============================================================
# Waveform Generators (torch, not learnable — precomputed)
# ============================================================

@torch.no_grad()
def torch_gen_saw(pitch, n_samples=N_SAMPLES, sr=SAMPLE_RATE, device='cuda'):
    t = torch.linspace(0, n_samples / sr, n_samples, device=device)
    audio = torch.zeros(n_samples, device=device)
    for h in range(1, 40):
        freq = pitch * h
        if freq > sr / 2:
            break
        audio += ((-1) ** (h + 1)) * torch.sin(2 * math.pi * freq * t) / h
    return _apply_env_torch(audio, sr)


@torch.no_grad()
def torch_gen_square(pitch, n_samples=N_SAMPLES, sr=SAMPLE_RATE, device='cuda'):
    t = torch.linspace(0, n_samples / sr, n_samples, device=device)
    audio = torch.zeros(n_samples, device=device)
    for h in range(1, 40, 2):
        freq = pitch * h
        if freq > sr / 2:
            break
        audio += torch.sin(2 * math.pi * freq * t) / h
    return _apply_env_torch(audio, sr)


@torch.no_grad()
def torch_gen_triangle(pitch, n_samples=N_SAMPLES, sr=SAMPLE_RATE, device='cuda'):
    t = torch.linspace(0, n_samples / sr, n_samples, device=device)
    audio = torch.zeros(n_samples, device=device)
    for i, h in enumerate(range(1, 40, 2)):
        freq = pitch * h
        if freq > sr / 2:
            break
        audio += ((-1) ** i) * torch.sin(2 * math.pi * freq * t) / (h * h)
    return _apply_env_torch(audio, sr)


def _apply_env_torch(audio, sr):
    attack = int(0.01 * sr)
    release = int(0.05 * sr)
    env = torch.ones_like(audio)
    env[:attack] = torch.linspace(0, 1, attack, device=audio.device)
    env[-release:] = torch.linspace(1, 0, release, device=audio.device)
    audio = audio * env
    peak = audio.abs().max().clamp(min=1e-6)
    return (audio / peak * 0.8)


TORCH_WF_FNS = {
    'saw': torch_gen_saw,
    'square': torch_gen_square,
    'triangle': torch_gen_triangle,
}


def torch_gen_waveform(wf_name, pitch, n_samples=N_SAMPLES, sr=SAMPLE_RATE, device='cuda'):
    return TORCH_WF_FNS[wf_name](pitch, n_samples, sr, device)


# ============================================================
# Differentiable ADSR Envelope (audio-rate)
# ============================================================

def diff_adsr_audio(attack_raw, decay_raw, sustain_raw, release_raw, note_off_raw,
                    n_samples, sr=SAMPLE_RATE):
    """Differentiable ADSR at audio sample rate.

    Raw params are transformed:
      attack/decay/release: softplus * 0.3 + 0.001 → seconds
      sustain: sigmoid → [0, 1]
      note_off: softplus * 0.8 + 0.1 → seconds
    """
    a = F.softplus(attack_raw) * 0.3 + 0.001
    d = F.softplus(decay_raw) * 0.3 + 0.001
    s = torch.sigmoid(sustain_raw)
    r = F.softplus(release_raw) * 0.3 + 0.001
    noff = F.softplus(note_off_raw) * 0.8 + 0.1

    t = torch.arange(n_samples, dtype=torch.float32, device=a.device) / sr

    attack_env = (t / a).clamp(0, 1)
    decay_progress = ((t - a) / d).clamp(0, 1)
    decay_env = 1.0 - (1.0 - s) * decay_progress
    release_progress = ((t - noff) / r).clamp(0, 1)
    release_env = s * (1.0 - release_progress)

    sharpness = 200.0  # higher for audio-rate (more samples to resolve)
    w_attack = torch.sigmoid(sharpness * (a - t))
    w_decay = torch.sigmoid(sharpness * (t - a)) * torch.sigmoid(sharpness * (a + d - t))
    w_sustain = torch.sigmoid(sharpness * (t - a - d)) * torch.sigmoid(sharpness * (noff - t))
    w_release = torch.sigmoid(sharpness * (t - noff))

    env = (w_attack * attack_env +
           w_decay * decay_env +
           w_sustain * s +
           w_release * release_env)
    return env.clamp(0, 1)


def diff_filter_env_audio(base_raw, peak_raw, attack_raw, decay_raw, sustain_raw,
                          release_raw, note_off_raw, n_samples, sr=SAMPLE_RATE):
    """Differentiable filter cutoff envelope, returns Hz at audio rate."""
    base_norm = torch.sigmoid(base_raw)
    peak_norm = torch.sigmoid(peak_raw)
    shape = diff_adsr_audio(attack_raw, decay_raw, sustain_raw, release_raw,
                            note_off_raw, n_samples, sr)
    cutoff_norm = base_norm + (peak_norm - base_norm) * shape
    cutoff_hz = 10 ** (cutoff_norm * (LOG_CUTOFF_MAX - LOG_CUTOFF_MIN) + LOG_CUTOFF_MIN)
    return cutoff_hz


# ============================================================
# Differentiable Time-Varying Filter (frequency domain)
# ============================================================

class DifferentiableFilter(nn.Module):
    """Frequency-domain time-varying lowpass filter using exact biquad transfer function.

    STFT → multiply by |H(e^jw)|^2 (two cascaded biquads = 24dB/oct) → ISTFT.
    Uses the exact cookbook biquad coefficients to match the DSP filter.
    Fully differentiable, avoids IIR recursion gradient issues.
    """

    def __init__(self, n_fft=1024, hop_length=256, sr=SAMPLE_RATE):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.register_buffer('window', torch.hann_window(n_fft))
        # Normalized digital frequencies: omega = 2*pi*f/sr for each FFT bin
        freqs_hz = torch.linspace(0, sr / 2, n_fft // 2 + 1)
        self.register_buffer('omega', 2 * math.pi * freqs_hz / sr)  # [F]

    def biquad_response(self, cutoff_hz, Q):
        """Compute exact biquad lowpass magnitude response (cookbook formula).

        Two cascaded passes (squared response) for 24dB/oct, matching DSP.

        Args:
            cutoff_hz: [T_stft] cutoff frequencies in Hz
            Q: scalar tensor, Q factor
        Returns:
            H: [F, T_stft] magnitude response (two passes combined)
        """
        # Cookbook biquad lowpass coefficients (vectorized over T_stft)
        w0 = 2 * math.pi * cutoff_hz.clamp(20, self.sr / 2 - 100) / self.sr  # [T_stft]
        alpha = torch.sin(w0) / (2 * Q)  # [T_stft]

        cos_w0 = torch.cos(w0)  # [T_stft]

        b0 = (1 - cos_w0) / 2  # [T_stft]
        b1 = 1 - cos_w0        # [T_stft]
        b2 = (1 - cos_w0) / 2  # [T_stft]
        a0 = 1 + alpha          # [T_stft]
        a1 = -2 * cos_w0       # [T_stft]
        a2 = 1 - alpha          # [T_stft]

        # Normalize by a0
        b0 = b0 / a0; b1 = b1 / a0; b2 = b2 / a0
        a1 = a1 / a0; a2 = a2 / a0

        # Evaluate |H(e^jw)| at each FFT bin frequency
        # H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
        # At z = e^jw: z^-1 = e^-jw, z^-2 = e^-2jw
        w = self.omega.unsqueeze(1)  # [F, 1]

        ej1 = torch.exp(-1j * w)     # [F, 1]
        ej2 = torch.exp(-2j * w)     # [F, 1]

        # Broadcast coefficients: [T_stft] -> [1, T_stft]
        b0 = b0.unsqueeze(0); b1 = b1.unsqueeze(0); b2 = b2.unsqueeze(0)
        a1 = a1.unsqueeze(0); a2 = a2.unsqueeze(0)

        numer = b0 + b1 * ej1 + b2 * ej2  # [F, T_stft] complex
        denom = 1.0 + a1 * ej1 + a2 * ej2  # [F, T_stft] complex

        # Single biquad magnitude
        H_single = numer.abs() / (denom.abs() + 1e-10)  # [F, T_stft]

        # Two cascaded passes = squared magnitude
        H = H_single * H_single

        return H.clamp(1e-8, 10)

    def forward(self, audio, cutoff_hz, resonance):
        """Apply time-varying lowpass filter (24dB/oct, matching DSP biquad).

        Args:
            audio: [N] raw waveform
            cutoff_hz: [N] cutoff in Hz at audio sample rate
            resonance: scalar [0, 1]
        Returns:
            filtered: [N] tensor
        """
        N = audio.shape[0]
        audio_2d = audio.unsqueeze(0)  # [1, N]

        # STFT
        X = torch.stft(audio_2d, self.n_fft, self.hop_length,
                        window=self.window, return_complex=True)  # [1, F, T_stft]
        T_stft = X.shape[-1]

        # Resample cutoff to STFT frame rate
        cutoff_frames = F.interpolate(
            cutoff_hz.unsqueeze(0).unsqueeze(0), size=T_stft,
            mode='linear', align_corners=False
        ).squeeze()  # [T_stft]

        # Q from resonance (matching DSP: 0.707 + resonance * 15.0)
        Q = 0.707 + resonance * 15.0

        # Compute and apply exact biquad response (24dB/oct)
        H = self.biquad_response(cutoff_frames, Q)  # [F, T_stft]
        Y = X.squeeze(0) * H  # [F, T_stft]

        # ISTFT
        filtered = torch.istft(Y.unsqueeze(0), self.n_fft, self.hop_length,
                                window=self.window, length=N)
        return filtered.squeeze(0)


# ============================================================
# DifferentiablePatch — full subtractive synth
# ============================================================

class DifferentiablePatch(nn.Module):
    """Differentiable subtractive synth: waveform → filter → amp envelope → audio.

    The waveform buffer is frozen (discrete choice). Only filter/amp params are learned.
    Same parameter interface as FixedSourcePatch in test_inverse_synth.py.
    """

    def __init__(self, waveform_audio, sr=SAMPLE_RATE, device='cuda'):
        super().__init__()
        self.sr = sr
        self.n_samples = waveform_audio.shape[0]
        self.register_buffer('waveform', waveform_audio.detach())

        # Filter envelope params (raw, transformed in forward)
        self.filter_base = nn.Parameter(torch.tensor(0.0, device=device))
        self.filter_peak = nn.Parameter(torch.tensor(2.0, device=device))
        self.filter_attack = nn.Parameter(torch.tensor(0.0, device=device))
        self.filter_decay = nn.Parameter(torch.tensor(0.0, device=device))
        self.filter_sustain = nn.Parameter(torch.tensor(0.0, device=device))
        self.filter_release = nn.Parameter(torch.tensor(0.0, device=device))
        self.filter_noteoff = nn.Parameter(torch.tensor(1.0, device=device))
        self.resonance_raw = nn.Parameter(torch.tensor(0.0, device=device))

        # Amp envelope params
        self.amp_attack = nn.Parameter(torch.tensor(0.0, device=device))
        self.amp_decay = nn.Parameter(torch.tensor(0.0, device=device))
        self.amp_sustain = nn.Parameter(torch.tensor(0.0, device=device))
        self.amp_release = nn.Parameter(torch.tensor(0.0, device=device))
        self.amp_noteoff = nn.Parameter(torch.tensor(1.0, device=device))

        # Differentiable filter
        self.diff_filter = DifferentiableFilter(n_fft=1024, hop_length=256, sr=sr).to(device)

    def forward(self):
        """Synthesize audio from current parameters. Returns: [N_SAMPLES]."""
        # Filter cutoff envelope in Hz
        cutoff_hz = diff_filter_env_audio(
            self.filter_base, self.filter_peak,
            self.filter_attack, self.filter_decay, self.filter_sustain,
            self.filter_release, self.filter_noteoff,
            self.n_samples, self.sr
        )

        # Apply filter
        resonance = torch.sigmoid(self.resonance_raw)
        filtered = self.diff_filter(self.waveform, cutoff_hz, resonance)

        # Apply amp envelope
        amp_env = diff_adsr_audio(
            self.amp_attack, self.amp_decay, self.amp_sustain,
            self.amp_release, self.amp_noteoff,
            self.n_samples, self.sr
        )
        audio = filtered * amp_env

        # Normalize with stop-gradient on the scaling factor so envelope
        # amplitude changes flow directly through the loss
        with torch.no_grad():
            peak = audio.abs().max().clamp(min=1e-6)
            scale = 0.8 / peak
        return audio * scale

    def get_readable_params(self):
        """Extract human-readable parameters (matches FixedSourcePatch interface)."""
        with torch.no_grad():
            base_c = torch.sigmoid(self.filter_base).item()
            peak_c = torch.sigmoid(self.filter_peak).item()
            base_hz = 10 ** (base_c * (LOG_CUTOFF_MAX - LOG_CUTOFF_MIN) + LOG_CUTOFF_MIN)
            peak_hz = 10 ** (peak_c * (LOG_CUTOFF_MAX - LOG_CUTOFF_MIN) + LOG_CUTOFF_MIN)
            resonance = torch.sigmoid(self.resonance_raw).item()

            fa = (F.softplus(self.filter_attack) * 0.3 + 0.001).item()
            fd = (F.softplus(self.filter_decay) * 0.3 + 0.001).item()
            fs = torch.sigmoid(self.filter_sustain).item()
            fr = (F.softplus(self.filter_release) * 0.3 + 0.001).item()
            fnoff = (F.softplus(self.filter_noteoff) * 0.8 + 0.1).item()

            aa = (F.softplus(self.amp_attack) * 0.3 + 0.001).item()
            ad = (F.softplus(self.amp_decay) * 0.3 + 0.001).item()
            a_s = torch.sigmoid(self.amp_sustain).item()
            ar = (F.softplus(self.amp_release) * 0.3 + 0.001).item()
            anoff = (F.softplus(self.amp_noteoff) * 0.8 + 0.1).item()

        return {
            'filter_base_hz': base_hz, 'filter_peak_hz': peak_hz,
            'resonance': resonance,
            'filter_adsr': (fa, fd, fs, fr, fnoff),
            'amp_adsr': (aa, ad, a_s, ar, anoff),
        }

    def init_from_readable(self, filter_base_hz, filter_peak_hz, resonance,
                           filter_adsr, amp_adsr):
        """Initialize from human-readable params (inverse of get_readable_params).
        Useful for warm-starting from known params or encoder output.
        """
        with torch.no_grad():
            # Filter base/peak: invert the log-scale sigmoid
            base_norm = (math.log10(max(filter_base_hz, 20)) - LOG_CUTOFF_MIN) / (LOG_CUTOFF_MAX - LOG_CUTOFF_MIN)
            peak_norm = (math.log10(max(filter_peak_hz, 20)) - LOG_CUTOFF_MIN) / (LOG_CUTOFF_MAX - LOG_CUTOFF_MIN)
            # sigmoid^-1(x) = log(x / (1-x))
            self.filter_base.fill_(_logit(base_norm))
            self.filter_peak.fill_(_logit(peak_norm))
            self.resonance_raw.fill_(_logit(resonance))

            fa, fd, fs, fr, fnoff = filter_adsr
            self.filter_attack.fill_(_softplus_inv((fa - 0.001) / 0.3))
            self.filter_decay.fill_(_softplus_inv((fd - 0.001) / 0.3))
            self.filter_sustain.fill_(_logit(fs))
            self.filter_release.fill_(_softplus_inv((fr - 0.001) / 0.3))
            self.filter_noteoff.fill_(_softplus_inv((fnoff - 0.1) / 0.8))

            aa, ad, a_s, ar, anoff = amp_adsr
            self.amp_attack.fill_(_softplus_inv((aa - 0.001) / 0.3))
            self.amp_decay.fill_(_softplus_inv((ad - 0.001) / 0.3))
            self.amp_sustain.fill_(_logit(a_s))
            self.amp_release.fill_(_softplus_inv((ar - 0.001) / 0.3))
            self.amp_noteoff.fill_(_softplus_inv((anoff - 0.1) / 0.8))


def _logit(x):
    x = max(min(x, 0.999), 0.001)
    return math.log(x / (1 - x))


def _softplus_inv(x):
    x = max(x, 0.001)
    return math.log(math.exp(x) - 1) if x < 20 else x


# ============================================================
# Audio-Domain Loss
# ============================================================

class AudioDomainLoss(nn.Module):
    """Combined audio-domain loss for parameter optimization.

    Components:
    1. Multi-resolution STFT (spectral convergence + log-magnitude L1) — 50%
    2. Mel-spectrogram L1 — 30%
    3. RMS envelope correlation — 20%
    """

    def __init__(self, fft_sizes=(512, 1024, 2048), sr=SAMPLE_RATE, device='cuda'):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = [s // 4 for s in fft_sizes]
        self.sr = sr

        # Precompute windows (not nn.Parameters — just buffers)
        self.windows = {}
        for fs in fft_sizes:
            self.register_buffer(f'win_{fs}', torch.hann_window(fs, device=device))
            self.windows[fs] = f'win_{fs}'

        # Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128,
            power=2.0
        ).to(device)

    def _get_window(self, fs):
        return getattr(self, self.windows[fs])

    def forward(self, pred, target):
        """Compute audio-domain loss.

        Args:
            pred: [N] predicted audio
            target: [N] target audio
        Returns:
            (loss, metrics_dict)
        """
        pred_2d = pred.unsqueeze(0)
        target_2d = target.unsqueeze(0)

        # 1. Multi-resolution STFT loss
        stft_loss = torch.tensor(0.0, device=pred.device)
        for fs, hs in zip(self.fft_sizes, self.hop_sizes):
            win = self._get_window(fs)
            P = torch.stft(pred_2d, fs, hs, window=win, return_complex=True).abs()
            T = torch.stft(target_2d, fs, hs, window=win, return_complex=True).abs()

            # Spectral convergence: ||T - P||_F / ||T||_F
            sc = (T - P).norm() / (T.norm() + 1e-8)
            # Log-magnitude L1
            lm = F.l1_loss(torch.log(P + 1e-7), torch.log(T + 1e-7))
            stft_loss = stft_loss + sc + lm
        stft_loss = stft_loss / len(self.fft_sizes)

        # 2. Mel-spectrogram loss
        mel_pred = self.mel_transform(pred_2d)
        mel_target = self.mel_transform(target_2d)
        mel_loss = F.l1_loss(
            torch.log(mel_pred.clamp(min=1e-7)),
            torch.log(mel_target.clamp(min=1e-7))
        )

        # 3. Envelope correlation (mean-abs per frame, avoids sqrt(0) grad issue)
        frame_size = self.sr // 100  # 10ms frames
        n_frames = min(pred.shape[0], target.shape[0]) // frame_size
        if n_frames >= 2:
            pred_env = pred[:n_frames * frame_size].reshape(n_frames, frame_size).abs().mean(-1)
            tgt_env = target[:n_frames * frame_size].reshape(n_frames, frame_size).abs().mean(-1)
            # L1 on log-envelopes (like mel loss, perceptually weighted)
            pred_log = torch.log(pred_env.clamp(min=1e-7))
            tgt_log = torch.log(tgt_env.clamp(min=1e-7))
            env_loss = F.l1_loss(pred_log, tgt_log)
        else:
            env_loss = torch.tensor(0.0, device=pred.device)

        total = 0.5 * stft_loss + 0.3 * mel_loss + 0.2 * env_loss

        metrics = {
            'total': total.item(),
            'stft': stft_loss.item(),
            'mel': mel_loss.item(),
            'env': env_loss.item(),
        }
        return total, metrics


# ============================================================
# Differentiable Effects
# ============================================================

class DifferentiableDistortion(nn.Module):
    """Differentiable soft-clip distortion with tone control."""

    def __init__(self, sr=SAMPLE_RATE, device='cuda'):
        super().__init__()
        self.sr = sr
        self.drive_raw = nn.Parameter(torch.tensor(0.0, device=device))
        self.tone_raw = nn.Parameter(torch.tensor(0.0, device=device))
        self.n_fft = 1024
        self.hop = 256
        self.register_buffer('window', torch.hann_window(self.n_fft, device=device))
        self.register_buffer('freqs', torch.linspace(0, sr / 2, self.n_fft // 2 + 1, device=device))

    def forward(self, audio):
        drive = F.softplus(self.drive_raw) + 1.0  # [1, ~20]
        driven = torch.tanh(audio * drive)

        # Tone filter: simple lowpass in frequency domain
        tone_hz = torch.sigmoid(self.tone_raw) * 9000 + 1000  # [1000, 10000]
        X = torch.stft(driven.unsqueeze(0), self.n_fft, self.hop,
                        window=self.window, return_complex=True)
        f = self.freqs.unsqueeze(1)  # [F, 1]
        H = 1.0 / (1.0 + (f / tone_hz).pow(2))  # gentle 12dB/oct
        Y = X.squeeze(0) * H
        out = torch.istft(Y.unsqueeze(0), self.n_fft, self.hop,
                           window=self.window, length=audio.shape[0])
        return out.squeeze(0)


class DifferentiableDelay(nn.Module):
    """Differentiable feedback delay via comb filter in frequency domain."""

    def __init__(self, sr=SAMPLE_RATE, n_fft=2048, hop=512, device='cuda'):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.time_raw = nn.Parameter(torch.tensor(0.0, device=device))
        self.feedback_raw = nn.Parameter(torch.tensor(-1.0, device=device))
        self.mix_raw = nn.Parameter(torch.tensor(-2.0, device=device))
        self.register_buffer('window', torch.hann_window(n_fft, device=device))

    def forward(self, audio):
        time_s = torch.sigmoid(self.time_raw) * 0.45 + 0.05  # [50, 500] ms
        feedback = torch.sigmoid(self.feedback_raw) * 0.8  # [0, 0.8]
        mix = torch.sigmoid(self.mix_raw)

        # Comb filter response
        freqs = torch.linspace(0, self.sr / 2, self.n_fft // 2 + 1, device=audio.device)
        omega = 2 * math.pi * freqs * time_s

        # H_wet(f) = g * exp(-jw) / (1 - g * exp(-jw))
        z_inv = torch.exp(-1j * omega)
        g = feedback.to(torch.complex64)
        H_wet = g * z_inv / (1 - g * z_inv + 1e-8)
        H_total = 1.0 + mix * H_wet  # dry + wet

        X = torch.stft(audio.unsqueeze(0), self.n_fft, self.hop,
                        window=self.window, return_complex=True)  # [1, F, T]
        Y = X * H_total.unsqueeze(0).unsqueeze(-1)
        out = torch.istft(Y, self.n_fft, self.hop,
                           window=self.window, length=audio.shape[0])
        return out.squeeze(0)


class DifferentiableReverb(nn.Module):
    """Differentiable reverb via learnable FIR (noise * decay envelope)."""

    def __init__(self, max_ir_samples=44100, sr=SAMPLE_RATE, device='cuda'):
        super().__init__()
        self.sr = sr
        self.max_ir = max_ir_samples
        self.size_raw = nn.Parameter(torch.tensor(0.0, device=device))
        self.decay_raw = nn.Parameter(torch.tensor(0.0, device=device))
        self.mix_raw = nn.Parameter(torch.tensor(-2.0, device=device))
        # Fixed noise for IR
        self.register_buffer('noise', torch.randn(max_ir_samples, device=device) * 0.01)

    def forward(self, audio):
        size = torch.sigmoid(self.size_raw) * 0.8 + 0.2  # [0.2, 1.0]
        decay = torch.sigmoid(self.decay_raw) * 0.7 + 0.2  # [0.2, 0.9]
        mix = torch.sigmoid(self.mix_raw)

        # Exponential decay envelope over full IR length, modulated by size/decay
        t = torch.arange(self.max_ir, device=audio.device, dtype=torch.float32)
        # Decay rate: shorter for small rooms, longer for large
        decay_rate = decay * self.sr / 3.0
        decay_env = torch.exp(-t / decay_rate.clamp(min=100))
        # Size controls effective IR length via a soft window
        ir_len_samples = size * self.sr
        soft_gate = torch.sigmoid(20.0 * (ir_len_samples - t) / self.sr)
        ir = self.noise * decay_env * soft_gate

        # Convolve in frequency domain
        n_fft = audio.shape[0] + self.max_ir - 1
        n_fft_pow2 = 2 ** int(math.ceil(math.log2(max(n_fft, 1))))
        X = torch.fft.rfft(audio, n=n_fft_pow2)
        H = torch.fft.rfft(ir, n=n_fft_pow2)
        wet = torch.fft.irfft(X * H, n=n_fft_pow2)[:audio.shape[0]]

        # Normalize wet to match dry level
        wet_peak = wet.abs().max().clamp(min=1e-6)
        dry_peak = audio.abs().max().clamp(min=1e-6)
        wet = wet / wet_peak * dry_peak

        return audio * (1 - mix) + wet * mix


class DifferentiablePatchWithEffects(DifferentiablePatch):
    """DifferentiablePatch with optional effects chain."""

    def __init__(self, waveform_audio, sr=SAMPLE_RATE, device='cuda'):
        super().__init__(waveform_audio, sr, device)
        self.diff_distortion = DifferentiableDistortion(sr, device)
        self.diff_delay = DifferentiableDelay(sr=sr, device=device)
        self.diff_reverb = DifferentiableReverb(sr=sr, device=device)

    def forward(self, active_effects=None):
        # Base subtractive synth (filter + amp)
        audio = super().forward()

        if active_effects:
            if 'distortion' in active_effects:
                audio = self.diff_distortion(audio)
            if 'delay' in active_effects:
                audio = self.diff_delay(audio)
            if 'reverb' in active_effects:
                audio = self.diff_reverb(audio)

        peak = audio.abs().max().clamp(min=1e-6)
        return audio / peak * 0.8


# ============================================================
# Optimization helpers
# ============================================================

def optimize_audio_domain(patch, target_audio, loss_fn, n_steps=300, lr=0.01,
                          active_effects=None, verbose=False):
    """Optimize a DifferentiablePatch against target audio.

    Args:
        patch: DifferentiablePatch or DifferentiablePatchWithEffects
        target_audio: [N] torch tensor
        loss_fn: AudioDomainLoss instance
        n_steps: optimization steps
        lr: learning rate
        active_effects: set of effect names (for PatchWithEffects)
    Returns:
        best_loss: float
    """
    optimizer = torch.optim.Adam(patch.parameters(), lr=lr)
    T_0 = max(n_steps // 3, 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1, eta_min=lr * 0.01
    )

    best_loss = float('inf')
    best_state = None

    for step in range(n_steps):
        optimizer.zero_grad()

        if active_effects is not None:
            pred_audio = patch(active_effects=active_effects)
        else:
            pred_audio = patch()

        loss, metrics = loss_fn(pred_audio, target_audio)
        loss.backward()
        # Per-parameter clip (not global norm, which lets large grads suppress small ones)
        for p in patch.parameters():
            if p.grad is not None:
                p.grad.clamp_(-5.0, 5.0)
        optimizer.step()
        scheduler.step()

        if metrics['total'] < best_loss:
            best_loss = metrics['total']
            best_state = {k: v.clone() for k, v in patch.state_dict().items()}

        if verbose and (step % 100 == 0 or step == n_steps - 1):
            print(f"      Step {step:3d}: loss={metrics['total']:.4f} "
                  f"stft={metrics['stft']:.4f} mel={metrics['mel']:.4f} "
                  f"env={metrics['env']:.4f}")

    if best_state:
        patch.load_state_dict(best_state)

    return best_loss
