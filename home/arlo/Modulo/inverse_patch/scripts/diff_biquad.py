#!/usr/bin/env python3
"""Differentiable time-varying biquad filter using torch.jit.script."""

import torch
import math

SAMPLE_RATE = 44100
N_SAMPLES = int(SAMPLE_RATE * 2.0)


@torch.jit.script
def tv_biquad_jit(audio: torch.Tensor, cutoff_curve: torch.Tensor, Q: torch.Tensor,
                  block_size: int = 64, sr: int = 44100) -> torch.Tensor:
    """Time-varying biquad lowpass. Matches DSP: coefficients updated per block, state carried."""
    N = audio.shape[0]
    out = torch.zeros_like(audio)
    w1 = torch.tensor(0.0)
    w2 = torch.tensor(0.0)

    n_blocks = (N + block_size - 1) // block_size
    pi2 = 2.0 * 3.141592653589793

    for block in range(n_blocks):
        start = block * block_size
        end = min(start + block_size, N)
        mid = min(start + block_size // 2, N - 1)

        fc = cutoff_curve[mid].clamp(20.0, float(sr) / 2.0 - 100.0)
        w0 = pi2 * fc / float(sr)
        alpha = torch.sin(w0) / (2.0 * Q)
        cos_w0 = torch.cos(w0)

        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        b0n = b0 / a0; b1n = b1 / a0; b2n = b2 / a0
        a1n = a1 / a0; a2n = a2 / a0

        for n in range(start, end):
            x_n = audio[n]
            y_n = b0n * x_n + w1
            w1 = b1n * x_n - a1n * y_n + w2
            w2 = b2n * x_n - a2n * y_n
            out[n] = y_n

    return out


def diff_apply_tv_filter(audio: torch.Tensor, cutoff_curve: torch.Tensor,
                         Q: torch.Tensor) -> torch.Tensor:
    """Apply time-varying lowpass. Two passes for 24dB/oct, with normalization between.
    Matches the DSP apply_tv_filter exactly."""
    filt1 = tv_biquad_jit(audio, cutoff_curve, Q)
    peak1 = filt1.abs().max().clamp(min=1e-6)
    filt1_n = filt1 / peak1 * 0.8

    filt2 = tv_biquad_jit(filt1_n, cutoff_curve, Q)
    peak2 = filt2.abs().max().clamp(min=1e-6)
    return filt2 / peak2 * 0.8
