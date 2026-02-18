"""
Latent Modular Synth — Module Library

Standardized interfaces for all modules in the latent modular synth.
Everything speaks two signal types:
  - z signals: [B, 8, 16, T] — audio in DCAE latent space
  - mod signals: [T] or [B, T] — control values per frame, typically [0, 1]

Modulation sources produce mod signals (pure math, no training).
Transform modules accept z + mod signals (trained neural networks).
The SAME mod signal can be routed to multiple modules with different scaling.

Usage:
    env = Envelope.adsr(attack=0.01, decay=0.1, sustain=0.5, release=0.3, T=22)
    lfo = LFO.sine(rate=4.0, T=22, sr_z=11.0)

    # Same envelope → filter AND amplitude
    z = vcf(z, cutoff=env.route(amount=0.8, base=0.2), resonance=0.3)
    z = vca(z, gain=env.route(amount=1.0, base=0.0))

    # LFO → filter wobble
    z = vcf(z, cutoff=lfo.route(amount=0.3, base=0.5), resonance=0.5)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Modulation Signal Framework
# ============================================================

class ModSignal:
    """A control signal that can be routed to any module input.

    Values are in [0, 1] by convention (unipolar).
    Use .bipolar() for [-1, 1] range.
    """

    def __init__(self, curve: np.ndarray):
        """curve: [T] numpy array, values in [0, 1]."""
        self.curve = curve.astype(np.float32)
        self.T = len(curve)

    def route(self, amount: float = 1.0, base: float = 0.0) -> np.ndarray:
        """Route this signal to a module parameter.

        output = base + curve * amount
        Clipped to [0, 1] for normalized parameters.

        Args:
            amount: modulation depth (-1 to 1, negative inverts)
            base: base value when modulation is at 0
        Returns:
            [T] numpy array
        """
        return np.clip(base + self.curve * amount, 0.0, 1.0).astype(np.float32)

    def route_bipolar(self, amount: float = 1.0, center: float = 0.5) -> np.ndarray:
        """Route with bipolar modulation around a center value.

        output = center + (curve - 0.5) * 2 * amount
        """
        bipolar = (self.curve - 0.5) * 2  # [-1, 1]
        return np.clip(center + bipolar * amount, 0.0, 1.0).astype(np.float32)

    def to_tensor(self, device='cuda') -> torch.Tensor:
        """Convert to tensor [T]."""
        return torch.from_numpy(self.curve).float().to(device)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ModSignal(self.curve * other)
        elif isinstance(other, ModSignal):
            return ModSignal(self.curve * other.curve)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return ModSignal(self.curve + other)
        elif isinstance(other, ModSignal):
            return ModSignal(self.curve + other.curve)
        return NotImplemented


class Envelope:
    """Envelope generator — ADSR shapes as ModSignals."""

    @staticmethod
    def adsr(attack: float, decay: float, sustain: float,
             release: float, note_off: float, T: int,
             duration: float = 2.0) -> ModSignal:
        """Generate ADSR envelope at z-frame rate.

        Args:
            attack, decay, release: time in seconds
            sustain: sustain level [0, 1]
            note_off: note-off time in seconds
            T: number of z frames
            duration: total duration in seconds
        """
        fps = T / duration
        env = np.zeros(T, dtype=np.float32)

        attack_f = max(1, int(attack * fps))
        decay_f = max(1, int(decay * fps))
        release_f = max(1, int(release * fps))
        note_off_f = int(note_off * fps)

        t = 0
        # Attack
        end = min(t + attack_f, T)
        if end > t:
            env[t:end] = np.linspace(0, 1, end - t)
        t = end

        # Decay
        end = min(t + decay_f, T)
        if end > t:
            env[t:end] = np.linspace(1, sustain, end - t)
        t = end

        # Sustain
        end = min(note_off_f, T)
        if end > t:
            env[t:end] = sustain
        t = end

        # Release
        end = min(t + release_f, T)
        if end > t:
            env[t:end] = np.linspace(sustain, 0, end - t)
        t = end

        if t < T:
            env[t:] = 0.0

        return ModSignal(env)

    @staticmethod
    def ar(attack: float, release: float, T: int,
           duration: float = 2.0) -> ModSignal:
        """Attack-release envelope (no sustain)."""
        return Envelope.adsr(attack, 0.001, 1.0, release, attack + 0.001, T, duration)

    @staticmethod
    def ramp(start: float, end: float, T: int) -> ModSignal:
        """Linear ramp from start to end."""
        return ModSignal(np.linspace(start, end, T).astype(np.float32))

    @staticmethod
    def constant(value: float, T: int) -> ModSignal:
        """Constant value (static parameter)."""
        return ModSignal(np.full(T, value, dtype=np.float32))


class LFO:
    """Low-frequency oscillator — periodic ModSignals."""

    @staticmethod
    def sine(rate: float, T: int, duration: float = 2.0) -> ModSignal:
        """Sine LFO, output [0, 1]."""
        t = np.linspace(0, duration, T, dtype=np.float32)
        curve = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)
        return ModSignal(curve)

    @staticmethod
    def triangle(rate: float, T: int, duration: float = 2.0) -> ModSignal:
        """Triangle LFO, output [0, 1]."""
        t = np.linspace(0, duration, T, dtype=np.float32)
        phase = (t * rate) % 1.0
        curve = np.where(phase < 0.5, phase * 2, 2 - phase * 2)
        return ModSignal(curve.astype(np.float32))

    @staticmethod
    def saw(rate: float, T: int, duration: float = 2.0) -> ModSignal:
        """Sawtooth LFO, output [0, 1]."""
        t = np.linspace(0, duration, T, dtype=np.float32)
        curve = (t * rate) % 1.0
        return ModSignal(curve.astype(np.float32))

    @staticmethod
    def square(rate: float, T: int, duration: float = 2.0,
               pw: float = 0.5) -> ModSignal:
        """Square LFO with pulse width, output [0, 1]."""
        t = np.linspace(0, duration, T, dtype=np.float32)
        phase = (t * rate) % 1.0
        curve = np.where(phase < pw, 1.0, 0.0)
        return ModSignal(curve.astype(np.float32))


class SampleAndHold:
    """Sample & Hold — random stepped modulation."""

    @staticmethod
    def random(rate: float, T: int, duration: float = 2.0,
               seed: int = None) -> ModSignal:
        """Random values held for 1/rate seconds."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        fps = T / duration
        hold_frames = max(1, int(fps / rate))
        n_steps = T // hold_frames + 1
        values = rng.rand(n_steps).astype(np.float32)
        curve = np.repeat(values, hold_frames)[:T]
        return ModSignal(curve)


class SlewLimiter:
    """Slew limiter — smooth a control signal (portamento)."""

    @staticmethod
    def apply(signal: ModSignal, rise_time: float, fall_time: float,
              duration: float = 2.0) -> ModSignal:
        """Limit rate of change of a mod signal.

        rise_time/fall_time: time in seconds to slew from 0→1 or 1→0.
        """
        T = signal.T
        fps = T / duration
        rise_rate = 1.0 / max(rise_time * fps, 1)
        fall_rate = 1.0 / max(fall_time * fps, 1)

        out = np.zeros(T, dtype=np.float32)
        out[0] = signal.curve[0]
        for i in range(1, T):
            target = signal.curve[i]
            diff = target - out[i - 1]
            if diff > 0:
                out[i] = out[i - 1] + min(diff, rise_rate)
            else:
                out[i] = out[i - 1] + max(diff, -fall_rate)
        return ModSignal(out)


class Quantizer:
    """Pitch quantizer — snap values to musical intervals."""

    # Common scales as semitone offsets from root
    CHROMATIC = list(range(12))
    MAJOR = [0, 2, 4, 5, 7, 9, 11]
    MINOR = [0, 2, 3, 5, 7, 8, 10]
    PENTATONIC = [0, 2, 4, 7, 9]
    BLUES = [0, 3, 5, 6, 7, 10]

    @staticmethod
    def snap(signal: ModSignal, scale: list = None,
             octaves: int = 4) -> ModSignal:
        """Quantize a [0,1] signal to scale degrees.

        Maps 0→1 across `octaves` octaves of the given scale.
        """
        if scale is None:
            scale = Quantizer.CHROMATIC
        # Build full set of valid semitones across octaves
        valid = []
        for octave in range(octaves):
            for note in scale:
                valid.append(octave * 12 + note)
        valid = np.array(valid, dtype=np.float32)
        valid_norm = valid / (octaves * 12)  # normalize to [0, 1]

        # Snap each value to nearest valid note
        curve = signal.curve.copy()
        for i in range(len(curve)):
            idx = np.argmin(np.abs(valid_norm - curve[i]))
            curve[i] = valid_norm[idx]
        return ModSignal(curve)


# ============================================================
# Neural Module Base Classes
# ============================================================

class PerFrameFiLMBlock(nn.Module):
    """Residual block with per-frame FiLM conditioning."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.film_scale = nn.Linear(cond_dim, channels)
        self.film_shift = nn.Linear(cond_dim, channels)

        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

    def forward(self, x, cond):
        """x: [B, C, T], cond: [B, T, cond_dim]"""
        h = self.norm(x)
        h = self.conv(h)
        scale = self.film_scale(cond).permute(0, 2, 1)
        shift = self.film_shift(cond).permute(0, 2, 1)
        h = h * (1 + scale) + shift
        return x + F.gelu(h)


class DilatedFiLMBlock(nn.Module):
    """Residual block with dilated convolution for time-based effects.

    Larger receptive field for delay/reverb/chorus that redistribute
    energy across time frames.
    """

    def __init__(self, channels: int, cond_dim: int, dilation: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.film_scale = nn.Linear(cond_dim, channels)
        self.film_shift = nn.Linear(cond_dim, channels)

        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

    def forward(self, x, cond):
        """x: [B, C, T], cond: [B, cond_dim] (global)"""
        h = self.norm(x)
        h = self.conv(h)
        scale = self.film_scale(cond).unsqueeze(-1)
        shift = self.film_shift(cond).unsqueeze(-1)
        h = h * (1 + scale) + shift
        return x + F.gelu(h)


# ============================================================
# Transform Modules (Trained)
# ============================================================

class LatentWavefolder(nn.Module):
    """Wavefolder — nonlinear waveform distortion.

    (z, fold_amount[T]) → z_folded

    Per-frame fold amount for time-varying wavefolding.
    """

    def __init__(self, n_channels=8, latent_dim=16, cond_dim=64, n_blocks=4):
        super().__init__()
        self.flat_channels = n_channels * latent_dim

        self.param_enc = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, cond_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            PerFrameFiLMBlock(self.flat_channels, cond_dim)
            for _ in range(n_blocks)
        ])

        self.out_proj = nn.Conv1d(self.flat_channels, self.flat_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, fold_amount):
        """z: [B,8,16,T], fold_amount: [B,T] in [0,1]"""
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T)

        cond = self.param_enc(fold_amount.unsqueeze(-1))  # [B,T,cond_dim]

        h = z_flat
        for block in self.blocks:
            h = block(h, cond)

        delta = self.out_proj(h)
        return (z_flat + delta).reshape(B, C, H, T)


class LatentDistortion(nn.Module):
    """Distortion/Saturation — overdrive with tone control.

    (z, drive[T], tone) → z_distorted

    Per-frame drive for dynamic distortion, global tone.
    """

    def __init__(self, n_channels=8, latent_dim=16, cond_dim=64, n_blocks=4):
        super().__init__()
        self.flat_channels = n_channels * latent_dim

        # Per-frame: drive. Global: tone.
        self.param_enc = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, cond_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            PerFrameFiLMBlock(self.flat_channels, cond_dim)
            for _ in range(n_blocks)
        ])

        self.out_proj = nn.Conv1d(self.flat_channels, self.flat_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, drive, tone):
        """z: [B,8,16,T], drive: [B,T], tone: [B]"""
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T)

        tone_exp = tone.unsqueeze(-1).expand(-1, T)  # [B, T]
        params = torch.stack([drive, tone_exp], dim=-1)  # [B, T, 2]
        cond = self.param_enc(params)

        h = z_flat
        for block in self.blocks:
            h = block(h, cond)

        delta = self.out_proj(h)
        return (z_flat + delta).reshape(B, C, H, T)


class LatentRingMod(nn.Module):
    """Ring Modulator — multiply two z streams.

    (z_carrier, z_modulator, depth[T]) → z_ringmod

    Depth controls wet/dry: 0=dry carrier, 1=full ring modulation.
    """

    def __init__(self, n_channels=8, latent_dim=16, cond_dim=64, n_blocks=4):
        super().__init__()
        self.flat_channels = n_channels * latent_dim

        # Input: concat carrier + modulator
        self.in_proj = nn.Conv1d(self.flat_channels * 2, self.flat_channels, 1)

        self.param_enc = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, cond_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            PerFrameFiLMBlock(self.flat_channels, cond_dim)
            for _ in range(n_blocks)
        ])

        self.out_proj = nn.Conv1d(self.flat_channels, self.flat_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_carrier, z_mod, depth):
        """z_carrier, z_mod: [B,8,16,T], depth: [B,T]"""
        B, C, H, T = z_carrier.shape
        zc = z_carrier.reshape(B, C * H, T)
        zm = z_mod.reshape(B, C * H, T)

        z_cat = torch.cat([zc, zm], dim=1)  # [B, 256, T]
        h = self.in_proj(z_cat)

        cond = self.param_enc(depth.unsqueeze(-1))

        for block in self.blocks:
            h = block(h, cond)

        delta = self.out_proj(h)
        # Baseline: dry carrier at depth=0
        d = depth.unsqueeze(1)  # [B, 1, T]
        z_out = zc * (1 - d) + (zc + delta) * d
        return z_out.reshape(B, C, H, T)


class LatentDelay(nn.Module):
    """Delay — echo/feedback delay.

    (z, time, feedback, mix) → z_delayed

    Uses dilated convolutions for large temporal receptive field.
    Global parameters (not per-frame) since delay is a global effect.
    """

    def __init__(self, n_channels=8, latent_dim=16, cond_dim=128, n_blocks=6):
        super().__init__()
        self.flat_channels = n_channels * latent_dim

        self.param_enc = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, cond_dim),
            nn.GELU(),
        )

        # Dilated convolutions: dilation 1,2,4,8,4,2 covers full T=22
        dilations = [1, 2, 4, 8, 4, 2]
        self.blocks = nn.ModuleList([
            DilatedFiLMBlock(self.flat_channels, cond_dim, d)
            for d in dilations[:n_blocks]
        ])

        self.out_proj = nn.Conv1d(self.flat_channels, self.flat_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, time, feedback, mix):
        """z: [B,8,16,T], time/feedback/mix: [B] each in [0,1]"""
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T)

        params = torch.stack([time, feedback, mix], dim=-1)
        cond = self.param_enc(params)

        h = z_flat
        for block in self.blocks:
            h = block(h, cond)

        delta = self.out_proj(h)
        # Mix: 0=dry, 1=full wet
        m = mix.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        z_out = z_flat * (1 - m) + (z_flat + delta) * m
        return z_out.reshape(B, C, H, T)


class LatentReverb(nn.Module):
    """Reverb — room/space effect.

    (z, size, decay, mix) → z_reverbed

    Dilated convolutions for temporal smearing.
    """

    def __init__(self, n_channels=8, latent_dim=16, cond_dim=128, n_blocks=6):
        super().__init__()
        self.flat_channels = n_channels * latent_dim

        self.param_enc = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, cond_dim),
            nn.GELU(),
        )

        dilations = [1, 2, 4, 8, 4, 2]
        self.blocks = nn.ModuleList([
            DilatedFiLMBlock(self.flat_channels, cond_dim, d)
            for d in dilations[:n_blocks]
        ])

        self.out_proj = nn.Conv1d(self.flat_channels, self.flat_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, size, decay, mix):
        """z: [B,8,16,T], size/decay/mix: [B]"""
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T)

        params = torch.stack([size, decay, mix], dim=-1)
        cond = self.param_enc(params)

        h = z_flat
        for block in self.blocks:
            h = block(h, cond)

        delta = self.out_proj(h)
        m = mix.unsqueeze(-1).unsqueeze(-1)
        z_out = z_flat * (1 - m) + (z_flat + delta) * m
        return z_out.reshape(B, C, H, T)


class LatentChorus(nn.Module):
    """Chorus/Flanger — modulated delay effect.

    (z, rate, depth, mix) → z_chorused

    Rate and depth control the modulation; mix blends wet/dry.
    """

    def __init__(self, n_channels=8, latent_dim=16, cond_dim=128, n_blocks=6):
        super().__init__()
        self.flat_channels = n_channels * latent_dim

        self.param_enc = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, cond_dim),
            nn.GELU(),
        )

        dilations = [1, 2, 4, 8, 4, 2]
        self.blocks = nn.ModuleList([
            DilatedFiLMBlock(self.flat_channels, cond_dim, d)
            for d in dilations[:n_blocks]
        ])

        self.out_proj = nn.Conv1d(self.flat_channels, self.flat_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z, rate, depth, mix):
        """z: [B,8,16,T], rate/depth/mix: [B]"""
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T)

        params = torch.stack([rate, depth, mix], dim=-1)
        cond = self.param_enc(params)

        h = z_flat
        for block in self.blocks:
            h = block(h, cond)

        delta = self.out_proj(h)
        m = mix.unsqueeze(-1).unsqueeze(-1)
        z_out = z_flat * (1 - m) + (z_flat + delta) * m
        return z_out.reshape(B, C, H, T)


class LatentFM(nn.Module):
    """FM Synthesis — frequency modulation in z-space.

    (z_carrier, z_modulator, fm_index[T]) → z_fm

    Per-frame FM index for time-varying modulation depth.
    Index 0 = clean carrier, higher = more sidebands.
    Any ModSignal (envelope, LFO) can control FM depth.
    """

    def __init__(self, n_channels=8, latent_dim=16, cond_dim=128, n_blocks=6):
        super().__init__()
        self.flat_channels = n_channels * latent_dim

        # Input: concat carrier + modulator
        self.in_proj = nn.Conv1d(self.flat_channels * 2, self.flat_channels, 1)

        # Per-frame FM index conditioning
        self.param_enc = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, cond_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            PerFrameFiLMBlock(self.flat_channels, cond_dim)
            for _ in range(n_blocks)
        ])

        self.out_proj = nn.Conv1d(self.flat_channels, self.flat_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_carrier, z_mod, fm_index):
        """z_carrier, z_mod: [B,8,16,T], fm_index: [B,T] in [0,1]"""
        B, C, H, T = z_carrier.shape
        zc = z_carrier.reshape(B, C * H, T)
        zm = z_mod.reshape(B, C * H, T)

        z_cat = torch.cat([zc, zm], dim=1)  # [B, 256, T]
        h = self.in_proj(z_cat)

        cond = self.param_enc(fm_index.unsqueeze(-1))  # [B, T, cond_dim]

        for block in self.blocks:
            h = block(h, cond)

        delta = self.out_proj(h)
        # At index=0, output should be pure carrier
        idx = fm_index.unsqueeze(1)  # [B, 1, T]
        z_out = zc * (1 - idx) + (zc + delta) * idx
        return z_out.reshape(B, C, H, T)


# ============================================================
# Universal Chain Corrector
# ============================================================

# Module ID vocabulary — every module type gets a unique ID
MODULE_IDS = {
    'vco': 0, 'vcf_static': 1, 'vcf_temporal': 2, 'vca': 3,
    'mixer': 4, 'wavefolder': 5, 'distortion': 6, 'ringmod': 7,
    'delay': 8, 'reverb': 9, 'chorus': 10, 'fm': 11,
}
N_MODULE_TYPES = len(MODULE_IDS)
PAD_ID = N_MODULE_TYPES  # padding token
MAX_CHAIN_LEN = 12


class ChainEncoder(nn.Module):
    """Encode an ordered chain of module IDs into a conditioning vector.

    Uses learned embeddings + positional encoding, processed by a small
    transformer to capture ordering effects (VCF→Delay ≠ Delay→VCF).
    """

    def __init__(self, n_types: int = N_MODULE_TYPES + 1, embed_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, out_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.module_embed = nn.Embedding(n_types, embed_dim, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(MAX_CHAIN_LEN, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            dropout=0.0, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, chain_ids):
        """chain_ids: [B, L] — padded with PAD_ID.
        Returns: [B, out_dim] conditioning vector.
        """
        B, L = chain_ids.shape
        positions = torch.arange(L, device=chain_ids.device).unsqueeze(0).expand(B, -1)
        pad_mask = chain_ids == PAD_ID  # True where padded

        emb = self.module_embed(chain_ids) + self.pos_embed(positions)
        h = self.transformer(emb, src_key_padding_mask=pad_mask)

        # Pool: mean over non-padded positions
        mask_f = (~pad_mask).float().unsqueeze(-1)  # [B, L, 1]
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return self.out_proj(pooled)


class UniversalChainCorrector(nn.Module):
    """Universal chain drift corrector.

    Takes the output z of ANY module chain + a description of which modules
    were applied (as ordered IDs), and corrects accumulated error.

    Architecture:
    1. Small transformer encodes chain topology → conditioning vector
    2. Mix of dilated + standard FiLM blocks correct z (dilated for
       temporal redistribution from delay/reverb chains)
    3. Zero-init output for clean residual start

    The same corrector works for any chain depth and any module combination.
    """

    def __init__(self, n_channels: int = 8, latent_dim: int = 16,
                 chain_cond_dim: int = 128, n_blocks: int = 8):
        super().__init__()
        self.flat_channels = n_channels * latent_dim  # 128

        # Chain topology encoder
        self.chain_enc = ChainEncoder(out_dim=chain_cond_dim)

        # Correction blocks: alternating standard and dilated convolutions
        # Standard blocks handle spectral/timbral drift
        # Dilated blocks handle temporal drift (from delay/reverb chains)
        self.blocks = nn.ModuleList()
        dilations = [1, 1, 2, 4, 8, 4, 2, 1]  # mix of local and wide
        for i in range(n_blocks):
            d = dilations[i % len(dilations)]
            self.blocks.append(
                DilatedFiLMBlock(self.flat_channels, chain_cond_dim, dilation=d)
            )

        # Zero-init output
        self.out_proj = nn.Conv1d(self.flat_channels, self.flat_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_chain, chain_ids):
        """
        Args:
            z_chain: [B, 8, 16, T] — output of module chain (has drift)
            chain_ids: [B, L] — padded module ID sequence (what was applied)
        Returns:
            z_corrected: [B, 8, 16, T]
        """
        B, C, H, T = z_chain.shape
        z_flat = z_chain.reshape(B, C * H, T)

        # Encode chain topology
        cond = self.chain_enc(chain_ids)  # [B, chain_cond_dim]

        # Apply correction
        h = z_flat
        for block in self.blocks:
            h = block(h, cond)

        delta = self.out_proj(h)
        z_corrected = z_flat + delta

        return z_corrected.reshape(B, C, H, T)


def make_chain_ids(module_names, device='cuda'):
    """Convert list of module name strings to padded tensor.

    Args:
        module_names: list of strings like ['vcf_temporal', 'delay', 'vca']
    Returns:
        [1, MAX_CHAIN_LEN] tensor of module IDs, padded with PAD_ID
    """
    ids = [MODULE_IDS[name] for name in module_names]
    padded = ids + [PAD_ID] * (MAX_CHAIN_LEN - len(ids))
    return torch.tensor([padded], dtype=torch.long, device=device)
