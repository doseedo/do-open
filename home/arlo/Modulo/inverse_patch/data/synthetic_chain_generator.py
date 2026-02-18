"""
Synthetic Effect Chain Generator.

Generates training data by applying random effect chains to dry audio.

NOTE: NablAFx processors are imported lazily inside EffectChainGenerator._init_effects()
to avoid CLAP/FAD argparse conflicts when this module is imported.
"""

import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Tuple, Optional, NamedTuple, Any
from dataclasses import dataclass, field

# NablAFx processors are imported lazily to avoid CLAP/FAD argparse hijacking
# See EffectChainGenerator._init_effects() for the actual import


@dataclass
class EffectParams:
    """Parameters for a single effect."""
    effect_type: str
    params: Dict[str, torch.Tensor]
    normalized_params: Dict[str, torch.Tensor]  # Params in [0, 1] range


@dataclass
class ChainSpec:
    """Specification for an effect chain."""
    effects: List[EffectParams]

    def __len__(self):
        return len(self.effects)

    def to_list(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Convert to list of (effect_type, params) tuples."""
        return [(e.effect_type, {k: v.item() if v.numel() == 1 else v.tolist()
                                  for k, v in e.params.items()})
                for e in self.effects]


class DifferentiableCompressor(nn.Module):
    """
    Differentiable compressor implementation.
    Uses a soft-knee compression curve that is fully differentiable.
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        min_threshold_db: float = -60.0,
        max_threshold_db: float = 0.0,
        min_ratio: float = 1.0,
        max_ratio: float = 20.0,
        min_attack_ms: float = 0.1,
        max_attack_ms: float = 100.0,
        min_release_ms: float = 10.0,
        max_release_ms: float = 1000.0,
        min_knee_db: float = 0.0,
        max_knee_db: float = 12.0,
        min_makeup_db: float = 0.0,
        max_makeup_db: float = 24.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.param_ranges = {
            'threshold_db': (min_threshold_db, max_threshold_db),
            'ratio': (min_ratio, max_ratio),
            'attack_ms': (min_attack_ms, max_attack_ms),
            'release_ms': (min_release_ms, max_release_ms),
            'knee_db': (min_knee_db, max_knee_db),
            'makeup_db': (min_makeup_db, max_makeup_db),
        }
        self.num_control_params = 6

    def denormalize(self, norm_val: torch.Tensor, param_name: str) -> torch.Tensor:
        """Denormalize parameter from [0, 1] to actual range."""
        min_val, max_val = self.param_ranges[param_name]
        return norm_val * (max_val - min_val) + min_val

    def forward(
        self,
        x: torch.Tensor,
        control_params: torch.Tensor,
        train: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply compression.

        Args:
            x: Input audio [B, 1, T]
            control_params: Normalized parameters [B, 6, 1]

        Returns:
            Compressed audio and parameter dict
        """
        bs, chs, seq_len = x.size()

        # Extract and denormalize parameters
        threshold_db = self.denormalize(control_params[:, 0, 0], 'threshold_db')
        ratio = self.denormalize(control_params[:, 1, 0], 'ratio')
        attack_ms = self.denormalize(control_params[:, 2, 0], 'attack_ms')
        release_ms = self.denormalize(control_params[:, 3, 0], 'release_ms')
        knee_db = self.denormalize(control_params[:, 4, 0], 'knee_db')
        makeup_db = self.denormalize(control_params[:, 5, 0], 'makeup_db')

        param_dict = {
            'threshold_db': threshold_db,
            'ratio': ratio,
            'attack_ms': attack_ms,
            'release_ms': release_ms,
            'knee_db': knee_db,
            'makeup_db': makeup_db,
        }

        output = self.process(x, **param_dict)
        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        threshold_db: torch.Tensor,
        ratio: torch.Tensor,
        attack_ms: torch.Tensor,
        release_ms: torch.Tensor,
        knee_db: torch.Tensor,
        makeup_db: torch.Tensor,
    ) -> torch.Tensor:
        """Apply compression with given parameters."""
        bs, chs, seq_len = x.size()

        # Convert to dB for level detection
        x_abs = torch.abs(x) + 1e-8
        x_db = 20 * torch.log10(x_abs)

        # Compute gain reduction with soft knee
        threshold = threshold_db.view(bs, 1, 1)
        r = ratio.view(bs, 1, 1)
        knee = knee_db.view(bs, 1, 1)

        # Soft knee compression curve
        over_threshold = x_db - threshold

        # Below knee
        below_knee = over_threshold < -knee / 2
        # In knee region
        in_knee = (over_threshold >= -knee / 2) & (over_threshold <= knee / 2)
        # Above knee
        above_knee = over_threshold > knee / 2

        gain_db = torch.zeros_like(x_db)

        # In knee region: quadratic interpolation
        knee_gain = (1 / r - 1) * (over_threshold + knee / 2).pow(2) / (2 * knee + 1e-8)
        gain_db = torch.where(in_knee, knee_gain, gain_db)

        # Above knee: linear compression
        above_gain = (1 / r - 1) * over_threshold
        gain_db = torch.where(above_knee, above_gain, gain_db)

        # Apply attack/release envelope follower (simplified differentiable version)
        attack_coef = torch.exp(-1.0 / (attack_ms.view(bs, 1, 1) * self.sample_rate / 1000 + 1e-8))
        release_coef = torch.exp(-1.0 / (release_ms.view(bs, 1, 1) * self.sample_rate / 1000 + 1e-8))

        # Smooth gain reduction
        gain_db_smooth = self._smooth_gain(gain_db, attack_coef, release_coef)

        # Apply gain and makeup
        gain_linear = 10 ** (gain_db_smooth / 20)
        makeup_linear = 10 ** (makeup_db.view(bs, 1, 1) / 20)

        return x * gain_linear * makeup_linear

    def _smooth_gain(
        self,
        gain_db: torch.Tensor,
        attack_coef: torch.Tensor,
        release_coef: torch.Tensor
    ) -> torch.Tensor:
        """Apply attack/release smoothing to gain curve."""
        # Simplified: use 1D convolution as approximation for differentiability
        bs, chs, seq_len = gain_db.size()

        # Create smoothing kernel (exponential decay)
        kernel_size = min(1024, seq_len // 4)
        if kernel_size < 2:
            return gain_db

        t = torch.arange(kernel_size, device=gain_db.device, dtype=gain_db.dtype)

        # Average of attack and release for smoothing
        avg_coef = (attack_coef + release_coef) / 2
        kernel = torch.exp(-t.view(1, 1, -1) / (kernel_size / 4))
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)

        # Apply convolution
        gain_db_padded = torch.nn.functional.pad(gain_db, (kernel_size - 1, 0), mode='replicate')
        gain_db_smooth = torch.nn.functional.conv1d(
            gain_db_padded,
            kernel.expand(1, 1, -1),
            groups=1
        )

        return gain_db_smooth[..., :seq_len]


class DifferentiableReverb(nn.Module):
    """
    Differentiable reverb using FDN (Feedback Delay Network).
    Simplified version for training purposes.
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        min_decay_time: float = 0.1,
        max_decay_time: float = 10.0,
        min_pre_delay_ms: float = 0.0,
        max_pre_delay_ms: float = 100.0,
        min_wet_mix: float = 0.0,
        max_wet_mix: float = 1.0,
        min_damping: float = 0.0,
        max_damping: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.param_ranges = {
            'decay_time': (min_decay_time, max_decay_time),
            'pre_delay_ms': (min_pre_delay_ms, max_pre_delay_ms),
            'wet_mix': (min_wet_mix, max_wet_mix),
            'damping': (min_damping, max_damping),
        }
        self.num_control_params = 4

        # FDN delay line lengths (in samples at 44100)
        self.delay_lengths = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]

    def denormalize(self, norm_val: torch.Tensor, param_name: str) -> torch.Tensor:
        min_val, max_val = self.param_ranges[param_name]
        return norm_val * (max_val - min_val) + min_val

    def forward(
        self,
        x: torch.Tensor,
        control_params: torch.Tensor,
        train: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, seq_len = x.size()

        decay_time = self.denormalize(control_params[:, 0, 0], 'decay_time')
        pre_delay_ms = self.denormalize(control_params[:, 1, 0], 'pre_delay_ms')
        wet_mix = self.denormalize(control_params[:, 2, 0], 'wet_mix')
        damping = self.denormalize(control_params[:, 3, 0], 'damping')

        param_dict = {
            'decay_time': decay_time,
            'pre_delay_ms': pre_delay_ms,
            'wet_mix': wet_mix,
            'damping': damping,
        }

        output = self.process(x, **param_dict)
        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        decay_time: torch.Tensor,
        pre_delay_ms: torch.Tensor,
        wet_mix: torch.Tensor,
        damping: torch.Tensor,
    ) -> torch.Tensor:
        """Apply reverb using convolution-based approximation."""
        bs, chs, seq_len = x.size()

        # Generate impulse response
        ir_length = int(decay_time.max().item() * self.sample_rate)
        ir_length = min(ir_length, seq_len, 88200)  # Max 2 seconds

        if ir_length < 100:
            ir_length = 100

        # Create exponentially decaying noise as IR approximation
        t = torch.arange(ir_length, device=x.device, dtype=x.dtype)

        # Decay envelope
        decay_samples = decay_time.view(bs, 1) * self.sample_rate
        envelope = torch.exp(-3.0 * t.view(1, -1) / (decay_samples + 1e-8))

        # Damping: low-pass effect (reduce high frequencies over time)
        damp = damping.view(bs, 1)
        damping_envelope = torch.exp(-damp * t.view(1, -1) / ir_length)
        envelope = envelope * damping_envelope

        # Random noise for diffusion
        noise = torch.randn(bs, ir_length, device=x.device, dtype=x.dtype)
        ir = noise * envelope

        # Normalize IR by RMS to preserve energy (not sum, which kills the signal)
        ir_rms = (ir ** 2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
        ir = ir / ir_rms * 0.1  # Scale to reasonable level

        # Pre-delay: shift IR start
        pre_delay_samples = (pre_delay_ms * self.sample_rate / 1000).long()

        # Convolve (using FFT for efficiency)
        n_fft = 2 ** int(math.ceil(math.log2(seq_len + ir_length - 1)))

        X = torch.fft.rfft(x.squeeze(1), n=n_fft)
        IR = torch.fft.rfft(ir, n=n_fft)

        wet = torch.fft.irfft(X * IR, n=n_fft)[..., :seq_len]
        wet = wet.unsqueeze(1)

        # Normalize wet to match dry level (reverb shouldn't change overall volume)
        wet_rms = (wet ** 2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
        dry_rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
        wet = wet * (dry_rms / wet_rms)

        # Mix dry and wet
        wet_mix_expanded = wet_mix.view(bs, 1, 1)
        output = (1 - wet_mix_expanded) * x + wet_mix_expanded * wet

        return output


class DifferentiableDistortion(nn.Module):
    """
    Differentiable distortion/saturation effect.
    Uses soft clipping with controllable drive.
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        min_drive: float = 0.0,
        max_drive: float = 1.0,
        min_tone: float = 0.0,
        max_tone: float = 1.0,
        min_mix: float = 0.0,
        max_mix: float = 1.0,
        min_output_gain_db: float = -12.0,
        max_output_gain_db: float = 12.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.param_ranges = {
            'drive': (min_drive, max_drive),
            'tone': (min_tone, max_tone),
            'mix': (min_mix, max_mix),
            'output_gain_db': (min_output_gain_db, max_output_gain_db),
        }
        self.num_control_params = 4

    def denormalize(self, norm_val: torch.Tensor, param_name: str) -> torch.Tensor:
        min_val, max_val = self.param_ranges[param_name]
        return norm_val * (max_val - min_val) + min_val

    def forward(
        self,
        x: torch.Tensor,
        control_params: torch.Tensor,
        train: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, seq_len = x.size()

        drive = self.denormalize(control_params[:, 0, 0], 'drive')
        tone = self.denormalize(control_params[:, 1, 0], 'tone')
        mix = self.denormalize(control_params[:, 2, 0], 'mix')
        output_gain_db = self.denormalize(control_params[:, 3, 0], 'output_gain_db')

        param_dict = {
            'drive': drive,
            'tone': tone,
            'mix': mix,
            'output_gain_db': output_gain_db,
        }

        output = self.process(x, **param_dict)
        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        drive: torch.Tensor,
        tone: torch.Tensor,
        mix: torch.Tensor,
        output_gain_db: torch.Tensor,
    ) -> torch.Tensor:
        bs, chs, seq_len = x.size()

        # Apply drive (pre-gain)
        drive_gain = 1.0 + drive.view(bs, 1, 1) * 20  # 1x to 21x gain
        x_driven = x * drive_gain

        # Soft clipping (tanh-based)
        x_clipped = torch.tanh(x_driven)

        # Tone control (simple high-shelf-like effect)
        # Higher tone = more highs preserved
        tone_expanded = tone.view(bs, 1, 1)

        # Simple differentiation for high freq emphasis
        x_diff = x_clipped[..., 1:] - x_clipped[..., :-1]
        x_diff = torch.nn.functional.pad(x_diff, (1, 0), mode='replicate')

        x_toned = x_clipped * (1 - tone_expanded * 0.5) + x_diff * tone_expanded * 0.5

        # Mix dry and wet
        mix_expanded = mix.view(bs, 1, 1)
        x_mixed = (1 - mix_expanded) * x + mix_expanded * x_toned

        # Output gain
        output_gain = 10 ** (output_gain_db.view(bs, 1, 1) / 20)

        return x_mixed * output_gain


class DifferentiableChorus(nn.Module):
    """
    Differentiable chorus effect using modulated delay lines.
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        min_rate: float = 0.1,
        max_rate: float = 10.0,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
        min_mix: float = 0.0,
        max_mix: float = 1.0,
        min_feedback: float = 0.0,
        max_feedback: float = 0.9,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.param_ranges = {
            'rate': (min_rate, max_rate),
            'depth': (min_depth, max_depth),
            'mix': (min_mix, max_mix),
            'feedback': (min_feedback, max_feedback),
        }
        self.num_control_params = 4
        self.base_delay_ms = 7.0  # Base delay in ms
        self.max_mod_ms = 3.0  # Max modulation depth in ms

    def denormalize(self, norm_val: torch.Tensor, param_name: str) -> torch.Tensor:
        min_val, max_val = self.param_ranges[param_name]
        return norm_val * (max_val - min_val) + min_val

    def forward(
        self,
        x: torch.Tensor,
        control_params: torch.Tensor,
        train: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, seq_len = x.size()

        rate = self.denormalize(control_params[:, 0, 0], 'rate')
        depth = self.denormalize(control_params[:, 1, 0], 'depth')
        mix = self.denormalize(control_params[:, 2, 0], 'mix')
        feedback = self.denormalize(control_params[:, 3, 0], 'feedback')

        param_dict = {
            'rate': rate,
            'depth': depth,
            'mix': mix,
            'feedback': feedback,
        }

        output = self.process(x, **param_dict)
        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        rate: torch.Tensor,
        depth: torch.Tensor,
        mix: torch.Tensor,
        feedback: torch.Tensor,
    ) -> torch.Tensor:
        bs, chs, seq_len = x.size()

        # Generate LFO
        t = torch.arange(seq_len, device=x.device, dtype=x.dtype) / self.sample_rate
        t = t.view(1, 1, -1).expand(bs, 1, -1)

        rate_expanded = rate.view(bs, 1, 1)
        lfo = torch.sin(2 * math.pi * rate_expanded * t)

        # Calculate delay in samples
        depth_expanded = depth.view(bs, 1, 1)
        base_delay_samples = self.base_delay_ms * self.sample_rate / 1000
        mod_samples = depth_expanded * self.max_mod_ms * self.sample_rate / 1000
        delay_samples = base_delay_samples + lfo * mod_samples

        # Create delayed signal using linear interpolation
        indices = torch.arange(seq_len, device=x.device, dtype=x.dtype).view(1, 1, -1)
        indices = indices.expand(bs, 1, -1)

        read_indices = indices - delay_samples
        read_indices = torch.clamp(read_indices, 0, seq_len - 1)

        # Linear interpolation
        read_floor = read_indices.floor().long()
        read_ceil = (read_floor + 1).clamp(max=seq_len - 1)
        frac = read_indices - read_floor.float()

        # Gather values
        x_floor = torch.gather(x, 2, read_floor)
        x_ceil = torch.gather(x, 2, read_ceil)

        delayed = x_floor * (1 - frac) + x_ceil * frac

        # Mix
        mix_expanded = mix.view(bs, 1, 1)
        output = (1 - mix_expanded) * x + mix_expanded * delayed

        return output


class DifferentiableDelay(nn.Module):
    """
    Differentiable delay effect.
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        min_delay_ms: float = 1.0,
        max_delay_ms: float = 2000.0,
        min_feedback: float = 0.0,
        max_feedback: float = 0.95,
        min_mix: float = 0.0,
        max_mix: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.param_ranges = {
            'delay_ms': (min_delay_ms, max_delay_ms),
            'feedback': (min_feedback, max_feedback),
            'mix': (min_mix, max_mix),
        }
        self.num_control_params = 3

    def denormalize(self, norm_val: torch.Tensor, param_name: str) -> torch.Tensor:
        min_val, max_val = self.param_ranges[param_name]
        return norm_val * (max_val - min_val) + min_val

    def forward(
        self,
        x: torch.Tensor,
        control_params: torch.Tensor,
        train: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bs, chs, seq_len = x.size()

        delay_ms = self.denormalize(control_params[:, 0, 0], 'delay_ms')
        feedback = self.denormalize(control_params[:, 1, 0], 'feedback')
        mix = self.denormalize(control_params[:, 2, 0], 'mix')

        param_dict = {
            'delay_ms': delay_ms,
            'feedback': feedback,
            'mix': mix,
        }

        output = self.process(x, **param_dict)
        return output, param_dict

    def process(
        self,
        x: torch.Tensor,
        delay_ms: torch.Tensor,
        feedback: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        bs, chs, seq_len = x.size()

        # Calculate delay in samples
        delay_samples = (delay_ms * self.sample_rate / 1000).long()
        max_delay = delay_samples.max().item()

        # Create output buffer
        output = torch.zeros_like(x)
        wet = torch.zeros_like(x)

        # Apply delay with feedback (simplified for differentiability)
        feedback_expanded = feedback.view(bs, 1, 1)

        # Create delayed versions with exponentially decaying feedback
        for tap in range(5):  # 5 feedback taps
            tap_delay = delay_samples * (tap + 1)
            tap_gain = feedback_expanded ** tap

            for b in range(bs):
                d = tap_delay[b].item()
                if d < seq_len:
                    wet[b, :, d:] += x[b, :, :seq_len-d] * tap_gain[b]

        # Mix
        mix_expanded = mix.view(bs, 1, 1)
        output = (1 - mix_expanded) * x + mix_expanded * wet

        return output


class EffectChainGenerator:
    """
    Generate training data: dry audio -> random effect chain -> wet audio.
    """

    EFFECT_TYPES = ['eq', 'compressor', 'distortion', 'reverb', 'chorus', 'delay']

    def __init__(
        self,
        sample_rate: float = 48000,
        max_chain_length: int = 4,
        effect_types: Optional[List[str]] = None,
        device: str = 'cpu',
    ):
        self.sample_rate = sample_rate
        self.max_chain_length = max_chain_length
        self.effect_types = effect_types or self.EFFECT_TYPES
        self.device = device

        # Initialize effect processors
        self.effects = self._init_effects()

    def _init_effects(self) -> Dict[str, nn.Module]:
        """Initialize all effect processors."""
        # Import NablAFx processors lazily and directly from ddsp module
        # to avoid CLAP/FAD argparse conflicts (nablafx/__init__.py imports core
        # which triggers FAD imports with argparse side effects)
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from nablafx.processors.ddsp import ParametricEQ

        effects = {}

        if 'eq' in self.effect_types:
            effects['eq'] = ParametricEQ(
                sample_rate=self.sample_rate,
                control_type='static',
            ).to(self.device)

        if 'compressor' in self.effect_types:
            effects['compressor'] = DifferentiableCompressor(
                sample_rate=self.sample_rate,
            ).to(self.device)

        if 'distortion' in self.effect_types:
            effects['distortion'] = DifferentiableDistortion(
                sample_rate=self.sample_rate,
            ).to(self.device)

        if 'reverb' in self.effect_types:
            effects['reverb'] = DifferentiableReverb(
                sample_rate=self.sample_rate,
            ).to(self.device)

        if 'chorus' in self.effect_types:
            effects['chorus'] = DifferentiableChorus(
                sample_rate=self.sample_rate,
            ).to(self.device)

        if 'delay' in self.effect_types:
            effects['delay'] = DifferentiableDelay(
                sample_rate=self.sample_rate,
            ).to(self.device)

        return effects

    # Magnitude tier ranges for balanced data generation
    MAGNITUDE_TIERS = {
        'subtle':   (0.02, 0.15),  # Very light effect - barely perceptible
        'mild':     (0.12, 0.35),  # Noticeable but gentle
        'moderate': (0.30, 0.60),  # Clear effect
        'strong':   (0.50, 0.95),  # Heavy effect
    }

    # Effect-specific parameter indices that control intensity
    # These are the "intensity" params that should be tier-constrained
    INTENSITY_PARAMS = {
        'eq': [0, 3, 6, 9, 12],  # gain params (low_shelf, band0-2, high_shelf)
        'compressor': [1, 5],    # ratio, makeup_db
        'reverb': [0, 2],        # decay_time, wet_mix
        'distortion': [0, 2],    # drive, mix
        'chorus': [1, 2],        # depth, mix
        'delay': [1, 2],         # feedback, mix
    }

    def sample_params(
        self,
        effect_type: str,
        magnitude_tier: str = 'random',
    ) -> torch.Tensor:
        """
        Sample random normalized parameters for an effect.

        Args:
            effect_type: Type of effect ('eq', 'reverb', etc.)
            magnitude_tier: 'subtle', 'mild', 'moderate', 'strong', or 'random'

        Returns:
            Tensor of shape [1, num_params, 1] with normalized params in [0, 1]
        """
        effect = self.effects[effect_type]
        num_params = effect.num_control_params

        # Select tier
        if magnitude_tier == 'random':
            magnitude_tier = random.choice(list(self.MAGNITUDE_TIERS.keys()))

        tier_low, tier_high = self.MAGNITUDE_TIERS[magnitude_tier]
        intensity_indices = self.INTENSITY_PARAMS.get(effect_type, [])

        params = torch.zeros(1, num_params, 1, device=self.device)

        for i in range(num_params):
            if i in intensity_indices:
                # Intensity params: constrain to tier range
                params[0, i, 0] = random.uniform(tier_low, tier_high)
            else:
                # Non-intensity params: sample normally (Beta or uniform)
                if random.random() < 0.3:
                    params[0, i, 0] = random.random()
                else:
                    import numpy as np
                    params[0, i, 0] = np.random.beta(2.0, 2.0)

        return params

    def generate_sample(
        self,
        dry_audio: torch.Tensor,
        chain_length: Optional[int] = None,
        specific_effects: Optional[List[str]] = None,
        magnitude_tier: str = 'random',
    ) -> Tuple[torch.Tensor, ChainSpec, List[torch.Tensor]]:
        """
        Generate a wet audio sample from dry audio.

        Args:
            dry_audio: Input dry audio [1, T] or [1, 1, T]
            chain_length: Optional specific chain length
            specific_effects: Optional specific effect order
            magnitude_tier: 'subtle', 'mild', 'moderate', 'strong', or 'random'

        Returns:
            wet_audio: Processed signal [1, 1, T]
            chain_spec: Chain specification with all parameters
            intermediate_signals: Signal after each effect
        """
        # Ensure correct shape
        if dry_audio.dim() == 2:
            dry_audio = dry_audio.unsqueeze(0)  # [1, 1, T]
        if dry_audio.dim() == 3 and dry_audio.size(0) > 1:
            dry_audio = dry_audio[0:1]  # Take first batch only

        dry_audio = dry_audio.to(self.device)

        # Determine chain
        if specific_effects is not None:
            chain_effects = specific_effects
        else:
            if chain_length is None:
                chain_length = random.randint(1, self.max_chain_length)
            chain_effects = random.choices(self.effect_types, k=chain_length)

        # Process through chain
        signal = dry_audio
        chain_spec_effects = []
        intermediates = []

        for fx_type in chain_effects:
            effect = self.effects[fx_type]

            # Sample parameters with magnitude tier
            norm_params = self.sample_params(fx_type, magnitude_tier=magnitude_tier)

            # Apply effect
            signal, param_dict = effect(signal, norm_params, train=True)

            # Store chain info
            chain_spec_effects.append(EffectParams(
                effect_type=fx_type,
                params=param_dict,
                normalized_params={f'param_{i}': norm_params[0, i, 0]
                                   for i in range(norm_params.size(1))},
            ))

            intermediates.append(signal.clone())

        chain_spec = ChainSpec(effects=chain_spec_effects)

        return signal, chain_spec, intermediates

    def generate_batch(
        self,
        dry_audio_batch: torch.Tensor,
        chain_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[ChainSpec], List[List[torch.Tensor]]]:
        """
        Generate a batch of wet audio samples.

        Args:
            dry_audio_batch: Batch of dry audio [B, 1, T]
            chain_length: Optional fixed chain length for all samples

        Returns:
            wet_audio_batch: Processed signals [B, 1, T]
            chain_specs: List of chain specifications
            intermediates_batch: List of intermediate signals per sample
        """
        batch_size = dry_audio_batch.size(0)
        wet_batch = []
        chain_specs = []
        intermediates_batch = []

        for i in range(batch_size):
            wet, chain_spec, intermediates = self.generate_sample(
                dry_audio_batch[i:i+1],
                chain_length=chain_length,
            )
            wet_batch.append(wet)
            chain_specs.append(chain_spec)
            intermediates_batch.append(intermediates)

        wet_audio_batch = torch.cat(wet_batch, dim=0)

        return wet_audio_batch, chain_specs, intermediates_batch
