#!/usr/bin/env python3
"""
Full White-Box Audio Chain

z → mel_mapper → mel → sms_mapper → sines → program (MDL) → sines → audio

Neural parts: mel_mapper, sms_mapper (trained)
Explicit parts: MDL compression, program synthesis, additive synth (white-box)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import sys
import os
import json

sys.path.insert(0, '/home/arlo/Data/ACE-Step')


# ============================================================================
# NEURAL COMPONENTS (trained - matching checkpoint architectures)
# ============================================================================

# Mel bin frequencies
def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def get_mel_frequencies(n_mels=128, f_min=40, f_max=16000):
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mels = np.linspace(mel_min, mel_max, n_mels)
    return mel_to_hz(mels)

MEL_FREQS = torch.tensor(get_mel_frequencies(128, 40, 16000), dtype=torch.float32)


class MelMapperV2(nn.Module):
    """z → mel (trained, matching checkpoint)."""

    def __init__(self, hidden_dim=256):
        super().__init__()

        # Quadratic encoder for frequency dims
        self.freq_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim),  # dims 48-63
            nn.GELU(),
        )
        self.freq_quad = nn.Linear(16, hidden_dim, bias=False)  # Quadratic term

        # Linear encoder for other dims
        self.other_encoder = nn.Sequential(
            nn.Linear(112, hidden_dim),  # dims 0-47 + 64-127
            nn.GELU(),
        )

        # Combine
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Output
        self.mel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
        )

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        # Split by discovered roles
        z_freq = z_flat[..., 48:64]  # [B, T, 16]
        z_other = torch.cat([z_flat[..., :48], z_flat[..., 64:]], dim=-1)  # [B, T, 112]

        # Encode with quadratic for freq dims
        h_freq = self.freq_encoder(z_freq) + self.freq_quad(z_freq ** 2)
        h_other = self.other_encoder(z_other)

        # Combine
        h = self.combine(torch.cat([h_freq, h_other], dim=-1))

        # Temporal
        h_attn, _ = self.temporal_attn(h, h, h)
        h = self.temporal_norm(h + h_attn)

        # Predict mel
        mel = self.mel_head(h)

        # Upsample
        mel = mel.permute(0, 2, 1)
        mel = F.interpolate(mel, scale_factor=8, mode='linear', align_corners=False)
        mel = mel.permute(0, 2, 1)

        return mel


class SMSMapperV2(nn.Module):
    """mel → (freqs, amps) (trained, matching checkpoint)."""

    def __init__(self, n_sines=64, hidden_dim=256):
        super().__init__()
        self.n_sines = n_sines

        self.register_buffer('mel_freqs', MEL_FREQS.clone())

        # Predict amplitude for each mel bin
        self.amp_refiner = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
        )

        # Predict frequency offset within each bin
        self.freq_offset = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
        )

    def forward(self, mel):
        """
        mel: [B, T, 128]
        Returns: freqs [B, T, n_sines], amps [B, T, n_sines]
        """
        B, T, n_mels = mel.shape
        device = mel.device

        # Refine amplitudes
        amp_refined = torch.sigmoid(self.amp_refiner(mel))  # [B, T, 128]

        # Get frequency offsets
        freq_offset = 0.5 * torch.tanh(self.freq_offset(mel))  # [B, T, 128]

        # Select top-k bins by amplitude
        topk_amps, topk_idx = amp_refined.topk(self.n_sines, dim=-1)  # [B, T, n_sines]

        # Get frequencies: bin center + offset
        mel_freqs = self.mel_freqs.to(device)

        # Compute bin widths
        bin_widths = torch.zeros(128, device=device)
        bin_widths[:-1] = mel_freqs[1:] - mel_freqs[:-1]
        bin_widths[-1] = bin_widths[-2]

        # Get base frequencies for selected bins
        base_freqs = mel_freqs[topk_idx]

        # Get offsets for selected bins
        selected_offsets = freq_offset.gather(-1, topk_idx)
        selected_widths = bin_widths[topk_idx]

        # Final frequencies
        freqs = base_freqs + selected_offsets * selected_widths

        return freqs, topk_amps


# ============================================================================
# EXPLICIT WHITE-BOX COMPONENTS
# ============================================================================

class MDLProgramExtractor:
    """
    Extract MDL program from sines (fully explicit).

    Sines → HarmonicGroups + DampedResonators + IndependentSines
    """

    def __init__(self, f0_threshold=20, ratio_tolerance=0.05):
        self.f0_threshold = f0_threshold
        self.ratio_tolerance = ratio_tolerance

    def extract(self, freqs, amps):
        """
        Extract program from (freqs, amps).

        Args:
            freqs: [T, n_sines] Hz
            amps: [T, n_sines]

        Returns:
            program dict
        """
        T, n_sines = freqs.shape

        # Find harmonic groups
        harmonic_groups = self._find_harmonic_groups(freqs, amps)

        # Find damped sines
        damped_sines = self._find_damped_sines(amps)

        # Remaining sines (not in harmonic groups)
        used_sines = set()
        for g in harmonic_groups:
            used_sines.update(g['sine_indices'])

        independent_sines = []
        for i in range(n_sines):
            if i not in used_sines and amps[:, i].mean() > 0.001:
                independent_sines.append({
                    'sine_idx': i,
                    'avg_freq': freqs[:, i].mean().item(),
                    'avg_amp': amps[:, i].mean().item()
                })

        return {
            'harmonic_groups': harmonic_groups,
            'damped_sines': damped_sines,
            'independent_sines': independent_sines,
            'n_frames': T,
            'n_sines': n_sines
        }

    def _find_harmonic_groups(self, freqs, amps):
        """Find groups of sines forming harmonic series."""
        avg_freqs = freqs.mean(dim=0).numpy()
        avg_amps = amps.mean(dim=0).numpy()
        amp_order = np.argsort(avg_amps)[::-1]

        groups = []
        used = set()

        for i in amp_order:
            if i in used or avg_amps[i] < 0.001:
                continue

            f0 = avg_freqs[i]
            if f0 < self.f0_threshold:
                continue

            group = {
                'f0': float(f0),
                'partials': [(1, int(i), float(avg_amps[i]))],
                'sine_indices': [int(i)]
            }
            used.add(i)

            for j in range(len(avg_freqs)):
                if j in used or avg_amps[j] < 0.0001:
                    continue

                fj = avg_freqs[j]
                ratio = fj / f0

                nearest_int = round(ratio)
                if 2 <= nearest_int <= 16:
                    if abs(ratio - nearest_int) / nearest_int < self.ratio_tolerance:
                        group['partials'].append((int(nearest_int), int(j), float(avg_amps[j])))
                        group['sine_indices'].append(int(j))
                        used.add(j)

            if len(group['partials']) >= 2:
                groups.append(group)

        return groups

    def _find_damped_sines(self, amps, min_decay=0.01):
        """Find sines with exponential decay."""
        T, n_sines = amps.shape
        amps_np = amps.numpy()

        damped = []
        for i in range(n_sines):
            amp_i = amps_np[:, i]
            if amp_i.max() < 0.01:
                continue

            first_half = amp_i[:T//2].mean()
            second_half = amp_i[T//2:].mean()

            if first_half > second_half * 1.5 and first_half > 0.01:
                ratio = second_half / (first_half + 1e-8)
                decay_rate = -np.log(ratio + 1e-8) / (T // 2)

                if decay_rate > min_decay:
                    damped.append({
                        'sine_idx': int(i),
                        'initial_amp': float(first_half),
                        'decay_rate': float(decay_rate)
                    })

        return damped


class ProgramSynthesizer:
    """
    Synthesize sines from program (fully explicit).

    Program → (freqs, amps)
    """

    def __init__(self, n_sines=64, sr=44100):
        self.n_sines = n_sines
        self.sr = sr

    def synthesize(self, program):
        """
        Generate (freqs, amps) from program.

        Args:
            program: dict from MDLProgramExtractor

        Returns:
            freqs: [T, n_sines] Hz
            amps: [T, n_sines]
        """
        T = program['n_frames']
        device = 'cpu'

        freqs = torch.zeros(T, self.n_sines)
        amps = torch.zeros(T, self.n_sines)

        sine_idx = 0

        # Synthesize harmonic groups
        for group in program['harmonic_groups']:
            f0 = group['f0']
            for partial_num, _, partial_amp in group['partials']:
                if sine_idx >= self.n_sines:
                    break
                freqs[:, sine_idx] = f0 * partial_num
                amps[:, sine_idx] = partial_amp
                sine_idx += 1

        # Synthesize damped sines (apply decay envelope)
        t = torch.linspace(0, 1, T)
        for damped in program['damped_sines']:
            idx = damped['sine_idx']
            if idx < self.n_sines:
                decay = damped['decay_rate']
                initial = damped['initial_amp']
                amps[:, idx] = initial * torch.exp(-decay * t * T)

        # Independent sines
        for ind in program['independent_sines']:
            idx = ind['sine_idx']
            if idx < self.n_sines:
                freqs[:, idx] = ind['avg_freq']
                amps[:, idx] = ind['avg_amp']

        return freqs, amps


class AdditiveSynth:
    """
    Additive synthesis (fully explicit).

    (freqs, amps) → audio
    """

    def __init__(self, sr=44100, hop_length=512):
        self.sr = sr
        self.hop_length = hop_length

    def synthesize(self, freqs, amps):
        """
        Args:
            freqs: [T, n_sines] Hz
            amps: [T, n_sines]

        Returns:
            audio: [samples]
        """
        T, n_sines = freqs.shape
        n_samples = T * self.hop_length

        # Upsample
        freqs_t = freqs.T.unsqueeze(0)  # [1, n_sines, T]
        amps_t = amps.T.unsqueeze(0)

        freqs_up = F.interpolate(freqs_t, size=n_samples, mode='linear', align_corners=True)
        amps_up = F.interpolate(amps_t, size=n_samples, mode='linear', align_corners=True)

        freqs_up = freqs_up.squeeze(0).T  # [n_samples, n_sines]
        amps_up = amps_up.squeeze(0).T

        # Phase accumulation
        dt = 1.0 / self.sr
        phase_inc = 2 * np.pi * freqs_up * dt
        phase = torch.cumsum(phase_inc, dim=0)

        # Synthesize
        sines = amps_up * torch.sin(phase)
        audio = sines.sum(dim=1)

        # Normalize
        audio = audio / (audio.abs().max() + 1e-8) * 0.9

        return audio


# ============================================================================
# FULL CHAIN
# ============================================================================

class WhiteBoxChain:
    """
    Full white-box chain.

    z → mel_mapper → mel → sms_mapper → sines → program → sines → audio
    """

    def __init__(self, device='cuda'):
        self.device = device

        # Neural components
        self.mel_mapper = MelMapperV2().to(device)
        self.sms_mapper = SMSMapperV2().to(device)

        # Explicit components
        self.mdl = MDLProgramExtractor()
        self.program_synth = ProgramSynthesizer()
        self.additive = AdditiveSynth()

    def load_checkpoints(self, mel_path, sms_path):
        """Load trained mappers."""
        mel_ckpt = torch.load(mel_path, weights_only=True)
        key = 'model_state_dict' if 'model_state_dict' in mel_ckpt else 'model'
        self.mel_mapper.load_state_dict(mel_ckpt[key])

        sms_ckpt = torch.load(sms_path, weights_only=True)
        key = 'model_state_dict' if 'model_state_dict' in sms_ckpt else 'model'
        self.sms_mapper.load_state_dict(sms_ckpt[key])

        self.mel_mapper.eval()
        self.sms_mapper.eval()

    def forward(self, z, return_intermediates=False):
        """
        Full chain: z → audio

        Returns:
            audio: numpy array
            program: dict (if return_intermediates)
            intermediates: dict (if return_intermediates)
        """
        with torch.no_grad():
            # Step 1: z → mel (neural)
            mel = self.mel_mapper(z)  # [B, T, 128]

            # Step 2: mel → sines (neural)
            freqs, amps = self.sms_mapper(mel)  # [B, T, 64]

        # Take first sample
        freqs = freqs[0].cpu()  # [T, 64]
        amps = amps[0].cpu()

        # Step 3: sines → program (explicit MDL)
        program = self.mdl.extract(freqs, amps)

        # Step 4: program → sines (explicit synthesis)
        synth_freqs, synth_amps = self.program_synth.synthesize(program)

        # Step 5: sines → audio (explicit additive)
        audio = self.additive.synthesize(synth_freqs, synth_amps)

        if return_intermediates:
            return audio.numpy(), program, {
                'mel': mel[0].cpu().numpy(),
                'freqs': freqs.numpy(),
                'amps': amps.numpy(),
                'synth_freqs': synth_freqs.numpy(),
                'synth_amps': synth_amps.numpy()
            }

        return audio.numpy(), program


# ============================================================================
# TEST
# ============================================================================

def test():
    import orjson
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load DCAE for comparison
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
        vocoder_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    )
    dcae.dcae.to(device).eval()
    dcae.vocoder.to(device).eval()

    # Create chain
    chain = WhiteBoxChain(device=device)

    # Check for checkpoints
    mel_ckpt = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_mapper/best_model.pt'
    sms_ckpt = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/mel_to_sms/best_model_v2.pt'

    if os.path.exists(mel_ckpt) and os.path.exists(sms_ckpt):
        print("Loading trained mappers...")
        try:
            chain.load_checkpoints(mel_ckpt, sms_ckpt)
        except Exception as e:
            print(f"  Failed to load checkpoints: {e}")
            print("  Using random weights")
    else:
        print("WARNING: No checkpoints found, using random weights!")
        print(f"  Expected: {mel_ckpt}")
        print(f"  Expected: {sms_ckpt}")

    # Load a test latent
    manifest_path = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json'
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    z = None
    for entry in manifest['entries'][:50]:
        path = entry['path']
        if 'drum' in path.lower():
            continue
        try:
            data = torch.load(path, weights_only=True, map_location='cpu')
            lat_path = data.get('latent_path')
            if lat_path and os.path.exists(lat_path):
                lat_data = torch.load(lat_path, weights_only=True, map_location='cpu')
                z = lat_data.get('latents', lat_data)
                if z.dim() == 3:
                    z = z.unsqueeze(0)
                z = z[:, :, :, :32].to(device)  # Limit length
                print(f"Loaded: {path}")
                print(f"  z shape: {z.shape}")
                break
        except:
            continue

    if z is None:
        print("No latent found!")
        return

    # Run chain
    print("\nRunning white-box chain...")
    audio, program, intermediates = chain.forward(z, return_intermediates=True)

    print(f"\nExtracted program:")
    print(f"  Harmonic groups: {len(program['harmonic_groups'])}")
    for i, g in enumerate(program['harmonic_groups'][:3]):
        partials = [(p[0], f"{p[2]:.3f}") for p in g['partials'][:5]]
        print(f"    Group {i}: f0={g['f0']:.1f}Hz, partials={partials}")
    print(f"  Damped sines: {len(program['damped_sines'])}")
    print(f"  Independent sines: {len(program['independent_sines'])}")

    # Compression ratio
    raw_params = program['n_frames'] * program['n_sines'] * 2
    program_params = (len(program['harmonic_groups']) * 10 +
                      len(program['damped_sines']) * 2 +
                      len(program['independent_sines']) * 2)
    print(f"\n  Raw params: {raw_params}")
    print(f"  Program params: {program_params}")
    print(f"  Compression: {raw_params / (program_params + 1):.1f}x")

    # Save audio
    out_dir = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/outputs'
    os.makedirs(out_dir, exist_ok=True)

    sf.write(f'{out_dir}/whitebox_chain.wav', audio, 44100)
    print(f"\nSaved: {out_dir}/whitebox_chain.wav")

    # DCAE comparison
    with torch.no_grad():
        z_denorm = z / dcae.scale_factor + dcae.shift_factor
        mel = dcae.dcae.decoder(z_denorm).mean(dim=1)
        mel_scaled = mel * 0.5 + 0.5
        mel_scaled = mel_scaled * (dcae.max_mel_value - dcae.min_mel_value) + dcae.min_mel_value
        audio_dcae = dcae.vocoder.decode(mel_scaled).squeeze()
        audio_dcae = audio_dcae / (audio_dcae.abs().max() + 1e-8) * 0.9

    sf.write(f'{out_dir}/whitebox_chain_dcae.wav', audio_dcae.cpu().numpy(), 44100)
    print(f"DCAE reference: {out_dir}/whitebox_chain_dcae.wav")

    # Save program
    with open(f'{out_dir}/whitebox_program.json', 'w') as f:
        json.dump(program, f, indent=2)
    print(f"Program: {out_dir}/whitebox_program.json")


if __name__ == "__main__":
    test()
