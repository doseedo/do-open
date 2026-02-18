#!/usr/bin/env python3
"""
Train Universal Chain Corrector — Fix accumulated drift in any module chain

The stress test showed:
  - Individual modules: cos 0.997+
  - 3-4 module chains: cos 0.95+
  - 5-6 module chains: cos 0.89-0.96 (problematic)
  - Worst offenders: mixer/ringmod sources + stacked time effects

This script trains a single universal corrector that:
  1. Takes ANY chain's z output + chain topology description
  2. Learns to correct the accumulated drift
  3. Generalizes to unseen chain configurations

Training data: 14 chain templates × ~40 parameter variations each = ~560 chains.
Each chain: run DSP ground truth, encode, run latent chain, collect pairs.

Architecture: small transformer encodes chain topology → FiLM conditioning
on dilated correction blocks. Zero-init residual output.
"""

import sys
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

sys.path.insert(0, "/home/arlo/Data/ACE-Step")
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

SCRIPTS_DIR = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts")
sys.path.insert(0, str(SCRIPTS_DIR))

# Import ALL module classes
from test_latent_vcf import LatentVCF, FiLMBlock
from test_latent_vca import LatentVCA, PerFrameFiLMBlock as VCA_PF
from test_latent_vcf_temporal import (
    LatentVCFTemporal, PerFrameFiLMBlock as VCFT_PF,
    apply_tv_filter, normalize_cutoff as tv_normalize_cutoff,
    make_filter_envelope, make_filter_lfo, resample_cutoff_to_z,
)
from test_latent_mixer import LatentMixer, MixerFiLMBlock
from latent_modules import (
    LatentWavefolder, LatentDistortion, LatentRingMod,
    LatentDelay, LatentReverb, LatentChorus, LatentFM,
    UniversalChainCorrector, MODULE_IDS, PAD_ID, MAX_CHAIN_LEN,
)
from test_latent_vca import make_adsr_envelope, apply_envelope, resample_envelope_to_z
from train_all_modules import (
    dsp_wavefold, dsp_distortion, dsp_ringmod,
    dsp_delay, dsp_reverb, dsp_chorus,
)
from train_fm import dsp_fm_static, dsp_fm_temporal, make_fm_envelope, MAX_FM_INDEX

DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100
DURATION = 2.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

BASE_DIR = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch")
OUTPUT_DIR = BASE_DIR / "test_outputs" / "chain_corrector"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Source Generation
# ============================================================

def _normalize(audio):
    peak = np.abs(audio).max()
    if peak > 1e-6:
        return (audio / peak * 0.8).astype(np.float32)
    return audio.astype(np.float32)


def _apply_env(audio):
    attack = int(0.01 * SAMPLE_RATE)
    release = int(0.05 * SAMPLE_RATE)
    env = np.ones_like(audio)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return _normalize(audio * env)


def gen_saw(pitch):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    audio = np.zeros_like(t)
    for h in range(1, 40):
        freq = pitch * h
        if freq > SAMPLE_RATE / 2: break
        audio += ((-1) ** (h + 1)) * np.sin(2 * np.pi * freq * t) / h
    return _apply_env(audio)


def gen_square(pitch):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    audio = np.zeros_like(t)
    for h in range(1, 40, 2):
        freq = pitch * h
        if freq > SAMPLE_RATE / 2: break
        audio += np.sin(2 * np.pi * freq * t) / h
    return _apply_env(audio)


def gen_triangle(pitch):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    audio = np.zeros_like(t)
    for i, h in enumerate(range(1, 40, 2)):
        freq = pitch * h
        if freq > SAMPLE_RATE / 2: break
        audio += ((-1) ** i) * np.sin(2 * np.pi * freq * t) / (h * h)
    return _apply_env(audio)


def gen_sine(pitch):
    t = np.linspace(0, DURATION, N_SAMPLES, dtype=np.float32)
    return _apply_env(np.sin(2 * np.pi * pitch * t))


WAVEFORM_FNS = {'saw': gen_saw, 'square': gen_square, 'triangle': gen_triangle, 'sine': gen_sine}


# ============================================================
# DCAE Helpers
# ============================================================

def encode_audio(dcae, audio, device='cuda'):
    audio_tensor = torch.from_numpy(audio).float().to(device)
    audio_stereo = audio_tensor.unsqueeze(0).unsqueeze(0).expand(-1, 2, -1)
    audio_lengths = torch.tensor([audio_stereo.shape[-1]], device=device)
    with torch.no_grad():
        z, _ = dcae.encode(audio_stereo, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return z.squeeze(0)


def decode_latent(dcae, z, device='cuda'):
    z_4d = z.unsqueeze(0) if z.dim() == 3 else z
    audio_lengths = torch.tensor([N_SAMPLES], device=device)
    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return wavs[0].mean(dim=0).cpu().numpy()


def _save_wav(path, audio):
    torchaudio.save(str(path), torch.from_numpy(audio).unsqueeze(0).float(), SAMPLE_RATE)


def resample_to_z(curve, T):
    frame_size = len(curve) // T
    out = np.zeros(T, dtype=np.float32)
    for i in range(T):
        s = i * frame_size
        e = min(s + frame_size, len(curve))
        out[i] = curve[s:e].mean()
    return out


# ============================================================
# Load All Modules (frozen — we only train the corrector)
# ============================================================

def load_all_modules(device='cuda'):
    modules = {}
    ckpt_base = BASE_DIR / "test_outputs"

    loaders = {
        'vcf': ('latent_vcf/latent_vcf.pt', LatentVCF),
        'vca': ('latent_vca/latent_vca.pt', LatentVCA),
        'vcf_t': ('latent_vcf_temporal/latent_vcf_temporal.pt', LatentVCFTemporal),
        'mixer': ('latent_mixer/latent_mixer.pt', LatentMixer),
        'wavefolder': ('latent_wavefolder/latent_wavefolder.pt', LatentWavefolder),
        'distortion': ('latent_distortion/latent_distortion.pt', LatentDistortion),
        'ringmod': ('latent_ringmod/latent_ringmod.pt', LatentRingMod),
        'delay': ('latent_delay/latent_delay.pt', LatentDelay),
        'reverb': ('latent_reverb/latent_reverb.pt', LatentReverb),
        'chorus': ('latent_chorus/latent_chorus.pt', LatentChorus),
        'fm': ('latent_fm/latent_fm.pt', LatentFM),
    }

    for name, (path, cls) in loaders.items():
        full_path = ckpt_base / path
        if not full_path.exists():
            print(f"  {name}: NOT FOUND, skipping")
            continue
        ckpt = torch.load(str(full_path), map_location=device, weights_only=True)
        model = cls(**ckpt['config']).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        modules[name] = model
        print(f"  {name}: loaded (frozen)")

    return modules


# ============================================================
# Chain Data Generation
#
# Each chain is defined by:
#   - source(s): waveform + pitch
#   - module_sequence: ordered list of module names
#   - module_params: dict of parameters per module
#
# For each chain we run:
#   1. DSP chain → audio_dsp → encode → z_gt
#   2. Latent chain → z_chain
#   3. Record (z_chain, chain_ids, z_gt)
# ============================================================

def run_latent_chain(modules, z_source, z_source2, chain_modules, params, T, device):
    """Run a chain of latent modules.

    z_source: [1, 8, 16, T] — encoded source (or carrier for FM)
    z_source2: optional second source for mixer/ringmod/FM
    chain_modules: list of module name strings
    params: dict of parameters per module
    """
    z = z_source

    for mod_name in chain_modules:
        if mod_name == 'vcf_temporal':
            cutoff_z = resample_cutoff_to_z(params['cutoff_curve'], T)
            cutoff_t = torch.from_numpy(cutoff_z).float().unsqueeze(0).to(device)
            res_t = torch.tensor([params['resonance']]).float().to(device)
            with torch.no_grad():
                z = modules['vcf_t'](z, cutoff_t, res_t)

        elif mod_name == 'vca':
            adsr = params['adsr_env']
            gain_z = torch.from_numpy(resample_to_z(adsr, T)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                z = modules['vca'](z, gain_z)

        elif mod_name == 'wavefolder':
            fn = (params['fold_amount'] - 1.0) / 7.0
            fold_t = torch.full((1, T), fn, device=device)
            with torch.no_grad():
                z = modules['wavefolder'](z, fold_t)

        elif mod_name == 'distortion':
            dn = (params['drive'] - 1.0) / 14.0
            tn = (np.log10(params['tone_hz']) - np.log10(500)) / (np.log10(10000) - np.log10(500))
            drive_t = torch.full((1, T), dn, device=device)
            tone_t = torch.tensor([tn]).float().to(device)
            with torch.no_grad():
                z = modules['distortion'](z, drive_t, tone_t)

        elif mod_name == 'delay':
            tn = (params['delay_ms'] - 50) / 450
            with torch.no_grad():
                z = modules['delay'](z,
                    torch.tensor([tn]).float().to(device),
                    torch.tensor([params['delay_fb']]).float().to(device),
                    torch.tensor([params['delay_mix']]).float().to(device))

        elif mod_name == 'reverb':
            with torch.no_grad():
                z = modules['reverb'](z,
                    torch.tensor([params['reverb_size']]).float().to(device),
                    torch.tensor([params['reverb_decay']]).float().to(device),
                    torch.tensor([params['reverb_mix']]).float().to(device))

        elif mod_name == 'chorus':
            rn = params['chorus_rate'] / 5.0
            dn = params['chorus_depth'] / 10.0
            with torch.no_grad():
                z = modules['chorus'](z,
                    torch.tensor([rn]).float().to(device),
                    torch.tensor([dn]).float().to(device),
                    torch.tensor([params['chorus_mix']]).float().to(device))

        elif mod_name == 'mixer':
            with torch.no_grad():
                z = modules['mixer'](z_source, z_source2,
                    torch.tensor([params['mix_ratio']]).float().to(device))

        elif mod_name == 'ringmod':
            depth_t = torch.full((1, T), params['ringmod_depth'], device=device)
            with torch.no_grad():
                z = modules['ringmod'](z_source, z_source2, depth_t)

        elif mod_name == 'fm':
            fm_curve_z = resample_to_z(params['fm_curve'], T) / MAX_FM_INDEX
            idx_t = torch.from_numpy(fm_curve_z).float().unsqueeze(0).to(device)
            with torch.no_grad():
                z = modules['fm'](z_source, z_source2, idx_t)

    return z


def run_dsp_chain(audio_source, audio_source2, chain_modules, params):
    """Run the equivalent DSP chain in audio domain."""
    audio = audio_source

    for mod_name in chain_modules:
        if mod_name == 'vcf_temporal':
            audio = apply_tv_filter(audio, params['cutoff_curve'], params['resonance'])
        elif mod_name == 'vca':
            audio = apply_envelope(audio, params['adsr_env'])
        elif mod_name == 'wavefolder':
            audio = dsp_wavefold(audio, params['fold_amount'])
        elif mod_name == 'distortion':
            audio = dsp_distortion(audio, params['drive'], params['tone_hz'])
        elif mod_name == 'delay':
            audio = dsp_delay(audio, params['delay_ms'], params['delay_fb'], params['delay_mix'])
        elif mod_name == 'reverb':
            audio = dsp_reverb(audio, params['reverb_size'], params['reverb_decay'], params['reverb_mix'])
        elif mod_name == 'chorus':
            audio = dsp_chorus(audio, params['chorus_rate'], params['chorus_depth'], params['chorus_mix'])
        elif mod_name == 'mixer':
            audio = _normalize(audio_source * (1 - params['mix_ratio']) +
                              audio_source2 * params['mix_ratio'])
        elif mod_name == 'ringmod':
            audio = dsp_ringmod(audio_source, audio_source2, params['ringmod_depth'])
        elif mod_name == 'fm':
            audio = dsp_fm_temporal(params['fm_carrier_hz'], params['fm_mod_hz'],
                                    params['fm_curve'], 'sine')

    return audio


# ============================================================
# Chain Templates + Parameter Sampling
# ============================================================

# Random parameter generators
def rand_cutoff_env():
    """Random filter envelope."""
    base = np.random.choice([200, 300, 400, 500, 800])
    peak = np.random.choice([2000, 4000, 6000, 8000])
    a = np.random.choice([0.005, 0.01, 0.05, 0.2, 0.4])
    d = np.random.choice([0.1, 0.2, 0.3, 0.5, 0.8])
    s = np.random.uniform(0.0, 0.6)
    r = np.random.choice([0.15, 0.3, 0.5])
    noff = np.random.choice([0.6, 0.8, 1.0, 1.2, 1.5])
    return make_filter_envelope(base, peak, a, d, s, r, noff)


def rand_cutoff_lfo():
    """Random filter LFO."""
    center = np.random.choice([1000, 1500, 2000, 3000])
    depth = np.random.choice([400, 800, 1000, 1500])
    rate = np.random.choice([0.5, 1.0, 2.0, 3.0, 5.0])
    return make_filter_lfo(center, depth, rate)


def rand_adsr():
    """Random VCA envelope."""
    a = np.random.choice([0.002, 0.005, 0.01, 0.05, 0.2, 0.4, 0.8])
    d = np.random.choice([0.08, 0.15, 0.2, 0.3, 0.5])
    s = np.random.uniform(0.0, 0.9)
    r = np.random.choice([0.1, 0.2, 0.3, 0.5])
    noff = np.random.choice([0.5, 0.8, 1.0, 1.2, 1.5])
    return make_adsr_envelope(a, d, s, r, noff)


def rand_fm_env():
    """Random FM index envelope."""
    peak = np.random.choice([2.0, 4.0, 6.0, 8.0])
    a = np.random.choice([0.002, 0.005, 0.05, 0.3])
    d = np.random.choice([0.1, 0.3, 0.8, 1.5])
    s = np.random.uniform(0.0, 0.3)
    r = np.random.choice([0.1, 0.3])
    noff = np.random.choice([0.5, 1.0, 1.5])
    return make_fm_envelope(peak, a, d, s, r, noff)


# Chain templates: (name, module_list, needs_two_sources, param_fn)
CHAIN_TEMPLATES = []

# --- Single-source chains ---
def params_subtractive():
    return {
        'cutoff_curve': rand_cutoff_env(),
        'resonance': np.random.choice([0.0, 0.2, 0.4, 0.6]),
        'adsr_env': rand_adsr(),
    }
CHAIN_TEMPLATES.append(('sub_3', ['vcf_temporal', 'vca'], False, params_subtractive))

def params_sub_delay():
    p = params_subtractive()
    p.update({'delay_ms': np.random.choice([150, 200, 300, 400]),
              'delay_fb': np.random.choice([0.3, 0.4, 0.5]),
              'delay_mix': np.random.choice([0.2, 0.3, 0.4])})
    return p
CHAIN_TEMPLATES.append(('sub_delay_4', ['vcf_temporal', 'delay', 'vca'], False, params_sub_delay))

def params_sub_delay_reverb():
    p = params_sub_delay()
    p.update({'reverb_size': np.random.choice([0.3, 0.5, 0.7]),
              'reverb_decay': np.random.choice([0.4, 0.5, 0.7]),
              'reverb_mix': np.random.choice([0.2, 0.3, 0.4])})
    return p
CHAIN_TEMPLATES.append(('sub_delay_rev_5', ['vcf_temporal', 'delay', 'reverb', 'vca'],
                         False, params_sub_delay_reverb))

def params_westcoast():
    p = params_subtractive()
    p['fold_amount'] = np.random.choice([2.0, 3.0, 4.0, 6.0])
    p.update({'delay_ms': np.random.choice([150, 200, 300]),
              'delay_fb': np.random.choice([0.3, 0.4]),
              'delay_mix': np.random.choice([0.2, 0.3])})
    return p
CHAIN_TEMPLATES.append(('westcoast_5', ['wavefolder', 'vcf_temporal', 'delay', 'vca'],
                         False, params_westcoast))

def params_westcoast_full():
    p = params_westcoast()
    p.update({'reverb_size': np.random.choice([0.3, 0.5, 0.7]),
              'reverb_decay': np.random.choice([0.4, 0.6]),
              'reverb_mix': np.random.choice([0.2, 0.3])})
    return p
CHAIN_TEMPLATES.append(('westcoast_6', ['wavefolder', 'vcf_temporal', 'delay', 'reverb', 'vca'],
                         False, params_westcoast_full))

def params_acid():
    p = params_subtractive()
    p['resonance'] = np.random.choice([0.5, 0.6, 0.7, 0.8])
    p['drive'] = np.random.choice([2.0, 4.0, 8.0])
    p['tone_hz'] = np.random.choice([2000, 3000, 5000])
    p.update({'delay_ms': np.random.choice([150, 200]),
              'delay_fb': np.random.choice([0.4, 0.5]),
              'delay_mix': np.random.choice([0.2, 0.3])})
    return p
CHAIN_TEMPLATES.append(('acid_5', ['vcf_temporal', 'distortion', 'delay', 'vca'],
                         False, params_acid))

def params_pluck_chain():
    p = params_subtractive()
    p['fold_amount'] = np.random.choice([1.5, 2.0, 3.0])
    p.update({'delay_ms': np.random.choice([200, 300]),
              'delay_fb': np.random.choice([0.3, 0.4]),
              'delay_mix': np.random.choice([0.2, 0.3]),
              'reverb_size': np.random.choice([0.4, 0.5, 0.7]),
              'reverb_decay': np.random.choice([0.4, 0.5]),
              'reverb_mix': np.random.choice([0.2, 0.3])})
    return p
CHAIN_TEMPLATES.append(('pluck_6', ['vcf_temporal', 'wavefolder', 'delay', 'reverb', 'vca'],
                         False, params_pluck_chain))

# --- Two-source chains ---
def params_mixed():
    p = params_subtractive()
    p['mix_ratio'] = np.random.choice([0.3, 0.4, 0.5, 0.6])
    return p
CHAIN_TEMPLATES.append(('mix_sub_4', ['mixer', 'vcf_temporal', 'vca'], True, params_mixed))

def params_mixed_chorus():
    p = params_mixed()
    p.update({'chorus_rate': np.random.choice([0.5, 1.0, 2.0]),
              'chorus_depth': np.random.choice([3.0, 5.0, 7.0]),
              'chorus_mix': np.random.choice([0.3, 0.5])})
    return p
CHAIN_TEMPLATES.append(('mix_chorus_5', ['mixer', 'chorus', 'vcf_temporal', 'vca'],
                         True, params_mixed_chorus))

def params_mixed_full():
    p = params_mixed_chorus()
    p.update({'delay_ms': np.random.choice([200, 300]),
              'delay_fb': np.random.choice([0.3, 0.4]),
              'delay_mix': np.random.choice([0.2, 0.3]),
              'reverb_size': np.random.choice([0.5, 0.7]),
              'reverb_decay': np.random.choice([0.5, 0.7]),
              'reverb_mix': np.random.choice([0.3, 0.4])})
    return p
CHAIN_TEMPLATES.append(('mix_full_7', ['mixer', 'chorus', 'vcf_temporal', 'delay', 'reverb', 'vca'],
                         True, params_mixed_full))

def params_ringmod():
    p = params_subtractive()
    p['ringmod_depth'] = np.random.choice([0.5, 0.7, 0.8, 1.0])
    return p
CHAIN_TEMPLATES.append(('ringmod_4', ['ringmod', 'vcf_temporal', 'vca'], True, params_ringmod))

def params_ringmod_dist():
    p = params_ringmod()
    p['drive'] = np.random.choice([4.0, 8.0])
    p['tone_hz'] = np.random.choice([2000, 3000])
    p.update({'delay_ms': np.random.choice([200, 300]),
              'delay_fb': np.random.choice([0.4, 0.5]),
              'delay_mix': np.random.choice([0.3, 0.4])})
    return p
CHAIN_TEMPLATES.append(('ringmod_dist_5', ['ringmod', 'distortion', 'delay', 'vca'],
                         True, params_ringmod_dist))

def params_industrial():
    p = params_ringmod()
    p['drive'] = np.random.choice([4.0, 8.0, 15.0])
    p['tone_hz'] = np.random.choice([2000, 3000])
    p['fold_amount'] = np.random.choice([2.0, 4.0])
    p.update({'delay_ms': np.random.choice([200, 300]),
              'delay_fb': np.random.choice([0.5, 0.6]),
              'delay_mix': np.random.choice([0.3, 0.4])})
    return p
CHAIN_TEMPLATES.append(('industrial_6', ['ringmod', 'distortion', 'wavefolder', 'delay', 'vca'],
                         True, params_industrial))

def params_ambient():
    p = params_mixed()
    p['cutoff_curve'] = rand_cutoff_lfo()  # LFO instead of envelope
    p.update({'chorus_rate': np.random.choice([0.5, 1.0]),
              'chorus_depth': np.random.choice([5.0, 7.0]),
              'chorus_mix': np.random.choice([0.4, 0.5]),
              'reverb_size': np.random.choice([0.7, 0.9]),
              'reverb_decay': np.random.choice([0.7, 0.8]),
              'reverb_mix': np.random.choice([0.4, 0.5])})
    return p
CHAIN_TEMPLATES.append(('ambient_6', ['mixer', 'vcf_temporal', 'chorus', 'reverb', 'vca'],
                         True, params_ambient))


# FM chains (need fm module)
def params_fm_bell():
    return {
        'fm_carrier_hz': np.random.choice([220, 330, 440]),
        'fm_mod_hz': None,  # set below based on ratio
        'fm_curve': rand_fm_env(),
        'adsr_env': rand_adsr(),
        'reverb_size': np.random.choice([0.5, 0.7]),
        'reverb_decay': np.random.choice([0.5, 0.7]),
        'reverb_mix': np.random.choice([0.3, 0.4]),
    }
CHAIN_TEMPLATES.append(('fm_bell_3', ['fm', 'reverb', 'vca'], True, params_fm_bell))


# Source selection
SOURCE_CONFIGS = [
    # Single source: (wf, pitch, wf2, pitch2)
    ('saw', 110, None, None),
    ('saw', 220, None, None),
    ('saw', 330, None, None),
    ('square', 110, None, None),
    ('square', 220, None, None),
    ('triangle', 220, None, None),
    ('triangle', 330, None, None),
    # Two sources
    ('saw', 220, 'saw', 223),       # detuned
    ('saw', 110, 'square', 110),     # different waveforms
    ('triangle', 220, 'saw', 330),   # different wf + pitch
    ('saw', 220, 'triangle', 220),   # same pitch, diff wf
    ('square', 110, 'saw', 165),     # fifth interval
]

FM_RATIOS = [
    (1.0, 'harmonic'),     # unison
    (2.0, 'harmonic'),     # octave
    (1.26, 'inharmonic'),  # minor third
    (1.414, 'inharmonic'), # sqrt(2)
]


def generate_training_data(dcae, modules, device):
    """Generate all training chains."""
    print("Generating chain training data...\n")

    # Pre-encode all sources
    print("  Encoding sources...")
    source_cache = {}
    for wf, pitch, wf2, pitch2 in SOURCE_CONFIGS:
        key1 = (wf, pitch)
        if key1 not in source_cache:
            audio = WAVEFORM_FNS[wf](pitch)
            z = encode_audio(dcae, audio, device)
            source_cache[key1] = {'audio': audio, 'z': z.cpu()}
            print(f"    {wf}@{pitch}Hz")
        if wf2 and (wf2, pitch2) not in source_cache:
            audio = WAVEFORM_FNS[wf2](pitch2)
            z = encode_audio(dcae, audio, device)
            source_cache[(wf2, pitch2)] = {'audio': audio, 'z': z.cpu()}
            print(f"    {wf2}@{pitch2}Hz")

    z_T = list(source_cache.values())[0]['z'].shape[-1]
    print(f"  z frames: T={z_T}\n")

    # Generate chains
    all_z_chain = []
    all_z_gt = []
    all_chain_ids = []
    count = 0
    n_per_template = 40

    for tname, mod_list, needs_two, param_fn in CHAIN_TEMPLATES:
        # Skip FM if not available
        if 'fm' in mod_list and 'fm' not in modules:
            print(f"  Skipping {tname} (no FM module)")
            continue

        print(f"  Template: {tname} [{' → '.join(mod_list)}]")

        # Pick sources appropriate for this template
        if needs_two:
            valid_sources = [s for s in SOURCE_CONFIGS if s[2] is not None]
        else:
            valid_sources = [s for s in SOURCE_CONFIGS if s[2] is None]

        for var_i in range(n_per_template):
            # Pick random source
            src_cfg = valid_sources[var_i % len(valid_sources)]
            wf, pitch, wf2, pitch2 = src_cfg

            src1 = source_cache[(wf, pitch)]
            src2 = source_cache.get((wf2, pitch2)) if wf2 else None

            # Sample random parameters
            params = param_fn()

            # Special handling for FM: set mod frequency
            if 'fm' in mod_list:
                ratio, _ = FM_RATIOS[var_i % len(FM_RATIOS)]
                params['fm_carrier_hz'] = pitch
                params['fm_mod_hz'] = int(pitch * ratio)
                # Encode FM modulator
                fm_mod_key = ('sine', params['fm_mod_hz'])
                if fm_mod_key not in source_cache:
                    audio_m = gen_sine(params['fm_mod_hz'])
                    z_m = encode_audio(dcae, audio_m, device)
                    source_cache[fm_mod_key] = {'audio': audio_m, 'z': z_m.cpu()}
                src2 = source_cache[fm_mod_key]

            # Run DSP chain
            audio_dsp = run_dsp_chain(
                src1['audio'],
                src2['audio'] if src2 else None,
                mod_list, params
            )

            # Encode DSP result → z_gt
            z_gt = encode_audio(dcae, audio_dsp, device)

            # Run latent chain
            z_src1 = src1['z'].unsqueeze(0).to(device)
            z_src2 = src2['z'].unsqueeze(0).to(device) if src2 else None

            z_chain = run_latent_chain(
                modules, z_src1, z_src2, mod_list, params, z_T, device
            )

            # Build chain ID tensor
            chain_id_list = [MODULE_IDS[m] for m in mod_list]
            padded = chain_id_list + [PAD_ID] * (MAX_CHAIN_LEN - len(chain_id_list))

            all_z_chain.append(z_chain.squeeze(0).cpu())
            all_z_gt.append(z_gt.cpu())
            all_chain_ids.append(torch.tensor(padded, dtype=torch.long))

            count += 1
            if count % 40 == 0:
                print(f"    {count} chains generated...")
                clear_memory()

    print(f"\n  Total training chains: {count}")
    return all_z_chain, all_z_gt, all_chain_ids, z_T


# ============================================================
# Training
# ============================================================

def train_corrector(model, z_chain, z_gt, chain_ids, device,
                    epochs=600, batch_size=16, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n_samples = z_gt.shape[0]
    print(f"  Training corrector: {epochs} epochs, {n_samples} samples")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]

            z_pred = model(z_chain[idx], chain_ids[idx])
            loss = F.mse_loss(z_pred, z_gt[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch:3d}: loss = {avg_loss:.6f}")

    return model


# ============================================================
# Testing
# ============================================================

def test_corrector(model, modules, dcae, device, z_T, out_dir):
    """Test the corrector on the 7 stress test patches."""
    print("\n" + "=" * 60)
    print("Testing Chain Corrector on Stress Test Patches")
    print("=" * 60)

    model.eval()

    # Recreate the stress test patches with fixed params
    test_patches = {
        'west_coast': {
            'wf': 'saw', 'pitch': 220, 'wf2': None, 'pitch2': None,
            'modules': ['wavefolder', 'vcf_temporal', 'delay', 'reverb', 'vca'],
            'params': {
                'fold_amount': 3.0,
                'cutoff_curve': make_filter_envelope(300, 4000, 0.01, 0.3, 0.2, 0.4, 1.2),
                'resonance': 0.4,
                'delay_ms': 200, 'delay_fb': 0.4, 'delay_mix': 0.3,
                'reverb_size': 0.5, 'reverb_decay': 0.5, 'reverb_mix': 0.3,
                'adsr_env': make_adsr_envelope(0.005, 0.2, 0.3, 0.4, 1.2),
            }
        },
        'acid_house': {
            'wf': 'saw', 'pitch': 110, 'wf2': None, 'pitch2': None,
            'modules': ['vcf_temporal', 'distortion', 'delay', 'vca'],
            'params': {
                'cutoff_curve': make_filter_envelope(300, 6000, 0.005, 0.1, 0.05, 0.15, 0.6),
                'resonance': 0.7,
                'drive': 4.0, 'tone_hz': 3000,
                'delay_ms': 150, 'delay_fb': 0.5, 'delay_mix': 0.3,
                'adsr_env': make_adsr_envelope(0.005, 0.08, 0.1, 0.15, 0.6),
            }
        },
        'super_saw_pad': {
            'wf': 'saw', 'pitch': 220, 'wf2': 'saw', 'pitch2': 223,
            'modules': ['mixer', 'chorus', 'vcf_temporal', 'delay', 'reverb', 'vca'],
            'params': {
                'mix_ratio': 0.5,
                'chorus_rate': 1.5, 'chorus_depth': 5.0, 'chorus_mix': 0.5,
                'cutoff_curve': make_filter_envelope(800, 5000, 0.4, 0.3, 0.6, 0.5, 1.5),
                'resonance': 0.2,
                'delay_ms': 300, 'delay_fb': 0.3, 'delay_mix': 0.2,
                'reverb_size': 0.7, 'reverb_decay': 0.7, 'reverb_mix': 0.4,
                'adsr_env': make_adsr_envelope(0.4, 0.2, 0.7, 0.5, 1.5),
            }
        },
        'industrial': {
            'wf': 'saw', 'pitch': 110, 'wf2': 'square', 'pitch2': 155,
            'modules': ['ringmod', 'distortion', 'wavefolder', 'delay', 'vca'],
            'params': {
                'ringmod_depth': 0.8,
                'drive': 8.0, 'tone_hz': 3000,
                'fold_amount': 4.0,
                'delay_ms': 250, 'delay_fb': 0.6, 'delay_mix': 0.4,
                'adsr_env': make_adsr_envelope(0.005, 0.3, 0.2, 0.2, 0.8),
            }
        },
        'ambient_drone': {
            'wf': 'triangle', 'pitch': 110, 'wf2': 'saw', 'pitch2': 165,
            'modules': ['mixer', 'vcf_temporal', 'chorus', 'reverb', 'vca'],
            'params': {
                'mix_ratio': 0.4,
                'cutoff_curve': make_filter_lfo(1500, 800, 0.5),
                'resonance': 0.3,
                'chorus_rate': 1.0, 'chorus_depth': 6.0, 'chorus_mix': 0.5,
                'reverb_size': 0.9, 'reverb_decay': 0.8, 'reverb_mix': 0.5,
                'adsr_env': make_adsr_envelope(0.8, 0.1, 0.9, 0.3, 1.5),
            }
        },
        'pluck_chain': {
            'wf': 'saw', 'pitch': 330, 'wf2': None, 'pitch2': None,
            'modules': ['vcf_temporal', 'wavefolder', 'delay', 'reverb', 'vca'],
            'params': {
                'cutoff_curve': make_filter_envelope(500, 8000, 0.005, 0.15, 0.2, 0.3, 1.2),
                'resonance': 0.3,
                'fold_amount': 2.0,
                'delay_ms': 250, 'delay_fb': 0.3, 'delay_mix': 0.3,
                'reverb_size': 0.5, 'reverb_decay': 0.5, 'reverb_mix': 0.3,
                'adsr_env': make_adsr_envelope(0.005, 0.15, 0.3, 0.5, 0.8),
            }
        },
    }

    test_dir = out_dir / "test_patches"
    test_dir.mkdir(parents=True, exist_ok=True)

    results_before = {}
    results_after = {}

    for pname, patch in test_patches.items():
        # Encode sources
        audio1 = WAVEFORM_FNS[patch['wf']](patch['pitch'])
        z1 = encode_audio(dcae, audio1, device).unsqueeze(0)

        audio2 = None
        z2 = None
        if patch['wf2']:
            audio2 = WAVEFORM_FNS[patch['wf2']](patch['pitch2'])
            z2 = encode_audio(dcae, audio2, device).unsqueeze(0)

        # DSP ground truth
        audio_dsp = run_dsp_chain(audio1, audio2, patch['modules'], patch['params'])
        z_gt = encode_audio(dcae, audio_dsp, device)

        # Latent chain (uncorrected)
        z_chain = run_latent_chain(modules, z1, z2, patch['modules'], patch['params'], z_T, device)

        # Measure before correction
        cos_before = F.cosine_similarity(
            z_chain.flatten().unsqueeze(0), z_gt.flatten().unsqueeze(0)
        ).item()
        mse_before = F.mse_loss(z_chain.squeeze(0), z_gt).item()

        # Apply corrector
        chain_id_list = [MODULE_IDS[m] for m in patch['modules']]
        padded = chain_id_list + [PAD_ID] * (MAX_CHAIN_LEN - len(chain_id_list))
        chain_ids_t = torch.tensor([padded], dtype=torch.long, device=device)

        with torch.no_grad():
            z_corrected = model(z_chain, chain_ids_t)

        cos_after = F.cosine_similarity(
            z_corrected.flatten().unsqueeze(0), z_gt.flatten().unsqueeze(0)
        ).item()
        mse_after = F.mse_loss(z_corrected.squeeze(0), z_gt).item()

        improvement = ((mse_before - mse_after) / mse_before * 100) if mse_before > 0 else 0

        results_before[pname] = {'cos': cos_before, 'mse': mse_before}
        results_after[pname] = {'cos': cos_after, 'mse': mse_after}

        print(f"\n  {pname}:")
        print(f"    Before: cos={cos_before:.4f}  mse={mse_before:.6f}")
        print(f"    After:  cos={cos_after:.4f}  mse={mse_after:.6f}  ({improvement:.1f}% improvement)")

        # Save audio
        pd = test_dir / pname
        pd.mkdir(parents=True, exist_ok=True)
        _save_wav(pd / "dsp_gt.wav", audio_dsp)
        _save_wav(pd / "latent_uncorrected.wav",
                  decode_latent(dcae, z_chain.squeeze(0), device))
        _save_wav(pd / "latent_corrected.wav",
                  decode_latent(dcae, z_corrected.squeeze(0), device))

        clear_memory()

    # Summary
    print("\n" + "=" * 60)
    print("CORRECTION SUMMARY")
    print("=" * 60)
    print(f"\n{'Patch':<20s} {'Before cos':>10s} {'After cos':>10s} {'MSE improve':>12s}")
    print("-" * 55)
    for pname in test_patches:
        b = results_before[pname]
        a = results_after[pname]
        imp = ((b['mse'] - a['mse']) / b['mse'] * 100) if b['mse'] > 0 else 0
        print(f"{pname:<20s} {b['cos']:>10.4f} {a['cos']:>10.4f} {imp:>11.1f}%")

    avg_before = np.mean([r['cos'] for r in results_before.values()])
    avg_after = np.mean([r['cos'] for r in results_after.values()])
    print(f"\n{'Average':<20s} {avg_before:>10.4f} {avg_after:>10.4f}")

    return results_before, results_after


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("UNIVERSAL CHAIN CORRECTOR — Train + Test")
    print("=" * 60)

    device = 'cuda'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_PATH,
        vocoder_checkpoint_path=VOCODER_PATH,
    ).to(device)
    dcae.eval()
    print("DCAE loaded!")

    print("\nLoading all modules (frozen)...")
    modules = load_all_modules(device)
    print(f"Loaded {len(modules)} modules.\n")

    # Generate training data
    np.random.seed(42)
    z_chain_list, z_gt_list, chain_ids_list, z_T = \
        generate_training_data(dcae, modules, device)
    clear_memory()

    # Stack into tensors
    z_chain_t = torch.stack(z_chain_list).to(device)
    z_gt_t = torch.stack(z_gt_list).to(device)
    chain_ids_t = torch.stack(chain_ids_list).to(device)

    del z_chain_list, z_gt_list, chain_ids_list
    clear_memory()

    # Before-correction baseline
    cos_before = F.cosine_similarity(
        z_chain_t.reshape(z_chain_t.shape[0], -1),
        z_gt_t.reshape(z_gt_t.shape[0], -1),
        dim=1
    ).mean().item()
    mse_before = F.mse_loss(z_chain_t, z_gt_t).item()
    print(f"\nBefore correction: avg cos={cos_before:.4f}, avg mse={mse_before:.6f}")

    # Build and train corrector
    model = UniversalChainCorrector(
        n_channels=8, latent_dim=16, chain_cond_dim=128, n_blocks=8
    ).to(device)

    model = train_corrector(model, z_chain_t, z_gt_t, chain_ids_t, device,
                            epochs=600, batch_size=16)

    # After-correction check
    model.eval()
    with torch.no_grad():
        z_corrected = []
        for i in range(0, z_chain_t.shape[0], 32):
            batch = model(z_chain_t[i:i+32], chain_ids_t[i:i+32])
            z_corrected.append(batch)
        z_corrected = torch.cat(z_corrected, dim=0)

    cos_after = F.cosine_similarity(
        z_corrected.reshape(z_corrected.shape[0], -1),
        z_gt_t.reshape(z_gt_t.shape[0], -1),
        dim=1
    ).mean().item()
    mse_after = F.mse_loss(z_corrected, z_gt_t).item()
    improvement = ((mse_before - mse_after) / mse_before * 100) if mse_before > 0 else 0

    print(f"\nAfter correction:  avg cos={cos_after:.4f}, avg mse={mse_after:.6f}")
    print(f"Improvement: {improvement:.1f}%")

    # Save checkpoint
    torch.save({
        'model': model.state_dict(),
        'config': {'n_channels': 8, 'latent_dim': 16, 'chain_cond_dim': 128, 'n_blocks': 8},
    }, str(OUTPUT_DIR / "chain_corrector.pt"))
    print(f"\nCheckpoint saved to {OUTPUT_DIR / 'chain_corrector.pt'}")

    # Free training data
    del z_chain_t, z_gt_t, chain_ids_t, z_corrected
    clear_memory()

    # Test on the stress test patches (matches the exact configs from stress test)
    test_corrector(model, modules, dcae, device, z_T, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("CHAIN CORRECTOR — COMPLETE")
    print("=" * 60)
    print(f"\nOutputs: {OUTPUT_DIR}")
    print("Listen to test_patches/<patch>/latent_corrected.wav vs latent_uncorrected.wav")


if __name__ == "__main__":
    main()
