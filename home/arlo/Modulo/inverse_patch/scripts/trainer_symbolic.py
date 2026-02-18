#!/usr/bin/env python3
"""
Symbolic Generation Trainer (Conditioned)

Generates z latents on the symbolic manifold, conditioned on performance signals:
  - Piano roll [128, T] — which notes to play
  - Amplitude [T] — dynamics
  - Instrument (group_id, subgroup_id) — what instrument

Pipeline:
  1. PerformerAIDataset loads (latents, piano_roll, amp, instrument) from manifest
  2. Flow-matching trains to generate z given conditioning
  3. Generated z is decomposable: z → SMS params + operation tree (editable!)
  4. z → DCAE decode → audio

The symbolic bottleneck (SMS codec + op tree) constrains the representation:
  z_real ≈ G(F(z)) + tree_decode(tree_encode(z - G(F(z)))) with ~0.99 cos_sim

Usage:
    python3 trainer_symbolic.py --validate              # Check decomposition quality
    python3 trainer_symbolic.py --train                  # Train conditioned model
    python3 trainer_symbolic.py --train --unconditioned  # Train unconditional (baseline)
    python3 trainer_symbolic.py --eval                   # Generate + decompose + audio
    python3 trainer_symbolic.py --eval --unconditioned   # Eval unconditional
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import torchaudio
import gc
import os
import random
import pretty_midi

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data')

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM
from phase2_operation_tree import OperationTreeCodec

OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "symbolic_trainer"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")
MANIFEST_PATH = "/home/arlo/gcs-bucket/Manifests/unified_manifest.json"

# Codec checkpoints
CODEC_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
TREE_PATH = SCRIPT_DIR.parent / "test_outputs" / "phase2_operation_tree" / "operation_tree.pt"
DATA_CACHE = OUTPUT_DIR / "symbolic_data_cache.pt"
MODEL_PATH = OUTPUT_DIR / "symbolic_diffusion.pt"
COND_MODEL_PATH = OUTPUT_DIR / "symbolic_diffusion_conditioned.pt"

Z_DIM = 128       # 8 * 16 flattened
MAX_FRAMES = 256
SAMPLE_RATE = 44100

DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

# Instrument vocab (matches dataloader.py)
APPROVED_GROUPS = ["piano", "guitar", "bass", "strings", "brass", "winds"]
APPROVED_SUBGROUPS = {
    "piano":   ["acoustic_piano", "keys", "undefined"],
    "guitar":  ["acoustic_guitar", "electric_guitar", "plucked", "undefined"],
    "bass":    ["electric_bass", "upright_bass", "undefined"],
    "strings": ["violin", "viola", "cello", "undefined"],
    "brass":   ["trumpet", "trombone", "french_horn", "tuba", "undefined"],
    "winds":   ["bassoon", "clarinet", "flute", "oboe", "sax"],
}
GROUP2ID = {g: i for i, g in enumerate(APPROVED_GROUPS)}
ALL_SUBS = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
SUB2ID = {sg: i for i, sg in enumerate(ALL_SUBS)}
N_GROUPS = len(APPROVED_GROUPS)
N_SUBGROUPS = len(ALL_SUBS)

SLOW_HZ = SAMPLE_RATE / 4096  # ~10.77 Hz (DCAE frame rate)


# ============================================================
# Dataset: loads from unified manifest + derives paths
# ============================================================

class SymbolicDataset(torch.utils.data.Dataset):
    """
    Loads (latent, piano_roll, amp, instrument) from the unified manifest.

    Path derivation from audio_path:
      - Latent: manifest['latent_path'] directly
      - Conditioning: audio_path with /protools/ → /Conditioning/protools/, .wav → .{amp,rframe,rbend}.npy
      - Piano roll: audio_path with /protools/ → /BasicPitch/protools/, .wav → .mid → pretty_midi
    """

    def __init__(self, manifest_path, window_slow=256, pr_dropout=0.15,
                 amp_dropout=0.10, seed=42):
        super().__init__()
        import orjson
        raw = orjson.loads(Path(manifest_path).read_bytes())

        if isinstance(raw, dict) and 'entries' in raw:
            entries = raw['entries']
        else:
            entries = raw

        # Unified manifest: entries is list of dicts
        if isinstance(entries, list):
            all_entries = entries
        elif isinstance(entries, dict):
            # Dict keyed by audio_path — convert to list
            all_entries = [{'audio_path': k, **v} for k, v in entries.items()]
        else:
            all_entries = []

        # Filter: approved group + has latent + has conditioning
        self.items = []
        for e in all_entries:
            g = (e.get('group') or '').lower()
            if g not in GROUP2ID:
                continue
            if not e.get('has_latent', False):
                continue
            if not e.get('has_conditioning', False):
                continue
            self.items.append(e)

        self.window_slow = window_slow
        self.pr_dropout = pr_dropout
        self.amp_dropout = amp_dropout
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def __len__(self):
        return len(self.items)

    def _derive_cond_path(self, audio_path, suffix):
        """audio_path → conditioning file path."""
        stem = Path(audio_path).stem
        parent = str(Path(audio_path).parent)
        cond_parent = parent.replace('/gcs-bucket/protools/', '/gcs-bucket/Conditioning/protools/')
        return f'{cond_parent}/{stem}.{suffix}.npy'

    def _derive_midi_path(self, audio_path):
        """audio_path → BasicPitch MIDI path."""
        stem = Path(audio_path).stem
        parent = str(Path(audio_path).parent)
        midi_parent = parent.replace('/gcs-bucket/protools/', '/gcs-bucket/BasicPitch/protools/')
        return f'{midi_parent}/{stem}.mid'

    def _load_npy(self, path):
        try:
            if Path(path).exists():
                return np.load(path)
        except Exception:
            pass
        return None

    def _midi_to_piano_roll(self, midi_path, T_slow):
        """Load MIDI file and convert to piano roll [128, T_slow]."""
        try:
            if not Path(midi_path).exists():
                return torch.zeros(128, T_slow)
            pm = pretty_midi.PrettyMIDI(midi_path)
            pr = pm.get_piano_roll(fs=SLOW_HZ)  # [128, T_midi]
            pr = torch.from_numpy(pr).float()
            # Normalize velocity to [0, 1]
            if pr.max() > 0:
                pr = pr / pr.max()
            # Pad/trim to T_slow
            if pr.shape[1] < T_slow:
                pr = F.pad(pr, (0, T_slow - pr.shape[1]))
            elif pr.shape[1] > T_slow:
                pr = pr[:, :T_slow]
            return pr
        except Exception:
            return torch.zeros(128, T_slow)

    def __getitem__(self, idx):
        e = self.items[idx]
        audio_path = e['audio_path']
        latent_path = e.get('latent_path', '')
        group = (e.get('group') or 'guitar').lower()
        subgroup = (e.get('subgroup') or e.get('sub_group') or 'undefined').lower()

        if group not in GROUP2ID:
            group = 'guitar'
        if subgroup not in SUB2ID:
            subgroup = 'undefined'

        # Load latent
        obj = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(obj, dict) and 'latents' in obj:
            latents = obj['latents']
        elif isinstance(obj, torch.Tensor):
            latents = obj
        else:
            raise ValueError(f"Bad latent at {latent_path}")

        # Ensure [8, 16, T]
        if latents.dim() != 3 or latents.shape[0] != 8 or latents.shape[1] != 16:
            raise ValueError(f"Bad latent shape {latents.shape} at {latent_path}")

        T_slow_all = latents.shape[2]

        # Window
        W = min(self.window_slow, T_slow_all)
        max_start = max(0, T_slow_all - W)
        start = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end = start + W
        latents = latents[:, :, start:end]
        T_slow = latents.shape[2]

        # Load conditioning
        amp_arr = self._load_npy(self._derive_cond_path(audio_path, 'amp'))
        rframe_arr = self._load_npy(self._derive_cond_path(audio_path, 'rframe'))

        if amp_arr is not None:
            amp = torch.from_numpy(amp_arr).float()
            if amp.shape[0] > start:
                amp = amp[start:min(start + W, amp.shape[0])]
            # Pad/trim to T_slow
            if amp.shape[0] < T_slow:
                amp = F.pad(amp, (0, T_slow - amp.shape[0]))
            elif amp.shape[0] > T_slow:
                amp = amp[:T_slow]
        else:
            amp = torch.zeros(T_slow)

        # Piano roll from MIDI
        midi_path = self._derive_midi_path(audio_path)
        piano_roll = self._midi_to_piano_roll(midi_path, T_slow_all)
        piano_roll = piano_roll[:, start:end]
        if piano_roll.shape[1] < T_slow:
            piano_roll = F.pad(piano_roll, (0, T_slow - piano_roll.shape[1]))

        # Conditioning dropout
        if random.random() < self.pr_dropout:
            piano_roll = torch.zeros_like(piano_roll)
        if random.random() < self.amp_dropout:
            amp = torch.zeros_like(amp)

        return {
            'latents': latents,            # [8, 16, T_slow]
            'piano_roll': piano_roll,      # [128, T_slow]
            'amp': amp,                    # [T_slow]
            'group_id': torch.tensor(GROUP2ID[group], dtype=torch.long),
            'subgroup_id': torch.tensor(SUB2ID[subgroup], dtype=torch.long),
        }


def collate_symbolic(batch):
    """Collate variable-length samples with padding."""
    max_T = max(b['latents'].shape[2] for b in batch)

    latents_list, pr_list, amp_list = [], [], []
    group_ids, subgroup_ids = [], []

    for b in batch:
        T = b['latents'].shape[2]
        pad = max_T - T

        lat = F.pad(b['latents'], (0, pad))         # [8, 16, max_T]
        pr = F.pad(b['piano_roll'], (0, pad))        # [128, max_T]
        amp = F.pad(b['amp'], (0, pad))              # [max_T]

        latents_list.append(lat)
        pr_list.append(pr)
        amp_list.append(amp)
        group_ids.append(b['group_id'])
        subgroup_ids.append(b['subgroup_id'])

    return {
        'latents': torch.stack(latents_list),         # [B, 8, 16, T]
        'piano_roll': torch.stack(pr_list),           # [B, 128, T]
        'amp': torch.stack(amp_list),                 # [B, T]
        'group_id': torch.stack(group_ids),           # [B]
        'subgroup_id': torch.stack(subgroup_ids),     # [B]
    }


# ============================================================
# Symbolic Bottleneck: frozen codec + tree
# ============================================================

class SymbolicBottleneck(nn.Module):
    """
    Wraps frozen SMS codec + operation tree.

    Provides:
      encode: z_flat → (sms_params, op_activations, op_params)
      decode: (sms_params, op_activations, op_params) → z_reconstructed
      bottleneck: z → symbolic → z_reconstructed (the full roundtrip)
    """

    def __init__(self, codec_path, tree_path, device='cuda'):
        super().__init__()
        self.device = device

        # Load codec
        print(f"Loading SMS codec from {codec_path}...")
        self.codec = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_DIM, g_hidden=384, f_hidden=256)
        codec_state = torch.load(codec_path, weights_only=True, map_location='cpu')
        self.codec.load_state_dict(codec_state)
        self.codec.to(device).eval()

        # Load operation tree
        print(f"Loading operation tree from {tree_path}...")
        tree_ckpt = torch.load(tree_path, weights_only=False, map_location='cpu')
        self.tree = OperationTreeCodec(
            z_dim=Z_DIM,
            n_ops=tree_ckpt['n_ops'],
            param_dim=tree_ckpt['param_dim'],
            encoder_hidden=256,
            top_k=tree_ckpt['top_k'],
        )
        self.tree.load_state_dict(tree_ckpt['model'])
        self.tree.to(device).eval()

        # Residual normalization stats
        self.res_mean = tree_ckpt['res_mean'].to(device)
        self.res_std = tree_ckpt['res_std'].to(device)

        self.n_ops = tree_ckpt['n_ops']
        self.param_dim = tree_ckpt['param_dim']
        self.top_k = tree_ckpt['top_k']

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

        print(f"  Codec: SMS_DIM={SMS_DIM}, Z_DIM={Z_DIM}")
        print(f"  Tree: n_ops={self.n_ops}, param_dim={self.param_dim}, top_k={self.top_k}")

    @torch.no_grad()
    def encode(self, z_flat):
        """z_flat [B, T, 128] → symbolic representation."""
        sms_params = self.codec.forward_F(z_flat)
        z_sms = self.codec.forward_G(sms_params)
        residual = z_flat - z_sms

        residual_norm = (residual - self.res_mean) / self.res_std
        hidden = self.tree.encode(residual_norm)
        z_recon_norm, activations, all_params = self.tree.decode(hidden)

        return {
            'sms_params': sms_params,
            'op_activations': activations,
            'op_params': all_params,
            'z_sms': z_sms,
            'residual': residual,
            'residual_norm': residual_norm,
            'tree_recon_norm': z_recon_norm,
        }

    @torch.no_grad()
    def decode(self, sms_params, op_activations, op_params):
        """Reconstruct z from symbolic params."""
        z_sms = self.codec.forward_G(sms_params)
        B, T, _ = z_sms.shape
        contributions = torch.zeros(B, T, Z_DIM, device=z_sms.device)

        for k in range(self.n_ops):
            params_k = op_params[:, :, k, :]
            contrib_k = self.tree.operations[k](params_k)
            alpha_k = op_activations[:, :, k:k+1]
            contributions = contributions + alpha_k * contrib_k

        residual = contributions * self.res_std + self.res_mean
        return z_sms + residual

    @torch.no_grad()
    def bottleneck(self, z_flat):
        """Full roundtrip: z → symbolic → z_reconstructed."""
        sym = self.encode(z_flat)
        return self.decode(sym['sms_params'], sym['op_activations'], sym['op_params'])


# ============================================================
# Validation: Check decomposition quality of real latents
# ============================================================

def gather_latents(max_samples=500):
    """Load real z latents from GCS bucket."""
    print(f"\nGathering latents from {LATENT_BASE}...")
    data = []
    skipped = 0

    for pt_file in LATENT_BASE.rglob("*.pt"):
        if len(data) >= max_samples:
            break
        try:
            loaded = torch.load(pt_file, weights_only=False, map_location='cpu')
            if isinstance(loaded, dict) and 'latents' in loaded:
                z = loaded['latents']
            elif isinstance(loaded, torch.Tensor):
                z = loaded
            else:
                skipped += 1
                continue

            if z.dim() != 3 or z.shape[0] != 8 or z.shape[1] != 16 or z.shape[2] < 10:
                skipped += 1
                continue

            if z.shape[2] > MAX_FRAMES:
                z = z[:, :, :MAX_FRAMES]

            data.append({'z': z, 'path': str(pt_file)})
            if len(data) % 100 == 0:
                print(f"    {len(data)} loaded...")
        except Exception:
            skipped += 1
            continue

    print(f"  Loaded {len(data)} latents (skipped {skipped})")
    return data


def validate(device='cuda', max_samples=200):
    """Validate that real z latents decompose cleanly through symbolic pipeline."""
    print("=" * 60)
    print("VALIDATING SYMBOLIC DECOMPOSITION")
    print("=" * 60)

    bottleneck = SymbolicBottleneck(CODEC_PATH, TREE_PATH, device)
    data = gather_latents(max_samples=max_samples)

    cos_sims_sms = []
    cos_sims_full = []
    mse_sms = []
    mse_full = []
    mag_ratios = []

    op_usage = np.zeros(bottleneck.n_ops)

    for i, sample in enumerate(data):
        z = sample['z'].unsqueeze(0).to(device)
        z_flat = bottleneck.codec.z_to_flat(z)

        sym = bottleneck.encode(z_flat)
        z_bottleneck = bottleneck.decode(
            sym['sms_params'], sym['op_activations'], sym['op_params']
        )

        cos_sms = F.cosine_similarity(z_flat, sym['z_sms'], dim=-1).mean().item()
        cos_sims_sms.append(cos_sms)
        mse_sms.append(F.mse_loss(sym['z_sms'], z_flat).item())

        cos_full = F.cosine_similarity(z_flat, z_bottleneck, dim=-1).mean().item()
        cos_sims_full.append(cos_full)
        mse_full.append(F.mse_loss(z_bottleneck, z_flat).item())

        mag_ratio = (z_bottleneck.norm() / z_flat.norm().clamp(min=1e-6)).item()
        mag_ratios.append(mag_ratio)

        activations = sym['op_activations'].squeeze(0)
        for k in range(bottleneck.n_ops):
            active = (activations[:, k] > 0.01).float().mean().item()
            op_usage[k] += active

        if i < 10:
            print(f"  Sample {i:3d}: SMS cos={cos_sms:.4f}  "
                  f"Full cos={cos_full:.4f}  "
                  f"gain={cos_full-cos_sms:+.4f}  "
                  f"mag={mag_ratio:.3f}")

    n = len(data)
    op_usage /= n

    print(f"\n{'='*40}")
    print(f"  Samples analyzed: {n}")
    print(f"\n  SMS-only decomposition:")
    print(f"    cos_sim:  {np.mean(cos_sims_sms):.4f} +/- {np.std(cos_sims_sms):.4f}")
    print(f"    MSE:      {np.mean(mse_sms):.6f}")
    print(f"\n  Full (SMS + op tree) decomposition:")
    print(f"    cos_sim:  {np.mean(cos_sims_full):.4f} +/- {np.std(cos_sims_full):.4f}")
    print(f"    MSE:      {np.mean(mse_full):.6f}")
    print(f"    mag:      {np.mean(mag_ratios):.4f}")
    print(f"\n  Improvement from op tree: {np.mean(cos_sims_full)-np.mean(cos_sims_sms):+.4f}")

    print(f"\n  Op usage (% frames active):")
    for k in range(bottleneck.n_ops):
        print(f"    Op {k}: {op_usage[k]*100:.1f}%")

    full_cos = np.mean(cos_sims_full)
    if full_cos > 0.98:
        print(f"\n  EXCELLENT: cos_sim={full_cos:.4f}")
    elif full_cos > 0.95:
        print(f"\n  GOOD: cos_sim={full_cos:.4f}")
    elif full_cos > 0.90:
        print(f"\n  FAIR: cos_sim={full_cos:.4f}")
    else:
        print(f"\n  POOR: cos_sim={full_cos:.4f}")

    return {'sms_cos': np.mean(cos_sims_sms), 'full_cos': full_cos, 'full_mse': np.mean(mse_full)}


# ============================================================
# Model: Performance Conditioner
# ============================================================

class PerformanceConditioner(nn.Module):
    """
    Lightweight performance conditioning for symbolic flow net.

    Produces:
      - global_cond [B, cond_dim*2]: timestep + instrument for FiLM modulation
      - frame_cond [B, T, z_dim]: per-frame additive conditioning from piano roll + amplitude
    """

    def __init__(self, z_dim=128, cond_dim=128, pr_dim=128,
                 n_groups=N_GROUPS, n_subgroups=N_SUBGROUPS):
        super().__init__()
        self.cond_dim = cond_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Instrument embedding (global)
        self.group_emb = nn.Embedding(n_groups, cond_dim // 2)
        self.subgroup_emb = nn.Embedding(n_subgroups, cond_dim // 2)
        self.inst_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Per-frame: piano roll → z_dim
        self.pr_proj = nn.Sequential(
            nn.LayerNorm(pr_dim),
            nn.Linear(pr_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, z_dim),
        )

        # Per-frame: amplitude → z_dim
        self.amp_proj = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, z_dim),
        )

        # Learnable gain for frame conditioning (starts small)
        self.frame_gain = nn.Parameter(torch.tensor(0.3))

    def forward(self, t, piano_roll=None, amp=None,
                group_id=None, subgroup_id=None):
        """
        Args:
            t: [B] float in [0, 1]
            piano_roll: [B, 128, T_slow] or None
            amp: [B, T_slow] or None
            group_id: [B] long or None
            subgroup_id: [B] long or None

        Returns:
            global_cond: [B, cond_dim * 2] for FiLM
            frame_cond: [B, T, z_dim] or None
        """
        B = t.shape[0]

        # Timestep
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, cond_dim]

        # Instrument
        if group_id is not None and subgroup_id is not None:
            g = self.group_emb(group_id.clamp(0, self.group_emb.num_embeddings - 1))
            sg = self.subgroup_emb(subgroup_id.clamp(0, self.subgroup_emb.num_embeddings - 1))
            inst = self.inst_proj(torch.cat([g, sg], dim=-1))  # [B, cond_dim]
        else:
            inst = torch.zeros_like(t_emb)

        global_cond = torch.cat([t_emb, inst], dim=-1)  # [B, cond_dim * 2]

        # Per-frame conditioning
        frame_cond = None
        if piano_roll is not None:
            pr = piano_roll.permute(0, 2, 1)  # [B, T, 128]
            frame_cond = self.pr_proj(pr)       # [B, T, z_dim]

            if amp is not None:
                amp_in = amp.unsqueeze(-1)  # [B, T, 1]
                frame_cond = frame_cond + self.amp_proj(amp_in)

            frame_cond = self.frame_gain.tanh() * frame_cond

        return global_cond, frame_cond


# ============================================================
# Model: Symbolic Flow Net (supports conditioning)
# ============================================================

class SymbolicFlowNet(nn.Module):
    """
    Flow-matching network for z latent generation.

    Architecture: temporal ConvNet + GRU with FiLM from global conditioning.
    Operates in flattened z-space [B, T, 128].

    Supports optional per-frame conditioning (piano roll + amp) added to input.
    """

    def __init__(self, z_dim=128, hidden=384, n_layers=4, cond_dim=128):
        super().__init__()
        self.z_dim = z_dim

        # Input: noisy z [B, T, z_dim] → hidden
        self.input_proj = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # FiLM from global conditioning (cond_dim * 2 = time + instrument)
        self.film_projs = nn.ModuleList([
            nn.Linear(cond_dim * 2, hidden * 2) for _ in range(n_layers)
        ])

        # Temporal convolutions
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(nn.Conv1d(hidden, hidden, kernel_size=7, padding=3))
            self.norms.append(nn.GroupNorm(1, hidden))

        # GRU for longer-range dependencies
        self.gru = nn.GRU(hidden, hidden // 2, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.1)

        # Output: predict velocity
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, x_t, global_cond, frame_cond=None):
        """
        Args:
            x_t: [B, T, z_dim] — noisy z
            global_cond: [B, cond_dim * 2] — timestep + instrument
            frame_cond: [B, T_cond, z_dim] or None — per-frame conditioning

        Returns:
            v_pred: [B, T, z_dim] — predicted velocity
        """
        B, T, _ = x_t.shape

        # Add per-frame conditioning to input
        if frame_cond is not None:
            if frame_cond.shape[1] != T:
                frame_cond = F.interpolate(
                    frame_cond.permute(0, 2, 1), size=T,
                    mode='linear', align_corners=False
                ).permute(0, 2, 1)
            x_t = x_t + frame_cond

        h = self.input_proj(x_t)  # [B, T, hidden]

        for conv, norm, film in zip(self.convs, self.norms, self.film_projs):
            # FiLM modulation
            params = film(global_cond)  # [B, hidden*2]
            gamma, beta = params.chunk(2, dim=-1)
            h_mod = gamma.unsqueeze(1) * h + beta.unsqueeze(1)

            # Conv (residual)
            h_t = h_mod.permute(0, 2, 1)  # [B, hidden, T]
            h_t = F.gelu(norm(conv(h_t)))
            h = h + h_t.permute(0, 2, 1)

        h, _ = self.gru(h)
        return self.output_proj(h)


# ============================================================
# Data loading helpers
# ============================================================

def z_4d_to_flat(z_4d):
    """[B, 8, 16, T] → [B, T, 128]"""
    B, C, H, T = z_4d.shape
    return z_4d.permute(0, 3, 1, 2).reshape(B, T, C * H)


def z_flat_to_4d(z_flat):
    """[B, T, 128] → [B, 8, 16, T]"""
    B, T, D = z_flat.shape
    return z_flat.reshape(B, T, 8, 16).permute(0, 2, 3, 1)


# ============================================================
# Training: Conditioned (main path)
# ============================================================

COND_DATA_CACHE = OUTPUT_DIR / "conditioned_data_cache.pt"


def prepare_conditioned_data(manifest_path=MANIFEST_PATH, max_samples=5000,
                             window_slow=256):
    """Pre-cache conditioned data from GCS to local disk."""
    print("=" * 60)
    print("PREPARING CONDITIONED TRAINING DATA")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if COND_DATA_CACHE.exists():
        print(f"\nCache exists at {COND_DATA_CACHE}. Loading...")
        cached = torch.load(COND_DATA_CACHE, weights_only=False, map_location='cpu')
        print(f"  {len(cached)} samples loaded from cache")
        return cached

    dataset = SymbolicDataset(
        manifest_path=manifest_path,
        window_slow=window_slow,
        pr_dropout=0.0,  # no dropout during caching
        amp_dropout=0.0,
        seed=42,
    )
    n_total = len(dataset)
    n_load = min(max_samples, n_total) if max_samples else n_total
    print(f"  Dataset: {n_total} total, loading {n_load}...")

    cached = []
    errors = 0
    for i in range(n_load):
        try:
            sample = dataset[i]
            cached.append(sample)
        except Exception as ex:
            errors += 1
            if errors <= 5:
                print(f"    Skip {i}: {ex}")
            continue

        if (len(cached)) % 200 == 0:
            print(f"    {len(cached)}/{n_load} loaded ({errors} errors)")

    print(f"\n  Cached {len(cached)} samples ({errors} errors)")
    print(f"  Saving to {COND_DATA_CACHE}...")
    torch.save(cached, COND_DATA_CACHE)
    sz = COND_DATA_CACHE.stat().st_size / 1e6
    print(f"  Done! ({sz:.0f} MB)")
    return cached


class CachedDataset(torch.utils.data.Dataset):
    """Wraps a list of pre-loaded samples with optional conditioning dropout."""

    def __init__(self, samples, pr_dropout=0.15, amp_dropout=0.10):
        self.samples = samples
        self.pr_dropout = pr_dropout
        self.amp_dropout = amp_dropout

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        out = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in s.items()}
        if random.random() < self.pr_dropout:
            out['piano_roll'] = torch.zeros_like(out['piano_roll'])
        if random.random() < self.amp_dropout:
            out['amp'] = torch.zeros_like(out['amp'])
        return out


def train_conditioned(epochs=100, batch_size=8, lr=3e-4,
                      sym_loss_weight=0.5, device='cuda',
                      manifest_path=MANIFEST_PATH, max_samples=5000,
                      window_slow=256):
    """Train conditioned symbolic flow-matching model."""
    print("=" * 60)
    print("TRAINING CONDITIONED SYMBOLIC DIFFUSION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-cache data from GCS
    all_samples = prepare_conditioned_data(
        manifest_path=manifest_path,
        max_samples=max_samples,
        window_slow=window_slow,
    )

    # Split train/test
    n_total = len(all_samples)
    n_test = min(50, n_total // 10)
    n_train = n_total - n_test
    train_samples = all_samples[:n_train]
    test_samples = all_samples[n_train:]

    train_set = CachedDataset(train_samples, pr_dropout=0.15, amp_dropout=0.10)
    test_set = CachedDataset(test_samples, pr_dropout=0.0, amp_dropout=0.0)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_symbolic, num_workers=0,
        pin_memory=True, drop_last=True,
    )
    print(f"  Train: {n_train}, Test: {n_test}")

    # Load symbolic bottleneck (frozen, for auxiliary loss)
    bottleneck = SymbolicBottleneck(CODEC_PATH, TREE_PATH, device)

    # Model
    cond_dim = 128
    conditioner = PerformanceConditioner(
        z_dim=Z_DIM, cond_dim=cond_dim,
        n_groups=N_GROUPS, n_subgroups=N_SUBGROUPS,
    ).to(device)
    flow_net = SymbolicFlowNet(
        z_dim=Z_DIM, hidden=384, n_layers=4, cond_dim=cond_dim,
    ).to(device)

    all_params = list(conditioner.parameters()) + list(flow_net.parameters())
    n_params = sum(p.numel() for p in all_params)
    print(f"  Model params: {n_params:,}")
    print(f"    Conditioner: {sum(p.numel() for p in conditioner.parameters()):,}")
    print(f"    Flow net:    {sum(p.numel() for p in flow_net.parameters()):,}")

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        conditioner.train()
        flow_net.train()

        total_loss = 0
        total_recon = 0
        total_sym = 0
        n_samples = 0

        for batch in train_loader:
            # Extract from batch
            latents = batch['latents'].to(device)           # [B, 8, 16, T]
            piano_roll = batch['piano_roll'].to(device)     # [B, 128, T]
            amp = batch['amp'].to(device)                   # [B, T]
            group_id = batch['group_id'].to(device)         # [B]
            subgroup_id = batch['subgroup_id'].to(device)   # [B]

            B_actual = latents.shape[0]
            T_lat = latents.shape[-1]

            # Flatten z: [B, 8, 16, T] → [B, T, 128]
            x0 = z_4d_to_flat(latents)  # [B, T, 128]

            # Align conditioning time to latent time
            if piano_roll.shape[-1] != T_lat:
                piano_roll = F.interpolate(piano_roll, size=T_lat, mode='nearest')
            if amp.shape[-1] != T_lat:
                amp = F.interpolate(amp.unsqueeze(1), size=T_lat, mode='nearest').squeeze(1)

            # Flow-matching: sample t, create x_t
            t = torch.rand(B_actual, device=device).clamp(1e-4, 1 - 1e-4)
            noise = torch.randn_like(x0)
            x_t = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * noise

            # Conditioning
            global_cond, frame_cond = conditioner(
                t, piano_roll=piano_roll, amp=amp,
                group_id=group_id, subgroup_id=subgroup_id,
            )

            # Predict velocity
            v_pred = flow_net(x_t, global_cond, frame_cond)

            # v target = noise - x0
            v_target = noise - x0

            # MSE loss on velocity
            recon_loss = F.mse_loss(v_pred, v_target)

            # Symbolic consistency loss (after warmup)
            sym_loss = torch.tensor(0.0, device=device)
            if sym_loss_weight > 0 and epoch >= 5:
                x0_hat = x_t - t.view(-1, 1, 1) * v_pred
                with torch.no_grad():
                    x0_bn = bottleneck.bottleneck(x0_hat.detach())
                sym_loss = F.mse_loss(x0_hat, x0_bn.detach()) * 0.1

            loss = recon_loss + sym_loss_weight * sym_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            total_loss += loss.item() * B_actual
            total_recon += recon_loss.item() * B_actual
            total_sym += sym_loss.item() * B_actual
            n_samples += B_actual

        scheduler.step()
        avg_loss = total_loss / max(n_samples, 1)
        avg_recon = total_recon / max(n_samples, 1)
        avg_sym = total_sym / max(n_samples, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save({
                'conditioner': conditioner.state_dict(),
                'flow_net': flow_net.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'conditioned': True,
                'cond_dim': cond_dim,
            }, COND_MODEL_PATH)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={avg_loss:.6f}  recon={avg_recon:.6f}  sym={avg_sym:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\n  Best loss: {best_loss:.6f} (epoch {best_epoch})")
    print(f"  Saved to {COND_MODEL_PATH}")

    return conditioner, flow_net


# ============================================================
# Training: Unconditioned (baseline)
# ============================================================

def prepare_unconditioned_data(device='cuda', max_samples=2000):
    """Pre-compute data for unconditional training from raw latent files."""
    print("=" * 60)
    print("PREPARING UNCONDITIONED TRAINING DATA")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_CACHE.exists():
        print(f"\nData cache exists at {DATA_CACHE}. Loading...")
        return torch.load(DATA_CACHE, weights_only=False, map_location='cpu')

    bottleneck = SymbolicBottleneck(CODEC_PATH, TREE_PATH, device)
    raw_data = gather_latents(max_samples=max_samples)

    processed = []
    cos_sims = []

    print(f"\nProcessing {len(raw_data)} latents through symbolic bottleneck...")

    for i, sample in enumerate(raw_data):
        z = sample['z'].unsqueeze(0).to(device)
        z_flat = bottleneck.codec.z_to_flat(z)

        sym = bottleneck.encode(z_flat)
        z_bottleneck = bottleneck.decode(
            sym['sms_params'], sym['op_activations'], sym['op_params']
        )

        cos = F.cosine_similarity(z_flat, z_bottleneck, dim=-1).mean().item()
        cos_sims.append(cos)

        processed.append({
            'z_original': z.squeeze(0).cpu(),
            'z_bottleneck': bottleneck.codec.flat_to_z(z_bottleneck).squeeze(0).cpu(),
            'sms_params': sym['sms_params'].squeeze(0).cpu(),
            'op_activations': sym['op_activations'].squeeze(0).cpu(),
            'op_params': sym['op_params'].squeeze(0).cpu(),
            'path': sample['path'],
        })

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(raw_data)}  avg cos_sim={np.mean(cos_sims):.4f}")

    print(f"\n  Processed {len(processed)} samples")
    print(f"  Average bottleneck cos_sim: {np.mean(cos_sims):.4f}")
    print(f"\n  Saving to {DATA_CACHE}...")
    torch.save(processed, DATA_CACHE)
    print(f"  Done! ({DATA_CACHE.stat().st_size / 1e6:.0f} MB)")

    return processed


def pad_batch_uncond(samples, key, device):
    """Pad variable-length tensors into a batch + mask."""
    if key in ('z_bottleneck', 'z_original'):
        tensors = [s[key].permute(2, 0, 1).reshape(-1, 128) for s in samples]
    elif key == 'sms_params':
        tensors = [s[key] for s in samples]
    else:
        tensors = [s[key] for s in samples]

    T_max = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[-1] if tensors[0].dim() >= 2 else 1
    B = len(tensors)

    if tensors[0].dim() == 2:
        padded = torch.zeros(B, T_max, D)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
    elif tensors[0].dim() == 3:
        D2 = tensors[0].shape[-1]
        D1 = tensors[0].shape[-2]
        padded = torch.zeros(B, T_max, D1, D2)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
    else:
        raise ValueError(f"Unsupported tensor dim: {tensors[0].dim()}")

    mask = torch.zeros(B, T_max)
    for i, t in enumerate(tensors):
        mask[i, :t.shape[0]] = 1.0

    return padded.to(device), mask.to(device)


def train_unconditioned(epochs=100, batch_size=16, lr=3e-4,
                        sym_loss_weight=0.5, device='cuda'):
    """Train unconditional symbolic flow-matching model (baseline)."""
    print("=" * 60)
    print("TRAINING UNCONDITIONED SYMBOLIC DIFFUSION")
    print("=" * 60)

    data = prepare_unconditioned_data(device=device)

    n_test = min(50, len(data) // 10)
    train_data = data[:-n_test]
    test_data = data[-n_test:]
    print(f"\n  Train: {len(train_data)}, Test: {n_test}")

    bottleneck = SymbolicBottleneck(CODEC_PATH, TREE_PATH, device)

    cond_dim = 128
    conditioner = PerformanceConditioner(z_dim=Z_DIM, cond_dim=cond_dim).to(device)
    flow_net = SymbolicFlowNet(z_dim=Z_DIM, hidden=384, n_layers=4, cond_dim=cond_dim).to(device)

    all_params = list(conditioner.parameters()) + list(flow_net.parameters())
    n_params = sum(p.numel() for p in all_params)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float('inf')

    for epoch in range(epochs):
        conditioner.train()
        flow_net.train()

        perm = np.random.permutation(len(train_data))
        total_loss = 0
        total_recon = 0
        total_sym = 0
        n_batches = 0

        for b_start in range(0, len(train_data), batch_size):
            batch_idx = perm[b_start:b_start + batch_size]
            batch = [train_data[i] for i in batch_idx]
            B_actual = len(batch)

            x0, mask = pad_batch_uncond(batch, 'z_original', device)
            sms_target, _ = pad_batch_uncond(batch, 'sms_params', device)

            t = torch.rand(B_actual, device=device).clamp(1e-4, 1 - 1e-4)
            noise = torch.randn_like(x0)
            x_t = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * noise

            # Unconditional: no piano_roll, no instrument
            global_cond, frame_cond = conditioner(t)
            v_pred = flow_net(x_t, global_cond, frame_cond)

            v_target = noise - x0
            sq_err = (v_pred - v_target).pow(2).mean(dim=-1)
            recon_loss = (sq_err * mask).sum() / mask.sum()

            sym_loss = torch.tensor(0.0, device=device)
            if sym_loss_weight > 0 and epoch >= 5:
                x0_hat = x_t - t.view(-1, 1, 1) * v_pred
                x0_hat_d = x0_hat.detach()
                sym_enc = bottleneck.encode(x0_hat_d)
                sms_err = (sym_enc['sms_params'] - sms_target).pow(2).mean(dim=-1)
                sym_loss = (sms_err * mask).sum() / mask.sum()

            loss = recon_loss + sym_loss_weight * sym_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            total_loss += loss.item() * B_actual
            total_recon += recon_loss.item() * B_actual
            total_sym += sym_loss.item() * B_actual
            n_batches += B_actual

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_recon = total_recon / max(n_batches, 1)
        avg_sym = total_sym / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'conditioner': conditioner.state_dict(),
                'flow_net': flow_net.state_dict(),
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'conditioned': False,
                'cond_dim': cond_dim,
            }, MODEL_PATH)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={avg_loss:.6f}  recon={avg_recon:.6f}  sym={avg_sym:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\n  Best loss: {best_loss:.6f}")
    print(f"  Saved to {MODEL_PATH}")

    return conditioner, flow_net


# ============================================================
# Generation
# ============================================================

def generate_sample(conditioner, flow_net, T_frames=64, steps=30,
                    device='cuda', piano_roll=None, amp=None,
                    group_id=None, subgroup_id=None):
    """Generate a z latent using flow-matching sampling with optional conditioning."""
    conditioner.eval()
    flow_net.eval()

    with torch.no_grad():
        x = torch.randn(1, T_frames, Z_DIM, device=device)

        dt = 1.0 / steps
        for i in range(steps, 0, -1):
            t = torch.full((1,), i * dt, device=device)
            global_cond, frame_cond = conditioner(
                t, piano_roll=piano_roll, amp=amp,
                group_id=group_id, subgroup_id=subgroup_id,
            )
            v_pred = flow_net(x, global_cond, frame_cond)
            x = x - dt * v_pred

    return x  # [1, T, 128]


# ============================================================
# Evaluation: Conditioned
# ============================================================

def evaluate_conditioned(device='cuda', manifest_path=MANIFEST_PATH, n_samples=5):
    """Generate conditioned samples, decompose symbolically, decode to audio."""
    print("=" * 60)
    print("EVALUATION: CONDITIONED SYMBOLIC GENERATION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    if not COND_MODEL_PATH.exists():
        print(f"No conditioned model at {COND_MODEL_PATH}. Run --train first.")
        return

    ckpt = torch.load(COND_MODEL_PATH, weights_only=False, map_location='cpu')
    cond_dim = ckpt.get('cond_dim', 128)

    conditioner = PerformanceConditioner(
        z_dim=Z_DIM, cond_dim=cond_dim,
        n_groups=N_GROUPS, n_subgroups=N_SUBGROUPS,
    ).to(device)
    flow_net = SymbolicFlowNet(
        z_dim=Z_DIM, hidden=384, n_layers=4, cond_dim=cond_dim,
    ).to(device)

    conditioner.load_state_dict(ckpt['conditioner'])
    flow_net.load_state_dict(ckpt['flow_net'])
    print(f"  Loaded model from epoch {ckpt['epoch']}, loss={ckpt['best_loss']:.6f}")

    # Load bottleneck
    bottleneck = SymbolicBottleneck(CODEC_PATH, TREE_PATH, device)

    # Load DCAE
    print("\n  Loading DCAE for audio decode...")
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_PATH,
        vocoder_checkpoint_path=VOCODER_PATH,
    ).to(device)
    dcae.eval()

    # Load test samples for conditioning (no dropout)
    print(f"\n  Loading test samples from {manifest_path}...")
    dataset = SymbolicDataset(
        manifest_path=manifest_path,
        window_slow=256,
        pr_dropout=0.0,
        amp_dropout=0.0,
        seed=99,
    )

    eval_dir = OUTPUT_DIR / "eval_conditioned"
    eval_dir.mkdir(parents=True, exist_ok=True)

    group2name = {i: g for i, g in enumerate(APPROVED_GROUPS)}

    print(f"\n  Generating {n_samples} conditioned samples...")
    gen_cos_sims = []

    for i in range(n_samples):
        # Get a real sample for conditioning
        sample = dataset[i * 100]  # spread across dataset
        latents = sample['latents'].unsqueeze(0).to(device)        # [1, 8, 16, T]
        piano_roll = sample['piano_roll'].unsqueeze(0).to(device)  # [1, 128, T]
        amp = sample['amp'].unsqueeze(0).to(device)                # [1, T]
        group_id = sample['group_id'].unsqueeze(0).to(device)
        subgroup_id = sample['subgroup_id'].unsqueeze(0).to(device)

        T_lat = latents.shape[-1]
        group_name = group2name.get(group_id.item(), "unknown")

        print(f"\n    Sample {i}: {group_name}, T={T_lat} frames")

        # Align conditioning
        if piano_roll.shape[-1] != T_lat:
            piano_roll = F.interpolate(piano_roll, size=T_lat, mode='nearest')
        if amp.shape[-1] != T_lat:
            amp = F.interpolate(amp.unsqueeze(1), size=T_lat, mode='nearest').squeeze(1)

        # Generate from same conditioning
        z_gen_flat = generate_sample(
            conditioner, flow_net, T_frames=T_lat, steps=50,
            device=device, piano_roll=piano_roll, amp=amp,
            group_id=group_id, subgroup_id=subgroup_id,
        )

        # Decompose symbolically
        sym = bottleneck.encode(z_gen_flat)
        z_recon_flat = bottleneck.decode(
            sym['sms_params'], sym['op_activations'], sym['op_params']
        )

        cos = F.cosine_similarity(z_gen_flat, z_recon_flat, dim=-1).mean().item()
        gen_cos_sims.append(cos)
        print(f"      Decomposition cos_sim: {cos:.4f}")

        # How close is generated z to the real z? (should be close if conditioning works)
        x0_real = z_4d_to_flat(latents)
        cos_vs_real = F.cosine_similarity(z_gen_flat, x0_real, dim=-1).mean().item()
        print(f"      vs real z cos_sim:     {cos_vs_real:.4f}")

        # Decode all to audio
        n_audio = int(T_lat * SAMPLE_RATE / 10.8)
        audio_lengths = torch.tensor([n_audio], device=device)

        # 1. Real z → audio (ground truth)
        sr, wavs = dcae.decode(latents, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
        torchaudio.save(str(eval_dir / f"sample_{i}_{group_name}_real.wav"),
                        wavs[0].float().cpu(), sr)

        # 2. Generated z → audio (raw)
        z_gen_4d = z_flat_to_4d(z_gen_flat)
        sr, wavs = dcae.decode(z_gen_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
        torchaudio.save(str(eval_dir / f"sample_{i}_{group_name}_gen_raw.wav"),
                        wavs[0].float().cpu(), sr)

        # 3. Generated z → symbolic → reconstructed → audio
        z_recon_4d = z_flat_to_4d(z_recon_flat)
        sr, wavs = dcae.decode(z_recon_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
        torchaudio.save(str(eval_dir / f"sample_{i}_{group_name}_gen_symbolic.wav"),
                        wavs[0].float().cpu(), sr)

    print(f"\n  Results:")
    print(f"    Generated z decomposition cos_sim: {np.mean(gen_cos_sims):.4f}")
    print(f"\n  Audio saved to {eval_dir}/")
    print(f"    sample_N_INST_real.wav          — ground truth (real z decoded)")
    print(f"    sample_N_INST_gen_raw.wav       — generated z decoded directly")
    print(f"    sample_N_INST_gen_symbolic.wav  — generated z → symbolic → decoded")


# ============================================================
# Evaluation: Unconditioned
# ============================================================

def evaluate_unconditioned(device='cuda'):
    """Generate unconditional samples, decode to audio."""
    print("=" * 60)
    print("EVALUATION: UNCONDITIONED SYMBOLIC GENERATION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"No model found at {MODEL_PATH}. Run --train --unconditioned first.")
        return

    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location='cpu')
    cond_dim = ckpt.get('cond_dim', 128)

    conditioner = PerformanceConditioner(z_dim=Z_DIM, cond_dim=cond_dim).to(device)
    flow_net = SymbolicFlowNet(z_dim=Z_DIM, hidden=384, n_layers=4, cond_dim=cond_dim).to(device)
    conditioner.load_state_dict(ckpt['conditioner'])
    flow_net.load_state_dict(ckpt['flow_net'])
    print(f"  Loaded model from epoch {ckpt['epoch']}, loss={ckpt['best_loss']:.6f}")

    bottleneck = SymbolicBottleneck(CODEC_PATH, TREE_PATH, device)

    print("\n  Loading DCAE for audio decode...")
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_PATH,
        vocoder_checkpoint_path=VOCODER_PATH,
    ).to(device)
    dcae.eval()

    eval_dir = OUTPUT_DIR / "eval_unconditioned"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Generating unconditioned samples...")
    gen_cos_sims = []

    for i in range(5):
        z_gen_flat = generate_sample(conditioner, flow_net, T_frames=64, steps=30, device=device)

        sym = bottleneck.encode(z_gen_flat)
        z_recon_flat = bottleneck.decode(
            sym['sms_params'], sym['op_activations'], sym['op_params']
        )

        cos = F.cosine_similarity(z_gen_flat, z_recon_flat, dim=-1).mean().item()
        gen_cos_sims.append(cos)
        print(f"    Gen sample {i}: decomposition cos_sim={cos:.4f}")

        z_gen_4d = z_flat_to_4d(z_gen_flat)
        z_recon_4d = z_flat_to_4d(z_recon_flat)
        T = z_gen_4d.shape[-1]
        n_audio = int(T * SAMPLE_RATE / 10.8)
        audio_lengths = torch.tensor([n_audio], device=device)

        sr, wavs = dcae.decode(z_gen_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
        torchaudio.save(str(eval_dir / f"gen_{i}_raw.wav"), wavs[0].float().cpu(), sr)

        sr, wavs = dcae.decode(z_recon_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
        torchaudio.save(str(eval_dir / f"gen_{i}_symbolic.wav"), wavs[0].float().cpu(), sr)

    print(f"\n  Decomposition cos_sim: {np.mean(gen_cos_sims):.4f}")
    print(f"  Audio saved to {eval_dir}/")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Symbolic Generation Trainer")
    parser.add_argument('--validate', action='store_true',
                        help='Validate symbolic decomposition quality on real latents')
    parser.add_argument('--train', action='store_true',
                        help='Train symbolic flow-matching model')
    parser.add_argument('--eval', action='store_true',
                        help='Generate samples and evaluate')
    parser.add_argument('--unconditioned', action='store_true',
                        help='Use unconditioned mode (baseline)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--prepare-data', action='store_true',
                        help='Pre-cache conditioned data from GCS to local disk')
    parser.add_argument('--max-samples', type=int, default=5000,
                        help='Max training samples')
    parser.add_argument('--manifest', type=str, default=MANIFEST_PATH,
                        help='Training manifest JSON path')
    parser.add_argument('--window-slow', type=int, default=256,
                        help='Window size in slow frames for conditioned training')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if not any([args.validate, args.prepare_data, args.train, args.eval]):
        print("No action specified. Use --validate, --prepare-data, --train, or --eval")
        parser.print_help()
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.validate:
        validate(device=args.device)

    if args.prepare_data:
        prepare_conditioned_data(
            manifest_path=args.manifest,
            max_samples=args.max_samples,
            window_slow=args.window_slow,
        )

    if args.train:
        if args.unconditioned:
            train_unconditioned(
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, device=args.device,
            )
        else:
            train_conditioned(
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, device=args.device,
                manifest_path=args.manifest,
                max_samples=args.max_samples,
                window_slow=args.window_slow,
            )

    if args.eval:
        if args.unconditioned:
            evaluate_unconditioned(device=args.device)
        else:
            evaluate_conditioned(
                device=args.device,
                manifest_path=args.manifest,
            )
