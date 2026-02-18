#!/usr/bin/env python3
"""
Program Synthesis: Discover the compositional tree of operations.

Goal: Find symbolic programs that explain z → sines
Not weights, but a tree of interpretable operations.

Example output:
  saxophone(z) =
    filter_harmonics(brightness=z[3],
      add_harmonics(f0=z[0], decay=z[1],
        envelope(attack=z[5], release=z[6])))

Approach:
1. Define atomic operations on sines
2. Use differentiable program search to find compositions
3. Extract symbolic tree from learned routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import importlib.util

# Direct import
spec = importlib.util.spec_from_file_location(
    'train_harmonic_ops',
    '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/training/train_harmonic_ops.py'
)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

HarmonicSMSDataset = train_module.HarmonicSMSDataset
collate_fn = train_module.collate_fn
sinkhorn_matching_loss = train_module.sinkhorn_matching_loss


# ============================================================================
# ATOMIC OPERATIONS ON SINES
# Each operation takes parameters (from z) and transforms sine representation
# ============================================================================

class AtomicOp(nn.Module):
    """Base class for atomic operations on sine representations."""
    name = "base"
    n_params = 0  # How many z dims this op needs

    def forward(self, sines, params):
        """
        Args:
            sines: dict with 'freqs' [B, T, N], 'amps' [B, T, N], 'phases' [B, T, N]
            params: [B, T, n_params] parameters from z
        Returns:
            transformed sines dict
        """
        raise NotImplementedError


class SetFundamental(AtomicOp):
    """Set the fundamental frequency from z."""
    name = "set_f0"
    n_params = 1

    def __init__(self):
        super().__init__()
        self.log_freq_min = np.log(20.0)
        self.log_freq_max = np.log(2000.0)  # f0 range

    def forward(self, sines, params):
        # params: [B, T, 1]
        f0_norm = torch.sigmoid(params[..., 0])
        log_f0 = self.log_freq_min + f0_norm * (self.log_freq_max - self.log_freq_min)
        f0 = torch.exp(log_f0)  # [B, T]

        # Set first sine to f0, rest are harmonics
        B, T, N = sines['freqs'].shape
        harmonics = torch.arange(1, N + 1, device=f0.device).float()
        sines['freqs'] = f0.unsqueeze(-1) * harmonics  # [B, T, N]
        return sines


class AddHarmonics(AtomicOp):
    """Add harmonic series with decay rate from z."""
    name = "add_harmonics"
    n_params = 1  # decay rate

    def forward(self, sines, params):
        # params[..., 0] = harmonic decay rate
        decay = torch.sigmoid(params[..., 0]) * 2 + 0.5  # [0.5, 2.5]

        B, T, N = sines['amps'].shape
        harmonics = torch.arange(1, N + 1, device=decay.device).float()

        # Amplitude decays as 1/n^decay
        decay_factors = 1.0 / (harmonics ** decay.unsqueeze(-1))  # [B, T, N]
        sines['amps'] = sines['amps'] * decay_factors
        return sines


class FilterHarmonics(AtomicOp):
    """Low/high pass filter on harmonics (brightness control)."""
    name = "filter"
    n_params = 2  # cutoff, resonance

    def forward(self, sines, params):
        cutoff = torch.sigmoid(params[..., 0])  # [0, 1] normalized cutoff
        resonance = torch.sigmoid(params[..., 1]) * 0.9  # [0, 0.9]

        B, T, N = sines['freqs'].shape

        # Normalized frequency position
        freq_norm = sines['freqs'] / 8000.0  # Normalize to [0, 1] range

        # Simple lowpass response
        cutoff_expanded = cutoff.unsqueeze(-1)
        distance = (freq_norm - cutoff_expanded).clamp(min=0)
        response = torch.exp(-distance * 10)  # Smooth rolloff

        # Resonance boost near cutoff
        resonance_expanded = resonance.unsqueeze(-1)
        near_cutoff = torch.exp(-((freq_norm - cutoff_expanded) ** 2) * 50)
        response = response + resonance_expanded * near_cutoff

        sines['amps'] = sines['amps'] * response.clamp(0, 1)
        return sines


class ApplyEnvelope(AtomicOp):
    """Apply amplitude envelope (attack/release shape)."""
    name = "envelope"
    n_params = 2  # attack, release

    def forward(self, sines, params):
        attack = torch.sigmoid(params[..., 0]) * 0.5  # [0, 0.5] of total time
        release = torch.sigmoid(params[..., 1]) * 0.5

        B, T, N = sines['amps'].shape

        # Time position normalized [0, 1]
        t = torch.linspace(0, 1, T, device=params.device)
        t = t.unsqueeze(0).expand(B, -1)  # [B, T]

        # Attack ramp
        attack_env = (t / attack.mean(dim=1, keepdim=True).clamp(min=0.01)).clamp(0, 1)

        # Release ramp
        release_start = 1.0 - release.mean(dim=1, keepdim=True).clamp(min=0.01)
        release_env = ((1.0 - t) / (1.0 - release_start)).clamp(0, 1)

        envelope = attack_env * release_env  # [B, T]
        sines['amps'] = sines['amps'] * envelope.unsqueeze(-1)
        return sines


class FrequencyVibrato(AtomicOp):
    """Add vibrato (frequency modulation)."""
    name = "vibrato"
    n_params = 2  # rate, depth

    def forward(self, sines, params):
        rate = torch.sigmoid(params[..., 0]) * 10 + 2  # [2, 12] Hz
        depth = torch.sigmoid(params[..., 1]) * 0.05  # [0, 5%] of freq

        B, T, N = sines['freqs'].shape

        # Time in seconds (assume ~11 fps)
        t = torch.arange(T, device=params.device).float() / 11.0
        t = t.unsqueeze(0).expand(B, -1)  # [B, T]

        # Sinusoidal modulation
        mod = torch.sin(2 * np.pi * rate.mean(dim=1, keepdim=True) * t)  # [B, T]
        mod = 1.0 + depth.mean(dim=1, keepdim=True) * mod

        sines['freqs'] = sines['freqs'] * mod.unsqueeze(-1)
        return sines


class AddBreathNoise(AtomicOp):
    """Add breath/air noise component."""
    name = "breath"
    n_params = 2  # bandwidth, amount

    def forward(self, sines, params):
        # This adds noise to amplitudes (simulating breath)
        bandwidth = torch.sigmoid(params[..., 0])
        amount = torch.sigmoid(params[..., 1]) * 0.3

        B, T, N = sines['amps'].shape

        # Random amplitude modulation (breath noise)
        noise = torch.randn(B, T, 1, device=params.device) * amount.unsqueeze(-1)
        sines['amps'] = (sines['amps'] + noise).clamp(0, 1)
        return sines


class ScaleAmplitude(AtomicOp):
    """Overall amplitude scaling."""
    name = "amplitude"
    n_params = 1

    def forward(self, sines, params):
        scale = torch.sigmoid(params[..., 0])  # [0, 1]
        sines['amps'] = sines['amps'] * scale.unsqueeze(-1)
        return sines


class OddEvenBalance(AtomicOp):
    """Balance between odd and even harmonics (instrument character)."""
    name = "odd_even"
    n_params = 1  # balance: 0=only odd, 1=only even, 0.5=equal

    def forward(self, sines, params):
        balance = torch.sigmoid(params[..., 0])  # [0, 1]

        B, T, N = sines['amps'].shape
        harmonics = torch.arange(1, N + 1, device=params.device)

        is_odd = (harmonics % 2 == 1).float()
        is_even = 1.0 - is_odd

        # Weight odd vs even
        balance_expanded = balance.unsqueeze(-1)
        weights = is_odd * (1 - balance_expanded) + is_even * balance_expanded

        sines['amps'] = sines['amps'] * weights
        return sines


# ============================================================================
# DIFFERENTIABLE PROGRAM TREE
# Learn which operations to apply and in what order
# ============================================================================

class OperationLibrary(nn.Module):
    """Library of all atomic operations."""

    def __init__(self):
        super().__init__()
        self.ops = nn.ModuleList([
            SetFundamental(),
            AddHarmonics(),
            FilterHarmonics(),
            ApplyEnvelope(),
            FrequencyVibrato(),
            AddBreathNoise(),
            ScaleAmplitude(),
            OddEvenBalance(),
        ])
        self.op_names = [op.name for op in self.ops]
        self.n_ops = len(self.ops)

        # Total params needed
        self.total_params = sum(op.n_params for op in self.ops)
        self.param_offsets = []
        offset = 0
        for op in self.ops:
            self.param_offsets.append(offset)
            offset += op.n_params


class DifferentiableProgramTree(nn.Module):
    """
    Learn a tree of operations via soft routing.

    The model learns:
    1. Which z dims feed into which operations (sparse selection)
    2. Which operations to apply (soft gating)
    3. Order of operations (learned sequential composition)
    """

    def __init__(self, n_sines=64, max_depth=4, hidden_dim=64):
        super().__init__()
        self.n_sines = n_sines
        self.max_depth = max_depth

        self.op_lib = OperationLibrary()
        n_ops = self.op_lib.n_ops

        # Learn which z dims feed each operation
        # Sparse: each op uses k dims
        self.k_dims = 8
        self.dim_selectors = nn.ParameterList([
            nn.Parameter(torch.randn(128) * 0.1)
            for _ in range(n_ops)
        ])

        # Small projector from selected z dims to op params
        self.param_projectors = nn.ModuleList([
            nn.Linear(self.k_dims, op.n_params) if op.n_params > 0 else None
            for op in self.op_lib.ops
        ])

        # Learn operation ordering/gating at each depth
        # gate[depth, op] = probability of applying op at this depth
        self.depth_gates = nn.Parameter(torch.zeros(max_depth, n_ops))

        # Temperature for gumbel-softmax
        self.tau = 1.0

    def get_selected_z(self, z, op_idx):
        """Get the k dims that op_idx uses from z."""
        logits = self.dim_selectors[op_idx]

        # Hard top-k selection
        _, topk_idx = logits.topk(self.k_dims)

        # Gather [B, T, k]
        B, T, _ = z.shape
        topk_idx_exp = topk_idx.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        z_selected = z.gather(-1, topk_idx_exp)

        return z_selected, topk_idx

    def forward(self, z):
        """
        z: [B, 8, 16, T]
        Returns: freqs [B, T, n_sines], amps [B, T, n_sines]
        """
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        # Initialize sines (will be set by operations)
        sines = {
            'freqs': torch.ones(B, T, self.n_sines, device=z.device) * 440,
            'amps': torch.ones(B, T, self.n_sines, device=z.device) * 0.5,
            'phases': torch.zeros(B, T, self.n_sines, device=z.device),
        }

        # Apply operations in learned order
        for depth in range(self.max_depth):
            # Soft gate: which ops to apply at this depth
            gate_logits = self.depth_gates[depth]

            if self.training:
                # Gumbel-softmax for differentiable selection
                gate = F.gumbel_softmax(gate_logits, tau=self.tau, hard=True)
            else:
                # Hard selection at inference
                gate = F.one_hot(gate_logits.argmax(), len(gate_logits)).float()

            # Apply weighted combination of ops
            new_sines = {k: torch.zeros_like(v) for k, v in sines.items()}

            for op_idx, op in enumerate(self.op_lib.ops):
                if gate[op_idx] < 0.01:
                    continue

                # Get params for this op from selected z dims
                z_selected, _ = self.get_selected_z(z_flat, op_idx)

                if op.n_params > 0:
                    params = self.param_projectors[op_idx](z_selected)
                else:
                    params = z_selected[..., :1]  # Dummy

                # Apply op
                op_sines = {k: v.clone() for k, v in sines.items()}
                op_sines = op(op_sines, params)

                # Weighted accumulate
                for k in new_sines:
                    new_sines[k] = new_sines[k] + gate[op_idx] * op_sines[k]

            sines = new_sines

        return sines['freqs'], sines['amps']

    def extract_program(self):
        """Extract the learned program as a symbolic tree."""
        program = []

        for depth in range(self.max_depth):
            gate_logits = self.depth_gates[depth]
            op_idx = gate_logits.argmax().item()
            op_name = self.op_lib.op_names[op_idx]

            # Which z dims does this op use?
            _, topk_idx = self.dim_selectors[op_idx].topk(self.k_dims)
            z_dims = topk_idx.cpu().numpy().tolist()

            program.append({
                'depth': depth,
                'operation': op_name,
                'z_dims': z_dims,
                'gate_value': gate_logits[op_idx].item(),
            })

        return program


class FullyDifferentiableTree(nn.Module):
    """
    Alternative: All operations run in parallel, tree structure learned via attention.
    """

    def __init__(self, n_sines=64, hidden_dim=128):
        super().__init__()
        self.n_sines = n_sines
        self.op_lib = OperationLibrary()

        # Encoder: z → hidden
        self.encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-operation parameter extractors
        self.op_param_extractors = nn.ModuleList([
            nn.Linear(hidden_dim, op.n_params) if op.n_params > 0 else None
            for op in self.op_lib.ops
        ])

        # Learn which operations to use (sparse selection)
        self.op_selector = nn.Linear(hidden_dim, len(self.op_lib.ops))

        # Final refinement
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim + n_sines * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_sines * 2),
        )

        self.log_freq_min = np.log(20.0)
        self.log_freq_max = np.log(8000.0)

    def forward(self, z):
        B, C, H, T = z.shape
        z_flat = z.reshape(B, C * H, T).permute(0, 2, 1)  # [B, T, 128]

        # Encode
        h = self.encoder(z_flat)  # [B, T, hidden]

        # Select operations
        op_logits = self.op_selector(h)  # [B, T, n_ops]
        op_weights = F.softmax(op_logits, dim=-1)

        # Initialize sines
        sines = {
            'freqs': torch.ones(B, T, self.n_sines, device=z.device) * 440,
            'amps': torch.ones(B, T, self.n_sines, device=z.device) * 0.5,
            'phases': torch.zeros(B, T, self.n_sines, device=z.device),
        }

        # Apply all operations weighted
        all_freqs = []
        all_amps = []

        for op_idx, op in enumerate(self.op_lib.ops):
            if self.op_param_extractors[op_idx] is not None:
                params = self.op_param_extractors[op_idx](h)
            else:
                params = h[..., :1]

            op_sines = {k: v.clone() for k, v in sines.items()}
            op_sines = op(op_sines, params)

            weight = op_weights[..., op_idx:op_idx+1]  # [B, T, 1]
            all_freqs.append(op_sines['freqs'] * weight)
            all_amps.append(op_sines['amps'] * weight)

        freqs = sum(all_freqs)
        amps = sum(all_amps)

        # Refinement pass
        combined = torch.cat([h, freqs, amps], dim=-1)
        delta = self.refine(combined)

        freq_delta = delta[..., :self.n_sines]
        amp_delta = delta[..., self.n_sines:]

        # Apply deltas
        freq_norm = torch.sigmoid(freq_delta)
        log_freq = self.log_freq_min + freq_norm * (self.log_freq_max - self.log_freq_min)
        freqs = torch.exp(log_freq)

        amps = torch.sigmoid(amp_delta)

        return freqs, amps, op_weights

    def extract_program(self, z_sample):
        """Extract which operations are most used for a sample."""
        with torch.no_grad():
            _, _, op_weights = self.forward(z_sample)

        mean_weights = op_weights.mean(dim=(0, 1))  # Average over batch and time

        program = []
        for op_idx, weight in enumerate(mean_weights):
            program.append({
                'operation': self.op_lib.op_names[op_idx],
                'weight': weight.item(),
            })

        program.sort(key=lambda x: -x['weight'])
        return program


def train_and_eval(model, dataloader, n_epochs=100, lr=1e-3, device='cuda'):
    """Train model and return final loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_freq_loss = float('inf')

    for epoch in range(n_epochs):
        if hasattr(model, 'tau'):
            model.tau = max(0.1, 1.0 - 0.9 * epoch / n_epochs)

        total_loss = 0
        total_freq_loss = 0
        n_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            z = batch['latent'].to(device)
            target_freqs = batch['freqs'].to(device)
            target_amps = batch['amps'].to(device)

            if isinstance(model, FullyDifferentiableTree):
                pred_freqs, pred_amps, _ = model(z)
            else:
                pred_freqs, pred_amps = model(z)

            loss, metrics = sinkhorn_matching_loss(
                pred_freqs, pred_amps,
                target_freqs, target_amps,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_freq_loss += metrics['freq_loss']
            n_batches += 1

        avg_freq_loss = total_freq_loss / n_batches
        if avg_freq_loss < best_freq_loss:
            best_freq_loss = avg_freq_loss

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/n_batches:.4f}, freq={avg_freq_loss:.4f}")

    return best_freq_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sines', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip_drums', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    dataset = HarmonicSMSDataset(
        sms_manifest_path='/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        max_samples=args.max_samples,
        skip_drums=args.skip_drums,
        n_sines=args.n_sines,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    # Test 1: Differentiable Program Tree
    print("\n" + "=" * 70)
    print("TEST 1: Differentiable Program Tree")
    print("Learn operation sequence: depth → which op to apply")
    print("=" * 70)

    model1 = DifferentiableProgramTree(n_sines=args.n_sines, max_depth=4)
    n_params = sum(p.numel() for p in model1.parameters())
    print(f"Params: {n_params:,}")

    freq_loss1 = train_and_eval(model1, dataloader, n_epochs=args.epochs, lr=args.lr, device=device)

    print("\nLearned program:")
    program = model1.extract_program()
    for step in program:
        print(f"  Depth {step['depth']}: {step['operation']} using z[{step['z_dims'][:4]}...]")

    # Test 2: Fully Differentiable (parallel ops)
    print("\n" + "=" * 70)
    print("TEST 2: Parallel Operations with Learned Weighting")
    print("All ops run, learn which to weight highly")
    print("=" * 70)

    model2 = FullyDifferentiableTree(n_sines=args.n_sines, hidden_dim=128)
    n_params = sum(p.numel() for p in model2.parameters())
    print(f"Params: {n_params:,}")

    freq_loss2 = train_and_eval(model2, dataloader, n_epochs=args.epochs, lr=args.lr, device=device)

    # Get a sample to analyze
    sample_batch = next(iter(dataloader))
    z_sample = sample_batch['latent'][:1].to(device)

    print("\nLearned operation weights (for sample):")
    program = model2.extract_program(z_sample)
    for step in program[:5]:
        print(f"  {step['operation']:15s}: {step['weight']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nFreq loss (lower is better):")
    print(f"  Program Tree (sequential): {freq_loss1:.4f}")
    print(f"  Parallel Ops (weighted):   {freq_loss2:.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if min(freq_loss1, freq_loss2) < 0.08:
        print("""
✓ COMPOSITIONAL STRUCTURE FOUND
  The operation tree achieves reasonable accuracy.

  The learned program shows which atomic operations
  (set_f0, add_harmonics, filter, envelope, etc.)
  compose to explain z → sines.

  This IS the interpretable tree of operations.
""")
    else:
        print("""
~ LIMITED COMPOSITIONAL STRUCTURE
  Our predefined operations don't fully capture
  the decoder's computation.

  Possible reasons:
  1. Need more atomic operations
  2. Need deeper composition
  3. Decoder learned different primitives

  Next: Try symbolic regression (PySR) to discover
  what operations the decoder actually uses.
""")


if __name__ == "__main__":
    main()
