#!/usr/bin/env python3
"""Train neural parameter predictor for inverse synthesis.

1. Generate training data: random synth params → render audio → mel spectrogram
2. Train MelParamPredictor or MelParamPredictorLarge (mel + pitch → params + waveform)

Usage:
    python scripts/train_param_predictor.py [--n-samples 50000] [--epochs 200] [--large]
"""

import sys
sys.path.insert(0, 'scripts')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import os

from fast_dsp import (
    full_render, ALL_WF_FNS, SAMPLE_RATE, N_SAMPLES, FILTER_TYPES,
)
from param_predictor import (
    MelParamPredictor, MelParamPredictorLarge, compute_mel_spectrogram,
    PARAM_BOUNDS, WAVEFORM_NAMES, N_PARAMS, N_WAVEFORMS,
    FILTER_TYPE_NAMES,
)


# ============================================================
# Data generation
# ============================================================

# Pitches: 3 octaves, chromatic
PITCHES = [55, 58.3, 61.7, 65.4, 69.3, 73.4, 77.8, 82.4, 87.3, 92.5, 98, 103.8,
           110, 116.5, 123.5, 130.8, 138.6, 146.8, 155.6, 164.8, 174.6, 185, 196, 207.7,
           220, 233.1, 246.9, 261.6, 277.2, 293.7, 311.1, 329.6, 349.2, 370, 392, 415.3,
           440, 466.2, 493.9, 523.3, 554.4, 587.3, 622.3, 659.3, 698.5, 740, 784, 830.6]


def generate_random_params(realistic=True):
    """Generate random synth params within bounds.

    If realistic=True, bias toward musically common parameter ranges:
    - Fast attacks (exponential distribution biased toward small values)
    - Moderate decays
    - Reasonable sustain levels
    - Note-off times that make musical sense
    """
    params = np.zeros(N_PARAMS)

    if not realistic:
        for i, (lo, hi) in enumerate(PARAM_BOUNDS):
            if i in [0, 1]:
                params[i] = 10 ** np.random.uniform(np.log10(lo), np.log10(hi))
            else:
                params[i] = np.random.uniform(lo, hi)
    else:
        # Filter base/peak Hz (log-scale) — bias toward mid-range
        params[0] = 10 ** np.random.uniform(np.log10(50), np.log10(10000))   # filter_base
        params[1] = 10 ** np.random.uniform(np.log10(200), np.log10(18000))  # filter_peak

        # Resonance — mostly low, occasionally high
        params[2] = np.random.beta(1.5, 3.0) * 0.95  # skewed toward low

        # Filter ADSR — biased toward fast/snappy
        params[3] = np.random.exponential(0.05) + 0.001   # attack: mostly fast
        params[4] = np.random.exponential(0.15) + 0.001   # decay: moderate
        params[5] = np.random.beta(2, 3)                   # sustain: biased low-mid
        params[6] = np.random.exponential(0.15) + 0.001   # release: moderate
        params[7] = np.random.uniform(0.1, 2.5)            # noteoff

        # Amp ADSR — similar bias
        params[8] = np.random.exponential(0.03) + 0.001    # attack: very fast common
        params[9] = np.random.exponential(0.12) + 0.001    # decay
        params[10] = np.random.beta(2, 2)                   # sustain: centered
        params[11] = np.random.exponential(0.12) + 0.001   # release
        params[12] = np.random.uniform(0.1, 2.5)            # noteoff

        # Clip to bounds
        for i, (lo, hi) in enumerate(PARAM_BOUNDS):
            params[i] = np.clip(params[i], lo, hi)

    # Ensure filter_peak >= filter_base
    if params[1] < params[0]:
        params[0], params[1] = params[1], params[0]

    return params


def _generate_one_sample(args):
    """Generate a single training sample (for multiprocessing)."""
    i, include_filter_types, sr = args

    wf_names = [n for n in WAVEFORM_NAMES if n != 'noise']
    n_wf = len(wf_names)

    wf_idx = i % n_wf
    wf_name = wf_names[wf_idx]
    wf_id = WAVEFORM_NAMES.index(wf_name)

    # Random filter type
    if include_filter_types:
        r = np.random.random()
        if r < 0.6:
            ft_name = 'lowpass'
        elif r < 0.8:
            ft_name = 'highpass'
        else:
            ft_name = 'bandpass'
    else:
        ft_name = 'lowpass'
    ft_id = FILTER_TYPE_NAMES.index(ft_name)

    pitch = PITCHES[np.random.randint(len(PITCHES))]
    params = generate_random_params(realistic=(np.random.random() < 0.8))

    if ft_name == 'highpass':
        params[0] = max(params[0], 200)
        params[1] = max(params[1], params[0])

    wf_fn = ALL_WF_FNS[wf_name]
    waveform = wf_fn(pitch) if wf_name != 'noise' else wf_fn()

    try:
        audio = full_render(params, waveform, filter_type=ft_name)
    except Exception:
        return None

    if np.abs(audio).max() < 0.01:
        return None

    mel = compute_mel_spectrogram(audio, sr)

    norm_params = np.zeros(N_PARAMS)
    for j, (lo, hi) in enumerate(PARAM_BOUNDS):
        if j in [0, 1]:
            norm_params[j] = (np.log10(params[j]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        else:
            norm_params[j] = (params[j] - lo) / (hi - lo + 1e-8)
    norm_params = np.clip(norm_params, 0, 1)

    return (mel, norm_params, wf_id, pitch, ft_id)


def generate_dataset(n_samples=200000, sr=SAMPLE_RATE, n_samples_audio=N_SAMPLES,
                     include_filter_types=True, n_workers=32):
    """Generate training data. Uses threads for parallelism (shares JIT cache).

    Returns:
        mels: list of [1, N_MELS, T] tensors
        params: [N, 13] array
        wf_ids: [N] array (waveform index)
        pitches: [N] array (Hz)
        ft_ids: [N] array (filter type index)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"Generating {n_samples} training samples ({n_workers} threads)...")
    t_start = time.time()

    # Warm up numba JIT first
    _warmup_params = generate_random_params(realistic=False)
    _warmup_wf = ALL_WF_FNS['saw'](220)
    _ = full_render(_warmup_params, _warmup_wf)
    # Also warm up all filter types
    for ft in FILTER_TYPES:
        try:
            full_render(_warmup_params, _warmup_wf, filter_type=ft)
        except Exception:
            pass
    print("  JIT warmup done")

    all_mels = []
    all_params = []
    all_wf_ids = []
    all_pitches = []
    all_ft_ids = []

    # Numba releases GIL during computation so threads work well
    def gen_one(i):
        return _generate_one_sample((i, include_filter_types, sr))

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(gen_one, i) for i in range(n_samples)]
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result is not None:
                mel, norm_params, wf_id, pitch, ft_id = result
                all_mels.append(mel)
                all_params.append(norm_params)
                all_wf_ids.append(wf_id)
                all_pitches.append(pitch)
                all_ft_ids.append(ft_id)

            if done % 5000 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (n_samples - done) / rate
                print(f"  {done}/{n_samples} ({rate:.0f}/s, ETA {eta:.0f}s) "
                      f"[{len(all_mels)} valid]")

    elapsed = time.time() - t_start
    print(f"Generated {len(all_mels)} samples in {elapsed:.0f}s "
          f"({len(all_mels)/elapsed:.0f}/s)")

    return (all_mels, np.array(all_params), np.array(all_wf_ids),
            np.array(all_pitches), np.array(all_ft_ids))


class SynthDataset(Dataset):
    """PyTorch dataset for synth parameter prediction."""

    def __init__(self, mels, params, wf_ids, pitches, ft_ids=None):
        self.mels = mels
        self.params = torch.from_numpy(params).float()
        self.wf_ids = torch.from_numpy(wf_ids).long()
        self.pitches = torch.from_numpy(pitches).float()
        self.ft_ids = torch.from_numpy(ft_ids).long() if ft_ids is not None else None

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        mel = self.mels[idx]  # [1, N_MELS, T]
        params = self.params[idx]
        wf_id = self.wf_ids[idx]
        pitch = self.pitches[idx]
        ft_id = self.ft_ids[idx] if self.ft_ids is not None else torch.tensor(0)
        return mel, params, wf_id, pitch, ft_id


def collate_fn(batch):
    """Collate with padding to max mel length."""
    mels, params, wf_ids, pitches, ft_ids = zip(*batch)

    # Pad mels to same length
    max_T = max(m.shape[-1] for m in mels)
    padded = []
    for m in mels:
        if m.shape[-1] < max_T:
            pad_size = max_T - m.shape[-1]
            m = torch.nn.functional.pad(m, (0, pad_size))
        padded.append(m)

    mels = torch.stack(padded)  # [B, 1, N_MELS, T]
    params = torch.stack(params)
    wf_ids = torch.stack(wf_ids)
    pitches = torch.stack(pitches).unsqueeze(1)  # [B, 1]
    ft_ids = torch.stack(ft_ids)

    return mels, params, wf_ids, pitches, ft_ids


# ============================================================
# Training
# ============================================================

def train(model, train_loader, val_loader, epochs=200, lr=1e-3, device='cuda',
          train_filter_type=False):
    """Train the predictor model."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Per-parameter weights: ADSR timing params matter more perceptually
    # [fb, fp, res, fa, fd, fs, fr, fnoff, aa, ad, as, ar, anoff]
    param_weights = torch.tensor(
        [1.0, 1.0, 1.5,             # filter freq + resonance
         2.0, 1.5, 1.5, 1.5, 1.0,   # filter ADSR (attack matters most)
         2.5, 1.5, 1.5, 1.5, 1.0],  # amp ADSR (attack matters most)
        device=device)

    def weighted_param_loss(pred, target):
        diff = (pred - target) ** 2
        return (diff * param_weights).mean()

    wf_loss_fn = nn.CrossEntropyLoss()
    ft_loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_param_loss = 0
        train_wf_loss = 0
        train_wf_correct = 0
        train_ft_correct = 0
        train_total = 0

        for mels, params, wf_ids, pitches, ft_ids in train_loader:
            mels = mels.to(device)
            params = params.to(device)
            wf_ids = wf_ids.to(device)
            pitches = pitches.to(device)
            ft_ids = ft_ids.to(device)

            pred_params_raw, wf_logits = model(mels, pitches)

            # Normalize predicted params to [0,1] for loss
            pred_params = torch.zeros_like(pred_params_raw)
            for j in range(N_PARAMS):
                lo, hi = PARAM_BOUNDS[j]
                if j in [0, 1]:
                    pred_params[:, j] = (torch.log10(pred_params_raw[:, j]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
                else:
                    pred_params[:, j] = (pred_params_raw[:, j] - lo) / (hi - lo + 1e-8)
            pred_params = pred_params.clamp(0, 1)

            p_loss = weighted_param_loss(pred_params, params)
            w_loss = wf_loss_fn(wf_logits, wf_ids)
            loss = p_loss + 0.5 * w_loss

            # Filter type loss (if model supports it)
            if train_filter_type and hasattr(model, 'predict_filter_type'):
                ft_logits = model.predict_filter_type(mels, pitches)
                ft_loss = ft_loss_fn(ft_logits, ft_ids)
                loss = loss + 0.3 * ft_loss
                train_ft_correct += (ft_logits.argmax(1) == ft_ids).sum().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_param_loss += p_loss.item() * len(mels)
            train_wf_loss += w_loss.item() * len(mels)
            train_wf_correct += (wf_logits.argmax(1) == wf_ids).sum().item()
            train_total += len(mels)

        scheduler.step()

        # Validation
        model.eval()
        val_param_loss = 0
        val_wf_correct = 0
        val_ft_correct = 0
        val_total = 0

        with torch.inference_mode():
            for mels, params, wf_ids, pitches, ft_ids in val_loader:
                mels = mels.to(device)
                params = params.to(device)
                wf_ids = wf_ids.to(device)
                pitches = pitches.to(device)
                ft_ids = ft_ids.to(device)

                pred_params_raw, wf_logits = model(mels, pitches)

                pred_params = torch.zeros_like(pred_params_raw)
                for j in range(N_PARAMS):
                    lo, hi = PARAM_BOUNDS[j]
                    if j in [0, 1]:
                        pred_params[:, j] = (torch.log10(pred_params_raw[:, j]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
                    else:
                        pred_params[:, j] = (pred_params_raw[:, j] - lo) / (hi - lo + 1e-8)
                pred_params = pred_params.clamp(0, 1)

                val_param_loss += weighted_param_loss(pred_params, params).item() * len(mels)
                val_wf_correct += (wf_logits.argmax(1) == wf_ids).sum().item()
                val_total += len(mels)

                if train_filter_type and hasattr(model, 'predict_filter_type'):
                    ft_logits = model.predict_filter_type(mels, pitches)
                    val_ft_correct += (ft_logits.argmax(1) == ft_ids).sum().item()

        avg_train_p = train_param_loss / train_total
        avg_train_w = train_wf_loss / train_total
        train_wf_acc = train_wf_correct / train_total
        avg_val_p = val_param_loss / val_total
        val_wf_acc = val_wf_correct / val_total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            extra = ''
            if train_filter_type:
                train_ft_acc = train_ft_correct / train_total
                val_ft_acc = val_ft_correct / val_total if val_total > 0 else 0
                extra = f' ft_acc={train_ft_acc:.2f}/{val_ft_acc:.2f}'
            print(f"  Epoch {epoch+1:3d}: train_p={avg_train_p:.4f} train_w={avg_train_w:.4f} "
                  f"wf_acc={train_wf_acc:.2f} | val_p={avg_val_p:.4f} val_wf_acc={val_wf_acc:.2f}"
                  f"{extra}")

        if avg_val_p < best_val_loss:
            best_val_loss = avg_val_p
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=100000)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--large', action='store_true', help='Use large model (~2.5M params)')
    parser.add_argument('--workers', type=int, default=32, help='Workers for data generation')
    parser.add_argument('--save-path', default=None,
                        help='Save path (default: auto-named by model size)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Default save path based on model size
    if args.save_path is None:
        if args.large:
            args.save_path = 'scripts/param_predictor_weights_large.pt'
        else:
            args.save_path = 'scripts/param_predictor_weights.pt'

    # Generate data
    mels, params, wf_ids, pitches, ft_ids = generate_dataset(
        args.n_samples, include_filter_types=args.large, n_workers=args.workers
    )

    # Train/val split (90/10)
    n_val = max(len(mels) // 10, 100)
    n_train = len(mels) - n_val

    indices = np.random.RandomState(42).permutation(len(mels))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = SynthDataset(
        [mels[i] for i in train_idx], params[train_idx],
        wf_ids[train_idx], pitches[train_idx], ft_ids[train_idx]
    )
    val_ds = SynthDataset(
        [mels[i] for i in val_idx], params[val_idx],
        wf_ids[val_idx], pitches[val_idx], ft_ids[val_idx]
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=8, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            persistent_workers=True)

    # Create model
    if args.large:
        model = MelParamPredictorLarge()
    else:
        model = MelParamPredictor()
    n_params_model = sum(p.numel() for p in model.parameters())
    print(f"Model: {'Large' if args.large else 'Base'} ({n_params_model:,} params)")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    t_start = time.time()
    model = train(model, train_loader, val_loader, epochs=args.epochs,
                  lr=args.lr, device=device, train_filter_type=args.large)
    elapsed = time.time() - t_start
    print(f"Training complete in {elapsed:.0f}s")

    # Save
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved to {args.save_path}")

    # Quick evaluation on target patches
    print("\n--- Quick eval on target patches ---")
    from test_audio_domain import generate_target_audio, TARGET_PATCHES
    from fast_dsp import spectral_similarity

    model.eval()
    model = model.to(device)

    eval_patches = ['pluck_saw220', 'bright_lead', 'warm_pad', 'sine_sub_bass',
                    'acid_bass', 'supersaw_pad', 'pulse_lead', 'hpf_lead']
    for pname in eval_patches:
        pdef = TARGET_PATCHES[pname]
        if pdef.get('synth_type') == 'fm':
            continue

        target = generate_target_audio(pdef)
        pitch = pdef['pitch'] if pdef.get('pitch', 0) > 0 else 220

        mel = compute_mel_spectrogram(target).unsqueeze(0).to(device)
        pitch_t = torch.tensor([[pitch]], dtype=torch.float32, device=device)

        with torch.inference_mode():
            pred_params, wf_logits = model(mel, pitch_t)

        pred_params = pred_params[0].cpu().numpy()
        pred_wf = WAVEFORM_NAMES[wf_logits[0].argmax().item()]

        # Render predicted audio with correct filter type
        ft = pdef.get('filter_type', 'lowpass')
        wf_fn = ALL_WF_FNS.get(pred_wf, ALL_WF_FNS['saw'])
        waveform = wf_fn(pitch) if pred_wf != 'noise' else wf_fn()
        pred_audio = full_render(pred_params.tolist(), waveform, filter_type=ft)
        spec = spectral_similarity(pred_audio, target)

        true_wf = pdef.get('waveform', '?')
        wf_match = 'OK' if pred_wf == true_wf else 'MISS'
        print(f"  {pname:<18s} true_wf={true_wf:<10s} pred_wf={pred_wf:<10s} "
              f"spec={spec:.4f} {wf_match}")


if __name__ == '__main__':
    main()
