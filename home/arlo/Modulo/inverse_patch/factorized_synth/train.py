#!/usr/bin/env python3
"""
Training script for factorized synth model.

Losses:
1. Reconstruction (mel MSE)
2. Flow likelihood (NLL)
3. Disentanglement (adversarial classifiers)
4. Cycle consistency
"""

import sys
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

sys.stdout.reconfigure(line_buffering=True)

# Memory constraints
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation
torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul with less memory

from dataset import SynthDataset, generate_dataset
from model import FactorizedSynthModel


def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_losses(model: FactorizedSynthModel, batch: Dict, epoch: int) -> Dict[str, torch.Tensor]:
    """Compute all training losses."""
    mel = batch['mel']
    cutoff_norm = batch['cutoff_norm']
    attack_norm = batch['attack_norm']
    cutoff_label = batch['cutoff_label']
    attack_label = batch['attack_label']

    # Forward pass
    mel_recon, z_filter, z_envelope = model(mel)

    losses = {}

    # 1. Reconstruction loss
    losses['recon'] = F.mse_loss(mel_recon, mel)

    # 2. Flow likelihood losses
    losses['flow_filter'] = -model.flow_filter.log_prob(z_filter, cutoff_norm).mean()
    losses['flow_env'] = -model.flow_envelope.log_prob(z_envelope, attack_norm).mean()

    # 3. Disentanglement losses (adversarial)
    # z_envelope should NOT predict cutoff
    cutoff_pred_from_env = model.aux_cutoff_from_env(z_envelope, use_grl=True)
    losses['disent_cutoff'] = F.cross_entropy(cutoff_pred_from_env, cutoff_label)

    # z_filter should NOT predict attack
    attack_pred_from_filter = model.aux_attack_from_filter(z_filter, use_grl=True)
    losses['disent_attack'] = F.cross_entropy(attack_pred_from_filter, attack_label)

    # 4. Cycle consistency (flow forward then inverse should recover z)
    base_filter, _ = model.flow_filter(z_filter)
    z_filter_recon = model.flow_filter.inverse(base_filter)
    losses['cycle_filter'] = F.mse_loss(z_filter_recon, z_filter.detach())

    base_env, _ = model.flow_envelope(z_envelope)
    z_env_recon = model.flow_envelope.inverse(base_env)
    losses['cycle_env'] = F.mse_loss(z_env_recon, z_envelope.detach())

    # Total loss with weights
    # Ramp up disentanglement weight over training
    disent_weight = min(1.0, epoch / 200)

    losses['total'] = (
        1.0 * losses['recon'] +
        0.1 * losses['flow_filter'] +
        0.1 * losses['flow_env'] +
        disent_weight * 0.5 * losses['disent_cutoff'] +
        disent_weight * 0.5 * losses['disent_attack'] +
        0.1 * losses['cycle_filter'] +
        0.1 * losses['cycle_env']
    )

    return losses


def train(n_epochs: int = 1000, lr: float = 1e-3, device: str = 'cuda'):
    """Train the factorized synth model."""

    print("="*60, flush=True)
    print("Training Factorized Synth Model", flush=True)
    print("="*60, flush=True)

    # Clear memory before starting
    clear_memory()

    # Generate dataset
    print("\nGenerating dataset...", flush=True)
    data = generate_dataset()
    dataset = SynthDataset(data)
    print(f"Dataset size: {len(dataset)}", flush=True)

    # Get mel shape from first sample
    sample = dataset[0]
    n_mels = sample['mel'].shape[0]
    n_frames = sample['mel'].shape[1]
    print(f"Mel shape: [{n_mels}, {n_frames}]", flush=True)

    # Use smaller batch size to reduce memory
    batch_size = min(8, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=False)
    print(f"Batch size: {batch_size}", flush=True)

    # Create model
    model = FactorizedSynthModel(n_mels=n_mels, n_frames=n_frames, latent_dim=16).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Optimizer with lower memory footprint
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Mixed precision scaler for memory efficiency
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # Training loop
    print("\nTraining...", flush=True)

    best_loss = float('inf')
    output_dir = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/factorized_synth/checkpoints")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)  # More memory efficient

            # Mixed precision forward pass
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                losses = compute_losses(model, batch, epoch)

            # Backward with scaler
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses['total'].item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Clear memory periodically
        if epoch % 50 == 0:
            clear_memory()

        # Logging
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            loss_str = " | ".join([f"{k}={v.item():.4f}" for k, v in losses.items() if k != 'total'])
            print(f"Epoch {epoch:4d}: total={avg_loss:.4f} | {loss_str}", flush=True)

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), str(output_dir / "best_model.pt"))

    # Save final
    torch.save(model.state_dict(), str(output_dir / "final_model.pt"))
    print(f"\nTraining complete! Best loss: {best_loss:.4f}", flush=True)
    print(f"Models saved to: {output_dir}", flush=True)

    return model


def evaluate_disentanglement(model: FactorizedSynthModel, device: str = 'cuda'):
    """Evaluate how well the model disentangles factors."""

    print("\n" + "="*60, flush=True)
    print("Evaluating Disentanglement", flush=True)
    print("="*60, flush=True)

    clear_memory()
    model.eval()

    # Generate dataset
    data = generate_dataset()
    dataset = SynthDataset(data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            mel = batch['mel']
            cutoff_label = batch['cutoff_label']
            attack_label = batch['attack_label']

            # Encode
            z_filter, z_envelope = model.encode(mel)

            # Test: can z_filter predict cutoff? (should be YES)
            cutoff_from_filter = model.aux_cutoff_from_env.classifier(z_filter)  # Without GRL
            cutoff_acc_correct = (cutoff_from_filter.argmax(dim=1) == cutoff_label).float().mean()

            # Test: can z_envelope predict attack? (should be YES)
            attack_from_env = model.aux_attack_from_filter.classifier(z_envelope)  # Without GRL
            attack_acc_correct = (attack_from_env.argmax(dim=1) == attack_label).float().mean()

            # Test: can z_envelope predict cutoff? (should be NO - disentangled)
            cutoff_from_env = model.aux_cutoff_from_env.classifier(z_envelope)  # Without GRL
            cutoff_acc_wrong = (cutoff_from_env.argmax(dim=1) == cutoff_label).float().mean()

            # Test: can z_filter predict attack? (should be NO - disentangled)
            attack_from_filter = model.aux_attack_from_filter.classifier(z_filter)  # Without GRL
            attack_acc_wrong = (attack_from_filter.argmax(dim=1) == attack_label).float().mean()

            print(f"z_filter predicts cutoff (should be high): {cutoff_acc_correct:.2%}", flush=True)
            print(f"z_envelope predicts attack (should be high): {attack_acc_correct:.2%}", flush=True)
            print(f"z_envelope predicts cutoff (should be ~25%): {cutoff_acc_wrong:.2%}", flush=True)
            print(f"z_filter predicts attack (should be ~25%): {attack_acc_wrong:.2%}", flush=True)


if __name__ == "__main__":
    # Limit CUDA memory to prevent OOM crashes
    if torch.cuda.is_available():
        # Reserve only 80% of GPU memory
        torch.cuda.set_per_process_memory_fraction(0.8)
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)

    try:
        model = train(n_epochs=1000, lr=1e-3, device=device)
        evaluate_disentanglement(model, device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\nOOM Error! Clearing cache and retrying with CPU...", flush=True)
            clear_memory()
            model = train(n_epochs=1000, lr=1e-3, device='cpu')
            evaluate_disentanglement(model, 'cpu')
        else:
            raise
