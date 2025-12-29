#!/usr/bin/env python3
"""
Paired Distribution Matching Training for Formant Correction

Uses exact pairs (shifted, natural) but with distribution matching loss
instead of L1. Also tries larger model and better architecture.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


class ResidualBlock1D(nn.Module):
    """1D Residual block with optional attention."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, use_attention: bool = False):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
            self.attn_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act(x + residual)

        if self.use_attention:
            # x: [B, C, T] -> [B, T, C] for attention
            x_t = x.transpose(1, 2)
            attn_out, _ = self.attn(x_t, x_t, x_t)
            x_t = self.attn_norm(x_t + attn_out)
            x = x_t.transpose(1, 2)

        return x


class FormantCorrectorV2(nn.Module):
    """
    Improved formant corrector with:
    - Larger capacity
    - Attention layers
    - Stronger direction conditioning
    - Direct output option
    """

    def __init__(
        self,
        latent_channels: int = 8,
        hidden_channels: int = 256,  # Larger
        num_blocks: int = 8,  # More blocks
        kernel_size: int = 7,  # Larger receptive field
        use_attention: bool = True,
        direct_output: bool = True,  # No residual - learn full mapping
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.direct_output = direct_output

        # Direction embedding - larger and more prominent
        self.direction_embed = nn.Embedding(2, hidden_channels)

        # Input projection with direction
        self.input_proj = nn.Sequential(
            nn.Conv1d(latent_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
        )

        # Residual blocks with attention on later layers
        self.blocks = nn.ModuleList([
            ResidualBlock1D(
                hidden_channels,
                kernel_size,
                dilation=2 ** (i % 4),  # Cycle dilations
                use_attention=(use_attention and i >= num_blocks // 2)
            )
            for i in range(num_blocks)
        ])

        # Direction modulation for each block (FiLM-style)
        self.direction_scales = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(num_blocks)
        ])
        self.direction_shifts = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels // 2, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels // 2, latent_channels, 1),
        )

        # Initialize output near identity for stability
        if not direct_output:
            nn.init.zeros_(self.output_proj[-1].weight)
            nn.init.zeros_(self.output_proj[-1].bias)

        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        self.input_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        latent: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, T = latent.shape

        # Get direction embedding
        dir_emb = self.direction_embed(direction)  # [B, hidden]

        # Flatten H into batch: [B*H, C, T]
        x = latent.permute(0, 2, 1, 3).reshape(B * H, C, T)

        # Input projection
        x = self.input_proj(x)

        # Expand direction for all H slices
        dir_emb_expanded = dir_emb.unsqueeze(1).expand(-1, H, -1).reshape(B * H, -1)

        # Process through blocks with FiLM conditioning
        for i, block in enumerate(self.blocks):
            # Apply FiLM: scale and shift based on direction
            scale = self.direction_scales[i](dir_emb_expanded).unsqueeze(-1)  # [B*H, hidden, 1]
            shift = self.direction_shifts[i](dir_emb_expanded).unsqueeze(-1)

            x = block(x)
            x = x * (1 + 0.1 * torch.tanh(scale)) + 0.1 * shift

        # Output projection
        output = self.output_proj(x)
        output = output.reshape(B, H, C, T).permute(0, 2, 1, 3)

        if self.direct_output:
            return output
        else:
            return self.input_scale * latent + self.residual_scale * output


class PairedDataset(Dataset):
    """Dataset using exact pairs."""

    def __init__(
        self,
        manifest_path: str,
        window_frames: int = 64,
        samples_per_epoch: int = 10000,
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch

        with open(manifest_path) as f:
            data = json.load(f)

        self.pairs = data['pairs']
        print(f"Loaded {len(self.pairs)} pairs")

        self._cache = {}

    def _load_pair(self, path):
        if path not in self._cache:
            try:
                data = torch.load(path, map_location='cpu')
                self._cache[path] = data
                if len(self._cache) > 500:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
            except:
                return None
        return self._cache.get(path)

    def _crop_window(self, latent1, latent2):
        """Aligned crop from both latents."""
        T = min(latent1.shape[-1], latent2.shape[-1])
        if T <= self.window_frames:
            pad = self.window_frames - T
            latent1 = F.pad(latent1, (0, pad))
            latent2 = F.pad(latent2, (0, pad))
            return latent1, latent2

        start = random.randint(0, T - self.window_frames)
        return latent1[..., start:start + self.window_frames], latent2[..., start:start + self.window_frames]

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        entry = random.choice(self.pairs)
        data = self._load_pair(entry['pair_path'])

        if data is None:
            return {
                'input': torch.zeros(8, 16, self.window_frames),
                'target': torch.zeros(8, 16, self.window_frames),
                'direction': 0,
                'valid': False,
            }

        shifted = data['shifted']
        natural = data['natural']
        direction = entry['direction']

        shifted_crop, natural_crop = self._crop_window(shifted, natural)

        return {
            'input': shifted_crop,
            'target': natural_crop,
            'direction': direction,
            'valid': True,
        }


class PairedDistMatchTrainer:
    """Train with paired data but distribution matching loss."""

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        window_frames: int = 64,
        samples_per_epoch: int = 10000,
        hidden_channels: int = 256,
        num_blocks: int = 8,
        use_attention: bool = True,
        direct_output: bool = True,
        use_amp: bool = True,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.num_epochs = num_epochs

        # Model
        self.model = FormantCorrectorV2(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            use_attention=use_attention,
            direct_output=direct_output,
        ).to(self.device)

        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Direct output: {direct_output}, Attention: {use_attention}")

        # Dataset
        self.dataset = PairedDataset(
            manifest_path=manifest_path,
            window_frames=window_frames,
            samples_per_epoch=samples_per_epoch,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        self.scaler = GradScaler('cuda') if use_amp else None

        self.log_file = self.output_dir / "training.log"
        self.best_loss = float('inf')

    def combined_loss(self, output, target, input_latent):
        """
        Combined loss: L1 for content + distribution matching for statistics.
        """
        B, C, H, T = output.shape

        # 1. L1 loss (content alignment) - weighted less heavily
        l1_loss = F.l1_loss(output, target)

        # 2. Per-channel mean/std matching
        out_mean = output.mean(dim=(2, 3))
        out_std = output.std(dim=(2, 3))
        tgt_mean = target.mean(dim=(2, 3))
        tgt_std = target.std(dim=(2, 3))

        mean_loss = F.mse_loss(out_mean, tgt_mean)
        std_loss = F.mse_loss(out_std, tgt_std)

        # 3. Per-frequency-band energy matching
        band_loss = 0
        for start_frac, end_frac, weight in [(0, 0.25, 1.0), (0.25, 0.5, 1.5), (0.5, 0.75, 2.0), (0.75, 1.0, 3.0)]:
            h_start = int(H * start_frac)
            h_end = int(H * end_frac)
            out_energy = output[:, :, h_start:h_end, :].pow(2).mean(dim=(2, 3))
            tgt_energy = target[:, :, h_start:h_end, :].pow(2).mean(dim=(2, 3))
            band_loss = band_loss + weight * F.mse_loss(out_energy, tgt_energy)

        # 4. Spectral difference - compare deltas
        # The delta from input→output should match input→target
        delta_pred = output - input_latent
        delta_target = target - input_latent
        delta_loss = F.mse_loss(delta_pred, delta_target)

        # 5. Cosine similarity per-frame
        out_flat = output.reshape(B, -1, T)  # [B, C*H, T]
        tgt_flat = target.reshape(B, -1, T)
        cos_sim = F.cosine_similarity(out_flat, tgt_flat, dim=1).mean()
        cos_loss = 1 - cos_sim

        # Combine with weights
        total = (
            l1_loss * 0.5 +       # Content alignment
            mean_loss * 1.0 +     # Mean matching
            std_loss * 1.0 +      # Std matching
            band_loss * 0.3 +     # Frequency bands
            delta_loss * 2.0 +    # Delta matching (important!)
            cos_loss * 0.5        # Cosine similarity
        )

        return total, {
            'l1': l1_loss.item(),
            'mean': mean_loss.item(),
            'std': std_loss.item(),
            'band': band_loss.item(),
            'delta': delta_loss.item(),
            'cos': cos_loss.item(),
        }

    def train_step(self, batch):
        input_latent = batch['input'].to(self.device)
        target = batch['target'].to(self.device)
        direction = batch['direction'].to(self.device)
        valid = batch['valid'].to(self.device)

        if not valid.any():
            return {'loss': 0.0}

        input_latent = input_latent[valid]
        target = target[valid]
        direction = direction[valid]

        if input_latent.shape[0] == 0:
            return {'loss': 0.0}

        self.model.train()
        self.optimizer.zero_grad()

        amp_context = autocast('cuda') if self.use_amp else nullcontext()

        with amp_context:
            output = self.model(input_latent, direction)
            loss, loss_dict = self.combined_loss(output, target, input_latent)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {'loss': loss.item(), **loss_dict}

    def log(self, msg):
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    def save_checkpoint(self, epoch, loss, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }

        torch.save(state, self.output_dir / 'latest.pt')

        if epoch % 10 == 0:
            torch.save(state, self.output_dir / f'epoch_{epoch}.pt')

        if is_best:
            torch.save(state, self.output_dir / 'best.pt')

    def train(self):
        self.log("=" * 60)
        self.log("PAIRED DISTRIBUTION MATCHING TRAINING V2")
        self.log("=" * 60)
        self.log(f"Output dir: {self.output_dir}")

        for epoch in range(1, self.num_epochs + 1):
            epoch_losses = []
            epoch_metrics = {k: [] for k in ['l1', 'mean', 'std', 'band', 'delta', 'cos']}

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                for k in epoch_metrics:
                    if k in metrics:
                        epoch_metrics[k].append(metrics[k])
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}", delta=f"{metrics.get('delta', 0):.4f}")

            self.scheduler.step()

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_delta = sum(epoch_metrics['delta']) / len(epoch_metrics['delta']) if epoch_metrics['delta'] else 0
            lr = self.scheduler.get_last_lr()[0]

            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss

            self.save_checkpoint(epoch, avg_loss, is_best)

            best_marker = " [BEST]" if is_best else ""
            self.log(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Delta: {avg_delta:.4f} | LR: {lr:.2e}{best_marker}")

        self.log("\nTraining complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/mnt/msdd2/pitchshift_v9_formant_pairs/manifest.json')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/msdd2/pitchshift_checkpoints/formant_paired_distmatch_v2')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--samples_per_epoch', type=int, default=10000)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--residual', action='store_true', help='Use residual output instead of direct')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    trainer = PairedDistMatchTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        use_attention=not args.no_attention,
        direct_output=not args.residual,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == '__main__':
    main()
