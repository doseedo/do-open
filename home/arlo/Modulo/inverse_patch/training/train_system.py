"""
Main training system for Inverse Audio Effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..models.effect_encoder import EffectEncoder, EffectEncoderConfig
from ..models.chain_estimator import IterativeChainEstimator, ChainEstimatorConfig
from ..models.differentiable_chain import DifferentiableFXChain
from .losses import InverseAFxLoss, MultiResolutionSTFTLoss


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Loss weights
    dry_loss_weight: float = 1.0
    wet_loss_weight: float = 1.0
    chain_loss_weight: float = 0.5
    contrastive_weight: float = 0.1
    cycle_weight: float = 0.3

    # Training stages
    use_curriculum: bool = True
    curriculum_stages: List[Dict] = None


class InverseAFxSystem(pl.LightningModule):
    """
    Full Inverse Audio Effects training system.

    Combines:
    - Effect encoder
    - Chain estimator
    - Differentiable FX chain for verification
    """

    def __init__(
        self,
        encoder_config: Optional[EffectEncoderConfig] = None,
        estimator_config: Optional[ChainEstimatorConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        effect_types: Optional[List[str]] = None,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder_config = encoder_config or EffectEncoderConfig()
        self.estimator_config = estimator_config or ChainEstimatorConfig()
        self.training_config = training_config or TrainingConfig()
        self.effect_types = effect_types or [
            'eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay'
        ]
        self.sample_rate = sample_rate

        # Initialize models
        self.encoder = EffectEncoder(self.encoder_config)
        self.chain_estimator = IterativeChainEstimator(self.estimator_config)
        self.fx_chain = DifferentiableFXChain(
            sample_rate=sample_rate,
            effect_types=effect_types,
        )

        # Losses
        self.loss_fn = InverseAFxLoss(
            audio_loss_weight=self.training_config.dry_loss_weight,
            chain_loss_weight=self.training_config.chain_loss_weight,
            contrastive_weight=self.training_config.contrastive_weight,
            cycle_weight=self.training_config.cycle_weight,
        )
        self.audio_loss = MultiResolutionSTFTLoss()

        # Curriculum tracking
        self.current_curriculum_stage = 0

        # Per-effect loss tracking for diagnostics
        self.effect_loss_accum = {fx: 0.0 for fx in self.effect_types}
        self.effect_loss_count = {fx: 0 for fx in self.effect_types}

    def forward(
        self,
        wet_audio: torch.Tensor,
        max_iterations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]]]:
        """
        Forward pass: estimate dry audio and effect chain from wet audio.

        Args:
            wet_audio: Wet audio [B, 1, T] or [B, T]
            max_iterations: Override max chain length

        Returns:
            dry_estimate: Estimated dry audio
            chain: Estimated effect chain
        """
        if wet_audio.dim() == 2:
            wet_audio = wet_audio.unsqueeze(1)

        dry_estimate, chain = self.chain_estimator(
            wet_audio,
            max_iterations=max_iterations,
        )

        return dry_estimate, chain

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        dry_audio = batch['dry_audio']
        wet_audio = batch['wet_audio']
        effect_types = batch.get('effect_types')
        effect_params = batch.get('effect_params')
        chain_length = batch.get('chain_length')

        # Ensure correct shape
        if dry_audio.dim() == 2:
            dry_audio = dry_audio.unsqueeze(1)
        if wet_audio.dim() == 2:
            wet_audio = wet_audio.unsqueeze(1)

        # Forward pass - use ground truth chain length, effect types, and params
        # This is full teacher forcing for inverter training
        gt_chain_len = int(chain_length.max().item()) if chain_length is not None else 1

        # Number of params per effect type (must match ParameterEstimator)
        param_counts = {
            'eq': 15, 'compressor': 6, 'reverb': 4,
            'distortion': 4, 'chorus': 4, 'delay': 3,
        }

        # Build GT effect type and params lists in REVERSE order (last effect first)
        # Use the first sample in batch as reference (simplified for batch processing)
        gt_effect_list = None
        gt_params_list = None
        if effect_types is not None and effect_params is not None:
            gt_effect_list = []
            gt_params_list = []
            cl = int(chain_length[0].item())
            for i in range(cl - 1, -1, -1):  # Reverse order
                if i < effect_types.size(1):
                    fx_idx = int(effect_types[0, i].item())
                    gt_effect_list.append(fx_idx)

                    # Extract GT params for this effect - clone with requires_grad for gradient flow
                    fx_type = self.effect_types[fx_idx] if fx_idx < len(self.effect_types) else 'eq'
                    num_params = param_counts.get(fx_type, 10)
                    # effect_params is [B, max_effects, max_params]
                    gt_p = effect_params[:, i, :num_params].clone().requires_grad_(True)  # [B, num_params]
                    gt_params_list.append(gt_p)

        dry_estimate, estimated_chain = self.chain_estimator(
            wet_audio,
            return_intermediates=False,
            force_iterations=gt_chain_len,
            gt_effect_types=gt_effect_list,  # Use GT effect types for correct inverter
            gt_params=gt_params_list,  # Use GT params for correct inversion
        )

        # Skip wet reconstruction for now - causes length mismatch errors
        # TODO: Fix inverters to preserve audio length
        wet_reconstructed = None
        wet_loss = torch.tensor(0.0, device=dry_audio.device)

        # Clamp dry_estimate to prevent NaN from bad inverter outputs
        dry_estimate = torch.clamp(dry_estimate, -10.0, 10.0)
        dry_estimate = torch.nan_to_num(dry_estimate, nan=0.0, posinf=1.0, neginf=-1.0)

        # Compute losses
        # 1. Dry reconstruction loss
        dry_loss = self.audio_loss(dry_estimate, dry_audio)

        # Skip batch if loss is NaN or extremely high (indicates bad inverter output)
        if torch.isnan(dry_loss) or dry_loss.item() > 100.0:
            return None

        # 3. Per-iteration classification loss (supervise which effect at each step)
        # Ground truth is in APPLICATION order, estimated is also in application order (reversed after loop)
        # But we estimated in REVERSE order (last effect first), so reverse GT to match
        iter_class_loss = dry_loss * 0.0  # Initialize with grad connection
        if effect_types is not None and len(estimated_chain) > 0:
            # Build GT chain in reverse order (last effect first) to match estimation order
            for iter_idx, chain_item in enumerate(reversed(estimated_chain)):
                pred_type, pred_params = chain_item[0], chain_item[1]
                pred_logits = chain_item[3] if len(chain_item) > 3 else None
                if pred_logits is None:
                    continue  # Skip if no logits available
                # For each sample in batch, get the GT effect at this iteration
                # effect_types[b, i] is the i-th effect in application order
                # We want the (chain_len - 1 - iter_idx)-th effect
                batch_size = effect_types.size(0)
                gt_effect_idx = torch.zeros(batch_size, dtype=torch.long, device=dry_audio.device)

                for b in range(batch_size):
                    cl = int(chain_length[b].item())
                    gt_idx_in_chain = cl - 1 - iter_idx  # Reverse index
                    if 0 <= gt_idx_in_chain < effect_types.size(1):
                        gt_effect_idx[b] = effect_types[b, gt_idx_in_chain]
                    else:
                        gt_effect_idx[b] = len(self.effect_types)  # "no effect" class

                # Classification loss for this iteration
                iter_class_loss = iter_class_loss + F.cross_entropy(pred_logits, gt_effect_idx)

            iter_class_loss = iter_class_loss / len(estimated_chain)

        # 4. Parameter regression loss (compare predicted params to ground truth)
        param_loss = torch.tensor(0.0, device=dry_audio.device)
        if effect_params is not None and len(estimated_chain) > 0:
            for iter_idx, chain_item in enumerate(reversed(estimated_chain)):
                pred_type, pred_params = chain_item[0], chain_item[1]
                batch_size = effect_params.size(0)
                for b in range(batch_size):
                    cl = int(chain_length[b].item())
                    gt_idx_in_chain = cl - 1 - iter_idx
                    if 0 <= gt_idx_in_chain < effect_params.size(1):
                        # Compare predicted params to GT params for this effect slot
                        gt_params = effect_params[b, gt_idx_in_chain, :pred_params.size(1)]
                        param_loss = param_loss + F.mse_loss(pred_params[b], gt_params)

            param_loss = param_loss / (len(estimated_chain) * batch_size)

        # Handle NaN in iter_class_loss - use zero that maintains grad graph
        if torch.isnan(iter_class_loss):
            iter_class_loss = dry_loss * 0.0  # Maintains grad connection

        # Stage 2c: Inverters only - disable classifier loss
        # Train inverters to reconstruct dry audio using GT effect types and GT params
        # Classifier will be trained separately in Stage 2a
        total_loss = self.training_config.dry_loss_weight * dry_loss
        # iter_class_loss disabled - classifier not trained in this stage

        # Final NaN check
        if torch.isnan(total_loss):
            return None

        # Per-effect loss tracking for diagnostics
        if effect_types is not None:
            batch_size = effect_types.size(0)
            for b in range(batch_size):
                cl = int(chain_length[b].item())
                for i in range(cl):
                    if i < effect_types.size(1):
                        fx_idx = int(effect_types[b, i].item())
                        if fx_idx < len(self.effect_types):
                            fx_name = self.effect_types[fx_idx]
                            self.effect_loss_accum[fx_name] += dry_loss.item()
                            self.effect_loss_count[fx_name] += 1

        # Log per-effect averages every 100 steps
        if batch_idx > 0 and batch_idx % 100 == 0:
            for fx_name in self.effect_types:
                if self.effect_loss_count[fx_name] > 0:
                    avg_loss = self.effect_loss_accum[fx_name] / self.effect_loss_count[fx_name]
                    self.log(f'train/loss_{fx_name}', avg_loss)
            # Reset accumulators
            self.effect_loss_accum = {fx: 0.0 for fx in self.effect_types}
            self.effect_loss_count = {fx: 0 for fx in self.effect_types}

        # Logging - Stage 2c focuses on dry_loss (inverter training)
        self.log('train/dry_loss', dry_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        # Disabled for Stage 2c:
        # self.log('train/iter_class_loss', iter_class_loss)
        # self.log('train/param_loss', param_loss)
        self.log('train/chain_len', float(len(estimated_chain)))
        self.log('train/gt_chain_len', float(gt_chain_len))

        return total_loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Validation step - uses GT effect types/params for Stage 2c evaluation."""
        dry_audio = batch['dry_audio']
        wet_audio = batch['wet_audio']
        effect_types = batch.get('effect_types')
        effect_params = batch.get('effect_params')
        chain_length = batch.get('chain_length')

        if dry_audio.dim() == 2:
            dry_audio = dry_audio.unsqueeze(1)
        if wet_audio.dim() == 2:
            wet_audio = wet_audio.unsqueeze(1)

        # Build GT effect type and params (same as training_step for Stage 2c)
        param_counts = {
            'eq': 15, 'compressor': 6, 'reverb': 4,
            'distortion': 4, 'chorus': 4, 'delay': 3,
        }

        gt_effect_list = None
        gt_params_list = None
        gt_chain_len = 1

        if effect_types is not None and effect_params is not None and chain_length is not None:
            gt_effect_list = []
            gt_params_list = []
            gt_chain_len = int(chain_length[0].item())
            for i in range(gt_chain_len - 1, -1, -1):
                if i < effect_types.size(1):
                    fx_idx = int(effect_types[0, i].item())
                    gt_effect_list.append(fx_idx)
                    fx_type = self.effect_types[fx_idx] if fx_idx < len(self.effect_types) else 'eq'
                    num_params = param_counts.get(fx_type, 10)
                    gt_p = effect_params[:, i, :num_params]
                    gt_params_list.append(gt_p)

        # Forward pass with GT (Stage 2c: evaluate inverters with correct params)
        dry_estimate, estimated_chain = self.chain_estimator(
            wet_audio,
            force_iterations=gt_chain_len,
            gt_effect_types=gt_effect_list,
            gt_params=gt_params_list,
        )

        # Safeguards (same as training_step) to prevent NaN/exploding values
        dry_estimate = torch.clamp(dry_estimate, -10.0, 10.0)
        dry_estimate = torch.nan_to_num(dry_estimate, nan=0.0, posinf=1.0, neginf=-1.0)

        # Reconstruct
        if len(estimated_chain) > 0:
            chain_spec = [
                (fx_type, params)
                for fx_type, params, *_ in estimated_chain  # Handle both 3 and 4 element tuples
            ]
            wet_reconstructed = self.fx_chain(dry_estimate, chain_spec)
        else:
            wet_reconstructed = dry_estimate

        # Compute losses
        dry_loss = self.audio_loss(dry_estimate, dry_audio)
        wet_loss = self.audio_loss(wet_reconstructed, wet_audio)

        # SI-SDR
        si_sdr = self._compute_si_sdr(dry_estimate, dry_audio)

        self.log('val/dry_loss', dry_loss, prog_bar=True)
        self.log('val/wet_loss', wet_loss)
        self.log('val/si_sdr', si_sdr, prog_bar=True)
        self.log('val/chain_len', float(len(estimated_chain)))

        return {
            'dry_loss': dry_loss,
            'wet_loss': wet_loss,
            'si_sdr': si_sdr,
        }

    def _compute_si_sdr(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scale-invariant SDR."""
        estimate = estimate.flatten(1)
        target = target.flatten(1)

        # Zero mean
        estimate = estimate - estimate.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)

        # Compute SI-SDR
        dot = (estimate * target).sum(dim=1, keepdim=True)
        s_target = (target ** 2).sum(dim=1, keepdim=True)
        proj = dot * target / (s_target + 1e-8)

        noise = estimate - proj
        si_sdr = 10 * torch.log10(
            (proj ** 2).sum(dim=1) / ((noise ** 2).sum(dim=1) + 1e-8)
        )

        return si_sdr.mean()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10000,
            T_mult=2,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Curriculum learning: advance stage if needed
        if self.training_config.use_curriculum:
            self._update_curriculum()

    def _update_curriculum(self):
        """Update curriculum stage based on epoch."""
        if self.training_config.curriculum_stages is None:
            return

        stages = self.training_config.curriculum_stages
        total_epochs = 0

        for i, stage in enumerate(stages):
            total_epochs += stage.get('epochs', 10)
            if self.current_epoch < total_epochs:
                if self.current_curriculum_stage != i:
                    self.current_curriculum_stage = i
                    self._apply_curriculum_stage(stage)
                break

    def _apply_curriculum_stage(self, stage: Dict):
        """Apply curriculum stage settings."""
        if 'max_chain_length' in stage:
            self.chain_estimator.config.max_iterations = stage['max_chain_length']

        if 'effect_types' in stage:
            # Filter to only use specified effects
            pass  # Would need to update data loader

        self.log('curriculum/stage', self.current_curriculum_stage)
