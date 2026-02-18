"""
Loss functions for training the Inverse Audio Effects System.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for audio reconstruction.
    Combines spectral convergence and log magnitude losses at multiple resolutions.
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        window: str = 'hann',
        sc_weight: float = 1.0,
        mag_weight: float = 1.0,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes or [s // 4 for s in fft_sizes]
        self.win_lengths = win_lengths or fft_sizes
        self.window = window
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight

        # Pre-compute windows
        self.windows = nn.ParameterDict()
        for fft_size, win_length in zip(fft_sizes, self.win_lengths):
            if window == 'hann':
                win = torch.hann_window(win_length)
            elif window == 'hamming':
                win = torch.hamming_window(win_length)
            else:
                win = torch.ones(win_length)
            self.windows[str(fft_size)] = nn.Parameter(win, requires_grad=False)

    def stft(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_length: int,
    ) -> torch.Tensor:
        """Compute STFT magnitude."""
        window = self.windows[str(fft_size)].to(x.device)

        # Handle shape
        if x.dim() == 3:
            x = x.squeeze(1)

        # Compute STFT
        stft_result = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

        return stft_result.abs()

    def spectral_convergence_loss(
        self,
        pred_mag: torch.Tensor,
        target_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Spectral convergence loss."""
        return torch.norm(target_mag - pred_mag, p='fro') / (torch.norm(target_mag, p='fro') + 1e-8)

    def log_magnitude_loss(
        self,
        pred_mag: torch.Tensor,
        target_mag: torch.Tensor,
    ) -> torch.Tensor:
        """Log magnitude loss."""
        log_pred = torch.log(pred_mag + 1e-8)
        log_target = torch.log(target_mag + 1e-8)
        return F.l1_loss(log_pred, log_target)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-resolution STFT loss.

        Args:
            pred: Predicted audio [B, 1, T] or [B, T]
            target: Target audio [B, 1, T] or [B, T]

        Returns:
            Total loss
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            pred_mag = self.stft(pred, fft_size, hop_size, win_length)
            target_mag = self.stft(target, fft_size, hop_size, win_length)

            sc_loss += self.spectral_convergence_loss(pred_mag, target_mag)
            mag_loss += self.log_magnitude_loss(pred_mag, target_mag)

        sc_loss /= len(self.fft_sizes)
        mag_loss /= len(self.fft_sizes)

        return self.sc_weight * sc_loss + self.mag_weight * mag_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for effect embedding learning.
    Similar chains should have similar embeddings.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        negative_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.negative_weight = negative_weight

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: Normalized embeddings [B, D]
            labels: Effect chain labels [B] (same label = positive pair)

        Returns:
            Contrastive loss
        """
        batch_size = embeddings.size(0)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask (same label = positive)
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()

        # Remove diagonal
        positive_mask.fill_diagonal_(0)

        # Negative mask
        negative_mask = 1.0 - positive_mask
        negative_mask.fill_diagonal_(0)

        # Compute loss
        exp_sim = torch.exp(sim_matrix)

        # Positive pairs
        pos_sum = (exp_sim * positive_mask).sum(dim=1)

        # Negative pairs
        neg_sum = (exp_sim * negative_mask).sum(dim=1)

        # Loss
        loss = -torch.log(pos_sum / (pos_sum + self.negative_weight * neg_sum + 1e-8))

        # Only compute for samples with positive pairs
        valid_mask = positive_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return loss


class ChainLoss(nn.Module):
    """
    Loss for effect chain estimation.
    Combines classification and parameter regression losses.
    """

    def __init__(
        self,
        num_effect_types: int = 6,
        classification_weight: float = 1.0,
        param_weight: float = 0.5,
        order_weight: float = 0.3,
    ):
        super().__init__()
        self.num_effect_types = num_effect_types
        self.classification_weight = classification_weight
        self.param_weight = param_weight
        self.order_weight = order_weight

        self.classification_loss = nn.BCEWithLogitsLoss()
        self.param_loss = nn.MSELoss()

    def forward(
        self,
        pred_effects: torch.Tensor,
        pred_params: Dict[str, torch.Tensor],
        target_effects: torch.Tensor,
        target_params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute chain estimation loss.

        Args:
            pred_effects: Predicted effect logits [B, num_types]
            pred_params: Dict of predicted parameters per effect
            target_effects: Target effect labels [B, num_types]
            target_params: Dict of target parameters per effect

        Returns:
            Total loss and loss components
        """
        # Classification loss
        class_loss = self.classification_loss(pred_effects, target_effects.float())

        # Parameter loss (only for present effects)
        param_loss = torch.tensor(0.0, device=pred_effects.device)
        param_count = 0

        for effect_type in pred_params:
            if effect_type in target_params:
                # Mask for samples where this effect is present
                effect_idx = list(pred_params.keys()).index(effect_type)
                mask = target_effects[:, effect_idx] > 0.5

                if mask.sum() > 0:
                    pred_p = pred_params[effect_type][mask]
                    target_p = target_params[effect_type][mask]
                    param_loss += self.param_loss(pred_p, target_p)
                    param_count += 1

        if param_count > 0:
            param_loss /= param_count

        # Total loss
        total_loss = (
            self.classification_weight * class_loss +
            self.param_weight * param_loss
        )

        return total_loss, {
            'classification': class_loss,
            'parameter': param_loss,
            'total': total_loss,
        }


class InverseAFxLoss(nn.Module):
    """
    Combined loss for the full Inverse AFx system.
    """

    def __init__(
        self,
        audio_loss_weight: float = 1.0,
        chain_loss_weight: float = 0.5,
        contrastive_weight: float = 0.1,
        cycle_weight: float = 0.3,
    ):
        super().__init__()
        self.audio_loss_weight = audio_loss_weight
        self.chain_loss_weight = chain_loss_weight
        self.contrastive_weight = contrastive_weight
        self.cycle_weight = cycle_weight

        self.audio_loss = MultiResolutionSTFTLoss()
        self.chain_loss = ChainLoss()
        self.contrastive_loss = ContrastiveLoss()

    def forward(
        self,
        dry_pred: torch.Tensor,
        dry_target: torch.Tensor,
        wet_recon: torch.Tensor,
        wet_target: torch.Tensor,
        pred_effects: Optional[torch.Tensor] = None,
        target_effects: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        dry_cycle: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.

        Args:
            dry_pred: Predicted dry audio
            dry_target: Target dry audio
            wet_recon: Reconstructed wet audio (chain applied to dry_pred)
            wet_target: Target wet audio
            pred_effects: Predicted effect logits (optional)
            target_effects: Target effect labels (optional)
            embeddings: Effect embeddings for contrastive loss (optional)
            labels: Labels for contrastive loss (optional)
            dry_cycle: Cycle-consistency dry audio (optional)

        Returns:
            Total loss and loss components
        """
        losses = {}

        # Dry audio reconstruction loss
        dry_loss = self.audio_loss(dry_pred, dry_target)
        losses['dry_reconstruction'] = dry_loss

        # Wet audio reconstruction loss
        wet_loss = self.audio_loss(wet_recon, wet_target)
        losses['wet_reconstruction'] = wet_loss

        total_loss = (
            self.audio_loss_weight * dry_loss +
            self.audio_loss_weight * wet_loss
        )

        # Chain loss (if provided)
        if pred_effects is not None and target_effects is not None:
            chain_loss = F.binary_cross_entropy_with_logits(
                pred_effects, target_effects.float()
            )
            losses['chain_classification'] = chain_loss
            total_loss += self.chain_loss_weight * chain_loss

        # Contrastive loss (if provided)
        if embeddings is not None and labels is not None:
            contrastive = self.contrastive_loss(embeddings, labels)
            losses['contrastive'] = contrastive
            total_loss += self.contrastive_weight * contrastive

        # Cycle consistency loss (if provided)
        if dry_cycle is not None:
            cycle_loss = self.audio_loss(dry_cycle, dry_pred.detach())
            losses['cycle_consistency'] = cycle_loss
            total_loss += self.cycle_weight * cycle_loss

        losses['total'] = total_loss

        return total_loss, losses


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained audio features.
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        layers: List[int] = [3, 6, 9],
    ):
        super().__init__()
        self.layers = layers

        # Default: use simple CNN feature extractor
        if feature_extractor is None:
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(1, 32, 1023, padding=511),
                nn.ReLU(),
                nn.Conv1d(32, 64, 511, padding=255),
                nn.ReLU(),
                nn.Conv1d(64, 128, 255, padding=127),
                nn.ReLU(),
                nn.Conv1d(128, 256, 127, padding=63),
                nn.ReLU(),
            )
        else:
            self.feature_extractor = feature_extractor

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss."""
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if target.dim() == 2:
            target = target.unsqueeze(1)

        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        return F.l1_loss(pred_features, target_features)
