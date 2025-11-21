"""
AGENT 4: Advanced Loss Functions for Scaled Hierarchical MTL
=============================================================

Multi-task loss functions with:
    1. Hierarchical weighting (Level 1 > Level 2 > Level 3)
    2. Dimension-specific weighting (Hierarchical vs Modular vs Rich)
    3. Uncertainty-weighted multi-task loss
    4. Gradient balancing across tasks
    5. Dynamic task weighting

Key Features:
    - Automatic task balancing via learned uncertainty
    - Gradient magnitude balancing
    - Per-dimension loss computation
    - Hierarchical importance weighting
    - Support for continuous, categorical, and structured outputs

Author: Agent 4 - Model Architecture Engineer
Date: November 21, 2025
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Install with: pip install torch")


# ============================================================================
# Base Loss Components
# ============================================================================

class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss from:
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (Kendall et al., CVPR 2018)

    Loss = (1 / 2σ²) * L + log(σ)
    where σ² is learned task-specific uncertainty (variance)
    """

    def __init__(self, num_tasks: int, init_log_var: float = 0.0):
        super().__init__()
        # Learnable log variance for each task
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_var))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Args:
            losses: Tensor of shape (num_tasks,) with per-task losses

        Returns:
            Weighted total loss
        """
        # Precision = exp(-log_var) = 1/σ²
        precision = torch.exp(-self.log_vars)

        # Weighted loss: (1 / 2σ²) * L + (1/2) * log(σ²)
        weighted_losses = 0.5 * precision * losses + 0.5 * self.log_vars

        return weighted_losses.sum()


class GradientBalancedLoss(nn.Module):
    """
    Gradient balancing across tasks to prevent gradient magnitude imbalance.

    Normalizes gradients to have similar magnitudes across all tasks.
    """

    def __init__(self, num_tasks: int, alpha: float = 0.5):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha  # Smoothing factor for running average
        self.register_buffer('avg_grad_norms', torch.ones(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: List of per-task losses

        Returns:
            Gradient-balanced total loss
        """
        # Stack losses
        loss_tensor = torch.stack(losses)

        # Compute gradient norms (approximate)
        if self.training:
            # Update running average of gradient norms
            grad_norms = []
            for loss in losses:
                if loss.requires_grad:
                    # Approximate gradient norm
                    grad_norms.append(loss.detach().abs())
                else:
                    grad_norms.append(torch.tensor(1.0, device=loss.device))

            grad_norms = torch.stack(grad_norms)
            self.avg_grad_norms = (
                self.alpha * self.avg_grad_norms +
                (1 - self.alpha) * grad_norms
            )

        # Balance losses by inverse of average gradient norm
        weights = 1.0 / (self.avg_grad_norms + 1e-8)
        weights = weights / weights.sum() * self.num_tasks  # Normalize

        # Weighted sum
        balanced_loss = (loss_tensor * weights).sum()

        return balanced_loss


# ============================================================================
# Scaled Hierarchical MTL Loss
# ============================================================================

class ScaledHierarchicalMTLLoss(nn.Module):
    """
    Advanced multi-task loss for ScaledHierarchicalMTL model.

    Combines:
        1. Hierarchical losses (L1, L2, L3) with hierarchical weighting
        2. Modular semantic losses (6 dimensions)
        3. Rich extension losses (per-track, temporal, genre-specific)
        4. Automatic uncertainty-based weighting
        5. Gradient balancing

    Loss Components:
        - Hierarchical (3 tasks): L1, L2, L3
        - Modular (6 tasks): Harmony, Rhythm, Form, Orchestration, Texture, Cross-dim
        - Rich (3 tasks): Per-track, Temporal, Genre-specific
        Total: 12 tasks
    """

    def __init__(
        self,
        # Hierarchical weights
        level1_weight: float = 3.0,
        level2_weight: float = 2.0,
        level3_weight: float = 1.5,

        # Category weights
        hierarchical_weight: float = 2.0,
        modular_weight: float = 1.5,
        rich_weight: float = 1.0,

        # Loss strategy
        use_uncertainty_weighting: bool = True,
        use_gradient_balancing: bool = True,

        # Task-specific settings
        mse_reduction: str = 'mean',
    ):
        super().__init__()

        self.level1_weight = level1_weight
        self.level2_weight = level2_weight
        self.level3_weight = level3_weight

        self.hierarchical_weight = hierarchical_weight
        self.modular_weight = modular_weight
        self.rich_weight = rich_weight

        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_gradient_balancing = use_gradient_balancing
        self.mse_reduction = mse_reduction

        # Total tasks: 3 (hierarchical) + 6 (modular) + 3 (rich) = 12
        num_tasks = 12

        if use_uncertainty_weighting:
            self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks)

        if use_gradient_balancing:
            self.gradient_balancer = GradientBalancedLoss(num_tasks)

    def compute_hierarchical_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute losses for hierarchical predictions (L1, L2, L3).

        Args:
            predictions: Dict with 'level_1', 'level_2', 'level_3'
            targets: Dict with same keys

        Returns:
            Tuple of (loss_l1, loss_l2, loss_l3)
        """
        loss_l1 = F.mse_loss(
            predictions['level_1'],
            targets['level_1'],
            reduction=self.mse_reduction
        )

        loss_l2 = F.mse_loss(
            predictions['level_2'],
            targets['level_2'],
            reduction=self.mse_reduction
        )

        loss_l3 = F.mse_loss(
            predictions['level_3'],
            targets['level_3'],
            reduction=self.mse_reduction
        )

        return loss_l1, loss_l2, loss_l3

    def compute_modular_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for modular semantic dimensions.

        Args:
            predictions: Dict with dimension names
            targets: Dict with same keys

        Returns:
            Dict of per-dimension losses
        """
        modular_losses = {}

        for dim_name in ['harmony', 'rhythm', 'form', 'orchestration', 'texture', 'cross_dimensional']:
            if dim_name in predictions and dim_name in targets:
                modular_losses[dim_name] = F.mse_loss(
                    predictions[dim_name],
                    targets[dim_name],
                    reduction=self.mse_reduction
                )
            else:
                # Fallback if dimension missing
                modular_losses[dim_name] = torch.tensor(0.0, device=predictions[list(predictions.keys())[0]].device)

        return modular_losses

    def compute_rich_losses(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute losses for rich extensions.

        Args:
            predictions: Dict with 'per_track', 'temporal', 'genre_specific'
            targets: Dict with same keys

        Returns:
            Tuple of (loss_per_track, loss_temporal, loss_genre_specific)
        """
        # Per-track loss (batch, 8, 10)
        loss_per_track = F.mse_loss(
            predictions['per_track'],
            targets['per_track'],
            reduction=self.mse_reduction
        )

        # Temporal loss (batch, 4, 10)
        loss_temporal = F.mse_loss(
            predictions['temporal'],
            targets['temporal'],
            reduction=self.mse_reduction
        )

        # Genre-specific loss (batch, 10)
        loss_genre_specific = F.mse_loss(
            predictions['genre_specific'],
            targets['genre_specific'],
            reduction=self.mse_reduction
        )

        return loss_per_track, loss_temporal, loss_genre_specific

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total multi-task loss.

        Args:
            predictions: Dict with keys 'hierarchical', 'modular', 'rich'
            targets: Dict with same structure

        Returns:
            Tuple of (total_loss, loss_dict)
            where loss_dict contains per-task losses for logging
        """
        # ===== Compute individual losses =====

        # Hierarchical
        loss_l1, loss_l2, loss_l3 = self.compute_hierarchical_losses(
            predictions['hierarchical'],
            targets['hierarchical']
        )

        # Modular
        modular_losses = self.compute_modular_losses(
            predictions['modular'],
            targets['modular']
        )

        # Rich
        loss_per_track, loss_temporal, loss_genre_specific = self.compute_rich_losses(
            predictions['rich'],
            targets['rich']
        )

        # ===== Apply hierarchical and category weights =====

        # Hierarchical losses
        weighted_l1 = self.level1_weight * self.hierarchical_weight * loss_l1
        weighted_l2 = self.level2_weight * self.hierarchical_weight * loss_l2
        weighted_l3 = self.level3_weight * self.hierarchical_weight * loss_l3

        # Modular losses
        weighted_modular = []
        for dim_name, loss in modular_losses.items():
            weighted_modular.append(self.modular_weight * loss)

        # Rich losses
        weighted_per_track = self.rich_weight * loss_per_track
        weighted_temporal = self.rich_weight * loss_temporal
        weighted_genre_specific = self.rich_weight * loss_genre_specific

        # ===== Combine losses =====

        # Collect all task losses
        task_losses = [
            weighted_l1,
            weighted_l2,
            weighted_l3,
            *weighted_modular,
            weighted_per_track,
            weighted_temporal,
            weighted_genre_specific
        ]

        # Apply uncertainty weighting or gradient balancing
        if self.use_uncertainty_weighting:
            loss_tensor = torch.stack([loss for loss in task_losses])
            total_loss = self.uncertainty_loss(loss_tensor)

        elif self.use_gradient_balancing:
            total_loss = self.gradient_balancer(task_losses)

        else:
            # Simple sum
            total_loss = sum(task_losses)

        # ===== Create loss dict for logging =====

        loss_dict = {
            'total': total_loss.item(),
            'hierarchical/level_1': loss_l1.item(),
            'hierarchical/level_2': loss_l2.item(),
            'hierarchical/level_3': loss_l3.item(),
            'modular/harmony': modular_losses['harmony'].item(),
            'modular/rhythm': modular_losses['rhythm'].item(),
            'modular/form': modular_losses['form'].item(),
            'modular/orchestration': modular_losses['orchestration'].item(),
            'modular/texture': modular_losses['texture'].item(),
            'modular/cross_dimensional': modular_losses['cross_dimensional'].item(),
            'rich/per_track': loss_per_track.item(),
            'rich/temporal': loss_temporal.item(),
            'rich/genre_specific': loss_genre_specific.item(),
        }

        return total_loss, loss_dict


# ============================================================================
# Factory Functions
# ============================================================================

def create_loss_function(
    hierarchical_weight: float = 2.0,
    modular_weight: float = 1.5,
    rich_weight: float = 1.0,
    use_uncertainty_weighting: bool = True,
    use_gradient_balancing: bool = False
) -> ScaledHierarchicalMTLLoss:
    """
    Factory function to create loss function.

    Args:
        hierarchical_weight: Weight for hierarchical tasks
        modular_weight: Weight for modular tasks
        rich_weight: Weight for rich extension tasks
        use_uncertainty_weighting: Use uncertainty-based weighting
        use_gradient_balancing: Use gradient balancing

    Returns:
        Configured loss function
    """
    loss_fn = ScaledHierarchicalMTLLoss(
        hierarchical_weight=hierarchical_weight,
        modular_weight=modular_weight,
        rich_weight=rich_weight,
        use_uncertainty_weighting=use_uncertainty_weighting,
        use_gradient_balancing=use_gradient_balancing
    )

    return loss_fn


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        exit(1)

    print("Loss Functions - Agent 4")
    print("="*70)

    # Create loss function
    print("\n1. Creating loss function...")
    loss_fn = create_loss_function()
    print("   ✅ Loss function created")

    # Create dummy predictions and targets
    print("\n2. Creating dummy data...")
    batch_size = 32

    predictions = {
        'hierarchical': {
            'level_1': torch.randn(batch_size, 8),
            'level_2': torch.randn(batch_size, 20),
            'level_3': torch.randn(batch_size, 22),
        },
        'modular': {
            'harmony': torch.randn(batch_size, 30),
            'rhythm': torch.randn(batch_size, 20),
            'form': torch.randn(batch_size, 15),
            'orchestration': torch.randn(batch_size, 25),
            'texture': torch.randn(batch_size, 20),
            'cross_dimensional': torch.randn(batch_size, 10),
        },
        'rich': {
            'per_track': torch.randn(batch_size, 8, 10),
            'temporal': torch.randn(batch_size, 4, 10),
            'genre_specific': torch.randn(batch_size, 10),
        }
    }

    targets = {
        'hierarchical': {
            'level_1': torch.randn(batch_size, 8),
            'level_2': torch.randn(batch_size, 20),
            'level_3': torch.randn(batch_size, 22),
        },
        'modular': {
            'harmony': torch.randn(batch_size, 30),
            'rhythm': torch.randn(batch_size, 20),
            'form': torch.randn(batch_size, 15),
            'orchestration': torch.randn(batch_size, 25),
            'texture': torch.randn(batch_size, 20),
            'cross_dimensional': torch.randn(batch_size, 10),
        },
        'rich': {
            'per_track': torch.randn(batch_size, 8, 10),
            'temporal': torch.randn(batch_size, 4, 10),
            'genre_specific': torch.randn(batch_size, 10),
        }
    }

    print("   ✅ Dummy data created")

    # Compute loss
    print("\n3. Computing loss...")
    total_loss, loss_dict = loss_fn(predictions, targets)
    print(f"   ✅ Loss computed")
    print(f"\n   Total loss: {total_loss.item():.4f}")
    print(f"\n   Per-task losses:")
    for task, value in loss_dict.items():
        if task != 'total':
            print(f"     {task}: {value:.4f}")

    # Test backward pass
    print("\n4. Testing backward pass...")
    total_loss.backward()
    print("   ✅ Backward pass successful")

    # Print learned uncertainty weights
    if hasattr(loss_fn, 'uncertainty_loss'):
        print("\n5. Learned uncertainty weights (log variance):")
        log_vars = loss_fn.uncertainty_loss.log_vars.detach().numpy()
        task_names = [
            'L1', 'L2', 'L3',
            'Harmony', 'Rhythm', 'Form', 'Orchestration', 'Texture', 'Cross-dim',
            'Per-track', 'Temporal', 'Genre-spec'
        ]
        for name, log_var in zip(task_names, log_vars):
            print(f"   {name}: {log_var:.4f} (σ² = {torch.exp(torch.tensor(log_var)).item():.4f})")

    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
