"""
Optimizer and Scheduler Factory for Hierarchical MTL Training.

Author: Agent 06
Date: November 20, 2025
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    ExponentialLR
)
from typing import Optional, Any

from midi_generator.training.hierarchical_mtl.config.training_config import (
    OptimizerConfig,
    SchedulerConfig,
    OptimizerType,
    SchedulerType
)


def create_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: Model to optimize
        config: Optimizer configuration

    Returns:
        Configured optimizer
    """
    parameters = model.parameters()

    if config.optimizer_type == OptimizerType.ADAM:
        optimizer = optim.Adam(
            parameters,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )

    elif config.optimizer_type == OptimizerType.ADAMW:
        optimizer = optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )

    elif config.optimizer_type == OptimizerType.SGD:
        optimizer = optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov
        )

    elif config.optimizer_type == OptimizerType.RMSPROP:
        optimizer = optim.RMSprop(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum
        )

    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

    print(f"Created optimizer: {config.optimizer_type.value}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    num_epochs: int
) -> Optional[Any]:
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        num_epochs: Total number of training epochs

    Returns:
        Configured scheduler or None if scheduler_type is NONE
    """
    if config.scheduler_type == SchedulerType.NONE:
        return None

    elif config.scheduler_type == SchedulerType.COSINE:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.T_max or num_epochs,
            eta_min=config.eta_min
        )

    elif config.scheduler_type == SchedulerType.STEP:
        scheduler = StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )

    elif config.scheduler_type == SchedulerType.PLATEAU:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode='rel',
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            verbose=True
        )

    elif config.scheduler_type == SchedulerType.EXPONENTIAL:
        scheduler = ExponentialLR(
            optimizer,
            gamma=config.decay_rate
        )

    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

    print(f"Created scheduler: {config.scheduler_type.value}")

    # Wrap with warmup if specified
    if config.warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer,
            scheduler,
            warmup_epochs=config.warmup_epochs,
            warmup_start_lr=config.warmup_start_lr
        )
        print(f"  With warmup: {config.warmup_epochs} epochs")

    return scheduler


class WarmupScheduler:
    """
    Learning rate scheduler with warmup period.

    Gradually increases learning rate from warmup_start_lr to base_lr
    over warmup_epochs, then follows the main scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        main_scheduler: Any,
        warmup_epochs: int,
        warmup_start_lr: float
    ):
        self.optimizer = optimizer
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None, metrics: Optional[float] = None):
        """Step the scheduler."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linearly increase LR
            alpha = self.current_epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
        else:
            # Main scheduler phase
            if isinstance(self.main_scheduler, ReduceLROnPlateau):
                if metrics is not None:
                    self.main_scheduler.step(metrics)
            else:
                self.main_scheduler.step()

    def state_dict(self):
        """Get scheduler state."""
        return {
            'main_scheduler': self.main_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.main_scheduler.load_state_dict(state_dict['main_scheduler'])
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.base_lrs = state_dict['base_lrs']
