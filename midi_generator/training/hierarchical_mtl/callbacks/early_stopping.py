"""
Early Stopping Callback for Hierarchical MTL Training.

Author: Agent 06
Date: November 20, 2025
"""

import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.

    Args:
        patience: Number of epochs with no improvement after which training will be stopped
        min_delta: Minimum change in the monitored quantity to qualify as an improvement
        mode: One of 'min', 'max'. In 'min' mode, training will stop when the quantity
              monitored has stopped decreasing; in 'max' mode it will stop when the
              quantity has stopped increasing
        verbose: If True, prints messages when early stopping condition is met
        restore_best_weights: If True, restores model weights from the epoch with the
                              best value of the monitored quantity
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.best_weights = None
        self.should_stop = False

        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.Inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.Inf

    def on_epoch_end(self, epoch: int, current_score: float, model_state_dict: Optional[dict] = None):
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            current_score: Current value of the monitored quantity
            model_state_dict: Model state dict to save if this is the best epoch
        """
        # Check if score has improved
        if self.mode == 'min':
            score_improved = current_score < (self.best_score - self.min_delta)
        else:
            score_improved = current_score > (self.best_score + self.min_delta)

        if score_improved:
            self.best_score = current_score
            self.wait = 0

            if self.restore_best_weights and model_state_dict is not None:
                # Deep copy of state dict
                import copy
                self.best_weights = copy.deepcopy(model_state_dict)

            if self.verbose:
                print(f"Epoch {epoch}: Validation score improved to {current_score:.6f}")

        else:
            self.wait += 1
            if self.verbose:
                print(f"Epoch {epoch}: Validation score did not improve ({current_score:.6f})")
                print(f"  Best score: {self.best_score:.6f}, Patience: {self.wait}/{self.patience}")

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True

                if self.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best score: {self.best_score:.6f} (epoch {epoch - self.wait})")

    def reset(self):
        """Reset early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.best_weights = None

        if self.mode == 'min':
            self.best_score = np.Inf
        else:
            self.best_score = -np.Inf

    def get_best_weights(self) -> Optional[dict]:
        """Return best model weights if restore_best_weights is True."""
        return self.best_weights

    def __repr__(self) -> str:
        return (f"EarlyStopping(patience={self.patience}, "
                f"min_delta={self.min_delta}, mode='{self.mode}')")
