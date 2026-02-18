"""
Gradient Reversal Layer for disentanglement training.

Multiplies gradient by -1 during backward pass.
Used to train auxiliary classifiers to FAIL at predicting cross-factor info.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient reversal function."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer.

    During forward: identity
    During backward: multiply gradient by -alpha

    Usage:
        grl = GradientReversalLayer(alpha=1.0)
        features = grl(features)
        pred = classifier(features)

        # Classifier will be trained to FAIL (adversarial)
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        """Update reversal strength (can schedule during training)."""
        self.alpha = alpha


if __name__ == "__main__":
    # Test
    grl = GradientReversalLayer(alpha=1.0)

    x = torch.randn(4, 16, requires_grad=True)
    y = grl(x)
    loss = y.sum()
    loss.backward()

    print(f"Input grad (should be negative): {x.grad.sum().item():.4f}")
    print("GRL working!" if x.grad.sum() < 0 else "GRL broken!")
