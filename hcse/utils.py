"""Utility functions for HCSE."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


def corrcoef(matrix: Tensor) -> Tensor:
    """Compute correlation matrix for 2D tensor."""
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    cov = matrix.t() @ matrix / (matrix.shape[0] - 1)
    std = matrix.std(dim=0, unbiased=True).clamp(min=1e-12)
    corr = cov / torch.outer(std, std)
    return corr


def info_nce_loss(features: Tensor, temperature: float = 0.1) -> Tensor:
    """Compute a simple InfoNCE loss given features."""
    features = nn.functional.normalize(features, dim=1)
    logits = features @ features.t() / temperature
    labels = torch.arange(features.shape[0], device=features.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

__all__ = ["corrcoef", "info_nce_loss"]
