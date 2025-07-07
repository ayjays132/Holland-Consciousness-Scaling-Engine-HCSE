"""Utility functions for HCSE."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


def corrcoef(matrix: Tensor) -> Tensor:
    """Return the correlation coefficient matrix for a 2D tensor."""
    if matrix.dim() != 2:
        raise ValueError("matrix must be 2D")

    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    # Compute unbiased covariance
    if matrix.size(0) > 1:
        cov = matrix.t() @ matrix / (matrix.size(0) - 1)
    else:
        cov = torch.zeros(matrix.size(1), matrix.size(1), device=matrix.device)

    var = cov.diagonal().clamp(min=1e-12)
    std = var.sqrt()
    corr = cov / (std[:, None] * std[None, :])
    return corr


def info_nce_loss(features: Tensor, temperature: float = 0.1) -> Tensor:
    """Compute a simple InfoNCE loss for a batch of feature vectors."""
    if features.dim() != 2:
        raise ValueError("features must be 2D")

    feats = nn.functional.normalize(features, dim=1)
    logits = feats @ feats.t() / temperature
    labels = torch.arange(feats.size(0), device=features.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

__all__ = ["corrcoef", "info_nce_loss"]
