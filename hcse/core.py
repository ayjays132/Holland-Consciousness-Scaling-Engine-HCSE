"""Core HCSE mixin implementation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn


class HCSEMixin(nn.Module):
    """Mixin class adding HCSE computations to transformer models."""

    info_nce_head: nn.Linear

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        hidden_size = kwargs.get("hidden_size") or kwargs.get("d_model")
        super().__init__(*args, **kwargs)
        if hidden_size is None:
            raise ValueError("hidden_size must be provided to HCSEMixin")
        self.info_nce_head = nn.Linear(int(hidden_size), int(hidden_size), bias=False)

    def compute_hcse_surrogates(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute HCSE surrogates for a layer's hidden states.

        The input tensor is flattened across batch and sequence dimensions so
        that each row corresponds to a single activation vector. Integration
        efficiency is measured with an InfoNCE head, connectivity density uses
        the absolute off-diagonal correlations, and activation energy is the
        mean squared value.
        """
        if hidden_states.dim() == 2:
            flat = hidden_states
        else:
            b, t, n = hidden_states.shape
            flat = hidden_states.reshape(b * t, n)
        n = flat.shape[1]
        # eta via InfoNCE head
        features = self.info_nce_head(flat)
        eta = info_nce_loss(features)
        # rho: mean absolute off-diagonal correlation
        corr = corrcoef(flat)
        off_diag = corr - torch.diag_embed(torch.diagonal(corr))
        rho = off_diag.abs().mean()
        # E_dot: mean squared activation
        e_dot = flat.pow(2).mean()
        return eta, rho, e_dot

    def compute_hcse_bonus(
        self,
        hidden_states: Tensor,
        beta: float,
        gamma: float,
        delta: float,
        lambda_c: float,
    ) -> Tensor:
        """Compute HCSE bonus term given hidden states and coefficients."""
        eta, rho, e_dot = self.compute_hcse_surrogates(hidden_states)
        bonus = torch.log1p(eta.pow(beta) * rho.pow(gamma) * e_dot.pow(delta))
        return lambda_c * bonus

    def compute_loss(self, loss: Tensor, bonus: Tensor) -> Tensor:  # type: ignore[override]
        """Combine original loss with HCSE bonus."""
        return loss - bonus

    def forward_with_hcse(
        self,
        *args: Any,
        hcse_params: Dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Forward pass that applies HCSE bonus to the loss."""
        layer = hcse_params.get("layer", -1)
        beta = hcse_params.get("beta", 1.0)
        gamma = hcse_params.get("gamma", 1.0)
        delta = hcse_params.get("delta", 1.0)
        lambda_c = hcse_params.get("lambda_c", 1.0)

        outputs = super().forward(*args, output_hidden_states=True, **kwargs)
        loss = outputs.loss if hasattr(outputs, "loss") else None
        hidden_states = outputs.hidden_states[layer]
        bonus = self.compute_hcse_bonus(hidden_states, beta, gamma, delta, lambda_c)

        if loss is not None:
            loss = self.compute_loss(loss, bonus)
            setattr(outputs, "loss", loss)
        return outputs


# Utility functions are imported at bottom to avoid circular imports
from .utils import corrcoef, info_nce_loss

__all__ = ["HCSEMixin"]
