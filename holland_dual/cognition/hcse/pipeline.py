"""Training pipeline components for HCSE."""

from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import Trainer


class HfTrainerWithHCSE(Trainer):
    """Trainer that integrates HCSE into the loss computation."""

    def __init__(self, *args: Any, hcse_params: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hcse_params = hcse_params or {}

    def compute_loss(
        self,
        model: Any,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        outputs = model.forward_with_hcse(**inputs, hcse_params=self.hcse_params)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

__all__ = ["HfTrainerWithHCSE"]
