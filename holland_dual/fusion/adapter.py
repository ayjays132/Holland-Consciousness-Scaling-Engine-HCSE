from __future__ import annotations

import numpy as np
import torch

from holland_dual.quantum.huqce.simulation import HuqceParams, HuqceSimulator
from hcse.core import HCSEMixin


def simulation_to_activation(params: HuqceParams) -> torch.Tensor:
    """Run HUQCE simulation and convert final wavefunction to activation tensor."""
    sim = HuqceSimulator(params)
    psi = sim.run()
    # Map complex wavefunction to real-valued activations: concatenate real/imag
    activations = np.stack([psi.real, psi.imag], axis=-1)
    return torch.from_numpy(activations.astype(np.float32))


__all__ = ["simulation_to_activation"]
