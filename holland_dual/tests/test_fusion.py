from holland_dual.fusion.adapter import simulation_to_activation
from holland_dual.quantum.huqce.simulation import HuqceParams
import numpy as np
import warnings
from numpy.exceptions import ComplexWarning


def test_adapter_shape():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ComplexWarning)
        acts = simulation_to_activation(HuqceParams(steps=1))
    assert acts.dim() == 2
