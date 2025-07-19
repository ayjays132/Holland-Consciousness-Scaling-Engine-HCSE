import numpy as np
import warnings
from numpy.exceptions import ComplexWarning
from huqce.simulation import HuqceParams, HuqceSimulator


def test_norm_conservation():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ComplexWarning)
        params = HuqceParams(steps=10)
        psi = HuqceSimulator(params).run()
    final_norm = np.linalg.norm(psi)
    assert abs(final_norm - 1.0) < 1e-5
