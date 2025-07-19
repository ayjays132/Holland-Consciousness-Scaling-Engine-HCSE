import numpy as np
import warnings
from numpy.exceptions import ComplexWarning
from huqce.simulation import HuqceParams, HuqceSimulator
from huqce.solver import compute_momentum_expectation


def test_norm_conservation():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ComplexWarning)
        params = HuqceParams(steps=10)
        psi = HuqceSimulator(params).run()
    final_norm = np.linalg.norm(psi)
    assert abs(final_norm - 1.0) < 1e-5


def test_momentum_expectation_type():
    psi = np.array([1 + 1j, 0j])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ComplexWarning)
        val = compute_momentum_expectation(psi, 1.0)
    assert np.iscomplexobj(val)
