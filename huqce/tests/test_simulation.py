import numpy as np
from huqce.simulation import HuqceParams, HuqceSimulator


def test_norm_conservation():
    params = HuqceParams(steps=10)
    psi = HuqceSimulator(params).run()
    final_norm = np.linalg.norm(psi)
    assert abs(final_norm - 1.0) < 1e-5
