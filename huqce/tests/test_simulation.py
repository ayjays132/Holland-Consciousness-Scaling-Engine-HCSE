import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from huqce.simulation import HuqceParams, HuqceSimulator


def test_norm_conservation():
    params = HuqceParams(steps=10)
    psi = HuqceSimulator(params).run()
    final_norm = np.linalg.norm(psi)
    assert abs(final_norm - 1.0) < 1e-5
