import numpy as np
from huqce.analysis import spectral_entropy


def test_spectral_entropy_constant():
    const_signal = np.ones(8)
    ent = spectral_entropy(const_signal)
    assert ent < 1e-6


def test_spectral_entropy_random():
    np.random.seed(0)
    sig = np.random.randn(128)
    ent = spectral_entropy(sig)
    assert ent > 0
