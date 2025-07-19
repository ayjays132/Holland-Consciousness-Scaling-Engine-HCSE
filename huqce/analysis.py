from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def spectral_entropy(signal: ArrayLike) -> float:
    """Compute spectral entropy of a real or complex 1D signal.

    Parameters
    ----------
    signal : ArrayLike
        Input array representing the wavefunction or time series.

    Returns
    -------
    float
        Shannon entropy of the normalized power spectrum.
    """
    arr = np.asarray(signal)
    spectrum = np.fft.fft(arr)
    power = np.abs(spectrum) ** 2
    if power.sum() == 0:
        return 0.0
    prob = power / power.sum()
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    return float(entropy)


__all__ = ["spectral_entropy"]
