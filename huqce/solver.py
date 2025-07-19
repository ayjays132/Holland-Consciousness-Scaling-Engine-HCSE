from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def crank_nicolson_step(
    psi: np.ndarray,
    laplacian: np.ndarray,
    dt: float,
    gamma: float,
    alpha: float,
    epsilon: float,
    momentum_expectation: float,
) -> np.ndarray:
    """Perform a single Crank-Nicolson step for 1D HUQCE.

    Parameters
    ----------
    psi : np.ndarray
        Current wave function samples (complex64/complex128).
    laplacian : np.ndarray
        Discrete Laplacian matrix.
    dt : float
        Time step.
    gamma : float
        Nonlinearity coefficient.
    alpha : float
        Chaos coefficient.
    epsilon : float
        Chaos strength scaling.
    momentum_expectation : float
        Current expectation value of momentum operator.

    Returns
    -------
    np.ndarray
        Updated wave function.
    """
    i = 1j
    hbar = 1.0
    n = psi.size
    # Hamiltonian matrix H = -(hbar^2/2m)L + gamma|psi|^2
    # Here we assume m=1 and potential V=0 for simplicity.
    diag_nl = gamma * np.abs(psi) ** 2
    A = np.eye(n, dtype=complex) + 0.5j * dt * (
        -0.5 * laplacian + np.diag(diag_nl)
    )
    B = np.eye(n, dtype=complex) - 0.5j * dt * (
        -0.5 * laplacian + np.diag(diag_nl)
    )
    rhs = B @ psi
    # chaos term: epsilon * alpha * (p - <p>)
    grad = -1j * np.gradient(psi)
    chaos = epsilon * alpha * (grad - momentum_expectation)
    rhs += dt * chaos * psi
    psi_next = np.linalg.solve(A, rhs)
    norm = np.linalg.norm(psi_next)
    if norm > 0:
        psi_next /= norm
    return psi_next


def compute_momentum_expectation(psi: ArrayLike, dx: float) -> complex:
    grad = np.gradient(psi, dx)
    expectation = np.sum(np.conj(psi) * (-1j * grad)) * dx
    return expectation


__all__ = ["crank_nicolson_step", "compute_momentum_expectation"]

