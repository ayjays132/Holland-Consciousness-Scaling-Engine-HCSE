from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .solver import crank_nicolson_step, compute_momentum_expectation


@dataclass
class HuqceParams:
    n: int = 256
    dx: float = 0.1
    dt: float = 0.01
    steps: int = 100
    gamma: float = 0.01
    alpha: float = 0.005
    epsilon: float = 0.1


class HuqceSimulator:
    def __init__(self, params: HuqceParams) -> None:
        self.params = params
        x = np.linspace(0, params.n * params.dx, params.n)
        self.psi = np.sqrt(2 / (params.n * params.dx)) * np.sin(np.pi * x / (params.n * params.dx))
        self.psi = self.psi.astype(complex)
        diag = -2 * np.ones(params.n)
        off = np.ones(params.n - 1)
        lap = (np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)) / params.dx**2
        self.laplacian = lap

    def step(self) -> None:
        p_exp = compute_momentum_expectation(self.psi, self.params.dx)
        self.psi = crank_nicolson_step(
            self.psi,
            self.laplacian,
            self.params.dt,
            self.params.gamma,
            self.params.alpha,
            self.params.epsilon,
            p_exp,
        )

    def run(self) -> np.ndarray:
        for _ in range(self.params.steps):
            self.step()
        return self.psi


__all__ = ["HuqceParams", "HuqceSimulator"]
