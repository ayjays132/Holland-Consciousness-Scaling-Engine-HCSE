from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from holland_dual.quantum.huqce.simulation import HuqceParams, HuqceSimulator
from holland_dual.fusion.adapter import simulation_to_activation

app = typer.Typer(help="Holland Dual CLI")


@app.command()
def hdq_sim(config: Optional[Path] = None) -> None:
    """Run HUQCE simulation."""
    params = HuqceParams()
    if config and config.exists():
        data = json.loads(config.read_text())
        params = HuqceParams(**data)
    sim = HuqceSimulator(params)
    psi = sim.run()
    print(psi[-5:])  # preview


@app.command()
def hdf_fuse() -> None:
    """Demonstrate fusion adapter."""
    acts = simulation_to_activation(HuqceParams(steps=10))
    print(acts.shape)


cli = app

__all__ = ["cli"]
