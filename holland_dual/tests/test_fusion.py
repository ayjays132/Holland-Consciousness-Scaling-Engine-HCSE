from holland_dual.fusion.adapter import simulation_to_activation
from holland_dual.quantum.huqce.simulation import HuqceParams


def test_adapter_shape():
    acts = simulation_to_activation(HuqceParams(steps=1))
    assert acts.dim() == 2
