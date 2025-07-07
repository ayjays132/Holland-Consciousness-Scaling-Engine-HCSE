import torch
from hcse.core import HCSEMixin


class BaseModel(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, output_hidden_states: bool = False):
        hidden = self.linear(input_ids)
        loss = hidden.mean()
        outputs = {
            "loss": loss,
            "hidden_states": (hidden,),
        }
        return type("Output", (), outputs)


class DummyModel(HCSEMixin, BaseModel):
    def __init__(self, hidden_size: int) -> None:
        HCSEMixin.__init__(self, hidden_size=hidden_size)


def test_surrogates_shape():
    model = DummyModel(4)
    data = torch.randn(2, 3, 4)
    eta, rho, e_dot = model.compute_hcse_surrogates(data)
    assert eta.dim() == 0
    assert rho.dim() == 0
    assert e_dot.dim() == 0


def test_bonus_application():
    model = DummyModel(4)
    input_ids = torch.randn(2, 3, 4)
    outputs = model.forward_with_hcse(input_ids, hcse_params={"layer": 0})
    assert hasattr(outputs, "loss")
