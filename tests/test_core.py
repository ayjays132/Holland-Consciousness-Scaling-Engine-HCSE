import torch
from hcse.utils import corrcoef, info_nce_loss


def test_corrcoef_basic():
    x = torch.randn(10, 4)
    c = corrcoef(x)
    assert c.shape == (4, 4)
    assert torch.allclose(torch.diag(c), torch.ones(4), atol=1e-6)


def test_info_nce_loss_scalar():
    feats = torch.randn(6, 8)
    loss = info_nce_loss(feats)
    assert loss.dim() == 0
    assert loss.item() > 0
