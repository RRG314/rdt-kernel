import torch
from rdt_kernel import step

def test_step_runs():
    L = 1 + 0.1 * torch.rand((1, 32, 32))
    out = step(L)
    assert out.shape == L.shape
    assert torch.all(out > 0)