import torch
from src.models.dora import DoRALinear

def test_reconstruction():
    in_f, out_f = 8, 5
    layer = DoRALinear(in_f, out_f, rank=2)
    x = torch.randn(3, in_f)
    y = layer(x)
    assert y.shape == (3, out_f)
    with torch.no_grad():
        delta_v = layer.A @ layer.B
        v_hat = layer.v + delta_v
        v_hat = torch.nn.functional.normalize(v_hat, p=2, dim=1)
        W_hat = layer.g * v_hat
        assert W_hat.shape == (out_f, in_f)
    print("DoRA unit test passed")

if __name__ == "__main__":
    test_reconstruction()

