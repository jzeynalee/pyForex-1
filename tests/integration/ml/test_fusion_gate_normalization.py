import pytest


torch = pytest.importorskip("torch")

from models.fusion import FusionNet


@pytest.mark.integration
def test_fusion_gate_weights_bounded_and_normalized():
    net = FusionNet(seq_dim=4, vit_dim=6, yolo_dim=3, hidden_dim=8)
    net.eval()

    seq = torch.zeros((2, 4))
    vit = torch.zeros((2, 6))
    yolo = torch.zeros((2, 3))

    logits, gates = net.forward_with_gates(seq, vit, yolo)

    assert gates.shape == (2, 3)
    assert torch.all(gates >= 0)
    assert torch.all(gates <= 1)
    sums = gates.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
