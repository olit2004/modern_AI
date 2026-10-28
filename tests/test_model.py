import pytest
import numpy as np
import tempfile
import os
from src.game.connect4 import Connect4Board

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for model tests")
def test_model_forward_and_masking():
    from src.model.net import Connect4Net
    model = Connect4Net(num_filters=16, hidden_dim=32)
    x = torch.zeros(2, 2, 6, 7)
    # Mask col 0 and 6 as illegal
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32)

    logits = model(x, legal_mask=mask)
    assert logits.shape == (2, 7)
    # Col 0 in first item should be very negative
    assert logits[0, 0].item() < -1e8
    assert logits[0, 1].item() > -1e8

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for model tests")
def test_predict_distribution():
    from src.model.net import Connect4Net
    model = Connect4Net(num_filters=16, hidden_dim=32)
    b = Connect4Board()
    b.make_move(3)

    probs = model.predict_distribution(b, temperature=1.0)
    assert len(probs) == 7
    assert np.isclose(probs.sum(), 1.0)
    assert (probs >= 0.0).all()

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for model tests")
def test_sanity_check_overfit():
    from src.model.train import sanity_check_overfit
    assert sanity_check_overfit() is True

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for model tests")
def test_save_and_load_checkpoint():
    from src.model.net import Connect4Net
    from src.model.train import save_checkpoint, load_checkpoint

    model1 = Connect4Net(num_filters=16, hidden_dim=32)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        save_checkpoint(model1, None, step=42, round_num=1, metrics={"acc": 0.8}, path=path)
        model2, meta = load_checkpoint(path, model_kwargs={"num_filters": 16, "hidden_dim": 32})
        assert meta["step"] == 42
        assert meta["round_num"] == 1
        assert meta["metrics"]["acc"] == 0.8
    finally:
        if os.path.exists(path):
            os.remove(path)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for model tests")
def test_student_net_forward_and_params():
    from src.model.net import Connect4Net, Connect4StudentNet
    teacher = Connect4Net()
    student = Connect4StudentNet()

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    assert student_params < teacher_params // 10, "Student should be at least 10x smaller than teacher"

    b = Connect4Board()
    b.make_move(3)
    probs = student.predict_distribution(b)
    assert len(probs) == 7
    assert np.isclose(probs.sum(), 1.0)
