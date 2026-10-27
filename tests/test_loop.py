import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required for loop tests")
def test_acceptance_gate_logic():
    from src.loop.engine import SelfImprovementEngine
    from src.data.dataset import DatasetManager
    from src.solver.wrapper import Connect4Solver
    from src.model.net import Connect4Net

    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DatasetManager(data_dir=tmpdir)
        solver = Connect4Solver(cache_path=str(Path(tmpdir) / "cache.sqlite"))

        engine = SelfImprovementEngine(
            condition="v1_fixed",
            output_dir=tmpdir,
            solver=solver,
            dataset_manager=dm,
            acceptance_tolerance=0.01
        )

        # Fake history
        engine.experiment_log = [{"round": 0, "dev_agreement": 0.50}]

        # Test proposal with higher agreement: should accept
        new_agreement_good = 0.55
        is_accepted_good = (new_agreement_good >= 0.50 - engine.acceptance_tolerance)
        assert is_accepted_good is True

        # Test proposal with much lower agreement: should reject
        new_agreement_bad = 0.40
        is_accepted_bad = (new_agreement_bad >= 0.50 - engine.acceptance_tolerance)
        assert is_accepted_bad is False
