import pytest
import tempfile
import numpy as np
from src.game.connect4 import Connect4Board
from src.solver.wrapper import Connect4Solver
from src.data.dataset import (
    generate_position_pool,
    split_dataset,
    DatasetManager,
    LockedDatasetAccessError
)

def test_dataset_generation_and_zero_leakage():
    # Use temporary sqlite cache
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as f:
        cache_path = f.name
    solver = Connect4Solver(cache_path=cache_path)

    # Generate small pool of positions
    records = generate_position_pool(num_games=10, solver=solver, seed=123)
    assert len(records) > 0

    train_data, dev_data, test_data = split_dataset(records, 0.6, 0.2, 0.2, seed=42)

    train_hashes = {r["canonical_hash"] for r in train_data}
    dev_hashes = {r["canonical_hash"] for r in dev_data}
    test_hashes = {r["canonical_hash"] for r in test_data}

    # Strict zero leakage assertion
    assert train_hashes.isdisjoint(dev_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert dev_hashes.isdisjoint(test_hashes)

    # Symmetry leakage check:
    # Ensure that no mirror of a train position appears in dev or test
    for r in train_data:
        grid = np.array(r["grid"], dtype=np.int8)
        b = Connect4Board(grid, current_player=r["current_player"])
        b_mirror = b.mirror()
        mirror_c_hash = b_mirror.to_canonical_hash()

        # The mirror MUST have the exact same canonical hash!
        assert mirror_c_hash == r["canonical_hash"]
        assert mirror_c_hash not in dev_hashes
        assert mirror_c_hash not in test_hashes

def test_locked_final_test_access():
    with tempfile.TemporaryDirectory() as tmpdir:
        dm = DatasetManager(data_dir=tmpdir)
        dm.save_splits([{"id": 1}], [{"id": 2}], [{"id": 3}])

        # Normal access to train and dev should succeed
        assert dm.load_train() == [{"id": 1}]
        assert dm.load_dev() == [{"id": 2}]

        # Without allow_final=True, access to final test MUST raise LockedDatasetAccessError
        with pytest.raises(LockedDatasetAccessError):
            dm.load_final_test(allow_final=False)

        # With allow_final=True, it succeeds
        assert dm.load_final_test(allow_final=True) == [{"id": 3}]
