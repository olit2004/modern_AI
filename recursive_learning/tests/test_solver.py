import pytest
from src.game.connect4 import Connect4Board
from src.solver.wrapper import Connect4Solver
import tempfile
import os

@pytest.fixture
def temp_solver():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        tmp_path = f.name
    solver = Connect4Solver(cache_path=tmp_path)
    yield solver
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

def test_solver_immediate_win(temp_solver):
    # Setup position where P1 has 3 pieces in col 3, P2 in col 0
    # Move history: 3 (P1), 0 (P2), 3 (P1), 0 (P2), 3 (P1), 0 (P2)
    # P1 now can win immediately in col 3!
    b = Connect4Board()
    for m in [3, 0, 3, 0, 3, 0]:
        b.make_move(m)

    scores, optimal_moves = temp_solver.evaluate(b)
    assert 3 in optimal_moves
    assert scores[3] > 0

def test_solver_symmetry(temp_solver):
    # P1 plays col 1, P2 plays col 2
    b = Connect4Board()
    b.make_move(1)
    b.make_move(2)

    mirror_b = b.mirror()

    scores1, opts1 = temp_solver.evaluate(b)
    scores2, opts2 = temp_solver.evaluate(mirror_b)

    # In mirror board, optimal moves should be 6 - col
    expected_opts2 = {6 - c for c in opts1}
    assert opts2 == expected_opts2

def test_solver_caching(temp_solver):
    b = Connect4Board()
    b.make_move(3)
    b.make_move(3)

    c_hash = b.to_canonical_hash()
    assert temp_solver.get_cached(c_hash) is None

    scores, opts = temp_solver.evaluate(b)
    cached = temp_solver.get_cached(c_hash)
    assert cached is not None
    assert cached[1] == opts

def test_solver_batch(temp_solver):
    b1 = Connect4Board()
    b1.make_move(3)

    b2 = Connect4Board()
    b2.make_move(2)

    results = temp_solver.evaluate_batch([b1, b2])
    assert len(results) == 2
    for scores, opts in results:
        assert len(opts) >= 1
