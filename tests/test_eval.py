import pytest
from src.game.connect4 import Connect4Board
from src.eval.opponents import RandomPlayer, HeuristicPlayer, MinimaxPlayer
from src.eval.tournament import run_matchup, play_game
from src.eval.metrics import compute_optimal_action_agreement, compute_diversity

def test_opponents_gameplay_ordering():
    # Minimax vs Random: Minimax should decisively beat Random
    minimax = MinimaxPlayer(depth=2)
    random_player = RandomPlayer(seed=42)

    res = run_matchup(minimax, random_player, num_games=6, seed=42)
    # Minimax should win most or all games
    assert res["wins"] >= res["losses"]
    assert res["win_rate"] >= 0.5

def test_optimal_action_agreement():
    # Mock records with optimal moves
    b = Connect4Board()
    rec = {
        "grid": b.grid.tolist(),
        "current_player": b.current_player,
        "optimal_moves": [3],
        "num_pieces": 0
    }
    heuristic = HeuristicPlayer() # prefers col 3 on empty board
    eval_res = compute_optimal_action_agreement(heuristic, [rec])
    assert eval_res["overall_agreement"] == 1.0
    assert eval_res["by_stage"]["early"]["agreement"] == 1.0

def test_diversity_metrics():
    sample_data = [
        {"canonical_hash": "hash1", "chosen_action": 3},
        {"canonical_hash": "hash2", "chosen_action": 2},
        {"canonical_hash": "hash1", "chosen_action": 3},
        {"canonical_hash": "hash3", "chosen_action": 4},
    ]
    div = compute_diversity(sample_data)
    assert div["total_positions"] == 4
    assert div["unique_canonical"] == 3
    assert div["diversity_ratio"] == 0.75
    assert div["action_entropy"] > 0
