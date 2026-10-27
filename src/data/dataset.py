"""
Dataset generation, strict zero-leakage hash splitting, and locked final-test loader.
Ensures zero overlap between train, dev, and test sets even under horizontal reflection symmetry.
"""

import os
import random
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np

from src.game.connect4 import Connect4Board, COLS, ROWS, PLAYER_1, PLAYER_2
from src.solver.wrapper import Connect4Solver
from src.utils.platform import set_seed

class LockedDatasetAccessError(PermissionError):
    """Raised when final test set is accessed without explicit --final authorization."""
    pass

class DatasetManager:
    """
    Manages loading and saving datasets with strict zero-leakage partitions
    and guarded final test set access.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.train_file = self.data_dir / "train.json"
        self.dev_file = self.data_dir / "dev.json"
        self.final_test_file = self.data_dir / "final_test.json"

    def save_splits(self, train_data: List[Dict], dev_data: List[Dict], test_data: List[Dict]):
        """Save train, dev, and test splits."""
        for path, data in [
            (self.train_file, train_data),
            (self.dev_file, dev_data),
            (self.final_test_file, test_data)
        ]:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def load_train(self) -> List[Dict]:
        if not self.train_file.exists():
            raise FileNotFoundError(f"Train split not found at {self.train_file}")
        with open(self.train_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_dev(self) -> List[Dict]:
        if not self.dev_file.exists():
            raise FileNotFoundError(f"Dev split not found at {self.dev_file}")
        with open(self.dev_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_final_test(self, allow_final: bool = False) -> List[Dict]:
        """
        Loads final test set ONLY if allow_final is True.
        Refuses access otherwise to prevent contamination during development.
        """
        if not allow_final:
            raise LockedDatasetAccessError(
                "Final test set is LOCKED. You must pass allow_final=True (or CLI flag --final) "
                "to evaluate on the unseen test split."
            )
        if not self.final_test_file.exists():
            raise FileNotFoundError(f"Final test split not found at {self.final_test_file}")
        with open(self.final_test_file, "r", encoding="utf-8") as f:
            return json.load(f)

def generate_position_pool(
    num_games: int,
    solver: Connect4Solver,
    seed: int = 42,
    mix_policy: bool = True
) -> List[Dict[str, Any]]:
    """
    Play games with diverse policies (random, 1-step lookahead heuristic)
    to collect non-terminal positions across early, mid, and late game stages.
    Labels each position with the solver.
    """
    set_seed(seed)
    rng = random.Random(seed)

    collected_by_canonical: Dict[str, Dict[str, Any]] = {}

    for game_idx in range(num_games):
        board = Connect4Board()
        # Randomize policy for this game: random vs random, random vs heuristic, etc.
        use_heuristic = mix_policy and (game_idx % 2 == 1)

        while not board.is_terminal():
            legal = board.legal_moves()
            if not legal:
                break

            c_hash = board.to_canonical_hash()
            if c_hash not in collected_by_canonical:
                collected_by_canonical[c_hash] = {
                    "canonical_hash": c_hash,
                    "grid": board.grid.tolist(),
                    "current_player": int(board.current_player),
                    "move_history": list(board.move_history),
                    "num_pieces": int(board.num_pieces()),
                    "board_obj": board.clone()
                }

            # Choose move to progress the game
            if use_heuristic:
                # 1-step winning move or blocking move
                move = None
                for col in legal:
                    b_win = board.clone()
                    b_win.make_move(col)
                    if b_win.winner() == board.current_player:
                        move = col
                        break
                if move is None:
                    # Block opponent immediate win
                    for col in legal:
                        b_block = Connect4Board(board.grid, current_player=-board.current_player)
                        b_block.make_move(col)
                        if b_block.winner() == -board.current_player:
                            move = col
                            break
                if move is None:
                    move = rng.choice(legal)
            else:
                move = rng.choice(legal)

            board.make_move(move)

    # Label all collected boards with solver
    all_boards = [item["board_obj"] for item in collected_by_canonical.values()]
    eval_results = solver.evaluate_batch(all_boards)

    dataset_records: List[Dict[str, Any]] = []
    for item, (scores, opts) in zip(collected_by_canonical.values(), eval_results):
        dataset_records.append({
            "canonical_hash": item["canonical_hash"],
            "grid": item["grid"],
            "current_player": item["current_player"],
            "move_history": item["move_history"],
            "num_pieces": item["num_pieces"],
            "scores": {str(k): v for k, v in scores.items()},
            "optimal_moves": sorted(list(opts))
        })

    return dataset_records

def split_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits records strictly by canonical_hash to ensure zero leakage across splits.
    Because canonical_hash maps any board and its mirror to the exact same hash,
    neither identical nor mirrored positions can ever appear across splits.
    """
    assert abs((train_ratio + dev_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    rng = random.Random(seed)

    # Map canonical_hash to its records
    unique_hashes = sorted(list({r["canonical_hash"] for r in records}))
    rng.shuffle(unique_hashes)

    n_total = len(unique_hashes)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)

    train_hashes = set(unique_hashes[:n_train])
    dev_hashes = set(unique_hashes[n_train:n_train + n_dev])
    test_hashes = set(unique_hashes[n_train + n_dev:])

    # Assert zero overlap
    assert len(train_hashes & dev_hashes) == 0, "Train-Dev leakage detected!"
    assert len(train_hashes & test_hashes) == 0, "Train-Test leakage detected!"
    assert len(dev_hashes & test_hashes) == 0, "Dev-Test leakage detected!"

    train_data = [r for r in records if r["canonical_hash"] in train_hashes]
    dev_data = [r for r in records if r["canonical_hash"] in dev_hashes]
    test_data = [r for r in records if r["canonical_hash"] in test_hashes]

    return train_data, dev_data, test_data
