"""
Inspect and verify Teacher Models for Task 3.
Loads both teacher models (Baseline Round 0 and Improved Round 3)
and evaluates their full output probability distributions for any Connect-4 position.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

from src.game.connect4 import Connect4Board
from src.model.train import load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Teacher Models for Distillation")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing experiment results and teachers")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline teacher checkpoint")
    parser.add_argument("--improved", type=str, default=None, help="Improved teacher checkpoint")
    parser.add_argument("--moves", type=str, default="3324", help="Sequence of 0-indexed column moves to test (e.g. '3324')")
    return parser.parse_args()

def main():
    args = parse_args()
    res_dir = Path(args.results_dir)
    b_path = Path(args.baseline) if args.baseline else (res_dir / "teachers" / "teacher_round0_baseline.pt")
    i_path = Path(args.improved) if args.improved else (res_dir / "teachers" / "teacher_round3_improved.pt")

    print("=" * 65)
    print("TASK 3 DISTILLATION TEACHER INSPECTOR")
    print("=" * 65)

    # Setup board
    board = Connect4Board()
    for ch in args.moves:
        col = int(ch)
        if board.is_legal(col):
            board.make_move(col)

    print("\nTest Board State:")
    print(board)
    print(f"Legal Moves: {board.legal_moves()}\n")

    # Load baseline
    if b_path.exists():
        model_base, _ = load_checkpoint(str(b_path))
        base_probs = model_base.predict_distribution(board, temperature=1.0)
    else:
        print(f"Note: Baseline checkpoint not found at {b_path} (mocking for test).")
        base_probs = np.full(7, 1.0 / 7)

    # Load improved
    if i_path.exists():
        model_imp, _ = load_checkpoint(str(i_path))
        imp_probs = model_imp.predict_distribution(board, temperature=1.0)
    else:
        print(f"Note: Improved checkpoint not found at {i_path} (mocking for test).")
        imp_probs = np.full(7, 1.0 / 7)

    print(f"{'Col':<5} | {'Baseline (R0) Prob':<20} | {'Improved (R3) Prob':<20}")
    print("-" * 52)
    for c in range(7):
        legal_tag = " (legal)" if board.is_legal(c) else " (ILLEGAL)"
        print(f"{c:<5} | {base_probs[c]:<20.4f} | {imp_probs[c]:<20.4f}{legal_tag}")
    print("-" * 52)
    print(f"Checksum sum(p): Base={base_probs.sum():.4f}, Imp={imp_probs.sum():.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
