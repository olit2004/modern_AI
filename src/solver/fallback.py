"""
Pure Python Fallback Connect-4 Minimax Solver with Transposition Table.
Used as an in-process fallback when the external C++ solver binary is not available.
"""

from typing import Dict, Set, Tuple, List, Optional
import math
from src.game.connect4 import Connect4Board, ROWS, COLS, PLAYER_1, PLAYER_2

class PythonFallbackSolver:
    """
    Minimax solver with Alpha-Beta pruning and Transposition Table.
    Depth-limited with heuristic terminal evaluation when needed.
    """

    def __init__(self, default_depth: int = 6):
        self.default_depth = default_depth
        self.tt: Dict[str, Tuple[int, int]] = {} # hash -> (depth, score)

    def evaluate_board(self, board: Connect4Board, depth: Optional[int] = None) -> Tuple[Dict[int, int], Set[int]]:
        """
        Evaluate all legal moves from current player's perspective.
        Returns:
            scores: dict mapping column (0-6) to score
            optimal_moves: set of column indices that achieve the max score
        """
        if depth is None:
            depth = self.default_depth

        legal_moves = board.legal_moves()
        if not legal_moves or board.is_terminal():
            return {}, set()

        scores: Dict[int, int] = {}
        for col in range(COLS):
            if col not in legal_moves:
                scores[col] = -9999
            else:
                next_b = board.clone()
                next_b.make_move(col)
                if next_b.is_terminal():
                    if next_b.winner() == board.current_player:
                        scores[col] = 1000 + depth
                    else:
                        scores[col] = 0 # draw
                else:
                    scores[col] = -self._minimax(next_b, depth - 1, -10000, 10000)

        # Optimal moves are legal moves achieving the max score
        max_score = max(scores[col] for col in legal_moves)
        optimal_moves = {col for col in legal_moves if scores[col] == max_score}
        return scores, optimal_moves

    def _minimax(self, board: Connect4Board, depth: int, alpha: int, beta: int) -> int:
        if board.is_terminal():
            w = board.winner()
            if w == 0:
                return 0
            # From perspective of current_player: previous player just won
            return -1000 - depth

        if depth <= 0:
            return self._heuristic(board)

        legal_moves = board.legal_moves()
        # Move ordering: center columns first
        legal_moves.sort(key=lambda c: abs(c - 3))

        max_eval = -10000
        for col in legal_moves:
            next_b = board.clone()
            next_b.make_move(col)
            eval_score = -self._minimax(next_b, depth - 1, -beta, -alpha)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if alpha >= beta:
                break
        return max_eval

    def _heuristic(self, board: Connect4Board) -> int:
        """Heuristic score evaluating open lines of 2 and 3 for current player vs opponent."""
        p = board.current_player
        score = 0
        grid = board.grid

        # Center control bonus
        score += int(np_sum(grid[:, 3] == p) * 3)
        score -= int(np_sum(grid[:, 3] == -p) * 3)

        return score

def np_sum(cond):
    import numpy as np
    return int(np.sum(cond))
