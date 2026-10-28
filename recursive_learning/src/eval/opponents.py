"""
Fixed Benchmark Opponents for Connect-4 Evaluation.
1. RandomPlayer: Baseline random legal moves.
2. HeuristicPlayer: 1-step tactical win & block with center preference.
3. MinimaxPlayer: Depth-limited minimax with positional evaluation.
4. ModelPlayer: Wrapper around Connect4Net policy.
"""

import random
from typing import Optional, List, Tuple
import numpy as np

from src.game.connect4 import Connect4Board, COLS, ROWS, PLAYER_1, PLAYER_2

class Player:
    def __init__(self, name: str):
        self.name = name

    def get_move(self, board: Connect4Board) -> int:
        raise NotImplementedError

class RandomPlayer(Player):
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Random")
        self.rng = random.Random(seed)

    def get_move(self, board: Connect4Board) -> int:
        legal = board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available.")
        return self.rng.choice(legal)

class HeuristicPlayer(Player):
    """
    1-step tactical player:
    1. If there's an immediate winning move, take it.
    2. If opponent has an immediate winning move, block it.
    3. Otherwise, prefer center columns [3, 2, 4, 1, 5, 0, 6].
    """
    def __init__(self, name: str = "Heuristic"):
        super().__init__(name)
        self.column_preference = [3, 2, 4, 1, 5, 0, 6]

    def get_move(self, board: Connect4Board) -> int:
        legal = board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available.")

        # 1. Immediate win
        for col in legal:
            b_win = board.clone()
            b_win.make_move(col)
            if b_win.winner() == board.current_player:
                return col

        # 2. Immediate block
        opp = -board.current_player
        for col in legal:
            b_block = Connect4Board(board.grid, current_player=opp)
            b_block.make_move(col)
            if b_block.winner() == opp:
                return col

        # 3. Preference order
        for col in self.column_preference:
            if col in legal:
                return col

        return legal[0]

class MinimaxPlayer(Player):
    """
    Depth-limited Minimax player with alpha-beta pruning and center-weighted evaluation.
    Default depth 3 or 4 runs in milliseconds per move.
    """
    def __init__(self, depth: int = 3, name: Optional[str] = None):
        super().__init__(name or f"Minimax(depth={depth})")
        self.depth = depth

    def get_move(self, board: Connect4Board) -> int:
        legal = board.legal_moves()
        if not legal:
            raise ValueError("No legal moves available.")
        if len(legal) == 1:
            return legal[0]

        # 1-ply immediate win shortcut
        for col in legal:
            b = board.clone()
            b.make_move(col)
            if b.winner() == board.current_player:
                return col

        best_score = -100000
        best_col = legal[0]
        # Center column ordering
        sorted_legal = sorted(legal, key=lambda c: abs(c - 3))

        for col in sorted_legal:
            b = board.clone()
            b.make_move(col)
            score = -self._alphabeta(b, self.depth - 1, -100000, 100000)
            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def _alphabeta(self, board: Connect4Board, depth: int, alpha: int, beta: int) -> int:
        if board.is_terminal():
            w = board.winner()
            if w == 0:
                return 0
            return -10000 - depth

        if depth <= 0:
            return self._evaluate(board)

        legal = board.legal_moves()
        sorted_legal = sorted(legal, key=lambda c: abs(c - 3))

        value = -100000
        for col in sorted_legal:
            b = board.clone()
            b.make_move(col)
            score = -self._alphabeta(b, depth - 1, -beta, -alpha)
            value = max(value, score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def _evaluate(self, board: Connect4Board) -> int:
        p = board.current_player
        grid = board.grid
        # Column 3 weight 4, columns 2,4 weight 2, others 1
        weights = np.array([1, 2, 3, 5, 3, 2, 1], dtype=np.int32)
        my_pieces = (grid == p).astype(np.int32)
        opp_pieces = (grid == -p).astype(np.int32)

        my_score = int(np.sum(my_pieces * weights))
        opp_score = int(np.sum(opp_pieces * weights))
        return (my_score - opp_score) * 10

class ModelPlayer(Player):
    """Wraps a trained Connect4Net as an opponent or playable agent."""
    def __init__(self, model, greedy: bool = True, temperature: float = 1.0, name: str = "Agent"):
        super().__init__(name)
        self.model = model
        self.greedy = greedy
        self.temperature = temperature

    def get_move(self, board: Connect4Board) -> int:
        return self.model.select_action(board, greedy=self.greedy, temperature=self.temperature)
