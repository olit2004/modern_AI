"""
Connect-4 Game Engine
Standard 6 rows x 7 columns board.
Supports:
- Legal move generation
- Fast win & draw detection (horizontal, vertical, both diagonals)
- Canonical position hashing with horizontal reflection symmetry
- Tensor encoding (2 planes: current player, opponent)
- Horizontal mirror transformation
"""

import numpy as np
from typing import List, Tuple, Optional

ROWS = 6
COLS = 7
PLAYER_1 = 1
PLAYER_2 = -1

class Connect4Board:
    """
    Representation of a Connect-4 board.
    Board grid is a 6x7 numpy array where:
      0: Empty
      1: Player 1 (Red / First mover)
     -1: Player 2 (Yellow / Second mover)
    Row 0 is the TOP row, Row 5 is the BOTTOM row.
    Pieces drop towards row 5.
    """

    def __init__(self, grid: Optional[np.ndarray] = None, current_player: int = PLAYER_1, move_history: Optional[List[int]] = None):
        if grid is None:
            self.grid = np.zeros((ROWS, COLS), dtype=np.int8)
        else:
            self.grid = grid.astype(np.int8).copy()
        self.current_player = current_player
        self.move_history: List[int] = list(move_history) if move_history is not None else []
        self._winner: Optional[int] = None
        self._is_terminal: Optional[bool] = None

    def clone(self) -> "Connect4Board":
        new_board = Connect4Board(self.grid, self.current_player, self.move_history)
        new_board._winner = self._winner
        new_board._is_terminal = self._is_terminal
        return new_board

    def to_move_str(self) -> str:
        """Returns 1-indexed move sequence string suitable for Pons solver."""
        return "".join(str(m + 1) for m in self.move_history)

    def legal_moves(self) -> List[int]:
        """Returns list of column indices (0-6) where top row is empty."""
        if self.is_terminal():
            return []
        return [col for col in range(COLS) if self.grid[0, col] == 0]

    def is_legal(self, col: int) -> bool:
        """Check if playing in column col is legal."""
        return 0 <= col < COLS and self.grid[0, col] == 0 and not self.is_terminal()

    def make_move(self, col: int) -> "Connect4Board":
        """
        Drop piece in column `col` for `current_player`.
        Modifies board in-place and returns self.
        Raises ValueError if move is illegal.
        """
        if not self.is_legal(col):
            raise ValueError(f"Illegal move in column {col}")

        # Find lowest available row in column
        for r in range(ROWS - 1, -1, -1):
            if self.grid[r, col] == 0:
                self.grid[r, col] = self.current_player
                self.move_history.append(col)
                # Check terminal state right after move
                self._check_win_at(r, col, self.current_player)
                break

        # Switch player
        self.current_player = -self.current_player
        return self

    def _check_win_at(self, r: int, c: int, player: int):
        """Check if move at (r, c) created a line of 4 for player."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # Forward direction
            nr, nc = r + dr, c + dc
            while 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc
            # Backward direction
            nr, nc = r - dr, c - dc
            while 0 <= nr < ROWS and 0 <= nc < COLS and self.grid[nr, nc] == player:
                count += 1
                nr -= dr
                nc -= dc

            if count >= 4:
                self._winner = player
                self._is_terminal = True
                return

        # Check for draw (full board)
        if not (self.grid[0, :] == 0).any():
            self._winner = 0  # Draw
            self._is_terminal = True
        else:
            self._winner = None
            self._is_terminal = False

    def is_terminal(self) -> bool:
        if self._is_terminal is None:
            self._recalculate_terminal()
        return bool(self._is_terminal)

    def winner(self) -> Optional[int]:
        """
        Returns:
          1: Player 1 won
         -1: Player 2 won
          0: Draw
          None: Game still in progress
        """
        if self._is_terminal is None:
            self._recalculate_terminal()
        return self._winner

    def _recalculate_terminal(self):
        """Full check of board state for win or draw."""
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                window = self.grid[r, c:c+4]
                if window[0] != 0 and (window == window[0]).all():
                    self._winner = int(window[0])
                    self._is_terminal = True
                    return

        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                window = self.grid[r:r+4, c]
                if window[0] != 0 and (window == window[0]).all():
                    self._winner = int(window[0])
                    self._is_terminal = True
                    return

        # Positive diagonal (\)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                p = self.grid[r, c]
                if p != 0 and p == self.grid[r+1, c+1] == self.grid[r+2, c+2] == self.grid[r+3, c+3]:
                    self._winner = int(p)
                    self._is_terminal = True
                    return

        # Negative diagonal (/)
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                p = self.grid[r, c]
                if p != 0 and p == self.grid[r-1, c+1] == self.grid[r-2, c+2] == self.grid[r-3, c+3]:
                    self._winner = int(p)
                    self._is_terminal = True
                    return

        # Full board check
        if not (self.grid == 0).any():
            self._winner = 0
            self._is_terminal = True
        else:
            self._winner = None
            self._is_terminal = False

    def mirror(self) -> "Connect4Board":
        """
        Returns horizontally mirrored board.
        Connect-4 board is symmetric horizontally (column 0 <-> column 6, 1 <-> 5, etc.)
        """
        mirrored_grid = np.fliplr(self.grid)
        mirrored_history = [COLS - 1 - m for m in self.move_history]
        new_board = Connect4Board(mirrored_grid, self.current_player, mirrored_history)
        new_board._winner = self._winner
        new_board._is_terminal = self._is_terminal
        return new_board

    def to_canonical_hash(self) -> str:
        """
        Returns a unique hash representing this position up to horizontal reflection symmetry
        and from the current player's perspective.
        This guarantees that identical and mirrored positions map to the exact same hash,
        preventing test/dev/train data leakage.
        """
        # Perspective of current player: 1 is self, -1 is opponent
        perspective_grid = self.grid * self.current_player
        mirrored_perspective = np.fliplr(perspective_grid)

        bytes_orig = perspective_grid.tobytes()
        bytes_mirror = mirrored_perspective.tobytes()

        # Canonical ordering: pick lexicographically smaller byte string
        canonical_bytes = min(bytes_orig, bytes_mirror)
        import hashlib
        return hashlib.sha256(canonical_bytes).hexdigest()

    def to_raw_hash(self) -> str:
        """Non-mirrored raw hash (useful for debugging)."""
        import hashlib
        perspective_grid = self.grid * self.current_player
        return hashlib.sha256(perspective_grid.tobytes()).hexdigest()

    def to_tensor_encoding(self) -> np.ndarray:
        """
        Encodes board from perspective of current player into 2 planes of shape (2, 6, 7):
        Plane 0: 1 where current player has pieces, 0 otherwise.
        Plane 1: 1 where opponent has pieces, 0 otherwise.
        dtype: float32
        """
        p = self.current_player
        plane_current = (self.grid == p).astype(np.float32)
        plane_opponent = (self.grid == -p).astype(np.float32)
        return np.stack([plane_current, plane_opponent], axis=0)

    def num_pieces(self) -> int:
        """Returns total number of pieces played on the board."""
        return int(np.count_nonzero(self.grid))

    def __str__(self) -> str:
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        for r in range(ROWS):
            row_str = " ".join(symbols[int(self.grid[r, c])] for c in range(COLS))
            lines.append(f"| {row_str} |")
        lines.append("+ " + "- " * COLS + "+")
        lines.append("  0 1 2 3 4 5 6")
        lines.append(f"Turn: {'X (1)' if self.current_player == 1 else 'O (-1)'}")
        return "\n".join(lines)
