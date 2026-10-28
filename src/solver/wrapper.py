"""
Connect-4 Perfect Solver Wrapper.
Interfaces with Pascal Pons' C++ bitboard solver executable with automatic compilation,
SQLite on-disk caching keyed by canonical position hash (with reflection symmetry support),
and pure Python fallback when C++ execution is unavailable.
"""

import os
import sys
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional, Union

from src.game.connect4 import Connect4Board, COLS, ROWS
from src.solver.fallback import PythonFallbackSolver

SOLVER_DIR = Path(__file__).resolve().parent / "cpp" / "pons"
SOLVER_BIN = SOLVER_DIR / "c4solver"
BOOK_PATH = SOLVER_DIR / "7x6.book"

class Connect4Solver:
    """
    Connect-4 Solver referee.
    Evaluates positions and returns optimal move sets and column scores.
    Uses SQLite cache to ensure expensive positions are solved at most once.
    """

    def __init__(self, cache_path: Optional[str] = None, use_weak: bool = False):
        self.use_weak = use_weak
        self.fallback_solver = PythonFallbackSolver()

        if cache_path is None:
            cache_dir = Path("results/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path = cache_dir / "solver_labels.sqlite"
        else:
            self.cache_path = Path(cache_path)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_cache()
        self.cpp_available = self._ensure_binary()

    def _init_cache(self):
        """Initialize SQLite database for cached evaluations."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_cache (
                    canonical_hash TEXT PRIMARY KEY,
                    raw_hash TEXT,
                    scores_json TEXT,
                    optimal_moves_json TEXT
                )
            """)
            conn.commit()

    def _ensure_binary(self) -> bool:
        """Ensure C++ solver binary is compiled and book is present."""
        if SOLVER_BIN.exists() and os.access(SOLVER_BIN, os.X_OK):
            return True

        # Try to compile with make
        if SOLVER_DIR.exists():
            try:
                subprocess.run(
                    ["make", "-C", str(SOLVER_DIR)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30
                )
                if SOLVER_BIN.exists() and os.access(SOLVER_BIN, os.X_OK):
                    return True
            except Exception as e:
                print(f"[Solver] Note: C++ compilation failed ({e}), using Python fallback solver.")
        return False

    def is_canonical_orientation(self, board: Connect4Board) -> bool:
        """
        Check if the board matches its canonical orientation.
        Canonical orientation is min(bytes, mirror_bytes).
        """
        perspective_grid = board.grid * board.current_player
        mirrored_perspective = perspective_grid[:, ::-1]
        return perspective_grid.tobytes() <= mirrored_perspective.tobytes()

    def get_cached(self, canonical_hash: str) -> Optional[Tuple[Dict[int, int], Set[int]]]:
        """Fetch evaluation from cache if present."""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT scores_json, optimal_moves_json FROM position_cache WHERE canonical_hash = ?",
                (canonical_hash,)
            )
            row = cursor.fetchone()
            if row:
                scores_raw = json.loads(row[0])
                scores = {int(k): v for k, v in scores_raw.items()}
                optimal_moves = set(json.loads(row[1]))
                return scores, optimal_moves
        return None

    def store_cache_batch(self, entries: List[Tuple[str, str, Dict[int, int], Set[int]]]):
        """Batch store canonical entries into SQLite."""
        if not entries:
            return
        with sqlite3.connect(self.cache_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO position_cache (canonical_hash, raw_hash, scores_json, optimal_moves_json) VALUES (?, ?, ?, ?)",
                [(c_hash, r_hash, json.dumps(scores), json.dumps(list(opts))) for c_hash, r_hash, scores, opts in entries]
            )
            conn.commit()

    def evaluate(self, board: Connect4Board) -> Tuple[Dict[int, int], Set[int]]:
        """
        Evaluate a single board position.
        Returns:
            scores: Dict[col, score] for col in 0..6 (-1000 or -9999 for illegal)
            optimal_moves: Set[col] containing all columns achieving max legal score
        """
        results = self.evaluate_batch([board])
        return results[0]

    def evaluate_batch(self, boards: List[Connect4Board]) -> List[Tuple[Dict[int, int], Set[int]]]:
        """
        Evaluate a batch of positions efficiently with caching.
        """
        outputs: List[Optional[Tuple[Dict[int, int], Set[int]]]] = [None] * len(boards)
        uncached_indices: List[int] = []

        # Step 1: Check cache for each position
        for i, b in enumerate(boards):
            if b.is_terminal():
                outputs[i] = ({c: -1000 for c in range(COLS)}, set())
                continue

            c_hash = b.to_canonical_hash()
            cached = self.get_cached(c_hash)
            if cached is not None:
                canonical_scores, canonical_opts = cached
                # Check orientation: if query board is mirrored relative to canonical, flip columns
                if self.is_canonical_orientation(b):
                    outputs[i] = (canonical_scores.copy(), set(canonical_opts))
                else:
                    flipped_scores = {COLS - 1 - col: s for col, s in canonical_scores.items()}
                    flipped_opts = {COLS - 1 - col for col in canonical_opts}
                    outputs[i] = (flipped_scores, flipped_opts)
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return [out for out in outputs if out is not None]

        # Step 2: Solve uncached positions
        uncached_boards = [boards[i] for i in uncached_indices]
        solved_canonical_entries: List[Tuple[str, str, Dict[int, int], Set[int]]] = []

        if self.cpp_available:
            solved_results = self._solve_with_cpp(uncached_boards)
        else:
            solved_results = [self.fallback_solver.evaluate_board(b) for b in uncached_boards]

        for idx, b, (scores, opts) in zip(uncached_indices, uncached_boards, solved_results):
            c_hash = b.to_canonical_hash()
            r_hash = b.to_raw_hash()

            # Determine canonical form to cache
            if self.is_canonical_orientation(b):
                canonical_scores = scores
                canonical_opts = opts
            else:
                canonical_scores = {COLS - 1 - col: s for col, s in scores.items()}
                canonical_opts = {COLS - 1 - col for col in opts}

            solved_canonical_entries.append((c_hash, r_hash, canonical_scores, canonical_opts))
            outputs[idx] = (scores, opts)

        # Step 3: Write newly solved entries to SQLite cache
        self.store_cache_batch(solved_canonical_entries)

        return [out for out in outputs if out is not None]

    def _solve_with_cpp(self, boards: List[Connect4Board]) -> List[Tuple[Dict[int, int], Set[int]]]:
        """Call Pascal Pons C++ solver binary over stdin/stdout in chunks to prevent subprocess timeouts."""
        chunk_size = 200
        all_results: List[Tuple[Dict[int, int], Set[int]]] = []
        for start_idx in range(0, len(boards), chunk_size):
            chunk = boards[start_idx:start_idx + chunk_size]
            all_results.extend(self._solve_chunk_with_cpp(chunk))
        return all_results

    def _solve_chunk_with_cpp(self, boards: List[Connect4Board]) -> List[Tuple[Dict[int, int], Set[int]]]:
        """Solve a chunk of positions with the C++ solver binary."""
        cmd = [str(SOLVER_BIN), "-a"]
        if BOOK_PATH.exists():
            cmd.extend(["-b", str(BOOK_PATH)])
        if self.use_weak:
            cmd.append("-w")

        # Prepare move sequences
        lines = []
        for b in boards:
            move_str = b.to_move_str()
            lines.append(move_str if move_str else " ")

        input_data = "\n".join(lines) + "\n"
        timeout_sec = max(60, len(boards) * 2)

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(SOLVER_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = proc.communicate(input=input_data, timeout=timeout_sec)
            if proc.returncode != 0:
                print(f"[Solver] C++ solver error: {stderr.strip()}, falling back to Python solver for this chunk.")
                return [self.fallback_solver.evaluate_board(b) for b in boards]

            results = []
            output_lines = [l.strip() for l in stdout.strip().split("\n") if l.strip()]
            # Filter out "Loading opening book..." line if present
            output_lines = [l for l in output_lines if not l.startswith("Loading opening book")]

            for i, b in enumerate(boards):
                if i < len(output_lines):
                    parts = output_lines[i].split()
                    # Pons output: <move_seq> <score_0> <score_1> ... <score_6>
                    # If move_seq is empty, parts will only have the 7 scores
                    if len(parts) >= 8:
                        score_vals = [int(x) for x in parts[1:8]]
                    elif len(parts) == 7:
                        score_vals = [int(x) for x in parts[:7]]
                    else:
                        score_vals = [-1000] * 7

                    scores = {col: score_vals[col] for col in range(COLS)}
                    legal = b.legal_moves()
                    legal_scores = [scores[c] for c in legal if scores[c] > -1000]
                    if legal_scores:
                        max_s = max(legal_scores)
                        opts = {c for c in legal if scores[c] == max_s}
                    else:
                        opts = set(legal)
                    results.append((scores, opts))
                else:
                    results.append(self.fallback_solver.evaluate_board(b))
            return results
        except Exception as e:
            print(f"[Solver] Subprocess chunk execution failed: {e}. Falling back to Python solver for this chunk.")
            return [self.fallback_solver.evaluate_board(b) for b in boards]
