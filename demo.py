"""
Interactive Presentation Demo for Task 1: Verified Self-Improvement on Connect-4.
Features:
1. Summary table of accepted and rejected rounds.
2. Line chart of dev accuracy and opponent win rates across rounds.
3. Live Head-to-Head tournament: Round 3 vs Round 0 (50-100 games) with score.
4. The 'Convincing Moment': Prints a board state where Round 0 blunders and Round 3 plays optimal.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.game.connect4 import Connect4Board, PLAYER_1, PLAYER_2, COLS, ROWS
from src.model.train import load_checkpoint
from src.eval.opponents import ModelPlayer
from src.data.dataset import DatasetManager
from src.solver.wrapper import Connect4Solver

def parse_args():
    parser = argparse.ArgumentParser(description="Task 1 Self-Improvement Presentation Demo")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing experiment results")
    parser.add_argument("--num_games", type=int, default=50, help="Number of head-to-head games (R3 vs R0)")
    parser.add_argument("--save_plot", type=str, default="results/figures/demo_progress.png", help="Path to save progress plot")
    return parser.parse_args()

def display_summary_table(experiment_log: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("📊 1. ACCEPTED & REJECTED ROUNDS SUMMARY TABLE")
    print("=" * 80)
    headers = ["Round", "Status", "Dev Agree (%)", "vs Random", "vs Heuristic", "vs Minimax", "Examples"]
    row_fmt = "{:<7} | {:<20} | {:<13} | {:<9} | {:<12} | {:<10} | {:<8}"
    print(row_fmt.format(*headers))
    print("-" * 80)

    for entry in experiment_log:
        r = entry.get("round", 0)
        status = entry.get("status", "ACCEPTED" if entry.get("accepted", True) else "REJECTED")
        agree = f"{entry.get('dev_agreement', 0.0) * 100:.1f}%"
        rnd = f"{entry.get('win_rate_random', 0.0) * 100:.0f}%"
        heu = f"{entry.get('win_rate_heuristic', 0.0) * 100:.0f}%"
        mini = f"{entry.get('win_rate_minimax', 0.0) * 100:.0f}%"
        examples = entry.get("examples_trained", 0)
        print(row_fmt.format(f"R{r}", status, agree, rnd, heu, mini, str(examples)))

    print("=" * 80 + "\n")

def plot_progress_curves(experiment_log: List[Dict[str, Any]], output_path: Path):
    if not HAS_MATPLOTLIB or len(experiment_log) < 2:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rounds = [entry["round"] for entry in experiment_log]
    agreements = [entry.get("dev_agreement", 0.0) * 100 for entry in experiment_log]
    wr_random = [entry.get("win_rate_random", 0.0) * 100 for entry in experiment_log]
    wr_heuristic = [entry.get("win_rate_heuristic", 0.0) * 100 for entry in experiment_log]
    wr_minimax = [entry.get("win_rate_minimax", 0.0) * 100 for entry in experiment_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Dev Agreement
    ax1.plot(rounds, agreements, marker="o", color="#2ca02c", linewidth=2.5, label="Verified Self-Improvement")
    ax1.set_title("Optimal-Move Agreement on Unseen Positions", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Self-Improvement Round", fontsize=11)
    ax1.set_ylabel("Agreement with Solver (%)", fontsize=11)
    ax1.set_xticks(rounds)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    # Panel 2: Win rates vs opponents
    ax2.plot(rounds, wr_random, marker="s", color="#1f77b4", linewidth=2, label="vs Random")
    ax2.plot(rounds, wr_heuristic, marker="^", color="#ff7f0e", linewidth=2, label="vs Heuristic")
    ax2.plot(rounds, wr_minimax, marker="d", color="#d62728", linewidth=2, label="vs Minimax (d=2)")
    ax2.set_title("Win Rates Across Self-Improvement Rounds", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Self-Improvement Round", fontsize=11)
    ax2.set_ylabel("Win Rate (%)", fontsize=11)
    ax2.set_xticks(rounds)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"📈 [Chart] Saved progress figure to: {output_path}")

def run_head_to_head(r0_path: Path, r3_path: Path, num_games: int = 50) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"⚔️  2. LIVE HEAD-TO-HEAD MATCH: Round 3 vs Round 0 ({num_games} Games)")
    print("=" * 80)

    if not r0_path.exists() or not r3_path.exists():
        print(f"Checkpoints missing! Looking for {r0_path} and {r3_path}")
        return {}

    m0, _ = load_checkpoint(str(r0_path))
    m3, _ = load_checkpoint(str(r3_path))

    p0 = ModelPlayer(m0, greedy=True, name="Round_0_Baseline")
    p3 = ModelPlayer(m3, greedy=True, name="Round_3_Improved")

    r3_wins = 0
    r0_wins = 0
    draws = 0

    for i in range(num_games):
        # Alternate sides: even games R3 is Player 1 (Red), odd games R0 is Player 1
        board = Connect4Board()
        if i % 2 == 0:
            player_map = {PLAYER_1: p3, PLAYER_2: p0}
        else:
            player_map = {PLAYER_1: p0, PLAYER_2: p3}

        while not board.is_terminal():
            active_player = player_map[board.current_player]
            move = active_player.get_move(board)
            board.make_move(move)

        winner = board.winner
        if winner is None:
            draws += 1
        elif player_map[winner] == p3:
            r3_wins += 1
        else:
            r0_wins += 1

    r3_pct = (r3_wins / num_games) * 100
    r0_pct = (r0_wins / num_games) * 100
    draw_pct = (draws / num_games) * 100

    print(f"  🏆 Round 3 (Improved): {r3_wins:>3} wins ({r3_pct:5.1f}%)")
    print(f"  📉 Round 0 (Baseline): {r0_wins:>3} wins ({r0_pct:5.1f}%)")
    print(f"  🤝 Draws:              {draws:>3}      ({draw_pct:5.1f}%)")
    print("=" * 80 + "\n")

    return {"r3_wins": r3_wins, "r0_wins": r0_wins, "draws": draws}

def find_convincing_blunder(r0_path: Path, r3_path: Path, data_dir: str = "data"):
    print("=" * 80)
    print("🔍 3. THE CONVINCING MOMENT: ROUND 0 BLUNDER VS ROUND 3 OPTIMAL MOVE")
    print("=" * 80)

    if not r0_path.exists() or not r3_path.exists():
        return

    dm = DatasetManager(data_dir)
    dev_records = dm.load_dev()
    if not dev_records:
        return

    m0, _ = load_checkpoint(str(r0_path))
    m3, _ = load_checkpoint(str(r3_path))

    p0 = ModelPlayer(m0, greedy=True)
    p3 = ModelPlayer(m3, greedy=True)

    found_position = None

    for rec in dev_records:
        board = Connect4Board(grid=rec["grid"], current_player=rec["current_player"])
        opts = rec.get("optimal_moves", [])
        if not opts:
            continue

        c0 = p0.get_move(board)
        c3 = p3.get_move(board)

        # Condition: R3 chooses optimal move, R0 makes a blunder (non-optimal)
        if c3 in opts and c0 not in opts:
            found_position = (board, c0, c3, opts, rec.get("scores", {}))
            break

    if found_position is None:
        print("No contrasting blunder found in dev samples.")
        return

    board, c0, c3, opts, scores = found_position
    curr = "RED (Player 1)" if board.current_player == PLAYER_1 else "YELLOW (Player 2)"

    print(f"Position Active Player: {curr}")
    print(f"Solver Optimal Move(s): {opts}")
    print(f"Solver Column Scores:   {scores}\n")
    print("--- BOARD STATE ---")
    print(board)
    print("  0 1 2 3 4 5 6   <-- Columns\n")
    print(f"❌ Round 0 (Baseline) played:  Column {c0}  [TACTICAL BLUNDER]")
    print(f"✅ Round 3 (Improved) played:  Column {c3}  [SOLVER-OPTIMAL MOVE]")
    print("\nInsight: Round 0 falls for a tactical trap or misses an immediate threat,")
    print("whereas Round 3's solver-verified training correctly identified the optimal column.")
    print("=" * 80 + "\n")

def main():
    args = parse_args()
    res_base = Path(args.results_dir)

    # Locate experiment log
    metrics_file = None
    for cand in [res_base / "verified_seed42" / "experiment_metrics.json",
                 res_base / "v3_verified_seed42" / "experiment_metrics.json",
                 res_base / "experiment_metrics.json"]:
        if cand.exists():
            metrics_file = cand
            break

    if metrics_file:
        with open(metrics_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        display_summary_table(log_data)
        plot_progress_curves(log_data, Path(args.save_plot))
    else:
        print(f"No experiment_metrics.json found in {res_base}. Run training first.")

    # Locate checkpoints
    r0_ckpt = None
    r3_ckpt = None
    for folder in [res_base / "verified_seed42", res_base / "v3_verified_seed42", res_base / "teachers", res_base]:
        c0 = folder / "checkpoint_round_0.pt" if (folder / "checkpoint_round_0.pt").exists() else (folder / "teacher_round0_baseline.pt")
        c3 = folder / "checkpoint_round_3.pt" if (folder / "checkpoint_round_3.pt").exists() else (folder / "teacher_round3_improved.pt")
        if c0.exists() and r0_ckpt is None:
            r0_ckpt = c0
        if c3.exists() and r3_ckpt is None:
            r3_ckpt = c3

    if r0_ckpt and r3_ckpt:
        run_head_to_head(r0_ckpt, r3_ckpt, num_games=args.num_games)
        find_convincing_blunder(r0_ckpt, r3_ckpt)
    else:
        print("Note: Run training to produce checkpoints before running head-to-head match.")

if __name__ == "__main__":
    main()
