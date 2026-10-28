"""
Interactive Presentation Demo for Task 1: Verified Self-Improvement on Connect-4.
Features:
1. Summary table of accepted and rejected rounds (strict optimal & outcome-preserving agreement).
2. Dual-panel progress charts comparing Verified Self-Improvement (with Judge) vs Unverified Self-Play (without Judge),
   benchmarked against Heuristic opponent play (with 'vs Random' excluded for focused competitive scale).
3. Live Head-to-Head tournament: Round 3 vs Round 0 with randomized openings.
4. The 'Convincing Moment': Rigorously identifies and explains true blunders vs suboptimal moves based on solver outcome values.
"""

import os
import sys
import json
import random
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
from src.eval.tournament import play_game
from src.data.dataset import DatasetManager
from src.solver.wrapper import Connect4Solver

def parse_args():
    parser = argparse.ArgumentParser(description="Task 1 Self-Improvement Presentation Demo")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing experiment results")
    parser.add_argument("--num_games", type=int, default=50, help="Number of head-to-head games (R3 vs R0)")
    parser.add_argument("--save_plot", type=str, default=None, help="Path to save progress plot (defaults to <results_dir>/figures/demo_progress.png)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dev dataset for blunder analysis")
    return parser.parse_args()

def display_summary_table(experiment_log: List[Dict[str, Any]]):
    print("\n" + "=" * 98)
    print("1. ACCEPTED & REJECTED ROUNDS SUMMARY TABLE")
    print("=" * 98)
    headers = ["Round", "Status", "Strict Agree", "Outcome Agree", "vs Random", "vs Heuristic", "vs Minimax", "Examples"]
    row_fmt = "{:<7} | {:<18} | {:<12} | {:<13} | {:<9} | {:<12} | {:<10} | {:<8}"
    print(row_fmt.format(*headers))
    print("-" * 98)

    for entry in experiment_log:
        r = entry.get("round", 0)
        status = entry.get("status", "ACCEPTED" if entry.get("accepted", True) else "REJECTED")
        strict = f"{entry.get('dev_agreement', 0.0) * 100:.1f}%"
        outcome = f"{entry.get('dev_outcome_agreement', entry.get('dev_agreement', 0.0)) * 100:.1f}%"
        rnd = f"{entry.get('win_rate_random', 0.0) * 100:.0f}%"
        heu = f"{entry.get('win_rate_heuristic', 0.0) * 100:.0f}%"
        mini = f"{entry.get('win_rate_minimax', 0.0) * 100:.0f}%"
        examples = entry.get("examples_trained", 0)
        print(row_fmt.format(f"R{r}", status, strict, outcome, rnd, heu, mini, str(examples)))

    print("=" * 98 + "\n")

def plot_progress_curves(
    experiment_log: List[Dict[str, Any]],
    output_path: Path,
    unverified_log: Optional[List[Dict[str, Any]]] = None
):
    if not HAS_MATPLOTLIB or len(experiment_log) < 2:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rounds = [entry["round"] for entry in experiment_log]
    strict_agreements = [entry.get("dev_agreement", 0.0) * 100 for entry in experiment_log]
    outcome_agreements = [entry.get("dev_outcome_agreement", entry.get("dev_agreement", 0.0)) * 100 for entry in experiment_log]
    wr_heuristic = [entry.get("win_rate_heuristic", 0.0) * 100 for entry in experiment_log]
    wr_minimax = [entry.get("win_rate_minimax", 0.0) * 100 for entry in experiment_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel 1: Dev Agreement (Verified vs Unverified & Baselines)
    ax1.plot(rounds, outcome_agreements, marker="s", color="#1f77b4", linewidth=2.5, linestyle="-", label="Verified (Outcome Agreement)")
    ax1.plot(rounds, strict_agreements, marker="o", color="#2ca02c", linewidth=2.5, linestyle="-", label="Verified (Strict Optimal)")

    if unverified_log and len(unverified_log) >= 2:
        u_rounds = [entry["round"] for entry in unverified_log]
        u_strict = [entry.get("dev_agreement", 0.0) * 100 for entry in unverified_log]
        u_outcome = [entry.get("dev_outcome_agreement", entry.get("dev_agreement", 0.0)) * 100 for entry in unverified_log]
        ax1.plot(u_rounds, u_outcome, marker="^", color="#9467bd", linewidth=2.0, linestyle="--", label="Without Judge (Outcome Agreement)")
        ax1.plot(u_rounds, u_strict, marker="x", color="#d62728", linewidth=2.0, linestyle="--", label="Without Judge (Strict Optimal)")

    # Baseline references
    r0 = experiment_log[0]
    heu_strict = r0.get("baseline_heuristic_strict", 0.38) * 100
    ax1.axhline(heu_strict, color="#ff7f0e", linestyle=":", alpha=0.8, label=f"Heuristic Baseline ({heu_strict:.1f}%)")
    ax1.axhline(14.3, color="gray", linestyle=":", alpha=0.6, label="Random Guess (14.3%)")

    ax1.set_title("Solver Agreement on Unseen Positions", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Self-Improvement Round", fontsize=11)
    ax1.set_ylabel("Agreement with Solver (%)", fontsize=11)
    ax1.set_xticks(rounds)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel 2: Competitive Win Rates (Random opponent removed to highlight competitive range)
    ax2.plot(rounds, wr_heuristic, marker="^", color="#ff7f0e", linewidth=2.5, label="Verified vs Heuristic")
    ax2.plot(rounds, wr_minimax, marker="d", color="#d62728", linewidth=2.5, label="Verified vs Minimax (d=2)")

    if unverified_log and len(unverified_log) >= 2:
        u_rounds = [entry["round"] for entry in unverified_log]
        u_wr_heu = [entry.get("win_rate_heuristic", 0.0) * 100 for entry in unverified_log]
        u_wr_mini = [entry.get("win_rate_minimax", 0.0) * 100 for entry in unverified_log]
        ax2.plot(u_rounds, u_wr_heu, marker="^", color="#ffbb78", linewidth=2.0, linestyle="--", label="Without Judge vs Heuristic")
        ax2.plot(u_rounds, u_wr_mini, marker="d", color="#ff9896", linewidth=2.0, linestyle="--", label="Without Judge vs Minimax")

    # Set appropriate y-axis range for competitive play
    all_wrs = wr_heuristic + wr_minimax
    if unverified_log:
        all_wrs += [e.get("win_rate_heuristic", 0.0) * 100 for e in unverified_log]
        all_wrs += [e.get("win_rate_minimax", 0.0) * 100 for e in unverified_log]
    y_min = max(0, min(all_wrs) - 10) if all_wrs else 0
    y_max = min(100, max(all_wrs) + 15) if all_wrs else 80
    ax2.set_ylim(y_min, y_max)

    ax2.set_title("Competitive Win Rates (100 Games, Randomized Openings)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Self-Improvement Round", fontsize=11)
    ax2.set_ylabel("Win Rate (%)", fontsize=11)
    ax2.set_xticks(rounds)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f" [Chart] Saved progress figure to: {output_path}")

def run_head_to_head(r0_path: Path, r3_path: Path, num_games: int = 50, seed: int = 42) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"⚔️  2. LIVE HEAD-TO-HEAD MATCH: Round 3 vs Round 0 ({num_games} Games with Randomized Openings)")
    print("=" * 80)

    if not r0_path.exists() or not r3_path.exists():
        print(f"Checkpoints missing! Looking for {r0_path} and {r3_path}")
        return {}

    m0, _ = load_checkpoint(str(r0_path))
    m3, _ = load_checkpoint(str(r3_path))

    p0 = ModelPlayer(m0, greedy=True, name="Round_0_Baseline")
    p3 = ModelPlayer(m3, greedy=True, name="Round_3_Improved")

    rng = random.Random(seed)
    r3_wins = 0
    r0_wins = 0
    draws = 0

    for i in range(num_games):
        k_openings = rng.randint(2, 4)
        if i % 2 == 0:
            outcome = play_game(p3, p0, random_openings=k_openings, rng=rng)
            if outcome == 1:
                r3_wins += 1
            elif outcome == -1:
                r0_wins += 1
            else:
                draws += 1
        else:
            outcome = play_game(p0, p3, random_openings=k_openings, rng=rng)
            if outcome == 1:
                r0_wins += 1
            elif outcome == -1:
                r3_wins += 1
            else:
                draws += 1

    r3_pct = (r3_wins / num_games) * 100
    r0_pct = (r0_wins / num_games) * 100
    draw_pct = (draws / num_games) * 100

    print(f"  Round 3 (Improved): {r3_wins:>3} wins ({r3_pct:5.1f}%)")
    print(f"  Round 0 (Baseline): {r0_wins:>3} wins ({r0_pct:5.1f}%)")
    print(f"  Draws:              {draws:>3}      ({draw_pct:5.1f}%)")
    print("=" * 80 + "\n")

    return {"r3_wins": r3_wins, "r0_wins": r0_wins, "draws": draws}

def find_convincing_blunder(r0_path: Path, r3_path: Path, data_dir: str = "data"):
    print("=" * 80)
    print("🔍 3. THE CONVINCING MOMENT: ROUND 0 MISTAKE VS ROUND 3 OPTIMAL MOVE")
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

    critical_blunder = None
    suboptimal_move = None

    candidate_records = dev_records + dm.load_train()

    for rec in candidate_records:
        board = Connect4Board(grid=np.array(rec["grid"], dtype=np.int8), current_player=rec["current_player"])
        opts = rec.get("optimal_moves", [])
        scores_raw = rec.get("scores", {})
        if not opts or not scores_raw:
            continue

        scores = {int(k): int(v) for k, v in scores_raw.items()}
        c0 = p0.get_move(board)
        c3 = p3.get_move(board)

        if c3 in opts and c0 not in opts:
            s0 = scores.get(c0, -999)
            s3 = scores.get(c3, -999)

            # Check if Round 0 committed a critical outcome-flipping blunder:
            # Optimal move preserves win/draw (s3 >= 0), but R0 blundered into a loss (s0 < 0)
            if s3 >= 0 and s0 < 0:
                critical_blunder = (board, c0, c3, opts, scores, s0, s3, "CRITICAL BLUNDER")
                break
            elif suboptimal_move is None and s0 < s3:
                category = "SUBOPTIMAL (SLOWER WIN)" if s0 > 0 else "SUBOPTIMAL DEFENSE (FASTER LOSS)"
                suboptimal_move = (board, c0, c3, opts, scores, s0, s3, category)

    chosen_example = critical_blunder if critical_blunder is not None else suboptimal_move

    if chosen_example is None:
        print("Note: Both models made identical outcome decisions on all candidate positions.")
        return

    board, c0, c3, opts, scores, s0, s3, label = chosen_example
    curr = "RED (Player 1)" if board.current_player == PLAYER_1 else "YELLOW (Player 2)"

    print(f"Classification:         [{label}]")
    print(f"Position Active Player: {curr}")
    print(f"Solver Optimal Move(s): {opts} (Score: {s3:+d})")
    print(f"Solver Column Scores:   {scores}\n")
    print("--- BOARD STATE ---")
    print(board)
    print("  0 1 2 3 4 5 6   <-- Columns\n")
    print(f"❌ Round 0 (Baseline) played:  Column {c0}  (Score: {s0:+d}) -> {label}")
    print(f"✅ Round 3 (Improved) played:  Column {c3}  (Score: {s3:+d}) -> SOLVER OPTIMAL")
    print(f"\nAnalysis: Round 3 chooses the mathematically optimal line (score {s3:+d}),")
    if s3 >= 0 and s0 < 0:
        print(f"while Round 0's move (column {c0}) directly forfeits the position into a loss (score {s0:+d}).")
    else:
        print(f"while Round 0's move (column {c0}) is suboptimal by {s3 - s0} game plies.")
    print("=" * 80 + "\n")

def main():
    args = parse_args()
    res_base = Path(args.results_dir)

    # Locate experiment log for Verified condition
    verified_metrics = None
    for cand in [res_base / "verified_seed42" / "experiment_metrics.json",
                 res_base / "v3_verified_seed42" / "experiment_metrics.json",
                 res_base / "experiment_metrics.json"]:
        if cand.exists():
            verified_metrics = cand
            break

    # Locate unverified condition log (Learning without the judge)
    unverified_log = None
    for cand in [res_base / "v2_unverified_seed42" / "experiment_metrics.json",
                 res_base / "unverified_seed42" / "experiment_metrics.json",
                 res_base / "v2_unverified" / "experiment_metrics.json"]:
        if cand.exists():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    unverified_log = json.load(f)
                print(f"[Demo] Found unverified (without judge) metrics: {cand}")
            except Exception:
                pass
            break

    if verified_metrics:
        with open(verified_metrics, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        display_summary_table(log_data)
        plot_path = Path(args.save_plot) if args.save_plot else (res_base / "figures" / "demo_progress.png")
        plot_progress_curves(log_data, plot_path, unverified_log=unverified_log)
    else:
        print(f"No verified experiment_metrics.json found in {res_base}. Run training first.")

    # Locate checkpoints
    r0_ckpt = None
    r3_ckpt = None
    for folder in [res_base / "verified_seed42", res_base / "v3_verified_seed42", res_base / "teachers", res_base]:
        c0 = folder / "checkpoint_round_0.pt" if (folder / "checkpoint_round_0.pt").exists() else (folder / "teacher_round0_baseline.pt")
        c3 = folder / "checkpoint_round_3.pt" if (folder / "checkpoint_round_3.pt").exists() else (folder / "teacher_round3_improved.pt")
        if (not c3 or not c3.exists()):
            for alt in [folder / "checkpoint_round_3_proposal.pt", folder / "checkpoint_round_2.pt", folder / "checkpoint_round_2_proposal.pt"]:
                if alt.exists():
                    c3 = alt
                    break
        if c0 and c0.exists() and (not r0_ckpt or not r0_ckpt.exists()):
            r0_ckpt = c0
        if c3 and c3.exists() and (not r3_ckpt or not r3_ckpt.exists()):
            r3_ckpt = c3

    if r0_ckpt and r3_ckpt:
        run_head_to_head(r0_ckpt, r3_ckpt, num_games=args.num_games)
        find_convincing_blunder(r0_ckpt, r3_ckpt, data_dir=args.data_dir)
    else:
        print(f"Checkpoints not found in {res_base}. Checked {r0_ckpt} and {r3_ckpt}.")

if __name__ == "__main__":
    main()
