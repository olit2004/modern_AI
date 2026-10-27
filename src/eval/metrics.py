"""
Evaluation Metrics:
1. Optimal-action agreement: fraction of positions where model's choice is in solver's optimal set.
2. Diversity metrics: unique canonical positions and column entropy (detects repetitive play/collapse).
3. Breakdown by game stage (early <= 8, mid 9-20, late > 20).
"""

from typing import List, Dict, Any, Optional
import math
import numpy as np
from src.game.connect4 import Connect4Board, COLS
from src.eval.opponents import Player

def compute_optimal_action_agreement(
    player: Player,
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates how often player picks a move in the solver's optimal set.
    Returns overall agreement and breakdown by game stage.
    """
    if not records:
        return {"overall_agreement": 0.0, "total_positions": 0}

    total = 0
    correct = 0

    stage_counts = {"early": 0, "mid": 0, "late": 0}
    stage_correct = {"early": 0, "mid": 0, "late": 0}

    for rec in records:
        grid = np.array(rec["grid"], dtype=np.int8)
        b = Connect4Board(grid, current_player=rec["current_player"])
        if b.is_terminal():
            continue

        optimal_moves = set(rec["optimal_moves"])
        chosen_col = player.get_move(b)

        is_optimal = chosen_col in optimal_moves
        total += 1
        if is_optimal:
            correct += 1

        pieces = rec.get("num_pieces", b.num_pieces())
        if pieces <= 8:
            stage = "early"
        elif pieces <= 20:
            stage = "mid"
        else:
            stage = "late"

        stage_counts[stage] += 1
        if is_optimal:
            stage_correct[stage] += 1

    stage_breakdown = {}
    for st in ["early", "mid", "late"]:
        st_tot = stage_counts[st]
        st_cor = stage_correct[st]
        stage_breakdown[st] = {
            "positions": st_tot,
            "correct": st_cor,
            "agreement": float(st_cor / max(1, st_tot))
        }

    return {
        "overall_agreement": float(correct / max(1, total)),
        "correct_positions": correct,
        "total_positions": total,
        "by_stage": stage_breakdown
    }

def compute_diversity(records_or_games: List[Any]) -> Dict[str, Any]:
    """
    Measures data diversity to flag mode collapse or repetition:
    - Number of unique canonical positions
    - Diversity ratio: unique / total
    - Move distribution entropy
    """
    total = len(records_or_games)
    if total == 0:
        return {"unique_canonical": 0, "total": 0, "diversity_ratio": 0.0, "entropy": 0.0}

    hashes = []
    actions = []

    for item in records_or_games:
        if isinstance(item, dict):
            hashes.append(item.get("canonical_hash", ""))
            if "chosen_action" in item:
                actions.append(item["chosen_action"])
        elif isinstance(item, str):
            hashes.append(item)

    unique_hashes = set(hashes)
    num_unique = len(unique_hashes)
    diversity_ratio = float(num_unique / total)

    # Entropy of action distribution if available
    entropy = 0.0
    if actions:
        counts = np.bincount(actions, minlength=COLS).astype(np.float64)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))

    return {
        "unique_canonical": num_unique,
        "total_positions": total,
        "diversity_ratio": diversity_ratio,
        "action_entropy": entropy
    }
