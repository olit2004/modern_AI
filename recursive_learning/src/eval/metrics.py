"""
Evaluation Metrics:
1. Optimal-action agreement: fraction of positions where model's choice is in solver's optimal set (fastest win).
2. Outcome-preserving agreement: fraction of positions where model's choice preserves the game-theoretic outcome (win remains win, draw remains draw).
3. Diversity metrics: unique canonical positions and column entropy (detects repetitive play/collapse).
4. Breakdown by game stage (early <= 8, mid 9-20, late > 20).
"""

from typing import List, Dict, Any, Optional, Set
import math
import numpy as np
from src.game.connect4 import Connect4Board, COLS
from src.eval.opponents import Player

def compute_optimal_action_agreement(
    player: Player,
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates player on both:
    - Strict optimal-move agreement (fastest win)
    - Outcome-preserving agreement (maintains win/draw outcome)
    Returns overall metrics and breakdown by game stage.
    """
    if not records:
        return {
            "overall_agreement": 0.0,
            "optimal_agreement": 0.0,
            "outcome_agreement": 0.0,
            "total_positions": 0
        }

    total = 0
    optimal_correct = 0
    outcome_correct = 0

    stage_counts = {"early": 0, "mid": 0, "late": 0}
    stage_optimal = {"early": 0, "mid": 0, "late": 0}
    stage_outcome = {"early": 0, "mid": 0, "late": 0}

    for rec in records:
        grid = np.array(rec["grid"], dtype=np.int8)
        b = Connect4Board(grid, current_player=rec["current_player"])
        if b.is_terminal():
            continue

        optimal_moves = set(rec.get("optimal_moves", []))
        scores_raw = rec.get("scores", {})
        # Parse scores dictionary to int keys and int values
        scores: Dict[int, int] = {}
        if isinstance(scores_raw, dict):
            for k, v in scores_raw.items():
                try:
                    scores[int(k)] = int(v)
                except (ValueError, TypeError):
                    pass

        chosen_col = player.get_move(b)
        is_optimal = chosen_col in optimal_moves

        # Determine outcome-preserving status
        if scores and chosen_col in scores:
            chosen_score = scores[chosen_col]
            # Max score among legal moves
            legal_scores = [scores[c] for c in b.legal_moves() if c in scores]
            max_score = max(legal_scores) if legal_scores else chosen_score

            if max_score > 0:
                # Theoretical WIN available: any winning move (>0) preserves the win
                is_outcome = chosen_score > 0
            elif max_score == 0:
                # Theoretical DRAW available: any drawing move (==0) preserves the draw
                is_outcome = chosen_score == 0
            else:
                # Theoretical LOSS: only optimal defense (slowest loss) preserves best outcome
                is_outcome = is_optimal
        else:
            is_outcome = is_optimal

        total += 1
        if is_optimal:
            optimal_correct += 1
        if is_outcome:
            outcome_correct += 1

        pieces = rec.get("num_pieces", b.num_pieces())
        if pieces <= 8:
            stage = "early"
        elif pieces <= 20:
            stage = "mid"
        else:
            stage = "late"

        stage_counts[stage] += 1
        if is_optimal:
            stage_optimal[stage] += 1
        if is_outcome:
            stage_outcome[stage] += 1

    stage_breakdown = {}
    for st in ["early", "mid", "late"]:
        st_tot = stage_counts[st]
        st_opt = stage_optimal[st]
        st_out = stage_outcome[st]
        stage_breakdown[st] = {
            "positions": st_tot,
            "optimal_correct": st_opt,
            "outcome_correct": st_out,
            "agreement": float(st_opt / max(1, st_tot)),
            "optimal_agreement": float(st_opt / max(1, st_tot)),
            "outcome_agreement": float(st_out / max(1, st_tot))
        }

    return {
        "overall_agreement": float(optimal_correct / max(1, total)),
        "optimal_agreement": float(optimal_correct / max(1, total)),
        "outcome_agreement": float(outcome_correct / max(1, total)),
        "optimal_correct_positions": optimal_correct,
        "outcome_correct_positions": outcome_correct,
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
        return {"unique_canonical": 0, "total": 0, "diversity_ratio": 0.0, "entropy": 0.0, "action_entropy": 0.0}

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
        counts = {}
        for a in actions:
            counts[a] = counts.get(a, 0) + 1
        for count in counts.values():
            p = count / len(actions)
            if p > 0:
                entropy -= p * math.log2(p)

    return {
        "unique_canonical": num_unique,
        "total": total,
        "total_positions": total,
        "diversity_ratio": diversity_ratio,
        "entropy": entropy,
        "action_entropy": entropy
    }
