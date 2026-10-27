"""
Self-Improvement Loop Engine for Connect-4.
Implements:
- Version 1: Fixed dataset baseline (matching data & step budget)
- Version 2: Unverified self-improvement (trains on own unverified moves)
- Version 3: Solver-verified self-improvement (solver verifies & labels collected positions)
Strict budget parity across all conditions.
Resumable execution and acceptance gate rollback.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.game.connect4 import Connect4Board, COLS, ROWS, PLAYER_1, PLAYER_2
from src.solver.wrapper import Connect4Solver
from src.data.dataset import DatasetManager
from src.model.net import Connect4Net
from src.model.train import train_model, save_checkpoint, load_checkpoint
from src.eval.opponents import ModelPlayer, RandomPlayer, HeuristicPlayer, MinimaxPlayer
from src.eval.tournament import run_tournament
from src.eval.metrics import compute_optimal_action_agreement, compute_diversity
from src.utils.platform import set_seed, get_device

class SelfImprovementEngine:
    """
    Manages self-improvement rounds with verified budgets,
    acceptance gating, and resumability.
    """

    def __init__(
        self,
        condition: str,
        output_dir: str,
        solver: Connect4Solver,
        dataset_manager: DatasetManager,
        seed: int = 42,
        num_rounds: int = 3,
        examples_per_round: int = 400,
        epochs_per_round: int = 5,
        batch_size: int = 32,
        acceptance_tolerance: float = 0.02,
        temperature: float = 0.8
    ):
        assert condition in ["v1_fixed", "v2_unverified", "v3_verified"], f"Unknown condition {condition}"
        self.condition = condition
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.solver = solver
        self.dataset_manager = dataset_manager
        self.seed = seed
        self.num_rounds = num_rounds
        self.examples_per_round = examples_per_round
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.acceptance_tolerance = acceptance_tolerance
        self.temperature = temperature

        self.metrics_file = self.output_dir / "experiment_metrics.json"
        self.experiment_log: List[Dict[str, Any]] = self._load_existing_log()

    def _load_existing_log(self) -> List[Dict[str, Any]]:
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_log(self):
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.experiment_log, f, indent=2)

    def get_latest_checkpoint_path(self, up_to_round: int) -> Optional[Path]:
        """Find the latest accepted checkpoint up to up_to_round."""
        for r in range(up_to_round, -1, -1):
            ckpt = self.output_dir / f"checkpoint_round_{r}.pt"
            if ckpt.exists():
                return ckpt
        return None

    def evaluate_model(self, model: Connect4Net, dev_records: List[Dict], round_num: int) -> Dict[str, Any]:
        """Evaluate model on dev agreement and gameplay tournament."""
        agent_player = ModelPlayer(model, greedy=True, name=f"Agent_R{round_num}")
        agreement_metrics = compute_optimal_action_agreement(agent_player, dev_records)

        opponents = [
            RandomPlayer(seed=self.seed),
            HeuristicPlayer(),
            MinimaxPlayer(depth=2)
        ]
        tournament_metrics = run_tournament(agent_player, opponents, num_games_per_opponent=10, seed=self.seed)

        return {
            "dev_agreement": agreement_metrics["overall_agreement"],
            "agreement_details": agreement_metrics,
            "tournament": tournament_metrics
        }

    def collect_self_play_positions(self, model: Connect4Net, num_target_positions: int) -> List[Dict[str, Any]]:
        """
        Let the model play against itself and fixed opponents to run into mistakes and real positions.
        """
        set_seed(self.seed + len(self.experiment_log))
        rng = random.Random(self.seed + len(self.experiment_log))

        collected: List[Dict[str, Any]] = []
        heuristic = HeuristicPlayer()

        while len(collected) < num_target_positions:
            board = Connect4Board()
            # 60% self-play, 40% vs heuristic opponent to encounter tough tactical threats
            play_mode = rng.random()

            while not board.is_terminal() and len(collected) < num_target_positions:
                legal = board.legal_moves()
                if not legal:
                    break

                curr_p = board.current_player
                # Save position before move
                pos_info = {
                    "grid": board.grid.tolist(),
                    "current_player": curr_p,
                    "canonical_hash": board.to_canonical_hash(),
                    "move_history": list(board.move_history),
                    "num_pieces": board.num_pieces(),
                    "board_obj": board.clone()
                }

                # Select action
                if play_mode < 0.6:
                    # Model self-play with temperature for diverse exploration
                    probs = model.predict_distribution(board, temperature=self.temperature)
                    p_legal = probs[legal] / probs[legal].sum()
                    action = int(rng.choices(legal, weights=p_legal)[0])
                else:
                    # Model vs heuristic
                    if curr_p == PLAYER_1:
                        probs = model.predict_distribution(board, temperature=self.temperature)
                        p_legal = probs[legal] / probs[legal].sum()
                        action = int(rng.choices(legal, weights=p_legal)[0])
                    else:
                        action = heuristic.get_move(board)

                pos_info["chosen_action"] = action
                collected.append(pos_info)
                board.make_move(action)

        return collected[:num_target_positions]

    def run(self) -> List[Dict[str, Any]]:
        """
        Executes Round 0 through Round N.
        Resumes automatically if prior rounds are completed.
        """
        set_seed(self.seed)
        train_pool = self.dataset_manager.load_train()
        dev_records = self.dataset_manager.load_dev()

        # Check existing completed rounds
        completed_rounds = {entry["round"]: entry for entry in self.experiment_log}

        # ---------------- Round 0: Initial Baseline ----------------
        if 0 not in completed_rounds:
            print(f"[{self.condition}] Running Round 0: Initial baseline training...")
            # Sample initial dataset
            rng = random.Random(self.seed)
            initial_train = rng.sample(train_pool, min(self.examples_per_round, len(train_pool)))

            model, train_hist = train_model(
                train_records=initial_train,
                dev_records=dev_records,
                epochs=self.epochs_per_round,
                batch_size=self.batch_size,
                seed=self.seed,
                round_num=0
            )

            eval_res = self.evaluate_model(model, dev_records, round_num=0)
            ckpt_path = self.output_dir / "checkpoint_round_0.pt"
            save_checkpoint(model, None, train_hist["total_steps"], 0, eval_res, str(ckpt_path))

            r0_entry = {
                "round": 0,
                "condition": self.condition,
                "status": "ACCEPTED_BASELINE",
                "examples_trained": len(initial_train),
                "total_steps": train_hist["total_steps"],
                "dev_agreement": eval_res["dev_agreement"],
                "overall_score": eval_res["tournament"]["overall_score"],
                "checkpoint": str(ckpt_path),
                "diversity": compute_diversity(initial_train)
            }
            self.experiment_log.append(r0_entry)
            self._save_log()
            current_model = model
            current_eval = eval_res
        else:
            latest_ckpt = self.output_dir / "checkpoint_round_0.pt"
            current_model, _ = load_checkpoint(str(latest_ckpt))
            current_eval = completed_rounds[0]

        # ---------------- Practice Rounds (1 to num_rounds) ----------------
        for r in range(1, self.num_rounds + 1):
            if r in completed_rounds:
                print(f"[{self.condition}] Round {r} already completed. Loading checkpoint...")
                ckpt_path = self.output_dir / f"checkpoint_round_{r}.pt"
                if ckpt_path.exists():
                    current_model, _ = load_checkpoint(str(ckpt_path))
                continue

            print(f"[{self.condition}] Starting Round {r}...")
            # 1. Practice & data collection
            if self.condition == "v1_fixed":
                # Version 1: sample next slice from fixed dataset (no practice rounds)
                rng = random.Random(self.seed + r)
                round_records = rng.sample(train_pool, min(self.examples_per_round, len(train_pool)))
            elif self.condition == "v2_unverified":
                # Version 2: practice games, learn from own moves WITHOUT solver verification
                raw_positions = self.collect_self_play_positions(current_model, self.examples_per_round)
                round_records = []
                for p in raw_positions:
                    round_records.append({
                        "grid": p["grid"],
                        "current_player": p["current_player"],
                        "canonical_hash": p["canonical_hash"],
                        "chosen_action": p["chosen_action"],
                        "optimal_moves": [p["chosen_action"]], # Self-label
                        "num_pieces": p["num_pieces"]
                    })
            elif self.condition == "v3_verified":
                # Version 3: practice games, VERIFIED and labeled by solver referee
                raw_positions = self.collect_self_play_positions(current_model, self.examples_per_round)
                boards_to_label = [p["board_obj"] for p in raw_positions]
                eval_results = self.solver.evaluate_batch(boards_to_label)

                round_records = []
                for p, (scores, opts) in zip(raw_positions, eval_results):
                    round_records.append({
                        "grid": p["grid"],
                        "current_player": p["current_player"],
                        "canonical_hash": p["canonical_hash"],
                        "scores": {str(k): v for k, v in scores.items()},
                        "optimal_moves": sorted(list(opts)), # Solver verified labels!
                        "num_pieces": p["num_pieces"]
                    })

            # Diversity measurement of the round's dataset
            round_diversity = compute_diversity(round_records)

            # 2. Train proposal model on collected examples
            # Start from current accepted checkpoint
            device = get_device()
            proposal_model = Connect4Net().to(device)
            proposal_model.load_state_dict(current_model.state_dict())

            _, train_hist = train_model(
                train_records=round_records,
                dev_records=dev_records,
                epochs=self.epochs_per_round,
                batch_size=self.batch_size,
                seed=self.seed + r,
                round_num=r
            )

            # 3. Evaluate proposal on dev set
            proposal_eval = self.evaluate_model(proposal_model, dev_records, round_num=r)
            prev_agreement = self.experiment_log[-1]["dev_agreement"]
            new_agreement = proposal_eval["dev_agreement"]

            # 4. Acceptance Gate
            # Accept if performance meets threshold, otherwise reject and rollback
            is_accepted = (new_agreement >= prev_agreement - self.acceptance_tolerance)

            round_entry = {
                "round": r,
                "condition": self.condition,
                "accepted": is_accepted,
                "status": "ACCEPTED" if is_accepted else "REJECTED_ROLLED_BACK",
                "dev_agreement": new_agreement if is_accepted else prev_agreement,
                "proposal_agreement": new_agreement,
                "previous_agreement": prev_agreement,
                "overall_score": proposal_eval["tournament"]["overall_score"],
                "examples_trained": len(round_records),
                "total_steps": train_hist["total_steps"],
                "diversity": round_diversity
            }

            if is_accepted:
                current_model = proposal_model
                ckpt_path = self.output_dir / f"checkpoint_round_{r}.pt"
                save_checkpoint(current_model, None, train_hist["total_steps"], r, proposal_eval, str(ckpt_path))
                round_entry["checkpoint"] = str(ckpt_path)
                print(f"[{self.condition}] Round {r} ACCEPTED. Dev agreement: {new_agreement:.3f} (prev: {prev_agreement:.3f})")
            else:
                # Rollback: save current_model as round checkpoint so resume works cleanly
                ckpt_path = self.output_dir / f"checkpoint_round_{r}.pt"
                save_checkpoint(current_model, None, train_hist["total_steps"], r, current_eval, str(ckpt_path))
                round_entry["checkpoint"] = str(ckpt_path)
                print(f"[{self.condition}] Round {r} REJECTED and ROLLED BACK. Proposal agreement: {new_agreement:.3f} < threshold.")

            self.experiment_log.append(round_entry)
            self._save_log()

        return self.experiment_log
