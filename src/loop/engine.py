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
        condition: str = "verified",
        output_dir: str = "results",
        solver: Optional[Connect4Solver] = None,
        dataset_manager: Optional[DatasetManager] = None,
        seed: int = 42,
        num_rounds: int = 3,
        examples_per_round: int = 400,
        epochs_per_round: int = 5,
        batch_size: int = 32,
        acceptance_tolerance: float = 0.0, # 0.0 means STRICT improvement required
        temperature: float = 0.8,
        learning_rate: float = 1e-3
    ):
        self.condition = "verified" if condition in ["verified", "v3_verified"] else condition
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.solver = solver if solver is not None else Connect4Solver()
        self.dataset_manager = dataset_manager if dataset_manager is not None else DatasetManager()
        self.seed = seed
        self.num_rounds = num_rounds
        self.examples_per_round = examples_per_round
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.acceptance_tolerance = acceptance_tolerance
        self.temperature = temperature
        self.learning_rate = learning_rate

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

    def _extract_win_rate(self, tourney_dict: Dict[str, Any], opponent_name_substr: str) -> float:
        """Safely extracts win rate for an opponent substring from tournament dict."""
        matchups = tourney_dict.get("matchups", tourney_dict.get("opponents", tourney_dict))
        for name, stats in matchups.items():
            if opponent_name_substr.lower() in str(name).lower() and isinstance(stats, dict):
                return float(stats.get("win_rate", 0.0))
        return 0.0

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

        # Experience Replay Buffer: maintains foundational knowledge while assimilating new verified practice
        replay_buf_path = self.output_dir / "replay_buffer.json"
        if replay_buf_path.exists():
            with open(replay_buf_path, "r", encoding="utf-8") as f:
                cumulative_train = json.load(f)
        else:
            rng = random.Random(self.seed)
            initial_train = rng.sample(train_pool, min(self.examples_per_round, len(train_pool)))
            cumulative_train = list(initial_train)

        # ---------------- Round 0: Initial Baseline ----------------
        if 0 not in completed_rounds:
            print(f"[{self.condition}] Running Round 0: Initial baseline training...")
            with open(replay_buf_path, "w", encoding="utf-8") as f:
                json.dump(cumulative_train, f)

            model, train_hist = train_model(
                train_records=cumulative_train,
                dev_records=dev_records,
                epochs=self.epochs_per_round,
                batch_size=self.batch_size,
                lr=self.learning_rate,
                seed=self.seed,
                round_num=0
            )

            eval_res = self.evaluate_model(model, dev_records, round_num=0)
            ckpt_path = self.output_dir / "checkpoint_round_0.pt"
            save_checkpoint(model, None, train_hist["total_steps"], 0, eval_res, str(ckpt_path))

            tourney = eval_res.get("tournament", {})
            r0_entry = {
                "round": 0,
                "condition": self.condition,
                "accepted": True,
                "status": "ACCEPTED_BASELINE",
                "examples_trained": len(cumulative_train),
                "total_steps": train_hist["total_steps"],
                "dev_agreement": eval_res["dev_agreement"],
                "overall_score": eval_res["tournament"]["overall_score"],
                "win_rate_random": self._extract_win_rate(tourney, "random"),
                "win_rate_heuristic": self._extract_win_rate(tourney, "heuristic"),
                "win_rate_minimax": self._extract_win_rate(tourney, "minimax"),
                "checkpoint": str(ckpt_path),
                "diversity": compute_diversity(cumulative_train)
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
                # Version 1 control: sample next slice from fixed dataset (no practice rounds)
                rng = random.Random(self.seed + r)
                round_records = rng.sample(train_pool, min(self.examples_per_round, len(train_pool)))
            elif self.condition == "v2_unverified":
                # Version 2 control: practice games, learn from own moves WITHOUT solver verification
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
            else:
                # Primary Track 1 Verified Loop: practice games, VERIFIED and labeled by solver referee
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

            # Append new practice experiences to replay buffer (prevents catastrophic forgetting)
            cumulative_train.extend(round_records)
            with open(replay_buf_path, "w", encoding="utf-8") as f:
                json.dump(cumulative_train, f)

            # Diversity measurement of the round's dataset
            round_diversity = compute_diversity(round_records)

            # 2. Train proposal model on cumulative experiences
            # Start from current accepted checkpoint weights
            proposal_model, train_hist = train_model(
                train_records=cumulative_train,
                dev_records=dev_records,
                epochs=self.epochs_per_round,
                batch_size=self.batch_size,
                lr=self.learning_rate,
                seed=self.seed + r,
                round_num=r,
                init_model=current_model
            )

            # 3. Evaluate proposal on dev set and opponents
            proposal_eval = self.evaluate_model(proposal_model, dev_records, round_num=r)
            prev_agreement = self.experiment_log[-1]["dev_agreement"]
            new_agreement = proposal_eval["dev_agreement"]
            prop_tourney = proposal_eval.get("tournament", {})

            # 4. Strict Acceptance Gate: must strictly improve over previous round
            is_accepted = (new_agreement > prev_agreement)

            round_entry = {
                "round": r,
                "condition": self.condition,
                "accepted": is_accepted,
                "status": "ACCEPTED" if is_accepted else "REJECTED_ROLLED_BACK",
                "dev_agreement": new_agreement if is_accepted else prev_agreement,
                "proposal_agreement": new_agreement,
                "previous_agreement": prev_agreement,
                "overall_score": proposal_eval["tournament"]["overall_score"] if is_accepted else current_eval["tournament"]["overall_score"],
                "win_rate_random": self._extract_win_rate(prop_tourney, "random"),
                "win_rate_heuristic": self._extract_win_rate(prop_tourney, "heuristic"),
                "win_rate_minimax": self._extract_win_rate(prop_tourney, "minimax"),
                "examples_trained": len(cumulative_train),
                "total_steps": train_hist["total_steps"],
                "diversity": round_diversity
            }

            if is_accepted:
                current_model = proposal_model
                current_eval = proposal_eval
                ckpt_path = self.output_dir / f"checkpoint_round_{r}.pt"
                save_checkpoint(current_model, None, train_hist["total_steps"], r, proposal_eval, str(ckpt_path))
                round_entry["checkpoint"] = str(ckpt_path)
                print(f"[{self.condition}] Round {r} STRICTLY ACCEPTED! Dev agreement: {new_agreement:.3f} (prev: {prev_agreement:.3f})")
            else:
                # Rollback: keep current_model as active and save checkpoint
                ckpt_path = self.output_dir / f"checkpoint_round_{r}.pt"
                save_checkpoint(current_model, None, train_hist["total_steps"], r, current_eval, str(ckpt_path))
                round_entry["checkpoint"] = str(ckpt_path)
                print(f"[{self.condition}] Round {r} REJECTED and ROLLED BACK. Proposal {new_agreement:.3f} <= prev {prev_agreement:.3f}")

            self.experiment_log.append(round_entry)
            self._save_log()


        return self.experiment_log
