"""
Main Experiment Runner for Task 1: Verified Self-Improvement on Connect-4.
Executes Version 1, 2, or 3 over multiple seeds.
Enforces strict budget parity, resumability, and writes unified results tables.
"""

import os
import sys
import argparse
import json
from pathlib import Path
try:
    import pandas as pd
except ImportError:
    pd = None
from typing import Dict, Any, List

from src.utils.platform import detect_platform, get_default_output_dir, load_config, set_seed
from src.solver.wrapper import Connect4Solver
from src.data.dataset import DatasetManager, generate_position_pool, split_dataset
from src.loop.engine import SelfImprovementEngine
from src.model.train import load_checkpoint
from src.eval.opponents import ModelPlayer
from src.eval.metrics import compute_optimal_action_agreement

def parse_args():
    parser = argparse.ArgumentParser(description="Connect-4 Self-Improvement Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/self_learning.yaml", help="Path to config YAML or JSON")
    parser.add_argument("--out", type=str, default=None, help="Output directory (auto-detected if None)")
    parser.add_argument("--condition", type=str, default=None, choices=["verified", "v1_fixed", "v2_unverified", "v3_verified", "compare", "all"], help="Override condition (use 'compare' for verified vs unverified)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override list of random seeds")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for datasets")
    parser.add_argument("--prepare_data", action="store_true", help="Generate and split dataset if missing")
    parser.add_argument("--final", action="store_true", help="Authorize evaluation on locked final test set")
    parser.add_argument("--clean", action="store_true", help="Wipe existing results directory and start fresh")
    parser.add_argument("--require_cpp", action="store_true", help="Fail loudly if Pascal Pons C++ solver binary is missing")
    return parser.parse_args()

def ensure_datasets(data_dir: Path, solver: Connect4Solver, seed: int = 42):
    """Generates initial pool and splits if not already present."""
    dm = DatasetManager(str(data_dir))
    if dm.train_file.exists() and dm.dev_file.exists() and dm.final_test_file.exists():
        print(f"[Data] Found existing datasets in {data_dir}.")
        return dm

    print(f"[Data] Generating position pool and zero-leakage splits in {data_dir}...")
    records = generate_position_pool(num_games=150, solver=solver, seed=seed)
    train_data, dev_data, test_data = split_dataset(records, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15, seed=seed)
    dm.save_splits(train_data, dev_data, test_data)
    print(f"[Data] Generated: {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test positions.")
    return dm

def main():
    args = parse_args()
    plat = detect_platform()
    print(f"[Platform] Running on: {plat.upper()}")

    # Output directory
    if args.out:
        out_base = Path(args.out)
    else:
        out_base = get_default_output_dir(base_name="results")

    if args.clean and out_base.exists():
        print(f"[Clean] Removing previous results at {out_base.resolve()}...")
        import shutil
        shutil.rmtree(out_base)

    out_base.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Results will be written to: {out_base.resolve()}")

    # Load configuration
    cfg = load_config(args.config)
    if args.condition == "compare":
        conditions = ["v3_verified", "v2_unverified"]
    elif args.condition == "all":
        conditions = ["v1_fixed", "v2_unverified", "v3_verified"]
    elif args.condition:
        conditions = [args.condition]
    else:
        conditions = [cfg.get("condition", "v3_verified")]
    seeds = args.seeds or cfg.get("seeds", [42])
    num_rounds = cfg.get("num_rounds", 3)
    examples_per_round = cfg.get("examples_per_round", 500)
    epochs_per_round = cfg.get("epochs_per_round", 5)
    batch_size = cfg.get("batch_size", 64)
    tolerance = cfg.get("acceptance_tolerance", 0.02)
    temp = cfg.get("temperature", 0.8)
    lr = cfg.get("learning_rate", 1e-3)

    # Initialize Solver & Cache
    cache_path = out_base / "cache" / "solver_labels.sqlite"
    solver = Connect4Solver(cache_path=str(cache_path), require_cpp=args.require_cpp)

    # Initialize Dataset
    data_dir = Path(args.data_dir)
    dm = ensure_datasets(data_dir, solver, seed=seeds[0])

    all_summaries = []

    for cond in conditions:
        for seed in seeds:
            print(f"\n{'='*60}\nRunning Condition: {cond} | Seed: {seed}\n{'='*60}")
            exp_dir = out_base / f"{cond}_seed{seed}"
            engine = SelfImprovementEngine(
                condition=cond,
                output_dir=str(exp_dir),
                solver=solver,
                dataset_manager=dm,
                seed=seed,
                num_rounds=num_rounds,
                examples_per_round=examples_per_round,
                epochs_per_round=epochs_per_round,
                batch_size=batch_size,
                acceptance_tolerance=tolerance,
                temperature=temp,
                learning_rate=lr
            )

            round_logs = engine.run()

            # Record final round summary
            latest_round = round_logs[-1]
            summary_entry = {
                "condition": cond,
                "seed": seed,
                "final_round": latest_round["round"],
                "status": latest_round.get("status", "ACCEPTED"),
                "dev_agreement": latest_round["dev_agreement"],
                "dev_outcome_agreement": latest_round.get("dev_outcome_agreement", latest_round["dev_agreement"]),
                "tournament_score": latest_round.get("overall_score", 0.0),
                "total_examples_trained": sum(r["examples_trained"] for r in round_logs),
                "checkpoint": latest_round.get("checkpoint", "")
            }

            # Optional final test evaluation (only if authorized with --final)
            if args.final:
                print(f"[{cond} seed {seed}] Evaluating on LOCKED final test set (--final flag active)...")
                test_records = dm.load_final_test(allow_final=True)
                ckpt_path = latest_round.get("checkpoint", "")
                if ckpt_path and Path(ckpt_path).exists():
                    model, _ = load_checkpoint(ckpt_path)
                    player = ModelPlayer(model, greedy=True)
                    test_eval = compute_optimal_action_agreement(player, test_records)
                    summary_entry["final_test_strict_agreement"] = test_eval["overall_agreement"]
                    summary_entry["final_test_outcome_agreement"] = test_eval.get("outcome_agreement", test_eval["overall_agreement"])
                    print(f"[{cond} seed {seed}] Final Test: Strict Optimal={test_eval['overall_agreement']:.2%} | Outcome-Preserving={test_eval.get('outcome_agreement', test_eval['overall_agreement']):.2%}")

            all_summaries.append(summary_entry)

    # Compile unified results table
    summary_csv = out_base / "summary_table.csv"
    summary_json = out_base / "summary_table.json"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    if pd is not None:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n{'='*60}\nEXPERIMENT COMPLETE. Summary Table:\n{'='*60}")
        print(summary_df.to_string(index=False))

        # Multi-seed aggregate stats if > 1 seed
        if len(seeds) > 1:
            print(f"\n{'='*60}\nMULTI-SEED AGGREGATE SUMMARY (Mean ± Std):\n{'='*60}")
            agg_cols = ["dev_agreement", "dev_outcome_agreement", "tournament_score"]
            if args.final:
                agg_cols += ["final_test_strict_agreement", "final_test_outcome_agreement"]
            available_cols = [c for c in agg_cols if c in summary_df.columns]
            grouped = summary_df.groupby("condition")[available_cols].agg(["mean", "std"])
            print(grouped.to_string())
    else:
        if all_summaries:
            headers = list(all_summaries[0].keys())
            with open(summary_csv, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                for row in all_summaries:
                    f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")
        print(f"\n{'='*60}\nEXPERIMENT COMPLETE. (Saved JSON to {summary_json})\n{'='*60}")

    print(f"\nSaved summary to: {summary_csv}")

if __name__ == "__main__":
    main()
