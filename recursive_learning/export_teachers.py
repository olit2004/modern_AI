"""
Export Teacher Models for Task 3 (Distillation).
Saves baseline (Round 0) and self-improved (Round 3) teachers into a clean directory
with model configuration and export manifest.
"""

import os
import shutil
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Export Teacher Models for Distillation")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory with experiment runs")
    parser.add_argument("--export_dir", type=str, default="results/teachers", help="Directory to export teachers")
    parser.add_argument("--best_condition", type=str, default="v3_verified", help="Condition of improved player")
    parser.add_argument("--seed", type=int, default=42, help="Seed to export")
    return parser.parse_args()

def main():
    args = parse_args()
    res_dir = Path(args.results_dir)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    exp_folder = res_dir / f"{args.best_condition}_seed{args.seed}"
    if not exp_folder.exists():
        for alt in [res_dir / f"verified_seed{args.seed}", res_dir / f"v3_verified_seed{args.seed}"]:
            if alt.exists():
                exp_folder = alt
                break

    r0_ckpt = exp_folder / "checkpoint_round_0.pt"
    r3_ckpt = exp_folder / "checkpoint_round_3.pt"

    if not r0_ckpt.exists():
        print(f"Warning: Baseline checkpoint not found at {r0_ckpt}")
    else:
        out_r0 = export_dir / "teacher_round0_baseline.pt"
        shutil.copyfile(r0_ckpt, out_r0)
        print(f"[Export] Saved baseline teacher to: {out_r0}")

    if not r3_ckpt.exists():
        print(f"Warning: Improved checkpoint not found at {r3_ckpt}")
    else:
        out_r3 = export_dir / "teacher_round3_improved.pt"
        shutil.copyfile(r3_ckpt, out_r3)
        print(f"[Export] Saved improved teacher to: {out_r3}")

    manifest = {
        "task": "Task 3 Distillation Teacher Export",
        "condition": args.best_condition,
        "seed": args.seed,
        "baseline_path": str(export_dir / "teacher_round0_baseline.pt"),
        "improved_path": str(export_dir / "teacher_round3_improved.pt"),
        "api_method": "predict_distribution(board: Connect4Board, temperature: float = 1.0) -> np.ndarray"
    }

    with open(export_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Export] Manifest written to {export_dir / 'manifest.json'}")

if __name__ == "__main__":
    main()
