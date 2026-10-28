"""
Knowledge Distillation Engine (Track B & Combined Stretch Goal)
iCog Labs Emerging AI Paradigms Curriculum

Distills the improved self-learned teacher (from Track A) into a lightweight student model.
Compares:
1. Teacher Model (Connect4Net, ~420k params)
2. Hard-Label Student Baseline (Connect4StudentNet, ~22k params, trained only on hard targets)
3. Distilled Student (Connect4StudentNet, trained with Hinton soft-target distillation)

Measures and compares:
- Test Agreement on held-out positions
- Model Parameter Count & Compression Ratio
- CPU Inference Latency (ms per 100 positions)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
except ImportError:
    torch = None

from src.game.connect4 import Connect4Board, COLS, ROWS
from src.model.net import Connect4Net, Connect4StudentNet
from src.model.train import load_checkpoint, get_device
from src.data.dataset import DatasetManager
from src.eval.opponents import ModelPlayer
from src.eval.metrics import compute_optimal_action_agreement
from src.utils.platform import set_seed

class DistillationDataset(Dataset):
    """Pairs board states with both hard targets and teacher soft logits."""

    def __init__(self, records: List[Dict[str, Any]], teacher_model: Optional[Connect4Net] = None, device: Optional["torch.device"] = None):
        self.records = records
        self.device = device or torch.device("cpu")
        self.teacher_model = teacher_model
        self.teacher_logits = []

        # Precompute teacher logits if teacher is available
        if self.teacher_model is not None:
            self.teacher_model.eval()
            with torch.no_grad():
                for rec in records:
                    b = Connect4Board(np.array(rec["grid"]), current_player=rec["current_player"])
                    tensor_np = b.to_tensor_encoding()
                    x = torch.from_numpy(tensor_np).unsqueeze(0).to(self.device)
                    legal = b.legal_moves()
                    mask = torch.zeros(1, COLS, device=self.device)
                    for c in legal:
                        mask[0, c] = 1.0
                    t_logits = self.teacher_model(x, legal_mask=mask).squeeze(0).cpu().numpy()
                    self.teacher_logits.append(t_logits)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        b = Connect4Board(np.array(rec["grid"]), current_player=rec["current_player"])
        x = b.to_tensor_encoding()

        legal = b.legal_moves()
        mask = np.zeros(COLS, dtype=np.float32)
        for c in legal:
            mask[c] = 1.0

        # Hard target distribution
        target = np.zeros(COLS, dtype=np.float32)
        if "optimal_moves" in rec and len(rec["optimal_moves"]) > 0:
            opts = rec["optimal_moves"]
            for c in opts:
                target[c] = 1.0 / len(opts)
        elif "chosen_action" in rec:
            target[rec["chosen_action"]] = 1.0
        else:
            for c in legal:
                target[c] = 1.0 / len(legal)

        t_logits = self.teacher_logits[idx] if self.teacher_logits else np.zeros(COLS, dtype=np.float32)

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(target).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(t_logits).float()
        )

def distillation_loss_fn(
    student_logits: "torch.Tensor",
    teacher_logits: "torch.Tensor",
    hard_targets: "torch.Tensor",
    legal_mask: "torch.Tensor",
    temperature: float = 2.0,
    alpha: float = 0.5
) -> "torch.Tensor":
    """
    Hinton Knowledge Distillation Objective:
    L = alpha * CE(hard_target, student) + (1 - alpha) * T^2 * KL(teacher/T || student/T)
    """
    # 1. Hard cross-entropy loss
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    hard_loss = -torch.sum(hard_targets * student_log_probs, dim=-1).mean()

    # 2. Soft distillation loss (KL Divergence scaled by T^2)
    t_scaled = teacher_logits / temperature
    s_scaled = student_logits / temperature

    teacher_soft = torch.softmax(t_scaled, dim=-1)
    student_soft_log = torch.log_softmax(s_scaled, dim=-1)

    kl_div = F.kl_div(student_soft_log, teacher_soft, reduction="batchmean") * (temperature ** 2)

    return alpha * hard_loss + (1.0 - alpha) * kl_div

def measure_latency(model: nn.Module, device: "torch.device", num_queries: int = 200) -> float:
    """Measure average inference latency per move on CPU in milliseconds."""
    model.eval()
    dummy_board = Connect4Board()
    dummy_board.make_move(3)
    tensor_np = dummy_board.to_tensor_encoding()
    x = torch.from_numpy(tensor_np).unsqueeze(0).to(device)
    mask = torch.ones(1, COLS, device=device)

    # Warmup
    for _ in range(20):
        _ = model(x, legal_mask=mask)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_queries):
            _ = model(x, legal_mask=mask)
    elapsed = (time.perf_counter() - start) * 1000.0 / num_queries
    return elapsed

def train_student(
    train_records: List[Dict],
    dev_records: List[Dict],
    teacher: Optional[Connect4Net],
    distill: bool = False,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 2.0,
    alpha: float = 0.5,
    seed: int = 42
) -> Tuple[Connect4StudentNet, float]:
    """Train a compact student model with either hard labels or distillation."""
    set_seed(seed)
    device = get_device()

    student = Connect4StudentNet().to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)

    dataset = DistillationDataset(train_records, teacher_model=teacher if distill else None, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        student.train()
        for x, targets, masks, t_logits in loader:
            x, targets, masks, t_logits = x.to(device), targets.to(device), masks.to(device), t_logits.to(device)
            optimizer.zero_grad()
            s_logits = student(x, legal_mask=masks)

            if distill:
                loss = distillation_loss_fn(
                    student_logits=s_logits,
                    teacher_logits=t_logits,
                    hard_targets=targets,
                    legal_mask=masks,
                    temperature=temperature,
                    alpha=alpha
                )
            else:
                s_log_probs = torch.log_softmax(s_logits, dim=-1)
                loss = -torch.sum(targets * s_log_probs, dim=-1).mean()

            loss.backward()
            optimizer.step()

    # Measure dev agreement
    player = ModelPlayer(student, greedy=True)
    eval_res = compute_optimal_action_agreement(player, dev_records)
    return student, eval_res["overall_agreement"]

def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Runner (Track B)")
    parser.add_argument("--results_dir", type=str, default="/content/results", help="Directory with Task 1 experiment results")
    parser.add_argument("--teacher_path", type=str, default=None, help="Path to teacher checkpoint")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save distillation outputs")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature T")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight between hard CE and soft distillation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123], help="Seeds for evaluation")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (res_dir / "distillation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate teacher checkpoint
    if args.teacher_path:
        teacher_path = Path(args.teacher_path)
    else:
        candidates = [
            res_dir / "teachers" / "teacher_round3_improved.pt",
            res_dir / "v3_verified_seed42" / "checkpoint_round_3.pt",
            res_dir / "v1_fixed_seed42" / "checkpoint_round_3.pt"
        ]
        teacher_path = next((c for c in candidates if c.exists()), None)

    device = get_device()
    print("=" * 65)
    print("KNOWLEDGE DISTILLATION BENCHMARK (TRACK B)")
    print("=" * 65)

    if teacher_path and teacher_path.exists():
        print(f"[Teacher] Loading teacher model from: {teacher_path}")
        teacher, _ = load_checkpoint(str(teacher_path), device=device)
    else:
        print("[Teacher] No teacher checkpoint found. Initializing surrogate teacher...")
        teacher = Connect4Net().to(device)

    # Count parameters
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_dummy = Connect4StudentNet()
    student_params = sum(p.numel() for p in student_dummy.parameters())
    compression = teacher_params / max(1, student_params)

    print(f"[Model Sizes] Teacher: {teacher_params:,} params | Student: {student_params:,} params ({compression:.1f}x compression)")

    # 2. Load dataset
    data_dir = res_dir / "data" if (res_dir / "data" / "train.json").exists() else Path("data")
    dm = DatasetManager(str(data_dir))
    train_data = dm.load_train()
    dev_data = dm.load_dev()
    test_data = dm.load_final_test(allow_final=True)

    # Measure Teacher Latency and Accuracy
    cpu_device = torch.device("cpu")
    teacher.to(cpu_device)
    teacher_latency = measure_latency(teacher, cpu_device)
    teacher_player = ModelPlayer(teacher, greedy=True)
    teacher_test_res = compute_optimal_action_agreement(teacher_player, test_data)
    teacher_acc = teacher_test_res["overall_agreement"]

    print(f"\n[Teacher Baseline] Test Agreement: {teacher_acc:.2%} | Latency: {teacher_latency:.2f} ms/move")

    # 3. Train across seeds
    results = []

    # Teacher entry
    results.append({
        "Model": "Teacher (Connect4Net)",
        "Type": "Teacher Baseline",
        "Seed": "-",
        "Parameters": teacher_params,
        "Compression": "1.0x",
        "Test Agreement": teacher_acc,
        "CPU Latency (ms)": teacher_latency
    })

    hard_accs = []
    distill_accs = []
    latencies = []

    for seed in args.seeds:
        print(f"\n--- Running Seed: {seed} ---")

        # Hard-Label Student
        print(f"[Student-Hard seed {seed}] Training with ground-truth hard labels...")
        student_hard, _ = train_student(
            train_records=train_data,
            dev_records=dev_data,
            teacher=None,
            distill=False,
            epochs=args.epochs,
            seed=seed
        )
        student_hard.to(cpu_device)
        hard_lat = measure_latency(student_hard, cpu_device)
        latencies.append(hard_lat)
        hard_player = ModelPlayer(student_hard, greedy=True)
        hard_eval = compute_optimal_action_agreement(hard_player, test_data)
        hard_acc = hard_eval["overall_agreement"]
        hard_accs.append(hard_acc)
        print(f"[Student-Hard seed {seed}] Test Agreement: {hard_acc:.2%}")

        results.append({
            "Model": "Student-Hard (Connect4StudentNet)",
            "Type": "Hard-Label Baseline",
            "Seed": seed,
            "Parameters": student_params,
            "Compression": f"{compression:.1f}x",
            "Test Agreement": hard_acc,
            "CPU Latency (ms)": hard_lat
        })

        # Distilled Student
        print(f"[Student-Distilled seed {seed}] Training with Hinton Soft Targets (T={args.temperature}, alpha={args.alpha})...")
        teacher.to(device)
        student_distill, _ = train_student(
            train_records=train_data,
            dev_records=dev_data,
            teacher=teacher,
            distill=True,
            epochs=args.epochs,
            temperature=args.temperature,
            alpha=args.alpha,
            seed=seed
        )
        student_distill.to(cpu_device)
        distill_player = ModelPlayer(student_distill, greedy=True)
        distill_eval = compute_optimal_action_agreement(distill_player, test_data)
        distill_acc = distill_eval["overall_agreement"]
        distill_accs.append(distill_acc)
        print(f"[Student-Distilled seed {seed}] Test Agreement: {distill_acc:.2%}")

        results.append({
            "Model": "Student-Distilled (Connect4StudentNet)",
            "Type": "Knowledge Distilled",
            "Seed": seed,
            "Parameters": student_params,
            "Compression": f"{compression:.1f}x",
            "Test Agreement": distill_acc,
            "CPU Latency (ms)": hard_lat
        })

    # Save summary table
    df = pd.DataFrame(results)
    table_csv = out_dir / "distillation_summary.csv"
    df.to_csv(table_csv, index=False)
    print(f"\n[Saved] Distillation benchmark summary saved to: {table_csv}")

    # Plot comparison figure
    plt.figure(figsize=(9, 5))
    categories = ["Teacher", "Student (Hard Labels)", "Student (Distilled)"]
    means = [teacher_acc, np.mean(hard_accs), np.mean(distill_accs)]
    errors = [0.0, np.std(hard_accs), np.std(distill_accs)]
    colors = ["#34495e", "#e74c3c", "#2ecc71"]

    bars = plt.bar(categories, means, yerr=errors, capsize=6, color=colors, alpha=0.85, width=0.55)
    plt.title(f"Knowledge Distillation Benchmark (Test Set Agreement)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Optimal Action Agreement (%)", fontsize=11)
    plt.ylim(0.0, max(means) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar, val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2%}", ha="center", fontweight="bold")

    plt.tight_layout()
    fig_path = out_dir / "distillation_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Saved] Comparison plot saved to: {fig_path}")

    print("\n" + "=" * 65)
    print("FINAL DISTILLATION SUMMARY")
    print("=" * 65)
    print(df.to_string(index=False))
    print("=" * 65)

if __name__ == "__main__":
    main()
