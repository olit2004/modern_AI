# Verified Self-Improvement on Connect-4

An end-to-end framework to teach a small neural network Connect-4 player to self-improve through autonomous practice rounds, and empirically measure whether solver verification prevents policy degradation.

---

## 🎯 The Core Scientific Question
Does autonomous self-play practice yield real policy improvement, and does **solver verification** (referee labeling) make a statistically significant difference compared to:
1. **Version 1 (Fixed Baseline):** Training on a fixed dataset with zero practice rounds.
2. **Version 2 (Unverified Self-Improvement):** Training on its own practice moves without checking them.
3. **Version 3 (Solver-Verified Self-Improvement):** Training on practice positions where every example is verified and labeled by a perfect Connect-4 solver.

All three versions are held to **strict parity**: exactly identical model architecture, identical total training examples, and identical gradient update steps.

---

## 🏗️ Repository Architecture

```
.
├── configs/                  # Experiment configurations (YAML & JSON)
│   ├── v1_baseline.yaml      # Version 1 config
│   ├── v2_unverified.yaml    # Version 2 config
│   ├── v3_verified.yaml      # Version 3 config
│   └── quick_test.json       # Fast smoke-test configuration
├── data/                     # Fixed datasets with strict zero-leakage partitions
│   ├── train.json            # 70% training pool
│   ├── dev.json              # 15% validation/gating pool
│   └── final_test.json       # 15% locked test set (guarded by --final flag)
├── notebooks/                # Thin execution notebooks
│   ├── colab_runner.ipynb    # Google Colab GPU runner
│   └── kaggle_runner.ipynb   # Kaggle GPU runner
├── src/
│   ├── game/                 # Connect-4 game engine (symmetries, tensor encoding)
│   ├── solver/               # Pascal Pons C++ bitboard solver wrapper & cache
│   ├── data/                 # Zero-leakage split generator & locked loader
│   ├── model/                # Connect4Net CNN, soft loss, checkpointing
│   ├── eval/                 # Opponents (Random, Heuristic, Minimax), tournaments
│   ├── loop/                 # Self-improvement engine, gating, rollback
│   ├── analysis/             # Publication charts and report generator
│   └── utils/                # Platform detection, seeds, config parsing
├── tests/                    # Unit tests running in seconds
├── export_teachers.py        # Exports teachers for Task 3 distillation
├── inspect_teachers.py       # Compares soft teacher distributions
├── play_agent.py             # Playable interactive CLI against the AI
├── run_experiments.py        # Main multi-seed experiment runner
├── requirements.txt          # Pinned dependencies
└── README.md
```

---

## 🛡️ Ground Rules & Safeguards

1. **Portable by Design:** Auto-detects Google Colab (Drive mounting), Kaggle (`/kaggle/working`), or local CPU/GPU.
2. **Strict Parity:** V1, V2, and V3 use identical total example counts and gradient steps so differences stem purely from the verification method.
3. **Zero Data Leakage:** Split strictly by **canonical position hash** (combining both perspective and horizontal reflection symmetry). No mirrored position can ever leak across train/dev/test splits.
4. **Guarded Final Test Set:** Code strictly refuses to read `final_test.json` unless the `--final` flag is supplied.
5. **Acceptance Gating & Rollback:** Each self-improvement proposal is evaluated on dev agreement; if performance drops beyond tolerance, the proposal is rejected, logged, and rolled back.
6. **Resumable:** Checkpoints and JSON logs saved after every round. Interrupted jobs resume without duplicating work.

---

## 🚀 Quickstart & Reproduction

### 1. Run Unit Tests (Runs in < 15 seconds)
```bash
pytest tests/ -v
```

### 2. Run Experiments
To run all three conditions across 3 seeds on Colab/Kaggle/Local:
```bash
python run_experiments.py --config configs/v3_verified.yaml --condition all --final
```

Or run a fast smoke test:
```bash
python run_experiments.py --config configs/quick_test.json
```

### 3. Generate Charts & Report
```bash
python src/analysis/generate_report.py --results_dir results
```
Generates:
- `results/figures/agreement_per_round.png`
- `results/figures/diversity_per_round.png`
- `results/REPORT.md`

### 4. Play Interactively
Play a game against your trained agent:
```bash
python play_agent.py --checkpoint results/teachers/teacher_round3_improved.pt
```
Or against benchmark opponents:
```bash
python play_agent.py --opponent minimax --first
```

---

## 🎓 Task 3 Distillation Handoff API

Task 3 requires exporting the baseline (Round 0) and improved (Round 3) teachers to distill into a smaller student network using soft output distributions.

### Exporting Teachers
```bash
python export_teachers.py --results_dir results --export_dir results/teachers
```

### Distillation Soft Distribution Interface
Any model checkpoint implements `predict_distribution()`:
```python
from src.model.train import load_checkpoint
from src.game.connect4 import Connect4Board

# Load teacher
teacher, _ = load_checkpoint("results/teachers/teacher_round3_improved.pt")

# Query soft distribution for any position
board = Connect4Board()
board.make_move(3) # e.g. center move

# Returns 7-element NumPy array summing to 1.0; illegal moves are guaranteed 0.0
probs = teacher.predict_distribution(board, temperature=1.0)
print("Soft Column Probabilities:", probs)
```

Verify teacher distributions side-by-side:
```bash
python inspect_teachers.py --moves "3324"
```
