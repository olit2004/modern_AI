# Modern AI Research & Capabilities: Recursive Self-Learning & Capability Distillation

This repository houses two self-contained research tracks investigating autonomous capability acquisition, verification, and efficiency in modern neural models:

1. **`recursive_learning/`**: **Task 1 — Verified Recursive Self-Improvement on Connect-4**  
   Autonomous self-play loop with an infallible C++ bitboard solver referee, experience replay buffer, strict acceptance gating with rollback, and zero-leakage reflection-symmetric data partitions.
2. **`Qwen_student/`**: **Task 2 — Low-Resource Amharic Capability Distillation**  
   Cross-model knowledge distillation from a larger teacher (`google/gemma-3-4b-it`) to a compact 494M-parameter student (`Qwen/Qwen2.5-0.5B-Instruct`) using 4-bit QLoRA.

---

## 📁 Repository Structure

```
.
├── README.md                      # Unified repository overview and execution guide
├── recursive_learning/            # Track 1: Connect-4 Recursive Self-Learning
│   ├── configs/                   # Experiment hyperparameters (rounds, budgets, tolerances)
│   ├── demo.py                    # Presentation demo (summary table, progress plots, live matches)
│   ├── notebooks/
│   │   └── colab_runner.ipynb     # Google Colab execution runner with GPU support
│   ├── requirements.txt           # Python dependencies
│   ├── run_experiments.py         # Main training and evaluation engine
│   ├── src/                       # Core engine, solver wrapper, neural net, tournament eval
│   └── tests/                     # Comprehensive test suite (game logic, dataset, solver, net)
└── Qwen_student/                  # Track 2: Amharic Capability Distillation
    ├── final_adapter/             # Trained 4-bit QLoRA adapter weights & tokenizer
    ├── evaluation_prompts.csv     # 60 held-out evaluation prompts
    ├── experiment_notes.txt       # Quantitative logs, training configs, qualitative observations
    ├── experiment_summary.json    # Machine-readable experiment and evaluation metrics
    ├── predictions_final.csv      # Model outputs on evaluation prompts
    ├── student trainer.ipynb      # End-to-end training and inference notebook
    └── training_data.csv          # Clean teacher-generated Amharic dataset (1,843 examples)
```

---

## 🕹️ Track 1: Verified Recursive Self-Improvement (`recursive_learning/`)

### Research Motivation
Can a neural network autonomously self-improve through practice games without falling victim to self-referential policy collapse or catastrophic forgetting? In Connect-4—a strongly solved zero-sum game—we investigate how external solver verification prevents degradation and guides genuine capability gains.

### Key Architectural Pillars
- **Infallible Referee (Pascal Pons C++ Bitboard Solver):** Evaluates every visited board position and computes exact game-theoretic win/draw/loss values and optimal move sets with sub-millisecond latency.
- **Experience Replay Buffer:** Retains foundational board positions from early exploration while assimilating newly corrected blunder positions from self-play, preventing catastrophic forgetting.
- **Strict Acceptance Gate with Rollback:** After each round, proposal weights are evaluated on a held-out dev split. If agreement degrades beyond tolerance ($\tau = 0.015$), the proposal is automatically rejected and rolled back to the previous accepted checkpoint.
- **Zero-Leakage Reflection-Invariant Hashing:** Connect-4 possesses bilateral reflection symmetry. Positions are partitioned using canonical byte hashes (`min(hash(board), hash(mirrored_board))`), guaranteeing zero leakage between train, dev, and test sets.
- **Anti-Determinism Randomized Openings:** Prevents greedy models and deterministic opponents from looping through identical games by initializing evaluation matches with 2–4 uniform random opening plies over 100+ games per opponent.

### Running Track 1 Locally

```bash
cd recursive_learning

# 1. Install dependencies
pip install -r requirements.txt

# 2. Build high-speed C++ solver & download opening book
make -C src/solver/cpp/pons
test -f src/solver/cpp/pons/7x6.book || curl -L -o src/solver/cpp/pons/7x6.book https://github.com/PascalPons/connect4/releases/download/book/7x6_small.book

# 3. Run validation test suite
PYTHONPATH=. pytest tests/ -v

# 4. Launch Verified Self-Improvement Loop
python run_experiments.py \
    --config configs/self_learning.yaml \
    --out results \
    --clean \
    --require_cpp

# 5. Run Live Presentation Demo
python demo.py --results_dir results --num_games 50
```

---

## 🇪🇹 Track 2: Amharic Capability Distillation (`Qwen_student/`)

### Research Motivation
Compact language models (~0.5B parameters) have minimal exposure to low-resource languages like Amharic (Ethiopic Ge'ez script) during pre-training. We distill instruction-following and generation capabilities from an advanced teacher model into a lightweight, deployable student.

### Model Setup
- **Teacher:** `google/gemma-3-4b-it` (High-resource generative reasoning)
- **Student:** `Qwen/Qwen2.5-0.5B-Instruct` (494M parameters, base Amharic generation corrupt/incoherent)
- **Training Method:** 4-bit QLoRA using `bitsandbytes` and PEFT
  - Rank ($r$): 16, Alpha ($\alpha$): 32, Dropout: 0.05
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - Optimizer: `paged_adamw_8bit`, Learning Rate: $2 \times 10^{-4}$
  - Effective batch size: 16 (batch size 4, gradient accumulation 4)
  - Trainable parameters: 2,162,688 (0.4359% of model)
  - Adapter footprint: 8.27 MB

### Quantitative & Qualitative Results
- **Training Data:** 1,843 clean instruction-response pairs; 0 sequence truncations at max length 1024.
- **Inference Efficiency (Tesla T4 GPU):**
  - Distilled student peak memory: **1.54 GB** (enables edge deployment)
  - Average inference latency: **14.19s per response** (10.03 tokens/sec)
- **Key Findings:**
  - Base student produced corrupted mixed-script tokens, prompt repetition, and English leakage.
  - Distilled student successfully internalized Amharic grammar structure and polite task-following patterns.
  - Remaining challenges include factual hallucinations and repetitive terminal phrasing on abstract open-ended prompts, identifying clear avenues for future data-filtering iterations.

---

## 🚀 Running on Google Colab

To train or demo the Connect-4 self-improvement engine on a free Google Colab T4/V100 GPU:

1. Open [Google Colab](https://colab.research.google.com).
2. Create a new notebook or upload `recursive_learning/notebooks/colab_runner.ipynb`.
3. Set runtime to **GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU`).
4. Execute the setup cells:

```python
# Clone the repository
!git clone https://github.com/olit2004/modern_AI.git /content/modern_AI
%cd /content/modern_AI/recursive_learning

# Install dependencies and compile solver
!pip install -r requirements.txt
!make -C src/solver/cpp/pons
!test -f src/solver/cpp/pons/7x6.book || curl -L -o src/solver/cpp/pons/7x6.book https://github.com/PascalPons/connect4/releases/download/book/7x6_small.book

# Run tests
!PYTHONPATH=. pytest tests/ -v

# Launch experiment loop
!python run_experiments.py --config configs/self_learning.yaml --out /content/results --clean --require_cpp

# Run interactive demo & render charts
!python demo.py --results_dir /content/results --num_games 50
from IPython.display import Image, display
display(Image(filename="/content/results/figures/demo_progress.png"))
```
