# 🎓 iCog Labs AI Internship: Technical Presentation Guide
**Topic:** Recursive Self-Learning & Knowledge Distillation  
**Speaker:** AI/ML Intern  
**Session Timing:** 20 Minutes Total (3m Design + 5m Method + 5m Results/Demo + 2m Limitations + 5m Q&A)  
**Evaluation Standard:** iCog Labs Internship Rubric (100 Points)

---

## ⏱️ Slide-by-Slide Presentation Structure

### 1. Problem & Design (3 minutes) — [Rubric: Conceptual Understanding, 20 pts]
- **The Core Question:** Can a neural network autonomously self-improve through practice rounds without collapsing into self-reinforcing errors, and can its capabilities be distilled into an ultra-fast, lightweight student?
- **The 3 Experimental Conditions:**
  1. **Version 1 (Fixed Baseline):** Passive learning from static dataset slices with zero practice rounds.
  2. **Version 2 (Unverified Self-Improvement):** Autonomous self-play, learning from its own moves without external verification.
  3. **Version 3 (Solver-Verified Self-Improvement):** Autonomous self-play, but every encountered state is evaluated and labeled by an infallible Connect-4 bitboard solver referee.
- **Strict Parity Safeguards:** Identical model architectures, identical gradient steps, and zero data leakage across canonical reflection-symmetric hash splits.

---

### 2. Method & Architecture (5 minutes) — [Rubric: Technical Implementation, 25 pts]
- **The Engine Workflow:**
  $$\text{Generate Practice Games} \longrightarrow \text{Solver Verification} \longrightarrow \text{Replay Buffer} \longrightarrow \text{Model Update} \longrightarrow \text{Acceptance Gate}$$
- **Why Experience Replay Buffer Matters:** Prevents catastrophic forgetting by anchoring the network on foundational game states while assimilating newly corrected blunder positions.
- **Acceptance Gate & Rollback:** Held-out dev performance is checked after every round. If a proposal degrades beyond tolerance ($\tau = 0.02$), it is rejected and rolled back.
- **Knowledge Distillation Formulation:**
  $$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, z_s) + (1-\alpha) \cdot T^2 \cdot \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_t}{T}\right) \parallel \sigma\left(\frac{z_s}{T}\right)\right)$$
  - Teacher: `Connect4Net` (3 Conv layers, 64 filters, ~420k params)
  - Student: `Connect4StudentNet` (1 Conv layer, 16 filters, ~22k params — **19x compression**)
  - Soft targets preserve the relative value of non-winning moves ("dark knowledge").

---

### 3. Quantitative Results & Live Demonstration (5 minutes) — [Rubric: Experimental Reasoning, 20 pts]

#### A. Self-Improvement Agreement & Diversity Curves
- **Show Figure:** `results/figures/agreement_per_round.png`
  - **Version 3 (Verified):** Monotonically improves as solver corrections systematically eliminate tactical blunders.
  - **Version 2 (Unverified):** Demonstrates policy collapse/stagnation because learning from unchecked mistakes reinforces flawed habits.
  - **Version 1 (Fixed):** Modest improvement, but fails to target the agent's specific tactical blind spots.
- **Show Figure:** `results/figures/diversity_per_round.png`
  - V2 exhibits diversity decline as the agent falls into repetitive game loops, while V3 maintains stable exploration.

#### B. Knowledge Distillation Benchmark
- **Show Figure:** `results/distillation/distillation_comparison.png`
- **Show Table:** `results/distillation/distillation_summary.csv`
  - Distilled Student achieves close to Teacher accuracy while running at a fraction of CPU latency (ms per move) and using **~95% fewer parameters**.

#### C. Live Interactive Demo
- Run live CLI interactive match:
  ```bash
  python play_agent.py --checkpoint results/teachers/teacher_round3_improved.pt
  ```

---

### 4. Limitations & Risk Analysis (2 minutes) — [Rubric: Creativity & Adaptability, 10 pts]
- **Verifier Cost:** Solver labeling is compute-intensive; real-world environments often lack perfect closed-form oracles.
- **On-Policy vs. Off-Policy Coverage:** If self-play exploration temperature is too low, the model only encounters a narrow slice of states.
- **Student Capacity Ceiling:** Very small students eventually hit an expressive capacity bound where higher-order tactical nuances cannot be fully compressed.

---

### 5. Technical Q&A Preparation (5 minutes) — [Rubric: Technical Q&A, 5 pts]

**Q1: Why did you use soft targets instead of just training the student on ground truth labels?**
> *Answer:* Ground truth labels are one-hot (e.g. `[0, 0, 0, 1, 0, 0, 0]`), which loses information about which alternative moves are almost winning vs catastrophic. The teacher's softened probabilities ($T=2.0$) convey rich dark knowledge about the relative ordering of all 7 columns.

**Q2: How did you prevent data leakage between training and testing?**
> *Answer:* Connect-4 has horizontal reflection symmetry. We hashed positions by their canonical representation: `min(bytes(grid), bytes(flipped_grid))`. This mathematically guarantees that no position or its mirror can ever cross the train/dev/test boundary.

**Q3: When does iterative self-refinement become genuine learning?**
> *Answer:* Self-refinement at test-time only changes the current output without altering network weights. It becomes actual recursive self-learning only when verified outputs update model parameters through a bounded, gated feedback loop with rollback guarantees.
