# 🎓 iCog Labs AI Internship: Technical Presentation Guide
**Topic:** Verified Recursive Self-Improvement on Connect-4 (Track 1)  
**Speaker:** AI/ML Intern  
**Session Timing:** 20 Minutes Total (3m Design + 5m Method + 5m Results/Demo + 2m Limitations + 5m Q&A)  
**Evaluation Standard:** iCog Labs Internship Rubric (100 Points)

---

## ⏱️ Slide-by-Slide Presentation Structure

### 1. Problem & Design (3 minutes) — [Rubric: Conceptual Understanding, 20 pts]
- **The Core Question:** Can a neural network autonomously self-improve through self-play practice rounds without collapsing into self-reinforcing errors, and does external solver verification prevent policy degradation?
- **The 3 Experimental Conditions:**
  1. **Version 1 (Fixed Baseline):** Passive learning from static dataset slices with zero practice rounds.
  2. **Version 2 (Unverified Self-Improvement):** Autonomous self-play, learning from its own moves without external verification.
  3. **Version 3 (Solver-Verified Self-Improvement):** Autonomous self-play, where every encountered position is evaluated and labeled by an infallible Connect-4 bitboard solver referee.
- **Strict Parity Safeguards:** Identical model architectures, identical gradient steps, identical total training examples, and zero data leakage across canonical reflection-symmetric hash splits.

---

### 2. Method & Architecture (5 minutes) — [Rubric: Technical Implementation, 25 pts]
- **The Engine Workflow:**
  $$\text{Generate Practice Games} \longrightarrow \text{Solver Verification} \longrightarrow \text{Replay Buffer} \longrightarrow \text{Model Update} \longrightarrow \text{Acceptance Gate}$$
- **Why Experience Replay Buffer Matters:** Prevents catastrophic forgetting by anchoring the network on foundational game states while assimilating newly corrected blunder positions.
- **Infallible Referee (Bitboard Solver):** Pascal Pons C++ solver identifies optimal moves and tactical blunders with sub-millisecond evaluation via an SQLite-backed opening book cache.
- **Acceptance Gate & Rollback:** Held-out dev performance is checked after every round. If a proposal degrades beyond tolerance ($\tau = 0.02$), it is rejected and rolled back to the previous accepted checkpoint.

---

### 3. Quantitative Results & Live Demonstration (5 minutes) — [Rubric: Experimental Reasoning, 20 pts]

#### A. The Progress Curves & Acceptance Table
- **Show Figure:** `results/figures/demo_progress.png`
  - **Panel 1 (Optimal-Move Agreement):** Agreement on unseen solver-verified positions monotonically improves from Round 0 to Round 3.
  - **Panel 2 (Opponent Win Rates):** Win rates systematically increase against Random, Heuristic (threat-detector), and Minimax opponents.
- **Show Table:** Accepted vs. rejected rounds under the strict improvement acceptance gate (`new_agreement > prev_agreement`).

#### B. Live Head-to-Head Showdown (Round 3 vs Round 0)
- Run 50–100 live matches between the initial baseline and the final improved agent:
  ```bash
  python demo.py --results_dir results --num_games 50
  ```
  - Demonstrates clear dominance of Round 3 over Round 0 with alternating first/second player turns.

#### C. The Convincing Moment: Blunder vs Optimal Move
- Display a concrete board position where Round 0 makes a fatal blunder, and Round 3 plays the solver-optimal move.
- Explains the exact tactical difference on the visual Connect-4 grid.

---

### 4. Limitations & Risk Analysis (2 minutes) — [Rubric: Creativity & Adaptability, 10 pts]
- **Verifier Cost & Oracle Availability:** Bitboard solver labeling is closed-form and infallible; real-world environments often lack perfect oracles.
- **Single-Variant Causal Attribution Trade-Off:** While the pipeline proves clear policy improvement across rounds, isolating whether solver verification was the sole causal driver requires an unverified self-play control ablation.
- **On-Policy vs. Off-Policy Coverage:** Self-play exploration temperature ($T=0.8$) must balance tactical blunder exposure with broad board diversity.

---

### 5. Technical Q&A Preparation (5 minutes) — [Rubric: Technical Q&A, 5 pts]

**Q1: When does iterative self-refinement become genuine learning?**
> *Answer:* Self-refinement at test-time only changes the current output without altering network weights. It becomes actual recursive self-learning only when verified outputs update model parameters through a bounded, gated feedback loop with rollback guarantees.

**Q2: How did you prevent data leakage between training and testing?**
> *Answer:* Connect-4 has horizontal reflection symmetry. We hashed positions by their canonical representation: `min(bytes(grid), bytes(flipped_grid))`. This mathematically guarantees that no position or its mirror can ever cross the train/dev/test boundary.

**Q3: How do you guarantee the agent doesn't suffer from policy collapse or catastrophic forgetting during self-improvement?**
> *Answer:* We employ a cumulative experience replay buffer that retains foundational board states while adding newly corrected mistakes, paired with an acceptance gate that automatically rolls back any proposal degrading validation agreement beyond tolerance $\tau = 0.02$.
