# Reflexion — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Reflexion: Language Agents with Verbal Reinforcement Learning |
| Venue | NeurIPS |
| Year | 2023 |
| Authors | Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao |
| Official Code | https://github.com/noahshinn024/reflexion |
| Venue Kind | paper |

## First Principles

This note is written based on the full text of the arXiv preprint `2303.11366` (v4); details in the formal camera-ready version (NeurIPS 2023) may differ slightly; the venue and year are taken from the DBLP record `conf/nips/ShinnCGNY23`.

### Problem: letting a language agent learn from failure without updating weights

When a large language model (LLM) is used as a goal-driven agent to operate external environments such as games, compilers, and APIs, the traditional way to make it "get better through trial-and-error" is reinforcement learning—but policy/value gradients require large numbers of interaction samples and expensive model finetuning, which is impractical for LLMs with tens of billions of parameters. This paper argues: rather than updating weights, update the agent's memory, writing down the lesson of each failure in natural language to serve as the context for the next round.

The paper calls this signal a kind of `semantic gradient`: the environment only gives binary or scalar success/failure, and the Self-Reflection model amplifies it into a concrete, actionable piece of textual feedback (which step was wrong, how to fix it next time), and this text makes credit assignment easier than a scalar reward. The authors also acknowledge the cost of this method: it relies entirely on the LLM's ability to self-evaluate (or on human heuristics), and it has no formal guarantee for success.

![Overview of Reflexion operating on the three task classes of decision-making, programming, and reasoning](imgs/reflexion_tasks.png)

### Three modules: Actor, Evaluator, Self-Reflection

Reflexion splits the agent into three independent LLM instances: the **Actor** $M_a$ generates text and actions from state; the **Evaluator** $M_e$ scores the produced trajectory; the **Self-Reflection** model $M_{sr}$ turns the score and trajectory into linguistic feedback. The policy is parameterized as "the Actor's weights plus a readable/writable memory," meaning that learning happens in memory rather than in $M_a$:

$$
a_t \sim \pi_\theta(a_t \mid s_t), \quad \theta = \{M_a, \mathit{mem}\}, \quad r_t = M_e(\tau_t)
$$

The Evaluator's implementation varies by task: reasoning tasks use exact match (EM) grading, decision-making uses human heuristics, and programming uses whether the agent's self-generated unit tests pass. This "adapt-to-the-situation" reward design is key to Reflexion working across the three task classes, but it also pushes the responsibility for evaluation quality onto each task's Evaluator.

After the Self-Reflection model receives the sparse success/failure signal, the current trajectory, and the existing memory, it generates a detailed piece of verbal feedback $sr_t$ and stores it in long-term memory $\mathit{mem}$; short-term memory is the current trajectory. To avoid exceeding the LLM's context limit, memory is capped at a maximum of $\Omega$ experiences (the paper often sets this to 1 to 3):

$$
\mathit{mem} \leftarrow \mathit{mem} \cup \{sr_t\}, \quad |\mathit{mem}| \le \Omega, \quad \Omega \in \{1,2,3\}
$$

![Reflexion's Actor–Evaluator–Self-reflection loop and short/long-term memory](imgs/reflexion_architecture.png)

### The Reflexion loop

The whole process is an iterative optimization: the Actor first produces a trajectory, the Evaluator scores it, Self-Reflection produces a lesson and writes it to memory, then the agent retries with the memory, until the Evaluator judges a pass or the trial limit is reached:

```text
Initialize Actor M_a, Evaluator M_e, Self-Reflection M_sr
policy π_θ, θ = {M_a, mem};  generate initial trajectory τ_0 with π_θ
r_0 = M_e(τ_0);  sr_0 = M_sr(τ_0, r_0);  mem ← [sr_0];  t = 0
while (M_e has not passed) and (t < max trials):
    τ_t = [a_0, o_0, ..., a_i, o_i]  produced by π_θ
    evaluate τ_t: r_t = M_e(τ_t)
    generate self-reflection sr_t = M_sr(τ_t, r_t)
    append sr_t to mem (discard the oldest if it exceeds Ω)
    t ← t + 1
return
```

### A full trial: using HumanEval code generation as an example

Let us walk through concretely with a programming task. Given a natural-language description, the Actor first writes a candidate program, then uses Chain-of-Thought prompting to generate a batch of tests with natural-language descriptions; the system filters out syntactically erroneous tests by whether a valid abstract syntax tree (AST) can be built, and finally samples at most $n=6$ to form a test suite $T=\{t_0,\dots,t_n\}$. The programming agent's memory cap is set to only 1 experience.

The Evaluator is simply "run this batch of self-generated tests": if all pass, it returns that solution early; if it fails, Self-Reflection reads the failed tests and program, writes a piece of text on "where it went wrong and how to fix it" and stores it in memory, and the Actor rewrites with it. Because programming can self-evaluate using its own generated tests, Reflexion can legitimately report pass@1. On HumanEval Python, this loop pushes pass@1 from a baseline of 0.80 to 0.91 (absolute +11%), surpassing GPT-4's then-SOTA of 80.1.

The same mechanism, however, actually loses to the baseline on MBPP Python (77.1 vs GPT-4's 80.1). The authors' diagnosis is false positives: the self-generated tests all pass on an incorrect solution, so the agent misjudges success and submits early. Quantitatively, the false-positive test execution rate on MBPP Python is as high as 16.3%, while on HumanEval Python it is only 1.4%—the same "use your own written tests as reward" design directly erodes performance on datasets with poor test quality; this is one of the few cells Reflexion loses.

### Main experimental results

The main pass@1 results for programming are as follows (units in %, bold is the best in each row):

| Benchmark + Language | Prev SOTA Pass@1 | SOTA Pass@1 | Reflexion Pass@1 |
|-|-|-|-|
| HumanEval (PY) | 65.8 (CodeT + GPT-3.5) | 80.1 (GPT-4) | **91.0** |
| HumanEval (RS) | -- | 60.0 (GPT-4) | **68.0** |
| MBPP (PY) | 67.7 (CodeT + Codex) | **80.1** (GPT-4) | 77.1 |
| MBPP (RS) | -- | 70.9 (GPT-4) | **75.4** |
| Leetcode Hard (PY) | -- | 7.5 (GPT-4) | **15.0** |

The other two task classes also show consistent gains: on decision-making's AlfWorld, ReAct + Reflexion uses a simple heuristic to detect hallucination and invalid planning, completing 130 of 134 tasks, an absolute improvement of 22% over a strong baseline (within 12 iterative learning steps); on reasoning's HotPotQA, it improves 20% over the baseline. The authors also do an ablation: replacing self-reflection with an episodic memory (EPM) that keeps only the most recent trajectory, the additional gain from self-reflection is an absolute 8%, corroborating that "refinement guided by reflection" beats "pure refinement."

## 🧪 Critical Assessment

### Whether the problem is real and important

"Letting an LLM agent get better from failure without finetuning" is a real need: for closed-source, ultra-large models, gradient-based RL is nearly infeasible, while in-context learning is indeed currently the only affordable lever. Amplifying the success/failure signal into a linguistic lesson, storing it in memory, and feeding it back—the pain point this framework captures is real, not a problem contrived to publish a paper.

### Whether baselines, ablations, and datasets are adequate

Each of the three task classes has reasonable baselines (ReAct, CoT, CoT(GT), GPT-4), and the ablation cleanly separates the two factors of test generation and self-reflection (on 50 HumanEval Rust problems, 0.60 → 0.52 → 0.60 → 0.68), which is more solid than most contemporaneous agent papers. But there is plenty to discount: many curves are reported only at a single temperature of 0.7 and a single run, without cross-seed variance or confidence intervals, while the output of self-reflection is itself highly stochastic; the evaluations on HotPotQA and AlfWorld use only 100–134 problems each, a small sample. The statistical robustness of these numbers is unproven, and the reader has no way to judge how much of the 8% or 22% gains fall within the noise range.

### Whether it is a genuine innovation or a recombination of existing components

Reflexion's components—self-evaluation, self-generated tests, episodic memory, retry loop—all appear in prior work such as Self-Refine, CodeT, and Self-Debugging, and the paper's own related-work comparison table faithfully lists them. The real novelty is stringing together "sparse reward → linguistic lesson → write to persistent memory → affect subsequent policy" into an explicit verbal RL loop, and claiming that memory is the policy parameter being optimized. My judgment is: this is closer to "reframing existing debugging/refinement techniques in RL vocabulary and adding persistent memory"; its contribution is conceptual integration and naming, not an entirely new mechanism; this still has value, but the headline number of 91% comes more from the strength of the GPT-4 base model and the self-evaluable programming setting than from the reflection mechanism being irreplaceable.

### The trade-off of LeetcodeHardGym and custom benchmarks

The paper introduces LeetcodeHardGym (40 hard-level problems covering multiple languages, with problems deliberately chosen to be released after GPT-4's pretraining cutoff to avoid memorization contamination), and its motivation—avoiding data contamination—is legitimate. But this is also a benchmark self-defined by the authors, only 40 problems in scale, on which Reflexion goes from 7.5 to 15.0, seemingly doubling, but in fact both absolute values are very low and lack third-party comparison; treating it as evidence that it "solved hard problems" would be overly optimistic, and the more conservative reading is "demonstrated relative improvement on a new and small sandbox."

### Whether the claimed problem is really solved, and how much real-world significance it has

Under the narrow "self-evaluable" setting (with a compiler, generatable unit tests, EM grading), Reflexion does reliably push scores upward. But the limitations the paper itself acknowledges point out the ceiling on extrapolation: memory is only a fixed-size sliding window, has no formal convergence guarantee, and may still get stuck in a local minimum; and the quality of the whole mechanism is entirely tied to the Evaluator—MBPP's false-positive collapse is proof: when self-evaluation is unreliable, reflection amplifies errors rather than correcting them. For real tasks without a cheap, trustworthy verifier (open-ended writing, most robotics, decisions without clear ground truth), whether this framework can transfer remains an open question, not one already solved.

## 🔗 Related notes

- [ChatGPT / InstructGPT (RLHF)](../ChatGPT/)
- [Instruction Tuning with GPT-4](../InstructioinTuningWithGPT4/)
