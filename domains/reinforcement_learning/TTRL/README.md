# TTRL: Test-Time Reinforcement Learning — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | TTRL: Test-Time Reinforcement Learning |
| Venue | arXiv preprint (2504.16084v3, cs.CL/cs.LG) |
| Year | 2025 |
| Authors | Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui et al. (16 authors in total; Tsinghua University, Shanghai AI Lab) |
| Official Code | https://github.com/PRIME-RL/TTRL |
| Venue Kind | paper |

## 🧭 First Principles: Running RL on Test Data With No Ground-Truth Answers

TTRL sets out to answer a very pointed question: when a batch of reasoning problems comes with "only the questions, no ground-truth labels," can we still use reinforcement learning to update an already-pretrained LLM? The premise of traditional RL is that a reward signal is available, yet the core difficulty here is precisely how to estimate the reward at inference time when no ground truth is accessible. The authors name this setting Test-Time Reinforcement Learning; it is a branch of Test-Time Training (TTT), sitting alongside inference-only Test-Time Inference (e.g., majority voting, Best-of-N) under the broad umbrella of Test-Time Scaling.

TTRL's key observation is that majority voting—commonly used within Test-Time Scaling—can itself produce a "good enough" reward to drive RL training. In other words, the model needs no external supervision: by repeatedly sampling the same problem and taking the majority vote, it can fabricate a proxy label (pseudo-label) and then use it to compute a rule-based reward. This connects "TTS's vote aggregation" with "RL's online updates," forming a self-reinforcing loop.

Formally, given a prompt $x$, the policy $\pi_\theta(y \mid x)$ samples $N$ candidate outputs $\{y_1,\dots,y_N\}$, and a consensus output $y^*$ is aggregated by voting as a proxy for the optimal action; the environment assigns a reward $r(y, y^*)$ depending on whether $y$ agrees with $y^*$. The optimization objective is simply to maximize the expected reward $\max_\theta \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r(y,y^*)]$, updating $\theta$ by gradient ascent $\theta \leftarrow \theta + \eta\nabla_\theta\mathbb{E}[r(y,y^*)]$. No $y_t$ label enters here; the training signal is produced entirely by the model's own majority vote.

The concrete reward computation is very plain: first, majority voting selects, from the $N$ sampled answers $P=\{\hat y_i\}_{i=1}^N$, the most frequent prediction as the estimated label $y$; then each candidate answer is scored by a rule—$1$ if correct (equal to the majority answer), $0$ otherwise. This corresponds to $R(\hat y_i, y)=1$ if $\hat y_i=y$, else $0$. The pseudo-code attached to the paper directly illustrates this "take the majority by voting, then score each one by comparison" procedure:

```python
from collections import Counter

def majority_voting_reward_fn(outputs):
    # Assigns reward 1 to each output whose answer matches the majority answer, else 0.
    answers = [extract_answer(output) for output in outputs]
    counts = Counter(answers)
    majority_answer, _ = counts.most_common(1)[0]
    rewards = [1 if ans == majority_answer else 0 for ans in answers]
    return rewards

outputs = llm.generate(problem, n=N)
rewards = majority_voting_reward_fn(outputs)
```

In implementation, TTRL runs GRPO once independently for each benchmark: a cosine schedule with peak learning rate $5\times10^{-7}$ and the AdamW optimizer; in the rollout stage it samples $64$ answers per problem (temperature $1.0$ for Qwen2.5-Math and the LRM, $0.6$ for the rest) to estimate the label by voting, then downsamples to $32$ answers for training. The maximum generation length is set to $32{,}768$ for the LRM and $3{,}072$ tokens for the rest; MATH-500 / AMC / AIME 2024 are run for $10$ / $30$ / $80$ episodes respectively. All experiments were conducted on 8 * NVIDIA A100 80GB GPUs, and this "vote first, then sample" strategy maintains strong results while lowering compute cost.

The main results are striking: measured by pass@1, Qwen2.5-Math-7B improves across the four benchmarks from $12.9$ / $35.6$ / $46.7$ / $29.1$ (AIME 2024 / AMC / MATH-500 / GPQA) to $40.2$ / $68.1$ / $83.4$ / $27.7$, with AIME 2024 improving relatively by about $211.6\%$ and the average improving by about $76.5\%$. Note that GPQA instead drops slightly by $1.4$ points—the only cell that regresses. The table below excerpts two representative backbones:

| Model | AIME 2024 | AMC | MATH-500 | GPQA | Avg |
|-|-|-|-|-|-|
| Qwen2.5-Math-1.5B | 7.7 | 28.6 | 32.7 | 24.9 | 23.5 |
| Qwen2.5-Math-1.5B w/ TTRL | 15.8 | 48.9 | 73.0 | 26.1 | 41.0 |
| Qwen2.5-Math-7B | 12.9 | 35.6 | 46.7 | 29.1 | 31.1 |
| Qwen2.5-Math-7B w/ TTRL | 40.2 | 68.1 | 83.4 | 27.7 | 54.9 |

### A Concrete Example: Why "Wrong Labels Can Still Give the Right Reward" on AIME 2024

Walk through a single training step on AIME 2024 with Qwen2.5-Math-7B: for one problem, sample $64$ answers, extract the answers, and vote. At this point the base model's outputs are extremely dispersed—the single most frequent answer accounts for only $16.6\%$ of all predictions, so the label estimated by majority voting agrees with the ground truth only $37\%$ of the time (label accuracy is only $37\%$). Intuitively such a pseudo-label should be too poor to use, but the measured reward accuracy is as high as $92\%$. The reason is what the authors call a "Lucky Hit": the math verifier assigns a rule-based reward by "comparison," so for a candidate that is itself wrong, as long as it "differs" from the (equally wrong) estimated label, the verifier still assigns a $0$ negative reward—which is exactly the correct reward we want. Thus even though the label is almost never estimated correctly, the dense and dispersed wrong answers keep most rewards correct; the paper also observes that label accuracy rarely exceeds $50\%$, yet reward accuracy stably stays above $75\%$. These $64$ answers are then downsampled to $32$ for the GRPO update, run for the full $80$ episodes, and pass@1 climbs all the way from $12.9$ to $40.2$.

A counterintuitive property of TTRL is that it can "surpass its own training signal." Since training relies on the initial model's majority vote, intuitively the initial maj@n should be the performance ceiling (this is also the ceiling of traditional self-training); but in practice the final avg@16 exceeds the initial maj@16 by more than $20$ points, and avg@64 also stably surpasses Qwen2.5-Math-7B's maj@64 across all benchmarks. The authors describe this phenomenon as the model "lifting itself up by its own bootstraps." Another ceiling is doing RL directly on the test data with true labels (which the authors call RL leakage), and TTRL's curve unexpectedly hugs this leakage ceiling; even the 1.5B model, starting from a subpar performance of $32.7$ on MATH-500, improved by $123.2\%$ to $73.0$.

TTRL is no panacea either. The authors explicitly state that, at the algorithmic level, it is no different from ordinary RL, and therefore inherits properties such as sensitivity to data difficulty, strong reliance on priors, and possible collapse. The most direct source of failure is insufficient priors: splitting MATH-500 by annotated difficulty into L1–L5 and training Qwen2.5-Math-1.5B separately, the accuracy gain decays monotonically from $+45.4$ ($\uparrow175.3\%$) at L1 to $+16.8$ ($\uparrow75.3\%$) at L5, and the degree of response-length compression declines in step, showing that the backbone's prior knowledge is insufficient to handle the complexity of the data. Another source of failure is RL hyperparameters: raising temperature from $0.6$ to $1.0$ increases entropy and exploration, but combined with an inappropriate batch size it can keep entropy from ever decreasing and cause training to collapse.

| MATH-500 difficulty | L1 | L2 | L3 | L4 | L5 |
|-|-|-|-|-|-|
| Backbone Accuracy | 25.9 | 33.0 | 36.3 | 32.5 | 22.3 |
| w/ TTRL Accuracy | 71.2 | 76.2 | 76.3 | 58.7 | 39.2 |
| Δ gain | +45.4 | +43.2 | +40.0 | +26.2 | +16.8 |

## 🧪 Critical Assessment

### The problem setting is a real need, but the "test-time" framing somewhat inflates its novelty
"Doing RL on unlabeled data" is indeed a practical pain point: large-scale annotation is expensive, while difficult new problems keep coming (the paper uses o3 solving only $4\%$ on ARC-AGI-2 as motivation). This problem is real. But note that TTRL's packaging of it as "test-time" is somewhat of a terminological amplification—the method itself is "use majority voting on a batch of unlabeled data to fabricate pseudo-labels, then run GRPO," which is not essentially tied to whether it is "test time"; it holds for any unlabeled training set just as well. The authors themselves also concede in related work that this is highly adjacent to self-rewarding, self-training, and concurrent self-play RL (e.g., Absolute Zero, Genius), the main difference being the use of majority voting to estimate reward so as to mitigate reward hacking. Its contribution is therefore more like "assembling existing components (majority voting + GRPO + rule-based reward) in a clean setting and doing serious empirical work," rather than a wholly new mechanism.

### The baselines are thin, and the evaluation benchmarks overlap heavily with the training data
What most deserves a question mark is the evaluation design. TTRL's main comparison group is only the backbone itself, and the authors themselves concede that "the comparison with previous SOTA looks unfair (different setup)." More critically: it "trains on the benchmark's problems and then evaluates on the same batch of problems." Although no answer labels are used, repeatedly doing online RL on this batch of prompts and then reporting pass@1 on this batch of prompts naturally overfits this data distribution—this is exactly an evaluation defined by the method's own strengths (the benchmark is framed by the authors around the advantages of their own method). The authors mitigate this concern with OOD transfer experiments and the RL leakage ceiling, which is the right direction, but the absolute numbers in the main table should still be understood as results "after self-adapting on that problem set," not as generalization to unseen problems. On GPQA, Qwen2.5-Math-7B instead regresses, and Mistral-Nemo drops to $0$ on AIME, which also hints that the effect depends heavily on the backbone's mathematical prior.

### "Surpassing the maj@n ceiling" is eye-catching, but the mechanistic explanation remains empirical
The paper's brightest selling point is that avg@16 exceeds the initial maj@16 by more than $20$ points and approaches the leakage ceiling RL leakage. If this phenomenon holds, it is significant. But note: the so-called "ceilings" are two references defined by the authors themselves (the initial model's maj@n, and leakage training), while "why it can be surpassed" is currently supported only by empirical arguments like the Lucky Hit and by the curves of a single backbone (Qwen2.5-Math), lacking a theoretical analysis of convergence or generalization bounds—the authors also list theoretical analysis as future work. The conditions for the Lucky Hit to hold are in fact quite delicate—it relies on "wrong answers being highly dispersed"; once the model's error modes concentrate (producing a consistent but wrong majority answer), reward accuracy may collapse, and the paper does not systematically sweep this boundary condition.

### Real-world relevance: effective for "verifiable tasks with strong priors," but extrapolate conservatively
For deployment, TTRL's sweet spot is tasks where "the answer can be compared by a rule-based verifier and the backbone already has sufficient prior" (math, multiple-choice questions). The difficulty ablation shows: once problems exceed the model's prior (L4–L5), the gain shrinks quickly, which amounts to saying it is more like "squeezing out the model's existing capability" than "learning new knowledge." For open-ended tasks without clean verifiable answers (dialogue, agentic, scientific discovery), whether the majority-voting reward proxy remains reliable is only listed as a future direction by the paper and not verified. My reading is therefore: this is a clean, reproducible, empirically solid preliminary study (and open-source), but reading it as "LLMs can self-evolve without limit" is a clear overextrapolation—its ceiling is precisely locked by the backbone's prior and the verifier's comparability.

## 🔗 Related notes

- [PPO: Proximal Policy Optimization](../ppo/) — Besides GRPO, TTRL also uses PPO as one of its compatible RL algorithms.
