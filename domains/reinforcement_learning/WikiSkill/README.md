# WikiSkill — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution |
| Venue | arXiv preprint (2608.27454v1) |
| Year | 2026 |
| Authors | Liyan Tang, Cyrus Rashtchian, Chun-Sung Ferng, Andrew Tomkins, Da-Cheng Juan, Tu Vu |
| Official Code | unknown |
| Venue Kind | paper |

> This note is written from the arXiv preprint `2608.27454v1` (2026-08-28); this version has not yet been peer-reviewed, and the official published version may differ. The authors are affiliated with Google Research and Virginia Tech.

## Introduction

The concrete problem WikiSkill sets out to solve is this: when an LLM agent "self-evolves" agent skills (a filesystem module that writes procedural knowledge into a `SKILL.md` file) by repeatedly executing tasks, the "insights" that guide skill improvement are scattered across each round's optimization history, making them hard to systematically reuse across iterations. For example, the lesson "do not return an object to its origin location" learned in round 2 — if it only exists as a rejected diff or a single feedback message — is very likely to be repeated by the proposer in round 5, or re-proposed after already having been rejected.

Why does this matter? Because the ceiling on skill evolution's effectiveness depends on whether the proposer can, each round, stand on "previously organized knowledge" rather than re-reading the raw trajectories from scratch every time. Inspired by Karpathy's "LLM Wiki" perspective, the authors argue for "compiling" experience into persistent, cumulative knowledge, and pose the core question: can an agent's experience be compiled into persistent knowledge to support long-term skill evolution?

WikiSkill's high-level solution is to insert a structured knowledge layer (wiki) between "raw experience" and "executable skills." It splits the agent workspace into three layers — a Raw Layer that stores immutable execution trajectories, a Wiki Layer that maintains structured knowledge and accumulates across iterations, and a Skill Layer that carries the evolving procedural knowledge — and keeps it running with a loop of four components: Inference Agent, Wiki Maintainer, Skill Proposer, and Gating and Rollback. The key design point is that skills can be rolled back, but the wiki is never rolled back, so subsequent updates can build on accumulated knowledge.

How does the paper measure whether the solution works? The authors compare a no-skill baseline against three skill-evolution baselines (Trace2Skill, EvoSkill, SkillOpt) across five benchmarks (LiveMathematicianBench, SealQA, SpreadsheetBench, OfficeQA, ALFWorld) and five models (Qwen-3.5-4B/9B, Qwen-3.6-27B, Gemma-4-31B, Gemini-3.5-Flash). All methods start from an empty skill set and inject the full text of the evolved skills into the Inference Agent's system prompt; the metric is average accuracy per benchmark, reported as the mean over three independent full evolution runs together with a paired bootstrap significance test. There are three core claims: WikiSkill beats existing methods in most settings, skill evolution and model scale are complementary, and evolved skills transfer across models and across families.

## First Principles

### Problem formalization: writing skill evolution as a gated search

The paper splits a task dataset $\mathcal{D} = \{(x_i, y_i)\}$ into three mutually exclusive splits, $\mathcal{D}_{\text{train}}$, $\mathcal{D}_{\text{val}}$, $\mathcal{D}_{\text{test}}$. An agent $\pi$ is equipped with a tool set $\mathcal{U}$ (e.g. bash, web search, file reader) and a set of active skills $S = \{s_1, \dots, s_M\}$, where each skill is a filesystem directory containing a `SKILL.md`; the skill set starts as the empty set $\emptyset$ and is evolved per dataset. When executing task $x_i$, the agent produces a trajectory $\tau_i \sim \pi(x_i; S)$, whose final action emits a prediction $\hat{y}_i$ scored by a scoring function $f(\hat{y}_i, y_i) \in [0,1]$, and the performance on a split $\mathcal{R}(\mathcal{T}_{\text{split}})$ is the average of all task scores on that split.

At round $k$, WikiSkill's system state is a pair $(S_k, W_k)$, where $S_k$ is the active skill set and $W_k$ is the persistent knowledge base (wiki). The key asymmetry here is: candidate skill updates must pass a validation gate and can be rolled back if the score regresses, but $W_k$ keeps accumulating and compounds across iterations, unaffected by rollback. The system starts from $(S_0, W_0) = (\emptyset, \emptyset)$, and the goal is to maximize the final performance on the unseen test set $\mathcal{R}(\mathcal{T}_{\text{test}})$. Notably, this entire process involves no gradient updates whatsoever — it is a black-box hill-climbing search over the discrete object "skill files," with model parameters frozen throughout, and "reward" is merely a $[0,1]$ score.

### The three-layer knowledge architecture: what is immutable, what compounds, what is reversible

![WikiSkill's three-layer architecture and four-step evolution loop](imgs/fig2_architecture.png)

*Figure 1: WikiSkill splits the workspace into three layers, Raw / Wiki / Skill, and forms one round of the loop from the four steps Inference Agent → Wiki Maintainer → Skill Proposer → Gating & Rollback. The three layers have different invariants: Raw is written once and permanent (Permanent, Write Once), Wiki only grows and never resets (Compounding, Never Reset), and Skill supports conditional update and rollback (Reversible, Conditional Update). Note the access asymmetry depicted by the arrow topology: the Inference Agent in Step 1 has only the two connections "Inject Skills / Write Traces" and no arrows at all to the Wiki Layer; the Skill Proposer in Step 3 is the only component that simultaneously "Read Skill, Wiki, Traces" across all three layers; Step 4 writes back conditionally to the Skill Layer (Update the Skill if Better) and unconditionally to the Wiki Layer (Update Wiki Logs). (Taken from the paper's Figure 2)*

The Raw Layer (`raw/`) stores the raw execution trajectories $\tau_i$ collected from training samples each round, containing the agent's step-by-step reasoning, tool calls, tool outputs, and final answers; to preserve history, this layer is immutable, and both the Wiki Maintainer and the Skill Proposer can read from it to analyze behavior.

The Wiki Layer (`wiki/`) compiles the raw trajectories into structured, compoundable knowledge, and is maintained continuously throughout the evolution process. It contains a `patterns/` directory holding individual markdown files that record specific failure modes or success strategies along with their actionable workarounds; and it provides long-term historical awareness across iterations through an evolution log (`logs.md`, updated by the Wiki Maintainer) and a skill-impact tracker (`skill-impact.md`, updated programmatically by the outer harness after the validation gate). These records let the two agents observe the full skill-acceptance history (avoiding re-proposing already-rejected interventions), track whether prior proposals succeeded, and identify errors that recur across iterations. The wiki is not reset between iterations, but instead keeps accumulating and compiling.

The Skill Layer (`skills/`) carries the active skill set $S$. Each skill directory contains two files: `SKILL.md` is the full text of the skill; `PURPOSE.md` links that skill back to the wiki pattern that inspired its creation or modification, forming an auditable chain from "skill" back to "motivating evidence."

### The evolution loop: four components and what each of them can see

One iteration runs four components in sequence. The Inference Agent executes tasks using the current `skills/` and produces immutable trajectories into `raw/`; during training rollout it is forbidden from accessing the Wiki Layer (an ablation shows that giving it the wiki is actually harmful). The Wiki Maintainer then analyzes the raw trajectories and the existing wiki, performs root-cause analysis of failures, extracts success strategies, and updates the pattern directory and the evolution log. The Skill Proposer next examines the updated wiki and reads the latest round's trajectories, and produces an "atomic" candidate proposal $P_k$ (targeting only a single skill, doing a creation or an incremental patch edit). Finally, Gating and Rollback evaluates the candidate skill on the validation split, accepting modifications that improve validation performance and rolling back otherwise.

The paper writes the full procedure as Algorithm 1, whose skeleton is as follows (adapted from the algorithm in the paper's Appendix, reusing the notation above):

```text
Input: D_train, D_val, metric R, number of iterations K
Initialize S_0 = ∅, W_0 = ∅
Baseline validation: T_val,0 = rollout(π(·; S_0)) on D_val;  R_best = R(T_val,0)
for k = 1..K:
    if R_best == 1.0: break                      # early stopping
    Inference: T_train,k = rollout(π(·; S_{k-1})) on D_train
    Sample a subset T_sample,k ⊂ T_train,k        # ≤5 failures + ≤3 successes
    Wiki Maintenance: W'_k = M_WM(W_{k-1}, T_sample,k)
    Skill Proposal:   P_k  = M_P(W'_k, S_{k-1}, T_train,k)   # ReAct, reads files on demand
    Apply:            S'_k = Apply(S_{k-1}, P_k)
    Validate:         T_val,k = rollout(π(·; S'_k)) on D_val
    if R(T_val,k) > R_best:                       # accept only on strict improvement
        S_k = S'_k;  R_best = R(T_val,k);  a_k = Accepted
    else:
        S_k = S_{k-1};  a_k = Rejected            # roll back the skill only, keep the wiki
    Update Wiki Log: W_k = Update(W'_k, P_k, R(T_val,k), a_k)
return S_K, W_K
```

The gate rule is strict improvement: a candidate is accepted only when its validation score is higher than the historical best $\mathcal{R}_{\text{best}}$, otherwise the skill rolls back to the last successful configuration. Formalized as

$$
S_k = \begin{cases} S'_k & \text{if } \mathcal{R}(\mathcal{T}_{\text{val},k}) > \mathcal{R}_{\text{best}} \\ S_{k-1} & \text{otherwise} \end{cases}
$$

$\mathcal{R}_{\text{best}}$ is initialized to the baseline score of the empty skill set on $\mathcal{D}_{\text{val}}$, i.e. $\mathcal{R}(\mathcal{T}_{\text{val},0})$; if the validation score reaches the ceiling $\mathcal{R}_{\text{best}} = 1.0$ at any point, the evolution loop terminates early. Whether accepted or not, the outer harness appends the proposal metadata, the target skill name, the modification's unified diff, the validation score, and the final result $a_k \in \{\text{Accepted}, \text{Rejected}\}$ to `skill-impact.md`, forming an objective audit trail for later proposers to consult so as to avoid repeating mistakes.

### What each agent can see: the access boundaries are the core of the design

The information-access boundaries of the four components are asymmetric, and this is precisely where WikiSkill's design claim lies. The Inference Agent only gets the full text of the active skills $S_{k-1}$ (injected directly into the system prompt), and is not given the wiki during training rollout; the authors follow prior work in using "full-text injection" rather than retrieval, in order to rule out "skill triggering or retrieval failure" as a confounding variable. The Wiki Maintainer gets the full wiki context $W_{k-1}$ plus the sampled trajectories $\mathcal{T}_{\text{sample},k}$, and updates patterns with incremental, patch-style edits (append / replace / insert_after) while synchronously rewriting the `index.md` table of contents and appending to `logs.md`; there is no hard cap on the number of patterns created or edited per round. The Skill Proposer operates in a multi-turn ReAct fashion: it is initially given only the wiki index $I(W'_k)$, `skill-impact.md`, and a condensed summary of all training-task results (pass/fail, prediction and ground truth), then autonomously uses the `read_file` tool to pick and read specific pattern pages and raw trajectories on demand, and only then synthesizes a proposal; the prompt requires it to "read at least 4 execution trajectories" before proposing.

### A concrete walkthrough of one iteration: Qwen-3.6-27B on ALFWorld

![Case study of wiki-guided skill evolution](imgs/fig3_case_study.png)

*Figure 2: A real evolution segment on ALFWorld (Qwen-3.6-27B), where four colored arrows each mark a causal path. On the left, the persistent Wiki Layer's `skill-impact.md` records that Iteration 0's `goal-directed-action` was rejected (REJECTED, val score = 0.72), and that Iteration 1 and Iteration 4's `break-repetition-loop` were accepted (ACCEPTED, val score = 0.78); `logs.md` records cross-round error recurrence and acceptance decisions; `patterns/` accumulates evidence such as `take-examine-move-loop.md` and `multi-operation-loop.md`. Red arrow: the rejection history explains skill motivation (`skill-impact.md`'s REJECTED → the right-side `PURPOSE.md`, "rejected for being too abstract"); green arrow: an accepted skill update (ACCEPTED → `SKILL.md`); yellow arrow: a specific pattern spawns a specific rule (`take-examine-move-loop` / `multi-operation-loop` → the rules of Iter 1 / Iter 4); blue arrow: the evolution log records patterns and decisions in chronological order (`logs.md` ↔ `patterns/`). On the right, the Skill Layer shows the accepted skill's `SKILL.md` (the rule "Never Return an Item to Its Origin Location") and the back-linked `PURPOSE.md`. (Taken from the paper's Figure 3)*

Let us ground the abstract loop above in a real example. In Iteration 0, the Wiki Maintainer identifies a basic looping behavior from the trajectories and writes it into `take-examine-move-loop.md`; the Skill Proposer proposes a more abstract `goal-directed-action`, but it fails to improve on the validation set (`logs.md` records val score = 0.72, not exceeding the baseline) and is therefore rejected. The key point is that `skill-impact.md` retains the diff and result of this rejected proposal, letting subsequent updates know "this path does not work."

Guided by this audit trail, in Iteration 1 the Skill Proposer creates `break-repetition-loop`, bringing in a concrete action rule "Never Return an Item to Its Origin Location," which this time passes (val score = 0.78 > 0.72) and is accepted. As new looping variants appear in rollouts (`multi-operation-loop.md`), the Wiki Maintainer keeps accumulating new evidence; then, guided by these accumulated patterns and new trajectories, the Skill Proposer in Iteration 4 further refines the skill with a new rule "Each Operation Type ONCE Per Item." Ultimately, on the ALFWorld test set, Qwen-3.6-27B improves from the no-skill 52.8% to WikiSkill's 77.6% (+24.8 points). This example makes concrete how "persistent knowledge feeds subsequent skill refinement across iterations": without the rejection record in `skill-impact.md`, Iteration 1 would very likely have re-proposed the rejected abstract approach.

### Experimental setup: small validation sets, full batch, three independent runs

The split sizes and tools of the five benchmarks are in the table below (taken from the paper's Table 6). Note that the validation sets are generally very small (10–40 items), which is crucial to the noise of the "strict improvement" gate.

| Benchmark | Interaction mode | Train | Val | Test | Environment tools |
|-|-|-|-|-|-|
| LiveMath | Single-Step | 35 | 18 | 124 | None (direct reasoning) |
| SealQA | Multi-Step | 16 | 10 | 85 | `web_search`, `read_file` |
| SpreadSheet | Multi-Step | 80 | 40 | 280 | `bash` |
| OfficeQA | Multi-Step | 50 | 24 | 172 | `glob`, `grep`, `read` |
| ALFWorld | Multi-Step | 39 | 18 | 134 | Admissible Actions |

On implementation details, each round applies stratified sampling for the Wiki Maintainer: at most 8 trajectories (up to 5 failures for root-cause analysis, up to 3 successes to prevent degradation), with each execution log truncated to 15,000 characters before injection. The Skill Proposer's ReAct rounds are roughly $10 \le T_{\text{ReAct}} \le 20$. The splits and tool sets of all methods are strictly aligned with prior work, and SealQA uniformly uses the July 2026 version. The significance test uses 1,000 paired bootstrap, and across benchmarks uses stratified macro-average resampling with each benchmark weighted equally.

### Main results: winning on average, but not a clean sweep cell by cell

The table below excerpts three model blocks from the paper's Table 1 (see the original for the full five models); each cell is the test-set mean over three independent evolutions.

| Model | Method | LiveMath | SealQA | SpreadSheet | OfficeQA | ALFWorld | Avg. |
|-|-|-|-|-|-|-|-|
| Qwen-3.5-4B | No skill | 29.1 | 32.5 | 14.6 | 30.2 | 24.4 | 26.2 |
| Qwen-3.5-4B | SkillOpt | 48.7 | 33.3 | 14.0 | 34.5 | 45.3 | 35.2 |
| Qwen-3.5-4B | WikiSkill | 49.7 | 39.4 | 21.1 | 28.5 | 53.7 | 38.5 |
| Qwen-3.6-27B | No skill | 33.9 | 27.5 | 40.8 | 42.1 | 52.8 | 39.4 |
| Qwen-3.6-27B | EvoSkill | 57.3 | 32.9 | 59.5 | 52.5 | 64.2 | 53.3 |
| Qwen-3.6-27B | WikiSkill | 61.9 | 41.6 | 81.7 | 53.7 | 77.6 | 63.3 |
| Gemini-3.5-Flash | No skill | 33.0 | 29.4 | 50.5 | 48.6 | 85.9 | 49.5 |
| Gemini-3.5-Flash | WikiSkill | 72.6 | 44.7 | 76.6 | 60.7 | 85.9 | 68.1 |

WikiSkill takes the highest score in the "average" column across all five models; relative to each model's strongest competing method, the averages improve by 3.3, 5.1, 10.0, 5.8, 12.0 points respectively (corresponding to Qwen-3.5-4B, Qwen-3.5-9B, Qwen-3.6-27B, Gemma-4-31B, Gemini-3.5-Flash). But "highest average" does not equal "cell-by-cell sweep": take Qwen-3.5-4B for example — on OfficeQA it actually drops from the no-skill 30.2% to 28.5%, also below SkillOpt's 34.5%; the authors explain this as small models regressing to their default reading behavior on long-context multi-step search. You can verify the average yourself: Qwen-3.5-4B WikiSkill's (49.7+39.4+21.1+28.5+53.7)/5 = 38.48 ≈ 38.5, consistent with the table. It is also worth mentioning that even the relative ranking of the two skill-evolution baselines is not fixed: on the smallest Qwen-3.5-4B, SkillOpt (average 35.2) beats EvoSkill (33.7), but from Qwen-3.5-9B onward EvoSkill (42.3 vs 40.2) overtakes and maintains the lead on subsequent models, showing that "which baseline is better" itself interacts with model capability.

![How WikiSkill's average accuracy changes with model scale](imgs/fig1_scaling.png)

*Figure 3: The average accuracy of no-skill / EvoSkill / SkillOpt / WikiSkill on four models (Qwen-3.5-4B, Qwen-3.5-9B, Qwen-3.6-27B, Gemini-3.5-Flash), with the Y axis fully marked from 30% to 75%. WikiSkill (yellow line, diamonds) is at the top on every model, and its lead over the stronger models progressively widens — leading the second-best baseline by +3.3, +5.1, +10.0, +12.0 points respectively. Meanwhile the two baselines EvoSkill (green) and SkillOpt (orange) cross in the figure: on the smallest 4B, SkillOpt (35.2) is slightly above EvoSkill (33.7), but from 9B onward EvoSkill overtakes and stays on top (9B 42.3 vs 40.2, 27B 53.3 vs 50.7, Flash 56.1 vs 55.9). Note that this figure plots only four models (omitting Gemma-4-31B) and the legend omits Trace2Skill — this does not correspond one-to-one with Table 1's full five models and five methods. (Taken from the paper's Figure 1)*

One of the paper's key claims is that "skill evolution and model scale are complementary." Within the Qwen family, WikiSkill's average gain over no-skill rises with scale: +12.3 points for 4B, +17.5 points for 9B, +23.9 points for 27B; on SpreadSheet it is especially dramatic, with the three being +6.5, +9.3, +40.9 points respectively. At the same time, evolved skills can compensate for scale gaps: Qwen-3.5-9B with WikiSkill reaches an average of 47.4%, beating the no-skill Qwen-3.6-27B's 39.4%.

### Persistent-knowledge ablation: the wiki helps the proposer but harms inference

The authors do a 2×2 ablation over "wiki access" using Gemini-3.5-Flash (whether or not to give it to the Inference Agent, whether or not to give it to the Skill Proposer; when the proposer has no wiki, the Wiki Maintainer is also removed, equivalent to turning off cross-iteration knowledge accumulation).

| Inference Agent wiki | Skill Proposer wiki | LiveMath | SealQA | SpreadSheet | OfficeQA | Avg. |
|-|-|-|-|-|-|-|
| No skill | — | 33.0 | 29.4 | 50.5 | 48.6 | 40.4 |
| ✓ | ✗ | 43.8 | 42.0 | 44.4 | 51.0 | 45.3 |
| ✗ | ✗ | 51.3 | 38.4 | 49.9 | 55.2 | 48.7 |
| ✓ | ✓ | 64.8 | 42.8 | 80.2 | 55.6 | 60.9 |
| ✗ | ✓ (default) | 72.6 | 44.7 | 76.6 | 60.7 | 63.7 |

Under the premise that the Inference Agent does not see the wiki, giving the Skill Proposer access to the persistent wiki lifts the average from 48.7% to 63.7% (+15.0 points), and LiveMath surges from 51.3% to 72.6%. Conversely, when the proposer already has the wiki, additionally letting the Inference Agent see the wiki during training rollout actually drops the average from 63.7% to 60.9% (LiveMath 72.6% → 64.8%). The authors hypothesize: when the Inference Agent holds both the skills and the wiki, part of the problem-solving knowledge may come directly from the wiki rather than the skills, making the resulting trajectories less informative for skill development. This ablation is the main evidence for the paper's "persistent knowledge is key" claim, but it is done only on a single model (Gemini-3.5-Flash) and four benchmarks (not including ALFWorld).

### Cross-model transfer: skill discovery and skill execution are two abilities

The paper distinguishes the "source model" (the one that evolves the skills) from the "inference model" (the one that executes the skills), and reports that evolved skills can often beat self-evolved skills. For example, the SpreadSheet skill evolved by Qwen-3.6-27B lifts Qwen-3.5-9B from the no-skill 24.3% to 50.5%, also higher than its self-evolved 33.6%; while skills evolved by Qwen-3.5-4B (a smaller model) push Gemma-4-31B to 73.1% on LiveMath and 66.9% on ALFWorld, showing that a stronger source does not necessarily produce better skills. But transfer can also be negative: Qwen-3.5-4B's SpreadSheet skill crashes Gemini-3.5-Flash from 50.5% down to 18.1%, whereas Qwen-3.6-27B's same-task skill raises it to 63.4%. The authors' error analysis points to two causes — small-model skills encode low-level workarounds (such as one-line Python or string-conversion rules) that tie down strong models from writing complete end-to-end scripts; and fragmented diagnostic steps introduce redundant tool calls that exhaust Gemini-3.5-Flash's interaction budget before the task completes. This pulls apart the two abilities that self-evolution usually conflates: "discovering" useful procedural knowledge from experience, and "executing" that knowledge at inference time.

### Cost: optimization calls that are O(1) in training-set size

The authors analyze the per-round optimizer API-call complexity $\mathcal{C}$. WikiSkill uses full batch on all datasets (batch size $B = N_{\text{train}}$, i.e. processing the entire training set at once per round), so each round requires only

$$
\mathcal{C}_{\text{WikiSkill}} = (1 + T_{\text{ReAct}}) \cdot \frac{N_{\text{train}}}{B} = 1 + T_{\text{ReAct}}
$$

optimization calls (1 Wiki Maintainer plus $T_{\text{ReAct}}$ ReAct rounds), independent of the training-set size, hence $\mathcal{O}(1)$ in $N_{\text{train}}$. By contrast, Trace2Skill requires an independent LLM analysis for each training trajectory, with a lower bound of $\mathcal{O}(N_{\text{train}})$; EvoSkill and SkillOpt, under their best minibatch settings, are $\mathcal{O}(N_{\text{train}}/B)$. Note that this counts only "optimizer" calls, not the inference cost of the rollout itself; the paper also admits that this constant complexity may bring higher inference overhead on some datasets.

## 🧪 Critical Assessment

### The problem is real, but the evidence base for "the persistent wiki is key" is narrow

The pain point "insights scattered across the optimization history and hard to reuse" is real and concrete: EvoSkill's flat feedback history, Trace2Skill's trajectory distillation, and SkillOpt's rejected-edit feedback indeed all fail to maintain "what has been learned" as a separate, evolvable knowledge representation. WikiSkill's three-layer split and `skill-impact.md` audit trail are a reasonable and clean engineering answer. But the core ablation (Table 3) that supports "persistent knowledge accumulation is key" is done only on a single model, Gemini-3.5-Flash, and only on four benchmarks (without ALFWorld); generalizing a single-model ablation into a causal conclusion holding for all models is a narrow evidence base. More subtly, that ablation ties "removing the wiki" together with "simultaneously removing the Wiki Maintainer," so the 48.7% → 63.7% gap mixes the two variables of "with or without persistent knowledge" and "with or without an extra analysis agent," rather than cleanly isolating the wiki alone.

### Small validation sets plus a strict-improvement gate are the main variance and overfitting risk

The gate makes "strict improvement" decisions on validation sets of 10–40 items (SealQA val is only 10, LiveMath and ALFWorld each 18). At this scale, a single item right or wrong can flip the accept/rollback decision, effectively letting skill evolution overfit a very small split. One direct symptom: Gemini-3.5-Flash already gets 100% on the ALFWorld validation set "before evolution," and therefore stops early and evolves no skill at all — this is precisely the false saturation caused by a small validation set. The authors do use three independent runs and a bootstrap test to mitigate this, but "three" is itself a very small number of seeds, and many of the bold cells in Table 1 are in fact statistical ties (no significant difference from the best) — for example on Qwen-3.5-9B's LiveMath, EvoSkill 58.1 and WikiSkill 56.3 are both bold — so cell by cell WikiSkill is not always the sole best. In addition, the test sets themselves are not large (85–280 items), and the cross-month-updated LiveMath and the versioned SealQA (July 2026 version) also make benchmark leakage and temporal drift hard to fully rule out.

### "Scale complementarity" holds mainly within the Qwen family; the cross-family scale axis is not clean

The trend that "stronger models benefit more from skills" has its cleanest evidence in Qwen 4B→9B→27B's +12.3 / +17.5 / +23.9. But the paper's Figure 1 also places Gemini-3.5-Flash at the far right of the same "scale" horizontal axis while omitting Gemma-4-31B, and Gemini is a closed-source model with an unknown parameter count — treating it as "larger scale" to support a monotonic trend is not rigorous. In fact Gemma-4-31B's (31B) average gain of +13.6 points is smaller than Qwen-3.6-27B's +23.9, showing that across families "the larger the scale, the more the benefit" is not monotonic — the gain more likely depends on the interaction between model and dataset, rather than on parameter count alone. This is a cherry-picking risk easily glossed over by the main figure's narrative.

### The novelty is at the "knowledge representation" level, not an entirely new mechanism; and the cost comparison is narrow in scope

WikiSkill's loop (rollout → analyze → propose → validation gate) is isomorphic to the three baselines; the real increment is that "persistent, auditable, only-grows-never-rolls-back" wiki layer and the `PURPOSE.md` backlink — this is a valuable representational design, but not an entirely new optimization mechanism, and the claim should avoid being read as a huge methodological leap. On cost, the elegant $\mathcal{O}(1)$ conclusion covers only "the number of optimizer calls," while its Skill Proposer is a 10–20-round ReAct that injects all training results in full batch, so the actual token and wall-clock cost may not be low; the paper reports no hardware, latency, or monetary/compute cost; and the paper's body never mentions any release of official framework code or reproducibility assets, and this note was also unable to verify an official implementation on the arXiv page or in the PDF (Official Code is therefore marked unknown, rather than confirmed "not released"), and with all models being versioned 2026 APIs/weights, reproducibility is accordingly discounted.

### To what extent the problem is solved, and the boundaries not yet touched

Within the scope of "limited rounds, five benchmarks," WikiSkill does demonstrate stable and often substantial gains, and this point is solid. But the paper's larger claims about "persistent, transferable, long-term" knowledge evolution actually exceed its evidence: Table 5 only buckets "accepted skill updates" into Early (Iter 0–1) / Mid (Iter 2–4) / Late (Iter 5–7), reports accepted updates only as late as Iter 5–7, yet never accounts in this table for the total iteration cap $K$ of each experiment (in the algorithm $K$ is always a symbol), so its "long-term" scale is in fact never quantified; the authors themselves also admit in the limitations section that the wiki only grows and never shrinks, that there is currently no automatic pruning mechanism, that over a long run patterns/logs/diffs will grow unboundedly, that the knowledge of rejected skills may still contaminate subsequent context, and that stale or wrong patterns may persist. A fairer reading is therefore: this is a skill-evolution benchmark result that holds up at medium scale and under limited iterations, rather than a proven conclusion about "persistent knowledge compounding over the long term."

## One-Minute Version

- **Experience gets forgotten**: the insights that guide skill improvement are scattered across each round's optimization history, and the model cannot accumulate across rounds and easily repeats old mistakes. For example, the lesson "do not return an object to its origin location" learned in round 2, if kept only as a single feedback message, is very likely to be re-proposed as an already-rejected ineffective approach in round 5.
- **Two memory layers are deliberately asymmetric**: skills that fail to meet validation get rolled back, but the induced knowledge base (wiki) is never rolled back and accumulates across iterations. In the ALFWorld case, a rejected proposal (val score 0.72) stays in the wiki as an audit record, guiding the next round to successfully propose the "never return an item to its origin location" rule (val score 0.78) and get accepted.
- **Skill evolution and model scale are complementary**: the stronger the model paired with evolved skills, the larger the gain; small models can also surpass no-skill large models via evolved skills. Across the Qwen family from 4B to 27B, the average gain widens from +12.3 to +23.9 points, and the skill-equipped 9B (47.4%) beats the no-skill 27B (39.4%).
- **The small validation set is the main risk**: with validation sets of only 10–40 items plus a "strict improvement" threshold, they are easily led astray by single-item noise or even false saturation. Gemini-3.5-Flash, on ALFWorld's validation set of only 18 items, happens to get 100% "before evolution," directly triggering early stopping and evolving no skill at all.
- **Transfer can be negative**: the low-level workaround rules that small models grope out tie down strong models, and their fragmented steps exhaust the interaction budget before completion. Qwen-3.5-4B's evolved SpreadSheet skill crashes Gemini-3.5-Flash from 50.5% down to just 18.1%.

## 🔗 Related notes

- [SkillOpt-Lite](../SkillOpt-Lite/) — another line of agent-skill self-evolution; SkillOpt is precisely one of this paper's baselines.
