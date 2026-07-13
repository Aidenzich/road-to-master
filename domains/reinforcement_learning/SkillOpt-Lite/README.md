# SkillOpt-Lite — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | SkillOpt-Lite: Better and Faster Agent Self-evolution via One Line of Vibe |
| Venue | arXiv (preprint) |
| Year | 2026 |
| Authors | Yifei Shen, Bo Li, Xinjie Zhang |
| Official Code | https://github.com/EvolvingLMMs-Lab/SkillOpt-Lite |
| Venue Kind | tech-report |

> This note is written against the arXiv preprint `2607.03451` (July 2026 version); the formal published version (if any) may differ. All experimental numbers in the text are quoted from that preprint. The paper uses the GPT-5.4 / GPT-5.5 family as the models under test and GitHub Copilot as the optimizer, but records no snapshot / version dates for these models nor the version and configuration of Copilot (see Critical Assessment). All figures in this note were transcoded from the vector PDFs bundled with the e-print.

## Introduction

Skill optimization tries to improve an autonomous agent without changing its frozen base model: it repeatedly edits the Markdown skill that guides planning and tool use. Existing approaches wrap those edits in increasingly elaborate reflection pools, update schedules, and rejection memories. SkillOpt-Lite asks a narrower question: what is the smallest closed-loop pipeline whose components are still justified by theory or by measured necessity?

Its answer is to treat rollout trajectories as ordinary files. A coding agent explores those files with native filesystem tools, extracts failure patterns shared across tasks, makes a minimal skill edit, and keeps the candidate only when it passes an independent validation gate. The paper measures this simplification against full SkillOpt across six benchmarks—SearchQA, Spreadsheet, ALFWorld, LiveMath, OfficeQA, and DocVQA—using benchmark scores and best-validation-so-far curves over ten SkillOpt-Lite batches; it then evaluates the same file-centric loop as HarnessOpt on SpreadsheetBench. The headline numbers are useful orientation, but their statistical and comparison limits are examined separately below.

## First Principles

![Original Figure 1a macro capability radar (averaged across all model scales): SkillOpt-Lite (blue) envelops SkillOpt (red) and the Init skill (grey dashed) along axes such as Spreadsheet 69.7 vs 57.1, ALFWorld 95.8 vs 93.4, LiveMath 63.5 vs 44.3, DocVQA 89.3 vs 87.6; only on SearchQA is blue 73.4 slightly below red 73.5, the two nearly tied. This is a macro view averaged across scales, which smooths away the single-point-without-error-bars and split-change problems flagged below in the Critical Assessment](imgs/fig1a_radar_macro_average.png)

### Problem setup: treat the "skill document" as the parameter to be optimized

An LLM agent's real-world capability comes not only from the underlying model `M`, but also from its execution harness `H` and its domain heuristics (skills `s`, usually a Markdown skill document). Because `M` is frozen at inference time, agent engineering effectively degenerates into "repeatedly rewriting the skill document" — and subtle differences in the skill text cause nonlinear swings in downstream task scores. The paper formalizes this as maximizing an expected reward:

$$f(s) = \mathbb{E}_{z \sim \mathcal{D}}\big[R\big(H(M, z, s)\big)\big]$$

where `s` is the textual skill, `z` is a task instance drawn from distribution `D`, and `R` is the score. Because `S_text` is discrete and `H∘M` is non-differentiable, the gradient `∇_s f` cannot be obtained analytically, so the whole agent–environment interaction is treated as a **Zeroth-Order (ZO) Oracle**: one can only query perturbed scalar scores like `f(s+μu)` and use them to infer "which direction to edit the text in".

### Core observation: existing methods are all reskins of the classic ZO toolbox

The paper's first contribution is a mapping table (original Table 1) that maps recent skill-optimization methods one by one back onto classic zeroth-order optimization operators. The correspondence is excerpted below (this table is a restatement of the paper's original table):

| ZO concept | Operator | Agent counterpart | Literature implementation |
|-|-|-|-|
| 1-Point Estimate | $\hat{\nabla}f(s) \propto f(s+\mu u)\,u$ | Single-trajectory reflection | Reflexion, Voyager |
| Multi-Point / Mini-batch | $\frac{1}{b}\sum_{i=1}^{b}[f(s+\mu u_i)-f(s)]\,u_i$ | Batch-trajectory consensus extraction | Trace2Skill, SkillOpt (batch $B_m{=}8$), SkillForge |
| Central Difference | $\frac{f(s+\mu u)-f(s-\mu u)}{2\mu}$ | Success/failure contrastive analysis | SkillCat |
| ZO Coordinate Descent | $\frac{f(s+\mu e_i)-f(s)}{\mu}\,e_i$ | Atomic edit localizing the wrong step | SkillAdapter |
| Trust Region | $\mathcal{B}(s_k,\Delta_k)$ | Structured edit constraint | SkillOpt (edit budget $L_t{:}4{\to}2$), SoftSkill |
| Control Variate | $\hat{g}_t - c_t + \mathbb{E}[c]$ | Historical-memory rejection buffer | SkillOpt (rejected-edit buffer) |

The paper then points out a key divergence: classic ZO faces a black box, obtaining only scalar scores with invisible intermediate states, so it can only "blindly" use numerical perturbations to approximate the gradient; but every agent rollout emits a **readable full trajectory** — planning rationale, environment state, error messages. Skill optimization is therefore more like "compile-and-debug of a program written in natural language": the LLM is simultaneously the compiler and the execution environment, the trajectory is the intermediate execution record, and it can be used for targeted semantic debugging rather than blind perturbation.

### Two PAC bounds motivate the "consensus" and "independent validation" principles

The paper uses algorithmic stability (Expected On-Average Stability) to give a generalization-error bound:

$$\epsilon(\mathcal{S}) \le \hat{\epsilon}_D(\mathcal{S}) + \mathcal{O}\!\left(\beta_{\exp} + \sqrt{\tfrac{\ln(1/\delta)}{N}}\right)$$

where `β_exp` is the stability coefficient for "how much the result changes when one training sample is removed". If the optimization overfits a single failure trajectory (e.g. hard-coding a trial-specific environment variable or case-by-case branch into the skill), `β_exp` blows up and generalization collapses. Hence **Principle 1 (Consensus Mining)**: the optimizer must act as a compression operator, extracting common failure modes across trajectories rather than memorizing a single one.

The other is a model-selection bound: once a validation set strictly disjoint from the training set is introduced, `β_exp` disappears entirely from the bound, leaving only $\mathcal{O}(\sqrt{\ln(1/\delta)/m})$ (`m` being the number of validation samples):

$$\epsilon(\mathcal{S}_{\text{val}}) \le \hat{\epsilon}_{\text{val}}(\mathcal{S}_{\text{val}}) + \mathcal{O}\!\left(\sqrt{\tfrac{\ln(1/\delta)}{m}}\right)$$

This motivates **Principle 2 (Independent Validation Gating)**: the validation gate must run on independent samples. The paper additionally criticizes that Reflexion has no dynamic validation at all, while SkillCat, SkillAdapter, and Trace2Skill validate on "copies or subsamples of the training failure instances", violating this independence premise.

### Principle 3 and the SkillOpt-Lite pipeline

The third principle comes from a pilot experiment (original Figure 2): the authors save the raw trajectories of GPT-5.4-nano's first batch of rollouts each as a plain-text file, and directly tell GitHub Copilot to browse the directory with native filesystem tools like `ls`/`cat`, find common failures, and edit the skill file directly — **without running any mini-batch, tree-structured merge, or validation loop**. The result: this "single-batch, unvalidated" approach on LiveMath and DocVQA actually beats the full SkillOpt run over 4 full epochs; but on Spreadsheet it drops below the initial baseline instead. The former yields **Principle 3 (the bitter lesson of skill optimization)**: when the model is strong enough, complex topology is worse than "treat everything as a flat file and give the model raw shell tools to read the logs directly"; the latter proves that **the closed-loop validation gate still cannot be dropped**.

![Original Figure 2 pilot: the y-axis is relative accuracy (best per benchmark = 100). The single-batch, unvalidated GitHub Copilot file-reading flow (blue bars) beats the full SkillOpt run over 4 epochs (orange bars) on LiveMath 50.9 vs SkillOpt 30.3 and DocVQA 83.7 vs 81.2; but on SpreadSheet the blue bar collapses to 15.3, below the initial skill (red bar) at 29.9 — this is exactly the visual evidence for the two conclusions of Principle 3, "flat file + raw tools suffice" and "the validation gate still cannot be dropped"](imgs/fig2_pilot_copilot_vs_skillopt.png)

Combining the three principles, SkillOpt-Lite removes mini-batch reflection pooling, the textual learning-rate schedule (slow update damping), and the rejected-edit buffer, keeping only four steps:

```
1. Trajectory Staging   After each batch of rollouts, save every trajectory (plan/environment state/score) as a separate log file
2. Trajectory Exploration The optimizer uses filesystem tools under a limited token budget to ls/cluster/pick high-leverage files (not dumping everything into context)
3. Consensus Mining + Minimal Edit Read files to find cross-task invariants, produce a concise diff/patch under a "minimal edit principle"
4. Validation Gating   Apply patches into candidate library S̃ → score on an independent validation set; accept only if it beats the current baseline,
                       overwrite best_skill.md only if it beats the historical best; rejected updates go into a history log for the record
```

![Original Figure 3(b) SkillOpt-Lite pipeline schematic: the left inputs are the frozen Agent (LLM harness) and the trainable Markdown skill S_t; each rollout trajectory is saved as a file, the optimizer uses bash ls and file-reading tools under a fixed token budget to pick files, find common failures, and directly rewrite the skill, then passes a Validation gate (Accept produces S_t+1, Reject goes into the edit-history log), and the best one is written to best_skill.md; mini-batch pooling / slow-update damping / rejected-edit buffer are already gone](imgs/fig3b_pipeline_lite.png)

The paper packages this as a VS Code Copilot extension that a developer can trigger with a one-line slash command: `/skillopt-loop rounds=10 batchsize=40 target=gpt5.4-nano`, which is what the title calls "one line of vibe".

### How one specific headline number is computed

Take the abstract's flagship "let the small model turn the tables" as an example and walk through SpreadsheetBench (original Table 3, accuracy as a 0–1 ratio). GPT-5.4-nano starts from the initial skill at **0.2989**, first goes to **0.6619** with pure SkillOpt-Lite skill optimization; then extends the same three-pillar "everything is a file" approach to also edit the execution harness (HarnessOpt): HarnessOpt editing only the harness (w.o. skill) jumps to **0.7651**, and optimizing skill and harness together (w. skill) reaches **0.7758**. This 0.7758 is higher than the "large model" GPT-5.5 running full SkillOpt on the base harness at **0.7620** — this is what the paper calls "capability inversion".

The paper's mechanistic explanation for this case is: HarnessOpt targets the "reasoning gets stuck in a repeated loop after a tool failure" pattern common to nano and GPT-5.5, automatically intercepting it and applying a `retry reasoning=low` fallback strategy to get the model unstuck; for GPT-5.4-mini/GPT-5.4 it instead widens the visible range of the spreadsheet preview and adds an output self-check step, reducing parsing and formatting errors. The entire harness rewrite is protected by three loop invariants: it can only edit skeleton scripts (allowlist), must pass compilation and an `N=5` smoke test before validation, and all changes are `git reset`-reversible and wrapped in an environment-variable toggle.

## 🧪 Critical Assessment

### The problem is real, but the "theory" is mostly post-hoc analogy rather than a predictive tool

The nonlinear sensitivity of scores to skills/prompts is a genuinely acknowledged practical problem, and formalizing it as zeroth-order optimization is a nice unifying view. But note: the paper's ZO mapping table and both PAC bounds are **applications of existing standard results**, proving no new theorem for this method nor any predictive bound; the quantities `β_exp`, `μ`, `u` are never actually estimated or measured in the text, and the equations remain at a narrative level. In other words, the theory is responsible for "explaining why cutting those modules is reasonable", not for "predicting how much better it will be after cutting" — the real persuasive weight rests entirely on the experimental numbers.

### The baseline is almost "fighting itself", and lacks variance

The main comparison target, SkillOpt, is prior work from the same author group. Table 1 clearly inventories Trace2Skill, SkillForge, SkillCat, SkillAdapter, etc., yet the formal experiments **do not compare directly against any of them** — only SkillOpt is a single line. More critically: the authors themselves admit LiveMath and OfficeQA "have only 10–20 validation samples and high validation variance", yet report only single-point numbers with no error bars or confidence intervals anywhere in the table. Using GitHub Copilot — an inherently non-deterministic commercial agent — as the optimizer, the jitter between single runs is quite possibly of the same order as part of the claimed gains, making the "small wins" of "+0.1 to +1.5 points" on semantic tasks nearly statistically meaningless.

![Original Figure 4 convergence curves: 8 (task × model-scale) panels track "best validation score so far" (y-axis) over step/batch 1–10. The paper only claims two things: at the "final step" the blue line (SkillOpt-Lite, square markers) equals or exceeds the red line (SkillOpt, circle markers); and on specific panels (e.g. LiveMath-GPT-5.5, LiveMath-GPT-5.4-nano) the blue line rises more steeply in the first 2–3 steps. The figure indeed has a counterexample against "leading stepwise throughout": in the bottom-row SearchQA-GPT-5.4 panel the blue line is below the red line at steps 1–9 and only ties at step 10 — so this figure only supports "final value not inferior, some panels start faster", not stepwise across-the-board superiority. Note that "best so far" is a monotone non-decreasing cumulative statistic; its steps only reflect when the historical best is refreshed and cannot themselves show run-to-run variance; all the figure can read off is the ordering, the final values, and the timing of best-score jumps (here only qualitative ordering and stair shape are taken, without checking the y-axis tick values one by one). The variance concern has a separate source: the authors admit LiveMath/OfficeQA "have only 10–20 validation samples, high variance", while the main table has no error bars or confidence intervals for these final values at all](imgs/fig4_convergence_skillopt_vs_lite.png)

### Changing the evaluation split is equivalent to moving the goalposts

To "mitigate instability", the authors changed the train:val:test of LiveMath and OfficeQA from `2:1:7` to `2:2:6`. To be clear, this new split is **shared** by SkillOpt and SkillOpt-Lite — the paper explicitly states the two methods use the same optimizer settings and are reported together in the same main table, so it is not "the baseline and the target each run a different split". The real problem lies elsewhere: the paper opens by saying the experiments "follow SkillOpt's existing evaluation protocol", yet privately changed the split for these two datasets, which deviates from the old protocol it claims to align with and makes the in-table numbers not directly comparable to the original SkillOpt paper or other existing results. And this is exactly where LiveMath shows its most exaggerated jump: GPT-4o's LiveMath goes from an initial 25.9 to 58.8 (+32.9) under SkillOpt-Lite — a **frozen** GPT-4o gets 30% more questions right just by editing the skill text. A jump of that magnitude looks more like the split change amplifying a test-set signal, or some task-format fix (e.g. answer extraction), than the skill itself getting smarter; the paper does not decompose the source of this gain.

### "Small model beats large model" is a narrative built on unequal conditions

The abstract and §5 headline nano's 0.7758 > GPT-5.5's 0.7620, which sounds like a capability inversion; but this compares "nano + full HarnessOpt" against "GPT-5.5 + a plain harness with only SkillOpt". Once conditions are aligned — both get HarnessOpt — GPT-5.5 is 0.8577, still comfortably above nano's 0.7758. So the conclusion that actually holds should be "the marginal benefit of harness optimization is larger for weaker models", not "small models can replace large models"; the latter is a headline assembled by cherry-picking the comparison target.

### Reproducibility and safety gaps

Although the paper releases code, it under-specifies the experimental environment that is key to reproduction: all experiments run on GPT-5.4-nano/mini, GPT-5.4, GPT-5.5, yet record no snapshot or version date for any model; the optimizer GitHub Copilot is given only by name, with no version, configuration, or call parameters. Because Copilot is itself a continuously updated commercial agent with randomness in single outputs, even with the source code a third party rerunning at a different time can almost never reproduce the same set of numbers — this is a reproducibility gap provable from the paper itself, not a conjecture about model availability. HarnessOpt lets the agent autonomously rewrite its own execution code; although there are guardrails such as an allowlist, smoke tests, and `git reset` rollback, the long-term drift and safety risk of "letting the model edit its own control flow" is only lightly touched on in future work; extrapolating this line to a vision of "automatically editing crawler controllers, collecting data, training better foundation models" has no supporting evidence at all and is speculation.

### One-minute wrap-up

This is an engineering-intuition-heavy, persuasively written tech report; the minimalist claim of "everything is a file + give the model raw tools" stands on its own and does beat the group's own more complex prior work on multiple benchmarks. But its persuasiveness comes mainly from single-point scores rather than reproducible statistical evidence: no error bars, a changed evaluation split, a comparison set limited to the group's own method, and a headline "inversion" assembled from unequal conditions. For the reader, the reasonable way to take it up is to accept the qualitative signal that "complex optimization topology has diminishing marginal returns on strong models", while reserving skepticism about the specific scores (especially LiveMath's huge jump and the capability inversion).

## One-minute version

- **The thing to optimize is not the model, but a skill document.** Because the underlying model is frozen at inference, agent engineering effectively degenerates into repeatedly rewriting a Markdown skill file; example: GPT-5.4-nano goes from 0.2989 to 0.6619 on SpreadsheetBench just by editing the skill text.
- **The method = treat everything as a flat file and give the model native shell tools to read logs directly.** SkillOpt-Lite removes mini-batch pooling, the learning-rate schedule, and the rejected-edit buffer, keeping only the four steps Staging→Exploration→Consensus/Minimal Edit→Validation Gating; example: a developer triggers it with one line `/skillopt-loop rounds=10 batchsize=40 target=gpt5.4-nano`, which is the title's "one line of vibe".
- **The headline is "the small model surpasses the large model" (capability inversion).** The paper flagships that after adding HarnessOpt the weak model scores above the large model's plain version; example: nano's 0.7758 is higher than GPT-5.5's 0.7620 on the base harness.
- **But this "small beats large" is assembled from unequal conditions.** Once both sides get HarnessOpt, the large model still leads comfortably, so what actually holds is only "harness optimization has larger marginal benefit for weaker models"; example: with conditions aligned, GPT-5.5 is 0.8577, still above nano's 0.7758.
- **The numbers themselves must be discounted too: no error bars and a changed evaluation split.** The authors admit the validation set has only 10–20 samples with high variance yet report only single points, and also changed LiveMath/OfficeQA's train:val:test from 2:1:7 to 2:2:6; example: GPT-4o's LiveMath jumps from 25.9 to 58.8 (+32.9), a frozen model getting 30% more questions right just by editing text — a magnitude that looks more like the split change than the skill getting smarter.
- **Practical uptake: accept the qualitative signal, be skeptical of specific scores.** The reasonable use is to trust the direction that "complex optimization topology has diminishing marginal returns on strong models", but because the paper records no model snapshots or Copilot version/configuration and the optimizer is a shifting, randomness-bearing commercial agent, the headline numbers are hard for a third party to reproduce and deserve reserved skepticism; example: be especially doubtful of LiveMath's huge jump and the capability inversion.

## 🔗 Related notes

<!-- No parseable related notes yet. -->
