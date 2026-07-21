# Ctx2Skill — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | From Context to Skills: Can Language Models Learn from Context Skillfully? |
| Venue | arXiv preprint (2604.27660v3) |
| Year | 2026 |
| Authors | Shuzheng Si, Haozhe Zhao, Yu Lei, Qingyi Wang, Dingwei Chen, Zhitong Wang, Zhenhailong Wang, Kangyang Luo, Zheng Wang, Gang Chen, Fanchao Qi, Minjia Zhang, Maosong Sun (THU / DeepLang AI / UIUC / FDU / CUHK) |
| Official Code | https://github.com/S1s-Z/Ctx2Skill |
| Venue Kind | paper |

> This note is written from the cached arXiv preprint `2604.27660v3` (version date 2026-06-02; v1 first submitted 2026-04-30); the official published version (if any) may differ from this.

## Introduction

The problem this paper tackles is **context learning**: many real-world tasks require a language model (LM) to learn new knowledge on the fly from a long context that lies "beyond its pretraining knowledge," and then use it to reason and solve problems, rather than relying on knowledge already baked into its parameters. The authors cite the definition from CL-bench, and stress that this differs from long-context (which mainly tests retrieval / reading comprehension) and from in-context learning (which learns simple task patterns from demonstrations)—context learning requires the model to genuinely induce the "implicit rules and procedures" hidden in the context. For example, having an LM read an unseen product document and then generate a step-by-step operating procedure or troubleshoot a fault.

An intuitive solution is **inference-time skill augmentation**: distill the rules and procedures in the context into a natural-language **skill** (a short Markdown document), and prepend it to the system prompt at inference time. But the authors point out that this route faces two fundamental difficulties in the context-learning setting: first, **manual skill annotation is prohibitively expensive**—contexts are long and technically dense, so having annotators fully internalize multi-paragraph documents is economically infeasible; second, **there is no external feedback signal**—unlike coding or math, where execution results or gold answers can automatically score outputs, here there is no automatic signal to tell you whether a skill faithfully and completely captures the context knowledge.

![Ctx2Skill concept illustration: facing diverse real-world contexts such as books, manuals, and reports, frontier LMs often cannot solve tasks directly; under "no manual annotation, no external feedback," Ctx2Skill distills the rules and procedures in the context into natural-language skills, prepended to the system prompt at inference time as inference-time skill augmentation.](imgs/concept.png)

The paper proposes **Ctx2Skill**, a self-evolving framework that, under the premise of "no human annotation and no external feedback," automatically discovers, refines, and selects context-specific skills from the context. At its core is a multi-agent self-play loop: the **Challenger** generates tasks and scoring rubrics from the context, the **Reasoner** answers based on the current skill set, a neutral **Judge** gives a binary verdict, and the failed and successful cases are routed to a **Proposer–Generator** pair on each side to diagnose weaknesses and rewrite that side's skill—each side evolves its own skill set rather than updating model parameters. On top of this is a **Cross-Time Replay** mechanism that selects the most generalizable skill set among the candidates from every round, to avoid adversarial collapse.

Success is measured by attaching the final Reasoner skill to an arbitrary LM and measuring the solving rate on CL-bench's four categories of context-learning tasks. CL-bench has 500 contexts, 1,899 tasks, and 31,607 scoring rubrics, with all-or-nothing scoring where a task counts as solved only if all rubrics pass. The main comparison baselines are two skill-construction baselines, *Prompting* (a single direct skill generation) and *AutoSkill4Doc* (window-wise skill extraction then recombination), plus a lineup of frontier LMs that use no skill. The paper's main result is that Ctx2Skill consistently lifts all three backbones: GPT-4.1 from 11.1% to 16.5%, GPT-5.1 from 21.1% to 25.8%, and GPT-5.2 from 18.2% to 21.4%.

## First Principles

### Problem formalization and the role of skills

A context-learning task consists of a context $C$, a set of tasks $\mathcal{T}=\{t_j\}$ whose answers depend on $C$, and a set of binary scoring rubrics $\mathcal{R}_j=\{r_{j,k}\}$ per task. Given an answer $a_j$ produced by LM $\pi$, a task counts as solved only when "all scoring rubrics pass," and the solving metric is defined as:

$$
y_j(\pi;C) = \prod_{k} \mathbb{I}\bigl[r_{j,k}(a_j)=\mathrm{pass}\bigr],
\qquad a_j \sim \pi(\cdot\mid C, t_j).
$$

Ctx2Skill's intervention point is simple: introduce a natural-language **skill set** $\mathcal{S}$ (a short Markdown prepended to the system prompt), turning the answering distribution into a skill-conditioned form $a_j \sim \pi(\cdot\mid \mathcal{S}, C, t_j)$. The key is that throughout, "no parameters are updated"—only this text evolves. During training $\mathcal{S}$ is split into two parts: the Reasoner's $\mathcal{S}^{\mathrm{R}}$ and the Challenger's $\mathcal{S}^{\mathrm{C}}$; at inference deployment only the Reasoner's $\mathcal{S}^{\mathrm{R}}$ is used.

### A self-play loop of five frozen LM roles

![Ctx2Skill overview: (a) the self-play loop where the Challenger poses tasks, the Reasoner answers, and the Judge routes results to update the two sides' skills separately; (b) Cross-Time Replay re-scores historical Reasoner skill candidates on representative tasks and picks the most balanced one for unseen tasks.](imgs/method.png)

The whole loop is built from five "frozen-parameter" LM roles, run for $N$ rounds. In round $i$: the **Challenger** uses context $C$ and its own skill $\mathcal{S}^{\mathrm{C}}_{i-1}$ to generate a batch of $M$ tasks and scoring rubrics, deliberately designed so that "answering correctly requires inducing the rules of $C$, not merely restating surface fragments"; the **Reasoner** answers based on $C$ and $\mathcal{S}^{\mathrm{R}}_{i-1}$; the neutral **Judge** gives a binary verdict rubric-by-rubric and computes the solving metric $y_m$, splitting the whole batch into a failure set $\mathcal{F}_i=\{m:y_m=0\}$ and a success set $\mathcal{P}_i=\{m:y_m=1\}$.

Then each side has a **Proposer** and **Generator** pair: the Proposer is responsible for "diagnosing why it failed/succeeded," synthesizing a batch of cases into a high-level diagnosis (specifying an action add or merge, a target skill name, a description and a rationale); the Generator then "materializes" the diagnosis into a fully replaced skill set, adding/editing only the relevant entries while preserving the rest. The Reasoner side consumes the failure set $\mathcal{F}_i$, diagnoses what context knowledge was missing, and produces an updated $\mathcal{S}^{\mathrm{R}}_{i}$; the Challenger side consumes the "too easily solved" success set $\mathcal{P}_i$ and tightens $\mathcal{S}^{\mathrm{C}}_{i}$ to keep adversarial pressure in the next round. Neither side's prompt ever sees the other's skill set, maintaining strict adversariality. Splitting Proposer and Generator into two agents (separating diagnosis from materialization) is a deliberate design; the ablation shows merging them drops the score slightly but consistently (GPT-4.1 16.5% → 15.9%).

### Cross-Time Replay: adversarial collapse and how to pick a skill

The authors point out an inherent tension in this design, called **adversarial collapse**: as rounds progress, the Challenger generates increasingly extreme tasks that drift away from $C$'s representative knowledge, and because the Reasoner's skill is updated in a "failure-driven" way, it over-specializes to these pathological cases, accumulating redundant skills that hurt generalization. Worse, this degradation cannot be detected inside the loop—each round's Judge only scores the newly generated tasks of that round, and cannot tell you whether knowledge learned earlier has been broken by later edits. So directly returning the last round's $\mathcal{S}^{\mathrm{R}}_{N}$ is unreliable.

The solution is Cross-Time Replay: during self-play, incrementally collect two small probe sets "along the way"—each round add the failure task with the lowest rubric pass rate to the hard probe set $\mathcal{Q}^{\mathrm{h}}$, and add the success task with the fewest scoring rubrics to the easy probe set $\mathcal{Q}^{\mathrm{e}}$. After the loop ends, have the Reasoner re-answer both probe sets using each candidate $\mathcal{S}^{\mathrm{R}}_{i}$, let the Judge re-score, and obtain Laplace-smoothed solving rates:

$$
\rho^h(i) = \frac{\sum_{q \in \mathcal{Q}^{\mathrm{h}}} y_q (\pi^\mathrm{R};C, \mathcal{S}^{\mathrm{R}}_{i}) + 1}{|\mathcal{Q}^{\mathrm{h}}| + 1},
\qquad
\rho^e(i) = \frac{\sum_{q \in \mathcal{Q}^{\mathrm{e}}} y_q (\pi^\mathrm{R};C, \mathcal{S}^{\mathrm{R}}_{i}) + 1}{|\mathcal{Q}^{\mathrm{e}}| + 1}.
$$

The final selected skill set is the round that maximizes the "product" of the two, $\mathcal{S}^{\mathrm{R}}_{\star}=\mathcal{S}^{\mathrm{R}}_{i^\star}$, where $i^{\star} = \arg\max_{i}\left(\rho^\mathrm{h}(i)\cdot\rho^\mathrm{e}(i)\right)$. The product form is the key: a skill that trades away easy-task performance to improve hard tasks will be penalized on $\rho^\mathrm{e}(i)$ and rejected, and vice versa. The authors verify this with an ablation—replacing the product with addition (Additive Scoring) drops 0.6% (16.5% → 15.9%). This $\mathcal{S}^{\mathrm{R}}_{\star}$ is computed only once per context, then reused on all unseen tasks in that context, amortizing the cost over $|\mathcal{T}|$ tasks.

### A concrete numerical walkthrough

Take GPT-4.1 as an example to walk through the magnitudes. The implementation sets $N=5$ rounds and $M=5$ tasks per round. In round 1 the Reasoner solves on average about $0.91/5$ (18.2%) tasks, rising to $1.17/5$ (23.3%) by round 5, but the failure rate stays above 76% throughout—meaning the Challenger's adversarial pressure never lets the Reasoner saturate. The skill file grows roughly linearly with rounds: the median word count of GPT-4.1's skill rises from 311 words at Iter-1 to 1,703 words at Iter-5, adding about one entry per round at roughly 340 words each. But the final skill selected by Cross-Time Replay has a median of only 705 words, falling between Iter-2 (656) and Iter-3 (1,002)—that is, the mechanism mostly picks the "earlier, more compact" rounds rather than the longest Iter-5. However, the selected round is not consistent across the three backbones: GPT-4.1 and GPT-5.2 clearly skew early (final skill median word counts 705 and 1,458, close to their respective Iter-2 values of 656 and 1,338), whereas GPT-5.1 skews later overall, with a selected median word count of 3,682 already approaching its Iter-3 (3,871). So "concentrated in the early rounds" holds only for the first two backbones; for GPT-5.1 the peak instead falls in the middle.

![Distribution of rounds selected by Cross-Time Replay: the number of contexts for which each backbone's final skill is selected from each round. GPT-4.1 (blue dashed line with circle markers) is clearly concentrated at Iter-1, GPT-5.1's (pink squares) peak falls at Iter-3, and GPT-5.2 (orange triangles) skews toward the earlier rounds overall.](imgs/skill_distribution.png)

### Main evidence

The main results (CL-bench, all-or-nothing solving rate, %) show that Ctx2Skill beats both baselines and the no-skill versions across all three backbones:

| Model | Overall | Domain Know. | Rule System | Procedural | Empirical |
|-|-|-|-|-|-|
| GPT-4.1 (no skill) | 11.1 | 10.6 | 14.8 | 10.4 | 4.6 |
| GPT-4.1 + Prompting | 12.3 | 12.4 | 12.3 | 13.9 | 8.2 |
| GPT-4.1 + AutoSkill4Doc | 13.2 | 13.3 | 13.1 | 15.0 | 8.7 |
| GPT-4.1 + Ctx2Skill | 16.5 | 16.8 | 17.6 | 17.6 | 9.7 |
| GPT-5.1 (no skill) | 21.1 | 22.4 | 21.0 | 22.8 | 13.6 |
| GPT-5.1 + Ctx2Skill | 25.8 | 27.9 | 24.9 | 26.9 | 19.1 |
| GPT-5.2 (no skill) | 18.2 | 19.5 | 18.0 | 19.1 | 12.1 |
| GPT-5.2 + Ctx2Skill | 21.4 | 22.2 | 20.4 | 25.4 | 12.6 |

One noteworthy comparison: GPT-4.1 equipped with Ctx2Skill skills (16.5%) actually surpasses the natively stronger but skill-less Gemini 3 Pro (15.8%), which the authors use to argue that context-specific skills can close the model capability gap. The skills are also transferable but asymmetrically so: skills generated by GPT-5.1 give GPT-4.1 16.1% (nearly matching its self-produced 16.5%), but skills generated by GPT-4.1 give GPT-5.1 only 23.1% (+2.0%, far below the +4.6% from self-production)—skills produced by the stronger model transfer well, whereas the weaker model cannot dig out knowledge that the stronger model can use.

## 🧪 Critical Assessment

### Problem realness and importance

The context-learning problem setting is defensible: real scenarios (a physician reading a new clinical guideline, an engineer following a document to execute a procedure) genuinely require the model to absorb an unseen context on the fly rather than apply parametric knowledge, and the CL-bench used by the paper is an external benchmark annotated by domain experts (500 contexts / 1,899 tasks / 31,607 scoring rubrics), not a benchmark defined by the authors, which substantially lowers the "shoot the arrow then paint the target" concern. The truly sharp question is not whether the tasks are real, but whether they are "solved": even with Ctx2Skill, the best GPT-5.1 reaches only 25.8% solving rate—in other words three-quarters of the tasks still fail, and the paper itself admits that context learning remains extremely challenging for current frontier LMs. This looks more like pushing a very hard problem forward by a few percentage points than solving it.

### Are the baselines, ablations, data, and metrics sufficient

The main results compare only two skill-construction baselines (Prompting and AutoSkill4Doc), while Related Work lists a whole lineup of automated skill methods such as AutoSkill, CoEvoSkills, EvoSkill, and SkillX. The authors' rationale is that those methods all rely on external feedback and do not apply to the no-feedback setting, which holds logically, but it also means the question "under the same no-feedback setting, is there a stronger no-feedback baseline" is left unanswered, leaving the comparison narrow. A more critical statistical issue: due to the API budget (total cost about $30K USD) the paper runs only a single trial with $N=M=5$, explicitly stating "we do not perform multiple independent runs to report error bars or confidence intervals." In this 11–26% low-absolute-value band, whether the mere +3.2% gain on GPT-5.2 is stable beyond randomness has no statistical evidence behind it—this is the warning I think most deserves to be retained.

### Circularity of the Judge and skill-quality scoring

The scoring protocol fixes the Judge as GPT-5.1 (following CL-bench's original protocol, claimed to have >90% agreement with humans). But this brings a structural concern: when the backbone itself is GPT-5.1, posing tasks (Challenger), answering (Reasoner), and judging (Judge) are all handled by the same model family, so the internal self-play signal risks having a same-source loop. The final solving rate is measured on the external CL-bench, which partly mitigates the concern; but Table 2's five-dimensional "skill quality" scoring (conciseness/faithfulness/clarity/effectiveness/reusability) uses GPT-4.1 as the judge, while the Ctx2Skill skills being scored are also produced by GPT-4.1, a self-judging setup prone to self-preference bias, so its +3.6 quality lead should be viewed cautiously and can hardly serve as independent evidence.

### The actual contribution of the mechanism: how small is Cross-Time Replay's contribution

Among the two innovations the paper touts, Cross-Time Replay's marginal contribution is actually quite limited. The authors' own data show: on GPT-4.1, fixing the skill from a single round, the solving rate declines gradually across rounds from 15.9% at Iter-1 to 14.7% at Iter-5 (15.9 → 15.6 → 15.6 → 15.2 → 14.7), while full Cross-Time Replay only reaches 16.5%, so the mechanism itself adds just +0.6% over the "best fixed round" (Iter-1's 15.9%). In other words, a much cheaper heuristic of "just use round 1 for everything" would capture most of the benefit, and the net gain from this "collect probe sets + product selection" machinery is thin. What really carries most of the gain is the Challenger's continual evolution (removing it drops GPT-4.1 to 13.8%, the largest single-ablation drop), which looks more like the idea of "continual adversarial task generation" itself doing the work, rather than the replay selection.

### Novelty and real-world relevance

Combining self-play, using scoring rubrics as reward signals, failure-driven skill editing, and a model-selection heuristic has a non-trivial engineering-integration component, and each individual component can find echoes in existing literature; what is genuinely more novel is the combined positioning of "under fully external-feedback-free conditions, using self-produced scoring rubrics as a proxy signal to drive skill evolution." On real-world relevance there is a detail easily obscured by the main results: from the per-sub-category figure one can see that skills are not uniformly beneficial—GPT-4.1 on Instructional Procedures actually drops from 8.8% to 5.3%, Game Mechanics from 11.8% to 10.2%, and Mathematical Formalism stays flat, with the largest gain (workflow orchestration +11.8%) coexisting with these regressions. The claim of "improving the vast majority of sub-categories" is true, but before treating it as a general plug-and-play gain, one must note that it can backfire on certain task types.

![CL-bench per-sub-category solving rate: GPT-4.1 with vs. without skill. Most sub-categories rise (workflow orchestration +11.8% being the largest), but Instructional Procedures, Game Mechanics, and others actually decline.](imgs/subcategory.png)

Additionally, the abstract writes the GPT-5.1 baseline as 21.2%, while the body and the main results table give 21.1%, a small internal inconsistency of the preprint; this note takes the value from the main results table.

## One-minute version

- **Context learning**: the model must induce implicit rules on the fly from a long context that lies "beyond its pretraining knowledge" to solve tasks. Example: having an LM read an unseen product document and then generate a step-by-step operating procedure or troubleshoot a fault.
- **Ctx2Skill framework**: under "no manual annotation and no external feedback," it uses multi-agent adversarial self-play to automatically refine natural-language skills. Example: the Challenger poses tasks from the context, the Reasoner answers based on the skill, and failure cases are routed to the Proposer to diagnose weaknesses and rewrite that side's skill.
- **Core finding**: automatically augmented skills can raise the solving success rate. Example: after adding skills, GPT-5.1 rises from 21.1% to 25.8%, and even lets GPT-4.1 (16.5%) overtake the natively stronger, skill-less Gemini 3 Pro (15.8%).
- **Biggest limitation**: this only pushes a hard problem forward by a few percentage points and does not truly solve it. Example: even with the best GPT-5.1 plus skills, the solving rate is still only 25.8%, meaning three-quarters of the tasks still fail.
- **Statistical and mechanism concerns**: the touted Cross-Time Replay's net gain is minuscule, and the single run lacks statistical error bars. Example: Replay adds only +0.6% over the best fixed round, and in the 11–26% low-score band the paper performs no multiple independent runs and reports no error bars or confidence intervals.

## 🔗 Related notes

- [Reflexion](../Reflexion/) — drives agent self-improvement via natural-language verbal feedback, sharing the same root as this paper's "failure-driven textual skill editing."
- [SELF-REFINE](../SelfRefine/) — single-model self-feedback iterative rewriting, contrast with this paper's multi-role division of labor.
- [Agent-as-a-Judge](../Agent-as-a-Judge/) — using an agent to judge an agent, echoing this paper's circularity concern around the Judge and scoring rubrics.
- [AutoMem](../AutoMem/) — treating memory as a learnable cognitive skill, related to the "context → skill" extraction orientation.
