# MetaHarness — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Meta-Harness: End-to-End Optimization of Model Harnesses |
| Venue | unknown |
| Year | 2026 |
| Authors | Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn |
| Official Code | https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact |
| Venue Kind | paper |

The authors are from Stanford, KRAFTON, and MIT. The paper's LaTeX template is `colm2026_conference`, indicating a submission target of COLM 2026, but at the time of writing there is no formal acceptance record, so the Venue is marked with the literal `unknown`, and the year is taken as 2026 from the arXiv version. This note is written based on the arXiv full text (`2603.28052`); the official conference version may differ.

## First Principles

The question this paper asks is very simple: for a large language model (LLM), beyond its weights, how much of its performance is determined by the surrounding "harness"? The harness the authors define refers to the code that decides "what to store, what to retrieve, and what to present to the model." The observation cited at the paper's outset is that—on the same benchmark, merely swapping the harness can produce a 6x performance gap, showing that the harness is often just as important as the model itself. In the past, harnesses were almost always designed by hand: engineers looked at failure cases, tuned heuristic rules, and iterated repeatedly across a handful of designs. What the authors ask is whether this iteration process itself can be automated.

A natural starting point is existing "text optimization" methods, such as OPRO, TextGrad, GEPA, AlphaEvolve/OpenEvolve, and Feedback Descent, which all use feedback from prior attempts to iteratively improve prompts or code. The authors' core critique is that these methods are unsuitable for harness engineering, because they compress feedback too aggressively—some only look at a scalar score, some only at the current candidate, and some restrict feedback to a short template or an LLM-generated summary. But a harness acts over long horizons: a decision about "when to store, when to retrieve, and how to present" may only affect behavior many reasoning steps later, and compressed feedback often discards the very information needed to trace downstream failures.

Meta-Harness's key design choice is not to compress: it lets the proposer access the "full history" through a **file system**. For every past candidate harness, the file system preserves its source code, evaluation score, and execution trace (prompts, tool calls, model outputs, state updates). Rather than stuffing all of this into a single prompt, the proposer selectively queries it with terminal tools like `grep` and `cat`. The proposer here is itself a coding agent (a language-model system that calls development tools and edits code), rather than a raw LLM running on a fixed prompt—because the volume of experience quickly exceeds the context limit, the proposer must decide for itself "what to look at" and validate its modifications through direct interaction with the codebase.

![Meta-Harness search loop](imgs/method_loop.png)

The search loop has only three steps, as shown above: (1) the proposer reads the file system containing the source code, execution traces, and scores of all prior candidates, and proposes a new harness program; (2) it runs this harness on the evaluation task; (3) it writes all logs from that run (the proposed code, reasoning trace, and evaluation score) into a new directory in the file system, and then the loop repeats. The authors deliberately keep the outer loop minimal: no parent-selection rule, no preset fixed scaffold, no mandated persistent-memory mechanism, and the proposer may inspect any prior harness. This minimalism is intentional—by delegating diagnosis and editing decisions to the proposer, Meta-Harness automatically improves as the coding agent gets stronger.

The formal objective is straightforward. Let $M$ be a fixed language model and $\mathcal{X}$ a task distribution; for a harness $H$ and task instance $x$, execute a rollout trajectory $\tau \sim p_M(H, x)$ and score it with the task reward $r(\tau, x)$. The goal of harness optimization is to find the harness that maximizes the expected final reward:

$$H^{*} = \arg\max_{H}\; \mathbb{E}_{x \sim \mathcal{X},\; \tau \sim p_M(H, x)}\; r(\tau, x)$$

When multiple objectives are of concern simultaneously (e.g., accuracy and context cost), the authors instead rank candidates by Pareto dominance and report the entire frontier. The outer loop of the search can be written as the following minimal pseudocode (adapted from the paper's Algorithm 1, with notation added by us):

```text
Input: task X, language model M, proposer P, number of iterations N
Initialize: population H  (a set of valid seed harnesses)
Initialize: file system D = empty   (stores code / scores / traces)
for H in population:
    E_H = Evaluate(H, M, X)
    D = D ∪ {(H, E_H)}
for t = 1 .. N:
    P queries file system D            # inspect prior harnesses, scores, and traces
    P proposes k new harnesses
    for H in these k candidates:
        if H passes interface validation:
            D = D ∪ {(H, Evaluate(H, M, X))}
return the Pareto frontier of harnesses in D
```

In the paper's implementation, each harness is a single-file Python program, the proposer $P$ is Claude Code paired with `Opus-4.6`, and the base model $M$ under evaluation depends on the domain and is always frozen. A typical search evaluates about 60 harnesses within 20 iterations. To quantify that "the file system is indeed used heavily," the authors logged file reads during the TerminalBench-2 search: the median number of files the proposer read per iteration is 82 (range 69–99), of which about 41% are prior harness source code and 40% are execution traces. In their most demanding setting, a single evaluation can produce up to 10,000,000 tokens of diagnostic information, about three orders of magnitude higher than the largest feedback budget of prior text optimization.

### A walkthrough with real numbers: online text classification

The clearest way to ground the mechanism above in concrete numbers is online text classification. The setup follows ACE's online protocol: the LLM (here `GPT-OSS-120B`) receives labeled examples one at a time, updates its memory, and is then evaluated on a held-out test set. Three datasets are deliberately chosen to be hard and messy: LawBench (law, 215 charge categories), Symptom2Disease (S2D, 22 classes), and USPTO-50k (chemical retrosynthesis, 180 classes). The search uses zero-shot, few-shot, ACE, and MCE as the four seed harnesses in the population, running 20 iterations with 2 candidates per iteration, producing 40 candidate harnesses in total.

The best harness selected is called Meta-Harness (Label-Primed Query). It constructs the prompt in three parts: first a "label primer" listing all valid output labels, then a "coverage" block placing one query-relevant retrieved example for each label, and finally "query-anchored contrastive pairs" that juxtapose highly similar but differently labeled examples. Take LawBench as an example: the model faces a case description and must choose one of 215 charges—the primer makes the full 215-label space visible, the coverage block provides per-category relevant precedents, and the contrastive block sharpens the decision boundary near the query. Ultimately, on LawBench this harness achieves 45.0% accuracy, far above ACE's 29.0%.

| Harness | USPTO | S2D | Law | Avg Acc | Ctx(K) ↓ |
|-|-|-|-|-|-|
| Zero-Shot | 12.0 | 63.2 | 7.0 | 27.4 | 0 |
| Few-Shot (all) | 15.0 | 78.3 | 29.0 | 40.8 | 12.3 |
| MCE | 14.0 | 83.0 | 23.0 | 40.0 | 28.5 |
| ACE | 16.0 | 77.8 | 29.0 | 40.9 | 50.8 |
| Meta-Harness | 14.0 | 86.8 | 45.0 | 48.6 | 11.4 |

![Text-classification search progress: Meta-Harness surpasses the prior best within very few evaluations](imgs/learning_curves.png)

Overall, the selected Meta-Harness has an average accuracy of 48.6%, 7.7 points higher than ACE and 8.6 points higher than MCE; and this improvement is not achieved by piling on more context—it uses only 11.4K context tokens, whereas ACE uses 50.8K and MCE 28.5K, meaning it is more accurate with nearly 4x less context. Notably, it does not lead across the board on every dataset: on USPTO, Meta-Harness scores only 14.0, losing to ACE's 16.0, with the bulk of its gains coming from S2D and Law.

The paper's most convincing ablation dissects what the proposer is allowed to see. Three conditions: scores-only, scores plus an LLM summary (scores + summary), and the full interface (able to read raw execution traces). The gap is large: scores-only median 34.6, best 41.3; scores + summary median 34.9, best 38.7; and the full interface median 50.0, best 56.7—even the full interface's "median candidate" beats the "best candidate" of both ablated versions. This strongly supports the paper's central claim: the truly critical ingredient is access to raw execution traces, and summaries not only fail to recover the missing signal but may even be harmful by compressing away diagnostic detail.

### Two other domains: retrieval-augmented math and agentic coding

On retrieval-augmented olympiad math, the authors give Meta-Harness a corpus of ≥500,000 solved problems (deduplicated and decontaminated), run 40 iterations on a 250-problem search set to produce 109 retrieval harnesses, select a single harness solely by search-set performance on `GPT-OSS-20B`, and then evaluate on 200 new IMO-level problems and on four models unseen during the search. This discovered retrieval harness beats the "no retrieval" baseline on all five held-out models, improving by 4.7 points on average; it runs on the same BM25 lexical retrieval stack as the sparse baseline, without additionally introducing a dense encoder.

On agentic coding, the authors use the 89 long-horizon tasks of TerminalBench-2, seeding with two strong open-source baselines, Terminus 2 and Terminus-KIRA. Here they treat the benchmark as a "discovery problem": both the search and the final evaluation are conducted on the same 89 problems. On `Claude Opus 4.6`, the discovered harness reaches a 76.4% pass rate, surpassing the hand-tuned Terminus-KIRA (74.7%); on the weaker `Claude Haiku 4.5` the improvement is larger, reaching 37.6%, beating the second-best Goose (35.5%) by 2.1 points. The qualitative trace in the appendix shows that the proposer's behavior is worth a look: the first two iterations bundle structural modifications together with prompt-template changes, and both regress relative to the seed baseline; by the 3rd iteration it explicitly hypothesizes that the regression is confounded by a shared prompt intervention, so it separates structural changes from prompt rewriting, and finally pivots to a "purely additive" modification (grabbing an environment snapshot with a single shell command before the first LLM call and appending it to the initial prompt), which becomes the best candidate of that search.

## 🧪 Critical Assessment

### The 6x harness gap makes automated search a real problem

The impact of the harness on LLM-system performance is real and widely observed. The citation that "swapping the harness causes a 6x gap," together with the public record of many teams continually hand-iterating harnesses on TerminalBench-2, both support that "automated harness search" is a problem of practical value rather than a manufactured need. On this point I think it holds up.

### The "average improvement" aggregate masks per-model unevenness

The ablation (scores-only vs summary vs full trace) is well designed and its conclusion is sharp, making it the paper's most solid piece of evidence. But several metrics are presented in a self-favoring way. The math-retrieval "average improvement of 4.7 points" is computed relative to "no retrieval"; compared against the nearest fixed baseline BM25, the overall lead is only 1.3 points, and it does not lead across the board on a per-model basis—on `Gem-3F`, dense retrieval (k=5) at 47.2 exceeds Meta-Harness's 46.3. Concentrating the advantage in the "average" aggregate easily masks per-model unevenness. The USPTO column in text classification likewise lags ACE, indicating that the gains come mainly from specific datasets.

### Exposing the full history via a file system is a system design unlocked by coding-agent capability

From the algorithmic skeleton, Meta-Harness belongs to the same family as AlphaEvolve/OpenEvolve—"program search with an LLM as the mutation operator"; what is genuinely claimed as new is "exposing the full history via a file system" plus "using a strong coding agent as the proposer." This looks more like an engineering configuration unlocked by the leap in coding-agent capability around 2026 than a wholly new search principle—the authors themselves make the "minimalism" of the outer loop a selling point. I therefore lean toward positioning the contribution as "a timely and effective system design with solid empirics," rather than a methodological breakthrough.

### Searching and evaluating on the same 89 problems, plus a selective ranking framing

The TerminalBench-2 experiments perform both search and final evaluation on the same 89 problems, without an independent held-out split. The authors acknowledge this and argue it is the community norm, that with so few problems a split would weaken the signal, and they check for string leakage via manual inspection plus regex auditing. This handling is honest, but it remains in essence "optimizing on the evaluation set," and the extrapolability of its scores should be discounted—this is precisely a case where success criteria are defined by the method's own strengths. Moreover, on Opus 4.6 the authors actually rank only 2nd (ForgeCode leads at 81.8% vs 76.4%), yet in the abstract and highlight boxes they selectively emphasize "rank 1st on Haiku 4.5" and downplay that higher score on the grounds that "ForgeCode cannot be reproduced from public code"—an understandable but reader-annotated framing choice. The qualitative appendix passage also writes Terminus-KIRA's baseline as 64.4%, inconsistent with the main table's 74.7%, with the source unclear and worth doubting.

### The proposition is supported, but the 10^7-token evaluation cost makes it hard for ordinary teams to replicate

The method does produce readable, transferable harnesses across all three domains, and generalizes to OOD datasets and unseen models—a substantive result. But the cost side is glossed over: a single evaluation can reach 10^7 tokens, a search involves about 60 harnesses, the proposer uses Opus-4.6 with "maximum reasoning," and the "a few hours wall-clock" phrasing masks considerable token and API costs, which is a major barrier to reproducibility for most teams. Combined with the fact that the whole paper only validates a single proposer (Claude Code) and rather small samples per domain (89 problems / 200 problems / 3 datasets), I consider the proposition "harness search can be automated" to be supported, but "practical for ordinary teams" still awaits cheaper, cross-proposer validation.

## 🔗 Related notes

<!-- Currently there are no safely resolvable directly related notes under domains/natural_language_processing; the heading is kept and left empty. -->
