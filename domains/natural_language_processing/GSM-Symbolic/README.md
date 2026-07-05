# GSM-Symbolic — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models |
| Venue | ICLR |
| Year | 2025 |
| Authors | Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar |
| Official Code | https://github.com/apple/ml-gsm-symbolic |
| Venue Kind | paper |

> This note is based on the arXiv preprint `2410.05229` (LaTeX source, including the `\iclrfinalcopy` marker), and was cross-confirmed via OpenReview (forum `AjXkRZIvjB`) as the officially accepted ICLR 2025 version. The citation count is marked `unavailable` because the Semantic Scholar API returned a 429 at the time of writing.

## First Principles

### The question this paper is really asking

GSM8K is one of the most popular benchmarks for evaluating an LLM's "grade-school math reasoning" ability, containing 7,473 training problems and 1,319 test problems. Its problem is that it provides only a **single score**, its problem set is **fixed and static**, and because it is so popular, its test problems have very likely leaked into the training data of various models (data contamination). When a model scores 95% on GSM8K, we cannot tell whether this is "able to reason" or "has memorized similar problems."

The authors' core claim is: **rather than viewing an LLM's math ability as a point estimate (single-point accuracy), we should view it as a distribution.** If a model is really doing formal reasoning, then swapping out the names and numbers of the same problem while keeping the reasoning steps completely unchanged should barely move accuracy; conversely, if accuracy shows clear variance, it suggests the model is doing something more like pattern-matching that maps a problem to "similar templates it has seen" in the training data.

### The data generation engine: symbolic templates

GSM-Symbolic's approach is not to collect more problems, but to rewrite **100 problems from the GSM8K test set** into parseable symbolic templates. Each template defines three things: the variables, the domain of each variable, and the conditions that guarantee both the problem and the answer hold. Taking the building-blocks problem from the paper's Figure 1 as an example, the template looks like this:

```text
When {name} watches her {family}, she gets out a variety of toys ...
The bag of building blocks has {x} blocks in it.
The bin of stuffed animals has {y} stuffed animals inside.
The tower of stacking rings has {z} multicolored rings on it.
{name} ... bringing her total number of toys ... up to {total}.
How many bouncy balls came in the tube?

#variables:
- name  = sample(names)
- family= sample(["nephew", "cousin", "brother"])
- x     = range(5, 100)
- y     = range(5, 100)
- z     = range(5, 100)
- total = range(100, 500)
- ans   = range(85, 200)

#conditions:
- x + y + z + ans == total
```

`ans` is the answer (the number of bouncy balls to buy). The condition `x + y + z + ans == total` guarantees that every randomly sampled instance is self-consistent. The authors deliberately keep the numeric ranges **close to those of the original GSM8K**, and the reason is crucial: what they want to test is **logical reasoning**, not **large-number arithmetic**. Ablation experiments in the appendix confirm that within these ranges the models' arithmetic accuracy remains stable, so the drop in accuracy cannot be attributed to "the numbers got bigger and became uncomputable."

Template quality is guarded by multiple layers of checking: automatic checks (the original variable values must not appear in the template, the original values must satisfy all conditions, and the final answer must match the original problem), a manual review of 10 random samples per template, and — after all models have been evaluated — a requirement that **at least two models answer each problem correctly**, otherwise that problem is sent for further manual review.

### A concrete forward example: from template to one evaluation

Substituting the original GSM8K values back into the template above gives the problem on the left of Figure 1. Let the number of bouncy balls be $T$; the problem gives $31+8+9+T = 62$, so

$$T = 62 - (31 + 8 + 9) = 62 - 48 = 14.$$

This computation requires only two steps: "add up the known quantities, then subtract from the total." What GSM-Symbolic does is replace the constants $31,8,9,62$ with `range(...)` sampling and replace the names with `sample(names)`, leaving the rest of the narrative and solution steps unchanged, and then **generate 50 instances from the same template**. The whole study uses 100 templates × 50 samples per template = 5,000 problems, equivalent to **50 datasets of 100 problems each**, running nearly 500 evaluations across 25 models (over 20 open-source models of 2B–27B, plus GPT-4o-mini, GPT-4o, o1-mini, o1-preview), all using 8-shot CoT and greedy decoding.

### Four findings, from shallow to deep

**(1) Across different instances of the same problem, accuracy is a distribution with variance.** Across the 50 datasets, every model's accuracy has non-negligible variance: for Gemma2-9B the gap between best and worst exceeds 12 percentage points, and for Phi-3.5-mini it is about 15 percentage points. More intriguingly, a model's accuracy on "the 100 original GSM8K problems that were used as templates" tends to fall on the **right side** of the GSM-Symbolic distribution (21 out of 25 models are like this); statistically such a shift is very unlikely, and the authors read it as a signal of **data contamination** — the original test problems may already have entered the training set, producing an optimistic bias.

![Accuracy drop from GSM8K → GSM-Symbolic (by model)](imgs/symbolic-drop.png)

The figure above shows each model's change in accuracy going from GSM8K to GSM-Symbolic. You can see that the older models most suspected of contamination (such as Mistral-7B-it-v0.1 dropping 9.2, Gemma2-2b dropping 7.4) drop the most, while o1-mini and GPT-4o barely move (dropping only 0.x).

**(2) Robust to names, sensitive to numbers.** Decomposing the effect of "what is changed" reveals: when only **proper nouns** are changed (people's names, places, foods), the distribution still stays close to the center of the original GSM8K; once **numeric values** are changed, the distribution shifts clearly to the left with larger variance; changing both is worse still. The authors' comment is sharp: even swapping only names causes accuracy to jitter, a phenomenon that should not occur for "a grade-schooler who truly understands math."

**(3) As difficulty rises, accuracy collapses faster than linearly.** Using GSM-Symbolic as the baseline, removing one clause gives M1, and adding one or two clauses gives P1 and P2. All models' distributions consistently shift left and grow more variable as difficulty rises, and **the rate of decline accelerates with difficulty**. Take Mathstral-7B as an example:

| Difficulty | M1 | Symbolic | P1 | P2 | NoOp |
|-|-|-|-|-|-|
| Accuracy (%) | 82.9 | 74.0 | 57.4 | 35.5 | 20.4 |

The number of reasoning steps increases roughly linearly with the number of clauses, but accuracy drops faster than linearly — this is consistent with the hypothesis that "the model is doing pattern-matching, and search difficulty explodes with problem complexity," rather than executing stable formal reasoning.

**(4) GSM-NoOp: one irrelevant but seemingly-relevant sentence can make the model collapse.** This is the most devastating experiment in the paper. The authors add to the problem a clause that is **semantically seemingly relevant but has no effect on the computation** (No-Op), for example the classic kiwi problem:

> On Friday Oliver picks 44 kiwis, on Saturday 58, and on Sunday twice as many as Friday ($2\times 44 = 88$), **but five of them are a bit smaller than average**. How many kiwis does Oliver have in total?

The correct approach should ignore the useless sentence "five are smaller," giving the answer $44+58+88 = 190$. But both o1-mini and Llama3-8B **blindly turn "five are smaller" into a subtraction**, computing $88-5=83$ for a total of $185$. The models tend to mechanically turn every noun clause into an operation, even uniformly reading "discount" as "multiplication," showing that they do not truly understand the concepts and are merely applying a "this sentence pattern → this operation" mapping from the training data.

![Accuracy drop from GSM8K → GSM-NoOp (by model)](imgs/noop-drop.png)

The result is a catastrophic drop: Phi-3-mini drops more than 65% (the "up to 65%" in this paper's abstract comes from here), and even o1-preview drops 17.5 percentage points. More critically, this drop **cannot be recovered with few-shot**: even stuffing the prompt with 8 examples of "the same problem, already demonstrating that irrelevant information should be ignored" (NoOp-Symb), or 8 examples of "different problems that all contain a No-Op" (NoOp-NoOp), most of the accuracies still stay within a standard deviation of the original level. On this basis the authors argue that this is not a surface problem that can be patched with in-context examples or fine-tuning, but a defect in the reasoning mechanism itself.

Combining the four points, the authors' conclusion is: an LLM's mathematical "reasoning" is more like sophisticated probabilistic pattern-matching than genuine formal logical reasoning.

## 🧪 Critical Assessment

### The problem is real, and it hits a sore point of evaluation methodology

"GSM8K's single-point accuracy is unreliable and carries contamination risk" is a genuine problem recognized by the community, and this paper turns it from a slogan into an operational measurement: using symbolic templates to expand one problem into a distribution, and using "the original problem falls on the right side of the distribution" as a statistical fingerprint of contamination — this design is clean and persuasive. Locking the numeric ranges to those of the original GSM8K, and using appendix ablations to rule out the obvious alternative explanation of "arithmetic overload," is responsible experimental control. The NoOp operationalization is even more elegant: it turns "does the model really understand the problem" into a reproducible, quantifiable probe.

### The adequacy of benchmarks, ablations, and metrics — and a few places that deserve a discount

The template scale is actually not large: only **100 templates**, all drawn from the GSM8K test set, so the "diversity" of the whole study is longitudinal (50 instances of the same problem) rather than horizontal (the problem types are still confined to grade-school four-operation narrative problems). This means the conclusions, strictly speaking, are about the robustness of "problems of the GSM8K kind," and extrapolating to broader mathematical reasoning requires caution. The quality-control rule "at least two models must answer each problem correctly, otherwise manual review" also carries a bit of circularity: it uses the consensus of the model pool to define problem validity, which may quietly cull some hard problems that are actually reasonable but that most models get wrong, thereby biasing the retained problem set slightly toward a model-friendly distribution. Whether the NoOp clause "five are smaller" is **really** irrelevant to the answer is to some extent a human judgment call; for human readers, in natural contexts such sentences often carry the ambiguity of "am I supposed to subtract this?", so attributing the model's failure entirely to "not understanding the concept" may overestimate the effect and underestimate the pragmatic ambiguity of the problem itself.

### Is this a new method, or an existing idea repackaged?

Symbolic templates and perturbation-based evaluation are not the first of their kind in this paper: GSM-IC long ago tested distraction with "irrelevant context," GSM-Plus created GSM8K variants, GSM1K used a control style to find systematic overfitting, and there are also functional variants of MATH. The paper itself honestly lists these prior works. The real increment lies in three points: making evaluation explicitly **distributional**, using **symbolic templates** to obtain controllable difficulty, and the NoOp probe that is especially sharp; combined, these are indeed more systematic than any single prior work. But two "drawing the target next to your own arrow" risks should be noted. First, the benchmark was designed by the authors around "exposing LLM weaknesses," and the NoOp clauses are deliberately written to be misleading, so "the model will collapse" is to some extent a designed-in expected result rather than a natural finding under neutral sampling. Second, the abstract's headline **"up to 65%" is a worst-case value** (the small model Phi-3-mini), while the strongest model on the same figure, o1-preview, drops only 17.5% — using the maximum drop as the headline makes the overall impression more pessimistic than "the real risk for frontier models."

### Has the claimed problem really been "solved"? And how relevant is it to the real world?

Two things need to be distinguished: this paper **diagnoses** very successfully, but it does not — and does not claim to — **solve** "making LLMs reason robustly." As a diagnosis, its strongest counterexample is actually hidden in its own data: the more capable the model, the smaller the degradation (the o1 series is far more stable than small models on both difficulty and NoOp), which makes the universal conclusion "LLMs lack formal reasoning" stand on somewhat shaky ground — the data equally supports a weaker and more likely reading: **the reasoning robustness of most current models improves with scale and training quality, rather than being fundamentally impossible.** In fact, directly equating behavioral degradation with "no reasoning inside" is an attribution leap: swapping numbers / adding clauses also changes the surface distribution of the problem statement, so degradation may come from prompt-format sensitivity or insufficient out-of-distribution generalization, and not necessarily "no reasoning at all." As for real-world relevance, the paper's greatest contribution is providing a **dynamic evaluation protocol that automatically becomes harder as models get stronger**, and this value is evergreen; but the statement "all SOTA models are fragile" has a shelf life — subsequent tracking (including frontier models before 2026) has shown that this problem is gradually being conquered, so this paper should be read as "a robustness lower bound at one time-slice plus a measurement tool," rather than a permanent verdict on LLM reasoning ability.

Overall, this is a paper with solid diagnosis and high tool value, but that slightly oversteps in causal attribution and headline narrative: it proves the unreliability of GSM8K's single-point accuracy and the models' sensitivity to irrelevant information, yet over-generalizes "behavioral fragility" into "lacking formal reasoning."

## 🔗 Related notes

<!-- There are currently no directly related notes that can be safely resolved under the NLP domain; the heading is kept as a placeholder to be filled in later. -->
