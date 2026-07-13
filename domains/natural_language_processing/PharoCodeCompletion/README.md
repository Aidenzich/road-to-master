# Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo |
| Venue | unknown |
| Year | 2026 |
| Authors | Kilian Kier, Alessandro Giagnorio, Omar AbedelKader, Oleksandr Zaitsev, Robert Peharz, Romain Robbes, Gabriele Bavota, Stéphane Ducasse |
| Official Code | https://doi.org/10.5281/zenodo.18833238 |
| Venue Kind | paper |

> This note is based on the arXiv preprint `2607.04939` (IEEEtran conference format, with no peer-reviewed publication information visible yet, hence Venue is marked `unknown`). The final published version may differ from this one.

## First Principles

### The problem: when a language has only ~2k GitHub repos

Mainstream code LLMs are trained on the assumption that "there is a massive amount of that language's code on the web." Python has about 26M public repos on GitHub (the authors use GitHub advanced search as of 2026-02-18 as the reference). This paper's subject, **Pharo** (a Smalltalk dialect), has only about 2k public repos — a full four orders of magnitude fewer than Python, and even one more order of magnitude below languages the literature has often called "low-resource," such as Lua (620k), Julia (85k), and Racket (23k). This extreme data scarcity directly shows up as a tooling gap: Pharo's IDE still only offers single-token completion, far behind the multi-token, context-aware completion of mainstream IDEs.

But the authors stress that Pharo is hard not merely because "there is little data," but because of three compounding factors:

1. **The Tonel file format mixes code with metadata.** Pharo developers do not normally operate on files directly; code is serialized into the Tonel format only when saved to git. A single Tonel file writes together "the Smalltalk-dialect-independent class definition" and "the alphabetically ordered methods," and the syntax used for the class definition differs from that in the IDE/documentation. If a model consumes this format directly, it easily learns the wrong pattern of mistaking "packaging metadata" for executable syntax.

```smalltalk
"
Class comment
"
Class {
    #name : 'ClassName',
    #superclass : 'SuperClassName',
    #instVars : [ 'var1', 'var2', ... ],
    #classVars : [ 'default', 'current', ... ],
    #category : 'CategoryName',
    #package : 'PackageName',
    #tag: 'Tag'
}

{ #category : 'MethodCategory' }
ClassName >> methodSelector [
    " Method comment"
    MethodBody
]
```
*The general structure of a Tonel file (redrawn from the paper's Figure 1): the upper `Class { ... }` is declaration metadata, and only the lower part is the actual method.*

2. **Smalltalk's syntax itself is very different from mainstream languages.** Control flow such as `if` and `while` are not syntactic keywords but ordinary methods (messages); method calls are keyword messages, with arguments and the method name interleaved, e.g. `at: aSymbol ifAbsentPut: aBlock`; statements are separated by a period `.` rather than a semicolon. These traits make transfer learning from high-resource languages difficult.

### The end-to-end specialization pipeline

The authors' core claim is: **under extreme data scarcity, "a specialized small model" is more practical than "directly using a giant general model,"** because the goal is in-IDE completion that can run on a developer's machine and meet real-time latency, not offline code generation. The pipeline has three stages.

**(1) Data curation.** Starting from GitHub repos with the `pharo` topic and an MIT license (748 of them), they filter by "whether it can be imported into at least one of Pharo 10–14" and "whether it is in Tonel format," reducing to 415 repos. To reduce data contamination, they use 2024-06-01 as the cutoff: repos created before it are used for training, and those created after are held out for repo-level evaluation. The authors additionally built two tools — a Pygments Pharo lexer and a tree-sitter Pharo grammar — to tokenize and parse out ASTs, extracting a total of **387,159 Pharo methods**.

**(2) Continued pre-training (teaching syntax).** Using causal language modeling. For 25% of the methods, they do full left-to-right prediction; for 75% they use fill-in-the-middle (FIM): a contiguous span is removed and the model completes it from the surrounding context. The masking uses an **AST-aware** strategy — randomly picking an AST node containing 3–10 tokens to mask (too short is meaningless, too long is too hard). When serialized, each model follows its own FIM template: Qwen uses prefix–suffix–middle (PSM), and Mellum uses suffix–prefix–middle (SPM).

**(3) Fine-tuning (aligning with the real completion scenario).** Switching to **Random-AST** masking: a token (or token fragment) is picked at random in the method body as a starting point and masking runs all the way to the end of the enclosing statement node. This simulates "the developer requesting completion at an arbitrary cursor position," rather than only at clean AST boundaries. To avoid forgetting the structural knowledge learned in pre-training, 20% AST-aware samples are mixed in as rehearsal, giving a final fine-tuning dataset of **324,725** entries. Training throughout uses LoRA (alpha=32, r=16, dropout=0.05), sequence length 2,048, learning rate 5×10⁻⁵, AdamW, typically converging in 3 epochs.

The models being specialized are two "small" open-source model families: Qwen2.5 Coder Base (0.5B / 1.5B / 3B / 7B) and Mellum-base (4B). The comparison group, besides their own base versions, includes two "giant" general models: Qwen3 Coder 480B A35B Instruct, and Claude Sonnet 4.5.

### Two kinds of benchmark

The authors built their own evaluation suite in two tiers:

- **Method-level (tests syntax, with executable tests):** the 164 HumanEval+ problems are first machine-translated into Pharo by GPT-4o and then human-proofread, plus 47 problems collected from the Exercism Pharo track, for 211 problems in total. Each problem keeps its canonical solution and tests; a span of the canonical solution is randomly masked, the model completes it, and then **the tests are run** to judge correctness, using `pass@1` (temperature=0.2, 20 repetitions per problem). Each problem is further split into AST-aware and Random-AST (r-AST) masking.
- **Repo-level (tests the real scenario, with similarity):** from 488 developer commits across 22 test repos, the changes are extracted, a span of the AST nodes newly added by the developer is masked (each impacted method masked at most 3 times), for 2,185 tasks total. Since there are no tests to run, **ChrF** and **CrystalBLEU** are used instead to measure similarity to the real code.

FIM task counts per benchmark:

| Tier | Benchmark | # FIM tasks |
|-|-|-:|
| Method-level | HumanEval+ AST-aware | 2,274 |
| Method-level | HumanEval+ r-AST | 990 |
| Method-level | Exercism AST-aware | 1,272 |
| Method-level | Exercism r-AST | 551 |
| Repo-level | Each context strategy (No/Class/Package/Impacted) each | 2,185 |

### A concrete forward example

Take **Qwen2.5 Coder 3B** on HumanEval+ AST-aware as an example: the base checkpoint's `pass@1` is 71.48%, and after pre-training + fine-tuning (SFT) it rises to **83.73% (+12.25)**, with an Odds Ratio = 3.81, meaning the odds of producing a correct completion are about 3× the original.

A more detailed failure-turned-success example is `TripleSumToZero` (HumanEval+ AST-aware, id `humanevalplus-40-10`): the task itself is simple (determine whether a collection has three elements summing to zero), the difficulty being to restore the correct parentheses and Pharo's message precedence. The masked fragment needs to complete as `(aCollection at: i)`. The specialized Qwen2.5 Coder 7B-SFT got it right in all 20 attempts; while all base models break the parenthesis structure (e.g. writing `aCollection at: i)`) or change the evaluation order, causing syntax errors. This echoes the authors' statistics on failure causes: 65.6% of base models' failures are syntax errors, followed by unexpected exceptions (17.9%) and assertion failures (16.5%); two-stage training reduced syntax errors by 33% on average.

### Main results

**Method-level (`pass@1`, excerpt):**

| Model | HE+ AST | HE+ r-AST | Exercism AST | Exercism r-AST |
|-|-:|-:|-:|-:|
| Qwen2.5 Coder 7B (base) | 71.12 | 45.24 | 68.58 | 36.12 |
| **Qwen2.5 Coder 7B - SFT** | **89.04** | **52.76** | **85.84** | **42.95** |
| Qwen2.5 Coder 3B - SFT | 83.73 | 51.83 | 78.11 | 41.81 |
| Qwen3 Coder 480B Instruct | 91.95 | 45.13 | 90.35 | 36.92 |
| Claude 4.5 Sonnet | 95.07 | 51.53 | 91.75 | 41.11 |

Notably the picture is split: on the "cleaner" AST-aware, the giant models (Claude 95.07, Qwen3 480B 91.95) still beat the specialized 7B (89.04); but on the more realistic r-AST the situation reverses — the specialized 3B / 7B not only **surpass** Qwen3 Coder 480B (which has roughly 60–320× more parameters, r-AST only 45.13 / 36.92), but also beat Claude 4.5 Sonnet (51.53 / 41.11): Qwen2.5 Coder 7B-SFT scores 52.76 on HumanEval+ r-AST and 42.95 on Exercism r-AST, with 3B-SFT at 51.83 / 41.81, both higher than Claude in the two r-AST columns.

**Repo-level (Qwen2.5 Coder 7B-SFT, different context):** going from "no context" to "providing the other methods changed within the same commit (impacted methods)," ChrF rises from 60.05% to 75.96% (+15.91), and CrystalBLEU from 35.96% to 58.99% (+23.03). The authors also set up a control: randomly picking the same number of methods as context (random methods) — it beats providing only class/package signatures in 37 of 48 model-metric comparisons (not a clean sweep), but is always worse than impacted methods. The conclusion is that **the relevance of the context matters more than the amount of context.** Claude 4.5 Sonnet is still the best overall (under impacted methods, ChrF 83.02%, CrystalBLEU 70.52%), but the authors note it already has an elevated score with no context, and does not rule out training-data contamination.

**Latency and quantization:** the 7B model uses llama.cpp's Q4_K_M quantization, dropping memory from 14.19 GiB to 4.36 GiB (about −70%), with method-level `pass@1` dropping only 0.61% on average. On latency, the quantized 7B is about 1.33 seconds on an M3/M4 Max CPU and about 0.53 seconds on a consumer-grade GPU RX 7800XT; the unquantized 3B is faster (0.62–0.73 seconds). The 7B's 1.3 seconds is slightly above the "sub-second" threshold common for interactive completion, but is already close to usable.

## 🧪 Critical Assessment

### The problem is real, but "beating a 60× larger model" is a cherry-picked framing

Pharo has only single-token completion and the community genuinely lacks tooling — this pain point is real. Specialization substantially raising the base models is also solid (most cells carry a statistically significant green underline mark). But the "specialized small model beats code LLMs over 60× larger" that the abstract and intro repeatedly stress needs to be put back into context: this "win" **only happens on r-AST and repo-level,** while on AST-aware the giant models still lead; moreover the Qwen3 Coder 480B being beaten is an **Instruct** model, which is not necessarily good at the specific FIM completion format the paper uses, making part of "small beats large" an asymmetry of "dedicated format vs general instruction model" rather than a pure capability gap. Claude 4.5 Sonnet is only firmly first in the AST-aware columns (HumanEval+ 95.07, Exercism 91.75); in the more realistic r-AST columns it (51.53 / 41.11) is actually surpassed by the specialized 7B (52.76 / 42.95) and 3B (51.83 / 41.81). So a more accurate statement would be "specialization can let a small model approach or even locally surpass a general large model on a specific completion format," rather than a sweeping "small beats large," nor a sweeping "the large model wins across the board."

### The double concern of a self-built benchmark and similarity metrics

Method-level judges correctness with executable tests, which deserves credit and avoids the self-serving evaluation of drawing one's own target with a custom similarity metric. But HumanEval+ was first auto-translated by GPT-4o and then human-proofread — translation quality, and the bias introduced by "forcibly converting a generation benchmark into completion," could both make the scores not fully equivalent to native Pharo ability. More critically, **repo-level has no executable tests at all,** using only ChrF/CrystalBLEU to measure literal similarity to the developer's source. The authors themselves admit: high similarity does not mean semantic correctness. Similarity metrics tend to reward "surface-close" completions, so a semantically wrong but lexically close completion may get an inflated score; conversely a semantically correct but differently-written completion is underrated. Therefore the absolute repo-level numbers (such as 75.96% ChrF) should not be read as "completion accuracy."

### Gaps in the comparison design

Two comparisons are omitted. First, the authors only evaluate the continued-pre-trained models and their fine-tuned versions, with **no fine-tuning-only comparison** — they skip it on the grounds that "prior research indicates FIM ability is mainly acquired in pre-training," but this is precisely the paper's own pipeline claim, and without this comparison it is hard to quantify how much the pre-training stage actually contributes. Second, data contamination is only handled for the self-trained small models via 8-gram deduplication and the time cutoff, and "cannot be guaranteed" for the two baselines Claude/Qwen3; Claude already scores high with no context, and the authors admit it may recognize the test repos, so treating Claude as a "ceiling" for comparison carries a contamination risk.

### The novelty is engineering integration, not a new method

Methodologically, continued pre-training + LoRA fine-tuning + FIM masking, and the conclusion "a small specialized model can beat a general large one," have all appeared in the low-resource code literature (MultiPL-T, Giagnorio et al., MonoCoder/MPIrigen, etc.). The paper's real contribution leans toward **engineering integration and domain deployment:** Tonel-aware data curation, Pharo's Pygments lexer and tree-sitter grammar, executable Pharo evaluation tooling, and converging the whole pipeline into a quantized model that runs in real time on a laptop. The authors also honestly write into the discussion that "the engineering investment far exceeds the training itself," which is the paper's positioning — a case study of deploying to a low-resource language, rather than a new algorithm.

### Still one real user study away from "solved"

Even with pretty numbers, the paper has not yet shown it is useful in a real IDE. All evaluations are "mask-then-complete" simulations, with no online study of developers actually accepting/modifying/rejecting suggestions; the latency evaluation is also only done on a few high-spec machines and does not cover low-end hardware. The authors list "integrating into the Pharo IDE and doing a human evaluation" as future work, and say they are developing a plugin. On the current evidence, the reasonable conclusion is: **under offline simulation, specialization has pushed a small model to near-deployable real-time completion quality** — this is a strong feasibility proof, but "whether it can actually improve Pharo developers' productivity" remains unverified.

## One-minute version

- **Extreme low-resource** = the target language has code on GitHub four orders of magnitude less than mainstream languages. Example: Pharo has only about 2k public repos, while Python has about 26M.
- **Specialization pipeline** = using three-stage training to teach a small open-source model a niche language, rather than directly using a giant general model. Example: on 387,159 Pharo methods, first do AST-aware FIM pre-training, then fine-tune with Random-AST masking, all with LoRA.
- **The effect of specialization** = two-stage training can substantially raise a small model's completion accuracy. Example: Qwen2.5 Coder 3B's pass@1 on HumanEval+ AST-aware rises from 71.48% to 83.73%, with an Odds Ratio of 3.81 (about 3× the odds).
- **"Small beats large" depends on which metric and setting** = no single model wins across all metrics: the specialized small model wins only in some columns, and the general large model leads in others. Example: on the two method-level r-AST pass@1 columns, the specialized 3B / 7B both surpass Qwen3 Coder 480B and Claude 4.5 Sonnet; but in the AST-aware columns Claude (95.07) and Qwen3 480B (91.95) beat the specialized 7B (89.04), and the repo-level rows providing impacted methods are led by Claude.
- **Repo-level numbers are not accuracy** = there are no executable tests, only literal similarity is measured, and a semantically wrong but lexically close completion can still score high. Example: the repo-level 75.96% ChrF should not be read as "completion accuracy."
- **Feasibility, not solved** = specialization proves a small model approaches deployable real-time quality under offline simulation, but there is not yet a real IDE user study. Example: the quantized 7B is about 1.33 seconds on an M3/M4 Max CPU, slightly above the "sub-second" threshold common for interactive completion.

## 🔗 Related notes

- [LoRA](../Lora/) — this paper conducts parameter-efficient fine-tuning with LoRA in both the pre-training and fine-tuning stages.
- [Fine-tuning vs In-context Learning vs RAG](../FineTuning-vs-ICL-vs-RAG/) — the Giagnorio et al. study cited by this paper compares the trade-offs of fine-tuning and in-context learning on low-resource code.
