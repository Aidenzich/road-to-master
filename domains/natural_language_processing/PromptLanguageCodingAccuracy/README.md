# PromptLanguageCodingAccuracy — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | How much does prompt language (English vs Chinese) change LLM coding accuracy, by task type and model generation? |
| Venue | unknown |
| Year | 2024–2026 |
| Authors | unknown |
| Official Code | unknown |
| Venue Kind | survey |

This is a cross-paper integrative survey note that answers a practical question: **when writing prompts for a coding agent, how much does using English versus Chinese affect output accuracy? And does the answer differ by "function-level vs repo-level task" and by "model generation"?** Rather than summarizing each paper one by one, we place four main sources on the same set of comparison dimensions and contrast them against each other.

## 📚 Sources

The table below lists the main sources cited in this note and their access status. All numbers have been re-located in the corresponding paper's `source/*.tex` original text and recorded in `ledger.json`; for any value retained in this note, the paper's original text is authoritative, and wherever it conflicts with the issue seed draft, the paper prevails and the correction is annotated in the text.

| # | Title | Venue | Year | arXiv | Access |
|-|-|-|-|-|-|
| 1 | HumanEval-XL: A Multilingual Code Generation Benchmark | LREC-COLING | 2024 | 2402.16694 | fetched |
| 2 | Exploring Multi-Lingual Bias of Large Code Models in Code Generation | preprint (unknown) | 2024 | 2404.19368 | fetched |
| 3 | From Effectiveness to Efficiency: Uncovering Linguistic Bias in LLM-based Code Generation | preprint (unknown) | 2024 | 2406.00602 | fetched |
| 4 | Mythbuster: Chinese Language Is Not More Efficient Than English in Vibe Coding | preprint (unknown) | 2026 | 2604.14210 | preprint-fetched |

Sources 2, 3, and 4 currently exist only as arXiv preprints (no peer-reviewed formal venue has been seen), so they are treated as preprints when cited, and values in the formal versions may differ; among them, source 4 lists the author affiliation as Scam.ai and self-describes as a "preliminary study." The secondary context source attached to the original task, `openai.com/index/introducing-gpt-5-5/` (the GPT-5.5 announcement page), returned HTTP 403 at fetch time and could not be accessed, so it is excluded; consequently this note does not cite any aggregate coding scores from that page. Its original purpose was merely to corroborate that "there is currently no official head-to-head Chinese-vs-English prompt comparison."

## Core Conclusion (bottom line first)

Stacking the four pieces of evidence together, the accuracy loss of Chinese prompts relative to English is **"small and unstable" in both direction and magnitude, and highly dependent on task level and model**; there is no general rule that "English always wins big":

- On **function-level** tasks (single problem, function signature already given), the gap usually falls between 0 and 8 percentage points, and the direction can even reverse: on the low-temperature-sampling average, Chinese is even slightly higher than English.
- On **repo-level** tasks (SWE-bench Lite and the like, which require reading an issue, locating files, and producing a patch), in the currently public data English wins more often, with a gap of about 4.5–9.9 percentage points, but the sample is very small and constitutes only preliminary preprint-level evidence.
- For **current-generation flagship models** (GPT-5.5 / Claude Opus, etc.), there is no public "same batch of tasks, English vs Chinese prompt" comparison experiment at all, so any claim that "you should still use English now" is extrapolation, not measurement.

## Function-Level Tasks: Small Gap, Reversible Direction

HumanEval-XL uses 80 parallel problems across 23 natural languages and 12 programming languages to measure pass@1. It is best suited to test the claim in the issue seed draft that "GPT-4 is actually higher in Chinese on Go (Chinese 67.50% vs English 63.75%)." After actually checking the Go table in the paper's appendix, this claim **does not hold**: on Go, GPT-4 scores 47.50 pass@1 in both English and Chinese (exactly the same), and GPT-3.5 scores 2.50 for both; the 63.75 / 67.50 and 7.50 / 6.25 cited by the seed draft are not in that table.

X-HumanEval-X (source 2) extends the observation to general code generation. Across nine code LLMs (StarCoder, CodeLlama, DeepSeek-Coder, in three sizes each), the authors report that switching instructions from English to Chinese causes pass@1 to "drop by at least 13%." Be careful: this 13% is a **relative decrease**, and it is a deviation computed relative to the lower Chinese baseline, (EN−ZH)/ZH, not absolute percentage points. Looking at the base-models average: Python English 37.32 → Chinese 31.84 (relative Δ17.25%, about 5.5 pp absolute), C++ 34.88 → 30.69 (Δ13.65%, about 4.2 pp), Java 33.47 → 25.74 (Δ30.03%, about 7.7 pp). Converted to absolute percentage points, the gap is about 4.2–7.7 pp.

A third angle comes from linguistic-bias (source 3): 52 parallel Chinese-English Python problems, ten models (eight open-source families plus GPT-3.5-Turbo and GPT-4). It has one key detail that overturns the intuition that "English is always better": **the average correctness at low temperature (t=0.2) is Chinese 0.58, actually slightly higher than English 0.56, and only at high temperature (t=0.8) does it become English 0.61 higher than Chinese 0.59**. Looking at GPT-4 alone, English leans higher at both temperatures (t=0.2 is 0.65 vs 0.63, t=0.8 is 0.71 vs 0.65). The authors also report that on average about 12% of problems exhibit an inconsistency where "one language is right and the other wrong," and another 39% exhibit efficiency (complexity) differences.

| Source / Setting | Metric | English | Chinese | Gap |
|-|-|-|-|-|
| HumanEval-XL · Go · GPT-4 | pass@1 | 47.50 | 47.50 | 0.0 pp (identical) |
| HumanEval-XL · Go · GPT-3.5 | pass@1 | 2.50 | 2.50 | 0.0 pp (identical) |
| X-HumanEval-X · base average · Python | pass@1 | 37.32 | 31.84 | 5.5 pp |
| X-HumanEval-X · base average · Java | pass@1 | 33.47 | 25.74 | 7.7 pp |
| linguistic-bias · overall average · t=0.2 | correctness | 0.56 | 0.58 | −2 pp (Chinese higher) |
| linguistic-bias · GPT-4 · t=0.8 | correctness | 0.71 | 0.65 | 6 pp |

## A Worked-Through Example: Go's "Chinese Advantage" Is Actually a Tie

Take the most eye-catching claim in the issue seed draft and walk through it end to end. The seed draft says: on the Go subset of HumanEval-XL, GPT-4's Chinese 67.50% is higher than English 63.75%, therefore "there is not enough evidence to say English beats Chinese on Go tasks, and Chinese may even be better." Checking the appendix Go table in the `humaneval-xl` cache (`\label{tab:appendix_go}`): both GPT-4's English row and Chinese row are 47.50 in the Go column, and GPT-3.5 is 2.50 for both. In other words, the correct reading is not "Chinese higher by 3.75 pp" but "on this table the two are exactly identical"; moreover, Go is a low-scoring programming language for all models (GPT-4 only ranges 38.75–50.00 across 23 languages), so this "sameness" more likely reflects that the problems themselves are hard with a low ceiling, rather than any Chinese advantage. The direction of the conclusion still holds (there is no evidence that English wins big on Go), but the specific numbers supporting it are corrected to a tie rather than a Chinese lead—which is exactly why the seed draft's numbers cannot be dropped straight into the note.

## Repo-Level Tasks: English Is Currently More Stable, but It Is Only Preliminary Evidence

Mythbuster (source 4) is the only source that touches repo-level real engineering tasks. It samples 50 problems from SWE-bench Lite and compares the English and Chinese prompt resolution rates of three models: MiniMax-2.7 English 66.0% vs Chinese 61.5% (gap 4.5 pp), GPT-5.4-mini 36.0% vs 26.1% (gap 9.9 pp), GLM-5 64.6% vs 55.1% (gap 9.5 pp). All three models are higher in English, with a magnitude of about 4.5–9.9 pp, in the same order of magnitude as the function-level gap.

But this evidence should be discounted: only 50 problems, the authors themselves state that no significance test was done, and the Chinese group has fewer evaluable problems (MiniMax-2.7 has only 39 evaluable problems in Chinese, 22% fewer than the 50 in English, because longer Chinese prompts more easily trigger the token limit and produce empty patches), which the authors explicitly say may cause the Chinese resolution rate to be "overestimated." At the same time, what it measures is MiniMax-2.7 / GPT-5.4-mini / GLM-5, not current-generation flagships like GPT-5.5 or Claude Opus. The authors' own convergent conclusion is actually: **the gap between models (about 30 pp between best and worst) is far larger than the language gap, so choosing the model matters much more than choosing the language.**

| Model · SWE-bench Lite | English resolution rate | Chinese resolution rate | Gap | Note |
|-|-|-|-|-|
| MiniMax-2.7 | 66.0% | 61.5% | 4.5 pp | Only 39/50 problems evaluable in Chinese |
| GPT-5.4-mini | 36.0% | 26.1% | 9.9 pp | No reasoning mode |
| GLM-5 | 64.6% | 55.1% | 9.5 pp | Chinese is actually more token-efficient (0.98×) |

## Why the Difference Exists: Tokenization, Not "Chinese Is More Economical"

Mythbuster's second contribution is debunking the myth that "Chinese saves tokens, so it is more cost-effective." It points out that token cost is **determined by the model (tokenizer) and the direction is inconsistent**: MiniMax-2.7 needs 1.28× the tokens for Chinese, but GLM-5 actually needs only 0.98× for Chinese. Measuring 23 SWE-bench descriptions with five tokenizers, GLM's tokenizer gives a ZH/EN ratio of 0.923 for Chinese (Chinese is more economical), whereas GPT/Llama's cl100k_base uses 15% more tokens for Chinese. What really matters is: even if a given language uses fewer tokens per attempt, as long as its resolution rate is lower it requires retries, so the overall "cost per successful problem" is actually higher—which explains why "looking only at input compression ratio" systematically underestimates the actual cost of Chinese. This tokenizer mechanism is also compatible with X-HumanEval-X's finding that "first translating Chinese instructions to English shrinks the bias": the bias mainly comes from the model's understanding and encoding of non-English input, not from Chinese itself being "harder to program in."

## Practical Recommendation (this note's inference, not the conclusion of any single source)

Returning to the reader's real question, "should I write my coding agent's prompts in English?": the answer the current evidence supports is "**the impact is small and context-dependent**." If you use an English-heavy model, do repo-level tasks, and care about stability, English is the safer default; for small function-level problems, or with models aligned to a CJK vocabulary (such as the GLM series), the loss from Chinese may approach zero or even reverse. No source directly validates the mixed writing style of "use Chinese for the background but keep the API / types / function signatures / test conditions in English," so it can only be treated as a reasonable inference this note draws from the tokenizer and understanding-bias mechanisms: keeping the technical symbols most easily distorted by tokenization and translation in their original English is usually a low-risk practice, but it currently lacks direct experimental support.

## 🧪 Critical Assessment

### Comparability of the sources' experimental settings
The four sources can hardly be added together directly. Their metrics differ (HumanEval-XL and X-HumanEval-X use pass@1, linguistic-bias uses correctness rate, Mythbuster uses resolution rate), their task levels differ (function vs repo), the model generations span from GPT-3.5 to 2026's MiniMax-2.7/GLM-5, and the language sets also differ. Therefore the "0–10 pp" this note gives is a rough generalization placing different measurements on the same order of magnitude—it is a cross-paper inference, not the measured value of any single paper; any comparison treating these numbers as the same ruler should be made with caution.

### Benchmark authenticity and whether it shoots at a self-drawn target
All three function-level sources are built on translated versions of the HumanEval family, whose problems are short and ceiling-limited (especially obvious for a low-scoring language like Go), and whether they extrapolate to real engineering is unclear. Both X-HumanEval-X and linguistic-bias are Chinese-English parallel datasets self-built by the authors, so translation quality is itself a confounding variable—part of the so-called "Chinese is worse" may be translation noise rather than model bias. Although Mythbuster uses the more realistic SWE-bench Lite, it samples only 50 problems with unaligned Chinese-English problem counts, which amounts to measuring on a narrow and imbalanced subset.

### Sample and statistical adequacy
Mythbuster self-reports no significance test; at a scale of 50 problems, a 4.5 pp gap very likely falls within noise; HumanEval-XL has only 80 problems per cell, so the smallest distinguishable granularity of pass@1 is 1.25 pp. All of this makes the "English higher by N pp" claim statistically fragile.

### External validity for the reader's practical question, and the parts still unanswered
The most critical gap is the **generation mismatch**: what the reader cares about is mostly the flagship model in use right now, but the evidence is concentrated on older or smaller, English-heavy models. No source provides a "same batch of tasks, English-Chinese comparison" head-to-head for current-generation models, and Mythbuster explicitly warns against over-extrapolating from three models. Therefore this note can only give "reasonable but unproven" confidence to "English should still be preferred now"; as model tokenizers and multilingual training improve, this conclusion is very likely to keep shrinking or even flip.

## 🔗 Related notes

<!-- No related note links can currently be safely resolved. -->
