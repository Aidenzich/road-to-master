# Large Language Models Are State-of-the-Art Evaluator for Grammatical Error Correction — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Large Language Models Are State-of-the-Art Evaluator for Grammatical Error Correction |
| Venue | Workshop on Innovative Use of NLP for Building Educational Applications (BEA), NAACL 2024 |
| Year | 2024 |
| Authors | Masamune Kobayashi, Masato Mita, Mamoru Komachi |
| Official Code | unknown |
| Venue Kind | paper |

> This note is written based on the LaTeX source of the arXiv preprint `2403.17540v2` (the 2024-05-26 version); the formal version is included in BEA 2024 (ACL Anthology `2024.bea-1.6`, pp. 68–77). The camera-ready version's details may differ slightly from the preprint. The authors did not provide a code repository specific to this paper; the SEEDA dataset used in the text comes from another paper (Kobayashi et al., 2024, *Revisiting Meta-evaluation for GEC*), so the Official Code in this table is recorded as `unknown`.

## First Principles

### What is this paper actually asking

This paper is **not** doing Grammatical Error Correction (GEC); rather, it is doing the thing of "**how to score a GEC system**"—that is, **meta-evaluation**.

To understand why this is a real problem, first look at the two-layer structure of GEC evaluation:

1. **First layer (ordinary evaluation)**: given the learner's source sentence, the system-corrected hypothesis, and the human gold reference, use some metric (such as M², ERRANT, GLEU) to assign the hypothesis a score.
2. **Second layer (meta-evaluation)**: how do we know whether the scores this metric gives are "accurate or not"? The approach is to take the metric's ranking and compute a correlation coefficient against **the human judges' ranking of the same batch of systems**. The higher the correlation, the closer the metric is to human judgment and the more trustworthy it is.

This paper's claim is: using **GPT-4 as that "scoring metric" in the second layer**, its correlation with human judgment is higher than all existing automatic metrics. The title's "state-of-the-art evaluator" refers to exactly this—GPT-4 is the best **evaluator**, not the best **corrector**.

### Two evaluation granularities and eight existing baselines

The paper divides existing metrics into two categories, and this classification is key later for understanding the prompt design:

- **Edit-Based Metrics (EBMs)**: only look at whether "the edits made" are correct. Includes M², ERRANT, GoToScorer, PT-M². For example, M² uses the Levenshtein algorithm to extract edits from the hypothesis, finds the maximum overlap with the gold edits, and then computes an F-score.
- **Sentence-Based Metrics (SBMs)**: look at the quality of the whole corrected sentence. Includes GLEU, Scribendi Score, SOME, IMPARA. For example, SOME fine-tunes BERT with human ratings, separately learning the three dimensions of grammaticality, fluency, and meaning preservation.

Correspondingly, the paper also has the LLM evaluate in two modes, marked with suffixes:

- **-E (edit-based)**: the prompt explicitly marks out the edits and has the LLM evaluate each edit.
- **-S (sentence-based)**: the prompt gives the whole sentence and has the LLM evaluate the sentence quality.

Three LLMs under test: LLaMa 2 (13B chat), GPT-3.5 (`gpt-3.5-turbo-1106`), GPT-4 (`gpt-4-1106-preview`). And on GPT-4, additional "add evaluation criteria" prompt variants are done: on the edit side add Difficulty / Impact; on the sentence side add Grammaticality / Fluency / Meaning Preservation. These variants just add one sentence at the end of the first paragraph of the base prompt, for example the Fluency version adds: "Please evaluate each target with a focus on the fluency of the sentence."

### How meta-evaluation is computed: turn the LLM's scores into rankings, then compute correlation

The evaluation uses the **SEEDA** dataset: for the outputs of 12 neural GEC systems + 3 human corrections, three annotators, under the two granularities of edit-based and sentence-based, each give a 5-point rating to pairwise corrections (A, B), producing a total of **5347 pairwise judgments** (A>B, A=B, A<B). Then, using scoring algorithms such as TrueSkill and Expected Wins, the pairwise judgments are turned into a human system ranking from 1st to 15th. SEEDA splits into two subsets, SEEDA-E (edit granularity) and SEEDA-S (sentence granularity).

The paper does meta-evaluation at two granularities:

- **System-level**: match each system's metric score (on the LLM side, the LLM's pairwise judgments are run through TrueSkill to obtain the system score) against the human system score, and compute **Pearson $r$** and **Spearman $\rho$**.
- **Sentence-level**: directly use SEEDA's pairwise judgments and compute **Accuracy** and **Kendall's $\tau$**.

Kendall's $\tau$ is this paper's main metric, intuitively defined as:

$$\tau = \frac{(\text{concordant pairs}) - (\text{discordant pairs})}{\text{total comparable pairs}}$$

where one "pair" is one human pairwise judgment; if the metric's ordering direction for this pair agrees with the human's it is concordant, if opposite it is discordant. $\tau \in [-1, 1]$, and the closer to 1, the more the metric's fine-grained judgments fit humans. The paper particularly emphasizes the sentence level, because it has a large sample size and can pull apart the differences between high-performance metrics—a point the critical assessment below will return to.

The paper additionally does a "+ Fluent corr." setting: on top of the 12 systems, it adds two "fluency-type corrections" (fluency edits, which pursue overall sentence fluency rather than minimal edits). This setting is specifically used to stress-test whether the metric goes off when facing "non-minimal edits, but more fluent" corrections.

### Reading the main results table

The table below is excerpted from the paper's Table 1 (`tab:meta`). It lists the key rows, with values cited directly from the LaTeX source. Bold marks the best value in that column.

| Metric | Sys SEEDA-E Base $r$ | Sys SEEDA-E Base $\rho$ | Sent SEEDA-E Base Acc | Sent SEEDA-E Base $\tau$ |
|-|-|-|-|-|
| M² (EBM) | 0.791 | 0.764 | 0.582 | 0.328 |
| ERRANT (EBM) | 0.697 | 0.671 | 0.573 | 0.310 |
| GLEU (SBM) | 0.911 | 0.897 | 0.695 | 0.404 |
| SOME (SBM) | 0.901 | 0.951 | 0.747 | 0.512 |
| IMPARA (SBM) | 0.889 | 0.944 | 0.742 | 0.502 |
| GPT-3.5-S | 0.878 | 0.916 | 0.633 | 0.265 |
| GPT-4-S | 0.960 | 0.958 | 0.798 | 0.595 |
| GPT-4-S + Fluency | **0.974** | 0.979 | **0.831** | **0.662** |
| GPT-4-E + Impact | 0.905 | **0.986** | 0.730 | 0.460 |

A few key observations:

1. **Only sentence-level $\tau$ can pull apart the differences**: at the system level, $r$ is crowded around 0.9 for everyone (GPT-4-S 0.960 vs SOME 0.901, a small gap); but at the sentence-level $\tau$, GPT-4-S + Fluency's 0.662 far surpasses SOME's 0.512 and GLEU's 0.404. This is exactly the 0.662 touted in the abstract.
2. **The word Fluency is key**: GPT-4-S base's sentence-level $\tau$ is 0.595, and just changing the prompt's final sentence from evaluating grammaticality to evaluating fluency makes $\tau$ jump to 0.662. The gap caused by changing one word hints that prompt engineering has a significant impact on evaluation performance.
3. **The smaller the scale, the worse**: GPT-3.5-S's $\tau$ is only 0.265, lower than the traditional GLEU's (0.404); LLaMa 2-S's $\tau$ drops even further to 0.042, almost uncorrelated with humans.

### A concrete walkthrough: how GPT-4-S + Fluency arrives at 0.662

Walk through the forward flow with the real prompt example from the paper's Appendix A to see how the numbers connect:

**Input**: three sentences of context from an English learner's essay, with the middle sentence being the source sentence to be evaluated. For it, 5 candidate correction targets are given (excerpted from the paper):

```
# targets
1. In conclude , socia media benefits people in several ways but in the same time harms people .
2. In conclusion , social media benefits people in several ways but at the same time harms people .
3. In conclusion , social media benefits people in several ways but , at the same time , harms people .
4. In conclude , social media benefits people in several ways but at the same time harms people .
5. In conclusion , socia media benefits people in several ways but , at the same time , harms people .
```

The prompt instructs the LLM: "assign a score from a minimum of 1 point to a maximum of 5 points to each target based on the quality of the sentence," and appends the fluency sentence, with the output being scores in JSON format. Target 1 retains the three errors `socia`, `in conclude`, `in the same time`, and should get a low score; target 3 corrects everything and breaks sentences naturally, and should get a high score.

**From per-sentence scores to system ranking**: feed GPT-4-S + Fluency's judgments on all pairwise corrections (A>B/A=B/A<B) into TrueSkill to obtain the system-level ranking (the sub-table corresponding to the paper's Table 5, with values cited from the source):

| # | Score | System |
|-|-|-|
| 1 | 0.721 | GPT-3.5 |
| 2 | 0.648 | REF-F |
| 3 | 0.230 | TransGEC |
| … | … | … |
| 5 | -0.308 | BART |
| 6 | -1.002 | INPUT |

The key is the top two: GPT-3.5 and REF-F are both **fluency-type corrections** (fluency edits). GPT-4-S + Fluency ranks them 1st and 2nd, which is exactly what the human ranking looks like; by contrast, GPT-3.5-S and LLaMa 2-S cannot rank these two fluency corrections high (Appendix B points out that small models tend to give many systems similar scores and cannot discriminate).

**From system ranking to correlation coefficient**: match this set of LLM rankings against the human ranking, and at the sentence level compute Kendall's $\tau$ using pairwise judgments, obtaining SEEDA-E Base's **0.662**—that is, the abstract's headline number. The whole chain is: *prompt (including the fluency instruction) → LLM scores each pair of corrections → TrueSkill aggregates into a system ranking → compute $\tau$ against the human ranking*.

### Window analysis: only compare systems of similar level

The paper worries that "everyone's system-level correlation is >0.9" is because the task is too easy (a mix of systems with widely differing strength, where any ordering is roughly correct). So it does a window analysis: in the human ranking, take only the subset of systems in **four consecutive ranks**, compute their Pearson $r$, and slide the window from 4th to 12th place. x=4 is the $r$ computed using only the 1st–4th ranked systems.

![Radar chart of the window analysis on SEEDA-E: the GPT-4 series (including +Fluency) maintains high and stable correlation at most window positions, while traditional metrics often show no correlation or negative correlation](imgs/radar_seedaE.png)

![Radar chart of the window analysis on SEEDA-S: overall correlation is high, but it drops significantly around x=10, hinting at the existence of hard-to-evaluate GEC systems](imgs/radar_seedaS.png)

On SEEDA-E, GPT-4-S + Fluency maintains high correlation at almost all windows, highlighting the importance of fluency; traditional metrics often drop to no correlation or even negative correlation, showing they are poorly robust when "comparing modern systems of similar level." On SEEDA-S, although the overall correlation is high, it clearly drops around x=10, hinting that some systems are hard to evaluate for all metrics.

## 🧪 Critical Assessment

### Whether the problem is real: the "resolution" crisis of meta-evaluation

This paper's problem setting is solid and inherits a real pain point: existing GEC metrics **cannot tell apart** high-performance systems. The authors cite Kobayashi et al. (2024) to point out that traditional metrics lack resolution, and this paper's own system-level data corroborates this point—the GPT-4 series' system-level correlation is >0.9 for most systems, and the authors honestly interpret this as "meta-evaluation over a dozen-odd systems has reached saturation," and warn that this will **underestimate** future high-performance metrics. This kind of self-negation of "even my own method has saturated the task" is more valuable than simply reporting SOTA, and makes the emphasis on the sentence-level $\tau$ appear reasonable rather than after-the-fact packaging.

### Sufficiency of baselines, data, and metrics

The baselines are quite complete: four EBMs + four SBMs, covering the F-score family, the n-gram family, and the BERT fine-tuning family (SOME/IMPARA), which is the comparison set this sub-field should have. The metrics report both Pearson/Spearman (system level) and Accuracy/Kendall (sentence level), without cherry-picking only what is favorable to itself.

But there are three obvious limitations worth flagging:

1. **Only one dataset, one language**: all conclusions are built on SEEDA (English, W&I+LOCNESS domain). The general assertion "GPT-4 is a SOTA evaluator" is entirely untested across languages and domains (such as native-speaker writing, formal documents)—this is the paper's biggest external validity gap.
2. **The number of systems is small enough to make the correlation coefficient unstable**: at the system level there are only 12 (or +2) systems. Computing Pearson $r$ on n=12, a single outlier system can swing the value substantially. Under the "+ Fluent corr." setting, traditional metrics' $r$ even flips to negative (e.g., M² drops from 0.791 to -0.239), and this dramatic flip itself shows n is too small and the conclusion is fragile.
3. **The LLM version is not reproducible**: `gpt-4-1106-preview` is a closed snapshot that will be deprecated/updated, and the authors themselves admit in the Limitations that results may be inconsistent across versions. The so-called "state-of-the-art" being tied to an API version that cannot be reproduced long-term is scientifically discounted.

### The fluency gain: is it a finding, or a pandering to its own metric?

The main thread most in need of scrutiny is "adding fluency makes it SOTA." $\tau$ going from 0.595 (base) to 0.662 (+Fluency) is indeed a genuine gain, and the window analysis corroborates it. But there is a circular-reasoning concern: **SEEDA's human annotation itself may already prefer fluency-type corrections** (the paper's Appendix B also says humans rank fluency corrections like REF-F and GPT-3.5 high). If the human gold standard already weights fluency, then "telling GPT-4 to also weight fluency makes it closer to humans" is almost a tautology—when you tune the evaluator's preference to be the same as the judge's, of course correlation rises. This looks more like **aligning with the preference of this specific annotation set** than discovering a general truth of GEC evaluation. The authors interpret it as "humans also prioritize fluency when comparing high-quality corrections," and this causal reading is possible, but the paper does not do an experiment that could distinguish the two explanations (for example, re-testing on an annotation set that deliberately lowers the fluency weight).

### Novelty: the method is simple, and the contribution lies in the rigorous comparison

In terms of method, this paper's novelty is limited—there is no new model, no new algorithm, and the core is "put GPT-4 with different prompts and use it as an evaluator, then measure correlation coefficients." The real contribution is not in the how, but in the what: it is (as the authors claim) the first work to do a **comprehensive** meta-evaluation of LLM evaluators using dozens of systems, whereas past work by Sottana et al. (2023) used only a few systems. This kind of contribution of "rigorously measuring existing tools clearly" is legitimate, but readers should position it as a solid **empirical and diagnostic** paper rather than a methodological innovation. In particular, the conclusion that "prompts with added criteria are better" is essentially a re-confirmation on GEC of an existing prompt-engineering observation.

### Is it really solved, and its real-world relevance

From the angle of "whether it can provide an automatic evaluator closer to humans than traditional metrics," the answer on SEEDA is affirmative. But the two cracks the paper itself reveals require discounting the claim that "the problem is solved": first, the system level is already saturated, meaning the **difficulty of this evaluation task needs to be redesigned** (the authors' solution is to turn to the sentence level and window analysis, but this is still within the same dataset); second, the deployment cost is high—using GPT-4 as an everyday evaluator requires paying for every comparison and is not reproducible, and the authors clearly acknowledge in the Limitations that this limits applicability, and supplement with LLaMa 2 as a "reproducible alternative," yet LLaMa 2's $\tau$ is only 0.042 and is essentially unusable in practice. In other words: the usable one (GPT-4) is not reproducible, and the reproducible one (LLaMa 2) is not usable. This tension has not yet been resolved and is the real hole that follow-up work should fill.

## 🔗 Related notes

<!-- No safely resolvable related notes yet. -->
