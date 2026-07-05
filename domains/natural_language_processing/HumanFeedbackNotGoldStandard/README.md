# Human Feedback is not Gold Standard — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Human Feedback is not Gold Standard |
| Venue | ICLR 2024 |
| Year | 2024 |
| Authors | Tom Hosking, Phil Blunsom, Max Bartolo |
| Official Code | https://github.com/cohere-ai/human-feedback-paper |
| Venue Kind | paper |

## First Principles

### 1. That single score treated as the "gold standard"

Open-ended generation tasks for large language models (LLMs) are hard to evaluate with automatic metrics, so "having people assign one overall score" has become the standard practice in effect: human evaluation using a single overall score has become the *de facto* standard, and this score is used not only for evaluation but also as the training objective for RLHF. This paper's core question is: what properties of the output does this `preference` score, compressed into a single scalar, actually capture? Can it really serve as a reliable "gold standard"?

The authors' starting point is a very practical observation: annotators cut corners. When facing a heavy judgment task, people look for shortcuts to make the task easier, and therefore tend to score based on easy-to-check surface properties (fluency, linguistic complexity) rather than properties that take effort to verify (such as factual correctness). If preference scores are systematically dominated by surface properties, then training models with them is tantamount to rewarding "looks good" rather than "is actually useful."

### 2. Part 1: The error coverage of a single preference score

The paper first defines a set of task-agnostic yet concrete-enough-to-be-annotated minimal requirements as "error types." This set of criteria synthesizes the critical evaluation of Xu et al. (2023), Grice's cooperative principle (the Maxim of Quantity corresponds to repetition, the Maxim of Quality corresponds to factuality), and the aspects users care about in production environments, ultimately yielding ten error types: Harmful, Fluency, Scope, Repetition, Refusal, Formatting, Relevance, Factuality, Inconsistency (i.e., faithfulness), Contradiction.

In the experimental setup, the authors construct inputs from three datasets (Curation Corpus summaries, Amazon product descriptions, WikiHow tutorials) and collect responses from MPT 30B Instruct, Falcon 40B Instruct, Command 6B/52B, and reference outputs. Annotations come from native English speakers on Prolific, with an interface based on Potato, and a protocol following the findings of RankME: giving only two outputs at a time and collecting absolute ratings better improves annotator agreement. Crucially, they use **two different groups of annotators**: one group labels each error type one by one (binary yes/no), and another group completely independently gives an overall quality score of 1–5 by its own standards. Part 1 annotates 900 distinct outputs, for a total of 4,440 annotations including quality checks.

Quality control uses two mechanisms. The first is a distractor: pairing an output with an output from "the same model but a different input" as an attention check — the result is that over 97% of distractors are correctly rated as worse, indicating that the vast majority of annotators are indeed working seriously. The second is measuring annotator agreement with Gwet's AC1, with scores ranging from 0.64 (Factuality, the hardest and most subjective) to 0.94 (Refusal, the easiest to determine); this gap itself hints that the reliability of different error types varies greatly.

To quantify "how much of each error is captured by the overall score," the authors fit a Lasso regression ($\alpha=0.01$) between the overall score and the individual error annotations. We can write the intuition of this linear model as (notation devised by this note):

$$\hat{s} \;=\; s_0 \;-\; \sum_{j=1}^{10} w_j \cdot \mathbb{1}[e_j = 1]$$

where $e_j$ indicates whether the $j$-th error appears, and $w_j \ge 0$ is the weight learned by Lasso — that is, "how much the overall score is expected to be docked when that error appears." Lasso's $\ell_1$ penalty pushes the weights of weakly contributing errors down to zero, so the errors with non-zero weights are exactly the ones that "made it into the overall score."

The result is that Six out of ten error types contribute to the overall scores, with refusal having the strongest weight; while Factuality and inconsistency do contribute, their weights are much smaller. In other words, a single preference score **masks** failures in the two key aspects of factuality and consistency — they are seen, but severely underweighted. The remaining errors that did not make it into the score (such as harmful, fluency) happen to also be the rarest (occurrence rate < 1%), because these strong models rarely make such errors on these well-formed tasks.

![Lasso weights](imgs/part1_lasso.png)

*Figure: The weight of each error type under Lasso; refusal is strongest, factuality is low.*

The distractor also reveals a subtler problem: annotators cannot cleanly separate the criteria in their judgments. In theory, criteria like factuality and contradiction, which are "independent of the input," should not be docked on a distractor (what should be docked are relevance and inconsistency). But in practice factuality and contradiction (within the output) are both rated worse for the distractor examples — they are erroneously docked by association. The authors speculate that annotators first form a "good/bad" first impression of a response, then let this impression contaminate every fine-grained judgment: once they feel a response is bad, they are more likely to feel it is wrong everywhere.

### 3. Part 2: Assertiveness is a confounding variable in factuality judgments

Since the fine-grained annotations themselves may also be biased, Part 2 directly examines the confounding variables in two hypotheses: assertiveness and linguistic complexity. Behind this lies the sociolinguistic concept of language ideology — a speaker's tone and style distort the listener's judgment of their credibility and intelligence. The authors use four preambles (system prompts) to manipulate output style: Assertiveness−− (cautious, defensive, uncertain), Assertiveness++ (authoritative, firm, persuasive), Complexity−− (using simple vocabulary), Complexity++ (using complex terminology), with the preamble hidden from annotators.

To estimate the "true" error rate, the authors themselves (the authors) carefully annotated 300 examples for each error type as expert annotations, used to contrast with the crowd annotations. Part 2 annotates 1,500 distinct outputs, with 7,200 annotations including quality checks, and the model set drops the reference and adds Llama 2 13B Chat trained with RLHF.

The core result is: crowd annotators systematically underestimate factuality and inconsistency errors, and the magnitude of this underestimation **expands as assertiveness rises** and shrinks as assertiveness falls. In the paper's words, annotators are more trusting of assertive responses, and are less likely to identify factuality or inconsistency errors within them. That is, assertiveness not only affects the overall score but also directly contaminates the fine-grained judgments that most need objectivity.

![Difference in crowd vs. expert error rates](imgs/part2_errors_by_preamble_diff.png)

*Figure: The error-rate difference δ between crowd and expert annotations under different preambles.*

Walking through the three Factuality columns extracted from the Table (`tab:full_error_rates`) makes this confounding effect clear. The numbers below are crowd (Ann.), expert (Exp.), and the difference δ = Ann. − Exp. (percentage points):

| Preamble | Factuality Ann. | Factuality Exp. | δ (Ann.−Exp.) |
|-|-|-|-|
| Baseline | 4.3 | 20.3 | -16.1 |
| Assertiveness−− | 7.1 | 12.5 | -5.4 |
| Assertiveness++ | 2.2 | 24.6 | -22.3 |

Read it like this: under Baseline, experts catch 20.3% of outputs as having factual errors, while the crowd catches only 4.3%, underestimating by 16.1 percentage points. At Assertiveness++, the true error rate caught by experts actually **rises** to 24.6% (more assertive outputs are actually more prone to fabricating facts), but what the crowd catches **falls** to just 2.2%, with the underestimation surging to 22.3 percentage points. Conversely, in the Assertiveness−− group, the expert true error rate drops to 12.5% and the crowd rises to 7.1%, with the underestimation down to just 5.4 percentage points. The same δ monotonically worsens from −5.4 → −16.1 → −22.3, corresponding exactly to assertiveness going from low to high — this is quantitative evidence that "the more assertive it is, the fewer people can catch it bluffing." The authors also re-plot error rates "binned by perceived assertiveness," reaching a consistent conclusion and ruling out the explanation that it is purely the preamble changing the true error rate.

![Error rates binned by assertiveness](imgs/part2_assertivebins.png)

*Figure: Error rates fall as perceived assertiveness rises.*

### 4. Part 3: Preference as a training objective may amplify assertiveness

If assertiveness raises perceived quality, then training with preference scores carries a risk of side effects. The authors measure that Assertiveness is strongly positively correlated with overall quality scores, with a Pearson correlation coefficient of 0.68, and complexity's correlation coefficient is 0.53. The causal direction is hard to pin down (are assertive responses really better, or are better responses perceived as more assertive?), but this strong correlation itself means: using human preference as a training objective may inadvertently push outputs toward being more assertive and more complex.

![Quality vs. assertiveness](imgs/all_models.png)

*Figure: Scatter and trend lines of quality vs. assertiveness for each model.*

The key difference the paper compares is that Command was fine-tuned on preference scores, while Llama 2 was trained using on-policy RLHF. On the quality-vs-assertiveness plot, Llama 2 13B exhibits higher assertiveness at the same quality (falling in the lower right), while Command 52B is relatively "humble" (upper left, lower assertiveness at the same quality). On this basis the authors offer preliminary evidence: while the RLHF objective raised Llama 2's quality, it may have raised assertiveness even more. This is only preliminary, because the training details of the two models cannot be directly compared — but it incidentally also proves that quality and assertiveness **can be decoupled**, and that a "humble yet high-quality" model should be regarded as more desirable than a "confident yet wrong" one.

## 🧪 Critical Assessment

### Is this problem real and important?

This paper's problem framing holds up, and it strikes at the heart of the current RLHF ecosystem: the preference score is simultaneously the mainstream evaluation metric and the training signal, and if it systematically favors surface properties, the whole alignment pipeline will be contaminated. The authors do not stop at a vague claim like "preference scores are subjective," but instead use expert-vs-crowd dual-track annotation to quantify "how much is underestimated, and by which variables it expands," and the monotonic worsening of δ from −5.4 to −22.3 is hard-to-refute evidence. This is more solid than many evaluation critiques that only make qualitative complaints.

### Are the baselines, controls, data, and metrics sufficient?

The strongest methodological design is treating "the authors' own expert annotations" as an approximation of ground truth and taking the difference against the crowd annotations. But this is exactly the most fragile link: the authors also acknowledge this is not strictly an unbiased set of ratings. The experts (who are also the paper's authors) know the research hypothesis and have an incentive to verify more strenuously on "assertive outputs," and this alone is enough to artificially amplify the trend of δ with assertiveness — a circular risk that cannot be ruled out from within the paper's data. The ideal control would be pre-registered third-party experts blinded to the hypothesis, not the authors themselves. Second, assertiveness and complexity are highly entangled in annotation (the authors themselves state the two dimensions are correlated), so Part 2's decomposition of these two "independent" confounding variables is therefore not clean. Third, Gwet AC1 for Factuality is only 0.64, meaning that even the factuality judgment used as the basis for ground truth is itself quite noisy, and the absolute values of δ need to leave room for error. The sample size (900 / 1,500 outputs) is sufficient for the main effect, but after slicing the data into preamble × error-type the per-cell sample becomes small, and the paper does not report confidence intervals or significance tests for δ, so how much of the "monotonic worsening" is sampling fluctuation remains unproven.

### Is it a new finding or an old observation repackaged?

Phenomena such as sycophancy, verbosity bias, and "reference is not ground truth" have already been raised in related work (Perez et al., Sharma et al., Kabir et al.), and this paper's assertiveness bias can be viewed as a concrete facet of sycophancy. The real increment lies in: moving the bias from the "model behavior" level to the "human annotation process" level, and using Lasso coverage + the expert-crowd difference to turn it from anecdote into a measurable quantity. This is not purely renaming terms, but it is not exactly overturning either — it is more like adding a clean quantitative skeleton to an existing intuition. It is worth noting that the ten error types the authors chose and the Lasso analysis are, to some extent, an evaluation framework designed around the predetermined conclusion "that factuality is underestimated can be highlighted": if a different set of criteria or a different aggregation method were used, refusal would not necessarily still be the strongest weight.

### Was the problem solved, and how much does it matter for the real world?

The paper honestly does not claim to solve the problem — it is a diagnosis, not a remedy, and Part 3 explicitly states that the RLHF part is only preliminary evidence, that the two models cannot be directly compared, and that it at most proposes a hypothesis to be verified, which cannot be taken as a definitive conclusion that RLHF amplifies assertiveness. The claim in the appendix that "reference quality is the lowest" also needs a discount: the unbiased column of `tab:scores_by_model1` actually has Command 6B (3.49) as the lowest, with reference at 3.62, and the claim only holds in the biased column (reference 3.50), where it differs from the next-lowest by only 0.01 — an over-generalized conclusion. In terms of real-world significance, the mitigation directions the authors propose (well-trained, incentivized annotator pools, multi-annotator aggregation, jury learning) are all reasonable but not shown to be effective in this paper, and remain open questions. Overall, this is a high-quality "problem-revealing" paper, whose value lies in making a vague concern measurable and discussable, rather than providing a directly deployable solution.

## 🔗 Related notes

- [Training language models to follow instructions with human feedback (InstructGPT)](../ChatGPT/)
