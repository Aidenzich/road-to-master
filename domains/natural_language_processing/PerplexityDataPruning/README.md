# Perplexed by Perplexity — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models |
| Venue | unknown |
| Year | 2024 |
| Authors | Zachary Ankner, Cody Blakeney, Kartik Sreenivasan, Max Marion, Matthew L. Leavitt, Mansheej Paul |
| Official Code | unknown |
| Venue Kind | paper |

> This note is written based on the arXiv preprint `2405.20541` (`https://arxiv.org/abs/2405.20541`). The authors are affiliated with Databricks, MIT, and DatologyAI; a paper footnote states "Code to be made public soon.", and at the time of writing there is no resolvable official code link, so Official Code is recorded as `unknown`; an official conference version, if it exists, may differ from this preprint.

## First Principles

### The core question and a counterintuitive claim

The question this paper asks is very simple: **can a very small language model be used to select pretraining data for a much larger model?** Prior perplexity-based pruning work (Marion et al.) typically uses a reference model that is "as large as the final model, or even larger" to score, and measures effectiveness with an upstream metric like "perplexity on a held-out test set of the pretraining data." This paper's counterintuitive conclusion is: a reference model with only 125 million parameters can prune data for a model 30x larger, improving the average downstream performance of a 3-billion-parameter model by up to 2.04, while shortening the number of pretraining steps needed to reach the baseline performance by up to 1.45x.

### The pruning algorithm: first train the reference model, then sample by perplexity percentile

The process has two stages. The first stage randomly splits the original dataset $D$ into two parts: one, $D_{\text{ref}}$, is used to train the reference model $\theta_{\text{ref}}$, and the other, $D_{\text{train}}$, is reserved for the final model. After the reference model is trained with the standard next-token-prediction objective, it computes the mean negative log-likelihood (NLL) for each sample in $D_{\text{train}}$ and converts it to perplexity. The second stage, following the empirical CDF $\hat{F}_P$ of these perplexities, keeps the samples that fall within a certain percentile interval and trains the final model. The overall pseudocode is as follows:

```text
Input: dataset D = {x^(i)}; selection_criteria ∈ {low, medium, high};
       selection rate r_s ∈ (0,1); reference split size R
D_ref, D_train ← random_split(D, R)
θ_ref* ← train(random_init, D_ref)
for x in D_train:
    NLL[x]  = (1/|x|) · Σ_j −log P(t_j | t_<j ; θ_ref*)
    PPLX[x] = 2 ^ NLL[x]
if   criteria == low:    min,max ← 0.0,      r_s
elif criteria == medium: min,max ← 0.5−r_s/2, 0.5+r_s/2
elif criteria == high:   min,max ← 1−r_s,    1.0
F̂_P ← empirical CDF of PPLX
D_pruned ← [ x for x in D_train if min < F̂_P(PPLX[x]) < max ]
θ_final* ← train(random_init, D_pruned)
return θ_final*
```

The perplexity of each sample is defined as base 2 raised to the mean negative log-likelihood:

$$\text{NLL}(x)=\frac{1}{|x|}\sum_{t_j \in x} -\log P(t_j \mid t_{<j};\theta_{\text{ref}}),\qquad \text{PPLX}(x)=2^{\text{NLL}(x)}$$

### The three selection criteria and the selection rate

The selection criteria determine which segment of the distribution to keep: low selects the samples with the lowest perplexity, high selects the highest, and medium selects the samples whose perplexity falls near the median, i.e., those in the $[50-\frac{r_s}{2},\,50+\frac{r_s}{2}]$ percentile. The selection rate $r_s$ determines how much to prune away—the experiments ultimately fix it at 50%. There is a key conceptual shift here: this "perplexity of the model after training on the same distribution" is not judging "whether this text is an outlier or dirty," but rather estimating "how hard this sample is for the model." High perplexity means the model finds it hard and information-rich, which also explains why on some datasets "keeping high-perplexity samples" is instead beneficial.

### Experimental setup: two datasets with very different compositions

The models are all based on the MPT transformer family, the reference model is fixed at 125 million parameters, and the final model comes in two sizes, 1 billion and 3 billion. The domain compositions of the two datasets are deliberately chosen to be very different: the Pile is composed of 22 domains, with only 15.61% from general web crawl; Dolma is composed of 7 domains, yet 81.31% is from CommonCrawl. All data are tokenized with the GPT-4 tokenizer. The reference model is fixed at 26 billion tokens of training, and the final model, unless otherwise specified, is trained to Chinchilla optimal (token count equal to 20x the parameter count). Evaluation uses the 33 downstream question-answering tasks on the MosaicML evaluation gauntlet.

Each task's score is first normalized against a random-guessing baseline, then averaged. The normalization formula converts "model accuracy $a_m$" and "random-guessing accuracy $a_r$" into scores on the same scale:

$$a_n=\frac{a_m-a_r}{1-a_r}$$

The paper groups the 33 tasks into five major categories—World Knowledge, Common Sense Reasoning, Language Understanding, Symbolic Problem Solving, Reading Comprehension—averaging within categories first and then across categories to obtain the final Average normalized accuracy. Note that this normalization differs from the byte-length-based normalization of the EleutherAI LM Evaluation Harness.

### Main results

The table below shows the average normalized accuracy of the best pruning setting versus the no-pruning baseline across the four "dataset × model size" settings (excerpted from the paper's Table 1, with Pile using high and Dolma using medium, both at a 50% selection rate):

| Setting | No Pruning (Average) | Best Pruning (Average) | Gain |
|-|-|-|-|
| 1B on Pile | 13.73 | 15.62 (High) | +1.89 |
| 3B on Pile | 18.63 | 20.67 (High) | +2.04 |
| 1B on Dolma | 13.84 | 15.35 (Medium) | +1.51 |
| 3B on Dolma | 19.20 | 19.79 (Medium) | +0.59 |

Across all datasets and model sizes, the model trained on the pruned data beats the baseline on average, and the paper accordingly claims that a small model's perplexity is an effective data-quality signal for a much larger model. But a detail easily obscured by the average is that the **best criterion is not universal** across datasets—the medium criterion that is best on Dolma, if applied to the Pile, actually loses 0.23 on average relative to no pruning.

### A complete pruning walkthrough (with the paper's real numbers)

Take the 1B on Pile cell as an example to walk through. First train a 125M reference model on 26 billion tokens; then compute the perplexity of each sample in the Pile's $D_{\text{train}}$. Because the best criterion on the Pile is high with $r_s=0.5$, the algorithm sets $\text{min\_percentile}=1-0.5=0.5$ and $\text{max\_percentile}=1.0$, i.e., **keeps only the 50% of samples with the highest perplexity** (the half the reference model finds hardest). Using this half of the data, train a 1-billion-parameter final model from scratch to Chinchilla optimal (about 20 billion tokens). Result: the average normalized accuracy improves from the baseline 13.73 to 15.62 (+1.89). Broken down by category, World Knowledge rises from 15.51 to 18.18 and Language Understanding from 28.11 to 33.2, which are the main drivers of the improvement; but Symbolic Problem Solving barely moves (3.53 vs. 3.36). This walkthrough also highlights a point the later critique will discuss: the overall average improvement is in fact highly concentrated in certain categories.

### Training efficiency: reaching the same level faster

![Intermediate downstream performance of each setting during pretraining](imgs/intermediate-eval.png)

Pruning improves not only final performance but also training dynamics. The paper performs intermediate evaluations on partially trained checkpoints and finds that the pruned model beats the baseline at all evaluated intermediate steps; and the pruned model reaches the baseline model's average normalized accuracy with 1.31x and 1.45x fewer steps on Pile 1B/3B respectively, and with 1.29x and 1.14x fewer steps on Dolma 1B/3B respectively.

### Non-standard scenarios: over-training and data-constrained

The paper further tests two non-standard scenarios. On over-training, it trains the 1B model to 130B tokens (5x Chinchilla optimal): on the Pile the pruning gain over the baseline slightly decreases from 1.89 to 1.74 (roughly maintained), but on Dolma it drops from 1.51 to 0.84 (a clear shrinkage). On the data-constrained side, pruning maintains its gain on both the Pile and Dolma until the base data is repeated about 2 times; because $r_s=0.5$ ($1/r_s=2$), the retained pruned subset is at this point effectively repeated about 4 times, echoing the finding of Muennighoff et al. that "returns approach zero beyond four repetitions."

### Upstream perplexity is a misleading evaluation metric

One observation from the paper worth remembering is: using "perplexity on a held-out test set of the pretraining data" to evaluate pruning gives the wrong conclusion. Take 1B on Pile as an example: the pruned model's perplexity on the test set worsens from 7.83 to 8.51, yet the downstream average accuracy improves from 13.73 to 15.62. The reason is that pruning changes the data distribution, making the model a biased estimator of the original distribution, so perplexity on the original distribution is inherently not a fair measure of quality. This is also the core argument for the paper's claim that "one should evaluate directly on downstream benchmarks."

### How pruning changes domain composition

![Log-perplexity distribution before and after pruning](imgs/pplx-dist.png)

![The proportion of each domain in the dataset before and after pruning](imgs/domain-composition.png)

From the log-perplexity distribution, the Pile is multimodal and asymmetric, whereas Dolma is unimodal and symmetric—which also explains why the Pile suits high and Dolma suits medium. From the domain composition, pruning tends to **increase** the proportion of general web-crawl domains and **decrease** the proportion of highly specialized technical domains: on the Pile, the proportions of Pile-CC and OpenWebText2 nearly double, while the proportions of Pubmed Central, ArXiv, Github, and other domains are cut to at least below one-third of the original. This raises a concern the authors themselves flag: for the domains that are heavily pruned away, will the corresponding downstream capabilities be harmed.

## 🧪 Critical Assessment

### The upstream-to-downstream evaluation switch is the paper's most substantive contribution

The problem itself is real: pretraining data quality is indeed a key lever for LLM performance, and "using a small model to save the scoring cost for a large model" is especially practical when the next-generation model is larger than any existing model. The paper's most substantive contribution relative to prior work is switching the evaluation from upstream perplexity to 33 downstream tasks, and explicitly demonstrating that upstream and downstream can move in **opposite** directions (test-set perplexity worsens while downstream improves)—this counterexample alone has methodological value and is worth remembering.

### A single 125M reference model and per-category regressions masked by the average

The breadth of the experiments is decent: two datasets with very different compositions, two model sizes, two trials per experiment, and sweeps over both the criterion and the selection rate. But there are several gaps worth doubting. First, the paper's title touts "small reference models," but it actually uses only a single 125M reference model, with no sweep of reference-model size at all—so the most valuable question, "how small is small enough," is in fact not answered. Second, the normalized average easily masks per-category regressions: on 3B on Pile, Symbolic Problem Solving drops from 4.88 to 2.91, and Reading Comprehension on Dolma 3B drops from 14.2 to 13.19, both substantive regressions covered up by the positive sign of the overall average. Third, the mere +0.59 gain on Dolma 3B, under only two trials and a "within one standard error counts as a tie" criterion, is of doubtful robustness.

### The algorithm follows Marion; the contribution is an empirical checkup

It must be viewed honestly: the pruning **algorithm itself** comes from Marion et al., and this paper does not propose a new pruning mechanism. Its contribution is empirical—switching to a downstream metric, using different domain compositions, adding non-standard scenarios—rather than an algorithmic innovation. This does not diminish its practical value, but reading it as a "new method" would lose focus; it is more like a systematic checkup of an existing method on "under what conditions it works and under what conditions it backfires."

### The best criterion is swept out on the same gauntlet and is not universal across datasets

The best criterion and selection rate are swept out on "the same gauntlet used to report the gains," so "the Pile uses high, Dolma uses medium" is to some extent a result of overfitting to the evaluation set; the paper also admits there is no theory that can predict the parameters in advance, and only shows that the best setting swept out on 1B can transfer to 3B. In other words, being non-universal across datasets and needing to re-sweep parameters for each new dataset is a cost that is not small in real applications, and the paper's mitigation of this (the setting can be transferred cheaply from 1B to 3B) only answers half the question.

### Domain-composition shift and tokenizer/family binding limit extrapolation

As for the claim that "a small model can prune for a large model and improve downstream," the paper does give convincing evidence within its setting. But several practical factors limit extrapolation: the domain composition is systematically skewed toward web text, cutting away code and scientific papers, and the impact on these downstream capabilities is left by the authors themselves only as future work; the code is not yet released at the time of writing, so reproduction requires self-implementation; and the conclusion is bound to a specific tokenizer and the MPT family. Overall this is a solid but clearly scoped empirical study, and reading it as "data pruning is solved" would overextend it—a more accurate positioning is: it advances perplexity pruning from "an upstream observation on a single dataset" to "a usable tool across multiple datasets, multiple scenarios, and judged by downstream."

## 🔗 Related notes

<!-- Currently there are no safely resolvable directly related notes under domains/natural_language_processing; the heading is kept and left empty. -->
