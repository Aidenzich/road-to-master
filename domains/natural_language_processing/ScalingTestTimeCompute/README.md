# Scaling LLM Test-Time Compute Optimally — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters |
| Venue | unknown |
| Year | 2024 |
| Authors | Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar |
| Official Code | unknown |
| Venue Kind | paper |

> This is the arXiv preprint `2408.03314` (UC Berkeley and Google DeepMind); if the camera-ready version differs, the formally published version takes precedence; the venue tier is marked `unknown` due to the lack of a citable ranking source.

## Core question and the two scaling axes

This paper asks a very concrete question: if we let an LLM spend a fixed but non-trivial amount of extra compute at inference (test time), how much can its performance improve on hard problems? The authors decompose "spending more inference compute" into two mutually independent mechanisms to study: one is searching against a process-based verifier reward model, and the other is letting the model "adaptively" revise its own distribution over answers at test time given the prompt. The effectiveness of both mechanisms depends strongly on problem difficulty, an observation that motivates a "compute-optimal" scaling strategy: allocating inference compute on a per-problem basis. Relative to a best-of-N baseline, this strategy improves the efficiency of inference compute by more than 4×; and in a FLOPs-matched comparison, for problems on which a small model already has a non-zero success rate, test-time compute can even outperform a model about 14× larger.

![Overall results of compute-optimal for revisions and search (top: revisions)](imgs/summary_revisions.png)

![Overall results of compute-optimal for revisions and search (bottom: PRM search)](imgs/summary_search.png)

### A unified view: the proposer and the verifier

The authors first unify all test-time compute methods into the single act of "adaptively modifying the model's output distribution at test time, conditioned on the prompt," and point out that there are only two knobs: one is at the input level, using extra tokens to augment the prompt so the model changes its conditional distribution (i.e., modifying the proposal distribution); the other is at the output level, first sampling multiple candidates and then operating on those candidates with a post-hoc verifier or scorer. The authors draw an analogy to MCMC sampling: using a simple proposal distribution together with a score function to approximate a complex target distribution. Modifying the proposal distribution and using a verifier constitute the two independent coordinate axes of this study.

With this unified view, the authors formalize "using test-time compute most effectively" as an optimization problem over the hyperparameter $\theta$. Let $\operatorname{Target}(\theta, N, q)$ be the model's distribution over output tokens for prompt $q$, test-time hyperparameter $\theta$, and budget $N$; the goal is to select the $\theta$ that maximizes the accuracy:

$$
\theta^{*}_{q,a^*(q)}(N) = \operatorname{argmax}_{\theta} \left( \mathbb{E}_{y \sim \operatorname{Target}(\theta, N, q)} \left[ \mathbbm{1}_{y = y^*(q)} \right] \right)
$$

where $y^*(q)$ is the correct answer to $q$. This expression itself cannot be solved directly, and the authors' key approximation is to express the optimal hyperparameter as a function of problem "difficulty." Difficulty is treated as a sufficient statistic: as long as one can estimate a problem's difficulty, one can look up and select the strategy that performs best on the validation set for that difficulty, and then apply it to the test set.

The definition of difficulty follows the approach of Lightman et al. and uses the base LLM as a reference: for each problem in the test set, the model's pass@1 rate is estimated using 2048 samples, and problems are binned into five difficulty quantiles accordingly. The authors distinguish two kinds of difficulty: oracle difficulty bins using a ground-truth correctness check, while model-predicted difficulty instead bins using the average final-answer score of a learned verifier over the same 2048 samples — the latter does not require knowing the answer and therefore can be used at deployment, at the cost that the binning itself also requires one round of inference compute.

### The verifier axis: PRM and three kinds of search

The authors found that directly reusing the PRM800k human-annotated data released by Lightman et al. to train a PRM works poorly for their PaLM 2 model (even best-of-N can easily exploit loopholes), which they attribute to a distribution shift between the GPT-4-generated samples and PaLM 2; they therefore switch to the annotation-free approach of Math-Shepherd, supervising the PRM using per-step correctness estimates obtained from a Monte Carlo rollout forward from each step, whose per-step prediction is equivalent to an estimate of the reward-to-go value of the base model's sampling policy. For aggregation, the step-wise score takes the score of "the last step" as the score of the entire solution (rather than taking the minimum or the product), while inter-answer aggregation uses best-of-N weighted — summing the verifier scores of identical final answers and selecting the one with the largest sum.

![Illustration of the three methods for PRM search (best-of-N, beam search, lookahead search)](imgs/search_methods.png)

The authors compare three ways of searching against the PRM: best-of-N weighted samples N complete solutions independently and then selects the best; beam search searches step by step over the PRM's per-step predictions; lookahead search additionally simulates k steps forward at each step of beam search, using the PRM score at the end of the rollout to score that step, so beam search can be viewed as the lookahead special case with $k=0$, and can also be viewed as MCTS with the exploration randomness removed, doing only exploitation. The core loop of beam search can be written as:

```
Given beam count N, beam width M:
  1. Sample N "first step" candidates
  2. Score each candidate step using the PRM's per-step reward-to-go estimate
  3. Keep the top N/M highest-scoring steps
  4. From each kept candidate, sample M "next steps" each, obtaining N/M × M = N prefixes
  Repeat 2–4 until the solution ends or 40 rounds of beam expansion are reached
Finally apply best-of-N weighted over the N final-answer candidates to select the answer
```

To fairly compare the generation budget across different search methods, the authors define one "generation" as sampling a single answer from the base LLM; the budget for best-of-N and beam search is N, while for lookahead search, because each step additionally simulates k steps, the cost is defined as $N \times (k+1)$ samples. The results show: at small generation budgets, beam search clearly outperforms best-of-N, but as the budget is scaled up the advantage disappears and even drops below best-of-N; lookahead search is generally the worst at the same generation budget, because the simulated rollouts consume extra compute. The authors attribute the diminishing returns to over-optimization of the PRM's predictions — for example, search can induce the model to produce low-information repeated steps at the end of the solution, or to squeeze out overly short solutions of only one or two steps. After binning by difficulty, one can see: on easy problems (levels 1 and 2), beam search regresses as the budget increases, indicating that it is amplifying spurious features of the verifier; on medium difficulty (levels 3 and 4), beam search stably beats best-of-N; and on the hardest level 5, no method makes substantive progress.

![Compute-optimal search vs. the best-of-N baseline (can use about 4× less compute)](imgs/compute_optimal_search.png)

Stitching together "picking the best search strategy for each difficulty bin" yields the compute-optimal curve for search. In the low-generation-budget regime, whether using oracle or predicted difficulty, compute-optimal scaling can nearly match best-of-N with up to 4× less test-time compute (e.g., 16 vs. 64 generations); in the higher-budget regime, part of the advantage of predicted difficulty shrinks, but oracle binning can still keep improving, showing that adaptively allocating compute does provide gains.

### The proposal-distribution axis: sequential revisions

![Illustration of parallel sampling (best-of-N) vs. sequential revisions](imgs/parallel_vs_sequential.png)

Simply prompting an off-the-shelf LLM to self-correct is almost ineffective on mathematical reasoning, so the authors fine-tune a revision model that revises its own answer step by step, following the recipe of Qu et al. The training data is produced as follows: for each problem, 64 responses are sampled in parallel, then post-hoc assembled into multi-turn trajectories — pairing each correct answer with a string of preceding incorrect answers placed in the context, with the number of incorrect answers sampled uniformly between 0 and 4, and using character edit distance to select incorrect answers related to the correct answer, so that the model learns to implicitly find and fix the errors in the in-context examples. At inference there is a distribution-shift problem: the model is trained only on "contexts that are all incorrect answers," but at test time the context may contain a correct answer, causing about 38% of originally correct answers to be revised into wrong ones on the next revision; therefore one needs to use sequential majority voting or verifier selection to pick the best answer across the entire revision sequence.

The authors argue that sequential and parallel have complementary properties: parallel sampling is like a global search, able to cover many different high-level solution approaches; sequential revision is like local refinement, suited to answers that are already roughly on the right track. Therefore, under a fixed generation budget, sweeping the "sequential/parallel" ratio yields an ideal ratio that maximizes accuracy, and this ideal ratio changes with problem difficulty: for easy problems it is best to put the entire budget into sequential revisions, while for harder problems one needs to strike a balanced ratio between sequential and parallel.

![Compute-optimal revisions vs. best-of-N (can use about 4× less compute)](imgs/compute_optimal_revisions.png)

Selecting the best sequential/parallel ratio per difficulty bin yields the compute-optimal strategy for revisions. At higher generation budgets, pure parallel sampling tends to saturate, while compute-optimal scaling keeps improving; whether using oracle or predicted difficulty, it can beat the best-of-N baseline with up to 4× less test-time compute (e.g., 64 vs. 256 samples).

### Exchanging test-time compute for pretraining compute

The authors then ask a resource-allocation question: if a model is pretrained with $X$ FLOPs and expected to run $Y$ FLOPs of inference, and we now want to multiply the total FLOPs by a factor $M$, should this extra budget go into pretraining or test-time compute? They use the common approximation, with pretraining $X = 6ND_{\text{pretrain}}$ and inference $Y = 2ND_{\text{inference}}$:

$$
X = 6 N D_{\text{pretrain}}, \qquad Y = 2 N D_{\text{inference}}
$$

where $N$ is the parameter count, and $D_{\text{pretrain}}$, $D_{\text{inference}}$ are the numbers of tokens for pretraining and inference respectively. Multiplying the parameters by $M$ multiplies both the pretraining and inference FLOPs by $M$. If instead one uses a small model plus test-time compute to match this amount of FLOPs, the small model's inference compute can be scaled up by a factor of $M + 3 \left(\frac{D_{\text{pretrain}}}{D_{\text{inference}}}\right)(M-1)$. This factor depends on the ratio, and the authors define $R = \frac{D_{\text{inference}}}{D_{\text{pretrain}}}$: large-scale production often has $R \gg 1$ (far more inference tokens than pretraining), while many self-improvement pipelines have $R \ll 1$.

![The trade-off between pretraining and test-time compute under FLOPs matching](imgs/pretrain_exchange.png)

The authors use a $\sim14×$ parameter scaling as the comparison, and pick three $R$ values: 0.16 ($R \ll 1$), 0.79 ($R \sim 1$), and 22 ($R \gg 1$). The conclusion is: if one will only encounter very hard problems (levels 4/5) or the inference load is high ($R$ large), it is more cost-effective to invest the budget in pretraining; if the problems are mostly easy to medium (levels 1/2/3, sometimes level 4) or the inference demand is low (such as self-improvement), test-time compute is more cost-effective. That is, the two are not exchangeable 1-to-1.

### A walk-through with real numbers

Take beam search as an example, with generation budget $N=64$ and beam width $M=4$: first sample 64 first-step candidates, score them with the PRM, and keep the top $N/M = 16$ highest-scoring; then from each kept candidate sample $M=4$ next steps each, obtaining $16 \times 4 = 64$ prefixes, cycling like this for up to 40 rounds, and finally apply best-of-N weighted over the 64 final-answer candidates. The "4×" compute-optimal benefit of search refers precisely to this: under adaptive allocation, 16 generations suffice to match the level that best-of-N needs 64 generations to reach.

Substituting $M=14$ into the FLOPs-exchange factor, one can compute the table below (the values here are derived by this note from the paper's formula $M + 3(D_{\text{pretrain}}/D_{\text{inference}})(M-1)$, not given directly in the paper):

| Scenario | $R = D_{\text{inference}}/D_{\text{pretrain}}$ | Factor by which small-model inference can scale |
|-|-|-|
| Self-improvement (little inference) | 0.16 | $14 + 3 \times 6.25 \times 13 \approx 258$ |
| Balanced | 0.79 | $14 + 3 \times 1.27 \times 13 \approx 63$ |
| Large-scale production (much inference) | 22 | $14 + 3 \times 0.045 \times 13 \approx 16$ |

This table quantifies the paper's intuition: when $R \ll 1$, a small model can employ about 258× its original inference compute to match the FLOPs of a 14×-larger model, and such a large test-time budget is enough for the compute-optimal strategy to win on most difficulties; but when $R \gg 1$, only about 16× inference budget remains, and the marginal benefit of test-time compute is insufficient to make up for parameter scaling, so pretraining wins. This precisely explains why the paper emphasizes that the two kinds of compute are "not freely exchangeable," and that the conclusion reverses with the deployment scenario ($R$).

## 🧪 Critical Assessment

### Should FLOPs go to pretraining or inference — a motivation that honestly confronts negative results

"Should FLOPs be spent on pretraining or inference" is a real question with direct impact on deployment: if a small model plus test-time compute can substitute for a large model, one could replace a data-center-scale LLM with an on-device small model, and it also offers a path to reduce human supervision via self-improvement. The paper's argument for this motivation is solid, and it honestly writes "test-time compute is not a panacea" into the conclusion — no method makes substantive progress on the hardest level 5 problems, and this refusal to dodge negative results improves credibility.

### A single dataset, a single model, and the compute cost hidden inside difficulty estimation

The methodological ablations are relatively complete: both search and revision have difficulty binning, sequential/parallel ratio sweeps, and oracle-vs-predicted difficulty comparisons, and the appendix additionally compares PRM aggregation strategies and PRM vs. ORM. But the external validity of the evaluation is an obvious soft spot — the entire paper draws conclusions only on a single dataset MATH and a single base model PaLM 2-S* (Codey), and the authors themselves only say they "believe" the findings can transfer to comparable models, which is an unexperimentally-verified inference, making it hard to determine whether the 4× and 14× hold on other model families or other reasoning tasks. Moreover, difficulty binning requires 2048 samples per problem, and even the model-predicted version admits it is costly in production, which means part of the "test-time compute" is spent on estimating difficulty but is not counted into the main comparison, which may cause the claimed efficiency gains to be overstated in practice.

### Old components, a new framework, and a benchmark bounded by the method's own difficulty definition

To be fair, the components the paper uses mostly come from existing work: the PRM comes from Lightman / Math-Shepherd, beam / lookahead search are variants of BFS-V and MCTS, the revision recipe follows Qu et al., and the FLOPs approximation comes from the Chinchilla line. The truly new contribution is the unified framework of "using problem difficulty as a sufficient statistic and adaptively allocating compute per problem" and its systematic measurement, rather than any single algorithm. One point that warrants caution here is that the evaluation design is biased toward the authors' own method: difficulty binning is defined precisely by "this base model's pass@1," and the compute-optimal gains are also measured under the same difficulty definition, selecting the best on the validation set and then measuring on the test set — this setting where "the benchmark is bounded by the method's own characteristics" makes it easy for the 4× gain to look prettier than it would under a neutral, exogenous difficulty standard.

### "can be" is a necessary qualifier: a constrained promise and the unsolved hurdle of difficulty-estimation cost

I believe the paper delivers a constrained version of its promise, not a universal claim. What it proves is: under specific conditions (easy-to-medium difficulty, low inference load $R$), simple test-time methods can beat pretraining scaling; but it simultaneously proves the reverse — at high difficulty or high $R$, pretraining is still better, and the two cannot be exchanged 1-to-1. Therefore the "can be" in the title's phrase "can be More Effective than Scaling Model Parameters" is a necessary qualifier, and reading it as a general conclusion would be misleading. In terms of real-world relevance, the most crucial unsolved problem is the cost of difficulty estimation: the current approach requires expensive sampling, and if difficulty cannot be judged in real time cheaply, the actual benefit of the compute-optimal strategy in online deployment remains questionable, a point the authors also list as future work.

## 🔗 Related notes
