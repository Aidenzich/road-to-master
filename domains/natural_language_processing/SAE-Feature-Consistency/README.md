# SAE Feature Consistency — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs |
| Venue | arXiv preprint (2505.20254) |
| Year | 2025 |
| Authors | Xiangchen Song, Aashiq Muhamed, Yujia Zheng, Lingjing Kong, Zeyu Tang, Mona T. Diab, Virginia Smith, Kun Zhang (Carnegie Mellon University; Kun Zhang is also affiliated with MBZUAI) |
| Official Code | https://github.com/xiangchensong/sae-feature-consistency |
| Venue Kind | paper |

> This note is written against the full arXiv version (`2505.20254`). The manuscript uses a NeurIPS 2025 preprint template, but no evidence of peer-review acceptance was found, so the formal publication venue is recorded as `unknown` and treated here as an "arXiv preprint"; if a formal accepted version appears, its content may differ from this draft.

## First Principles

This is a **position paper**. When reading it, it is essential to separate it into two layers: one is the set of testable **empirical claims** (the PW-MCC numbers, TopK's 0.80, ground-truth recovery in the synthetic model organism, and the correlation between consistency and semantic similarity), and the other is the **normative stance** (that the field as a whole "should" treat consistency as a first-class objective). This section first lays out the method from first principles; the critique is left to the next section.

### The Problem: Why SAE Features "Swap Out on Every Run"

Sparse Autoencoders (SAEs) are used to decompose a neural network's activations into more interpretable, more monosemantic features, and the community's long-standing ambition is to recover a **canonical** (unique, complete, atomic) feature dictionary. The problem is that even with identical data and architecture, changing only the random initialization, the dictionaries learned by different training runs are often markedly different — for a Standard SAE the overlap is sometimes as low as **30%**. This run-to-run instability (accompanied by phenomena such as feature splitting and feature absorption) directly erodes the scientific premise that "explanations obtained from an SAE can be reproduced."

### Formalizing SAEs and Consistency

An SAE consists of an encoder and a decoder: the encoder maps the input $\mathbf{x} \in \mathbb{R}^{m}$ to a sparse latent code $\mathbf{f} \in \mathbb{R}^{d_{\text{sae}}}$, where $d_{\text{sae}} > m$ makes the dictionary **overcomplete**; the decoder reconstructs the input using the dictionary $\mathbf{A}=\mathbf{W}_{\text{dec}}$. Training jointly approximates reconstruction and sparsity, minimizing the following loss:

$$
L(\mathbf{x}) = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \lambda S(\mathbf{f}(\mathbf{x}))
$$

The paper formalizes "consistent" as **Strong Feature Consistency**: two dictionaries $\mathbf{A}$, $\mathbf{A}'$ learned by two independent runs are regarded as equivalent if they can be aligned under a **permutation plus a per-feature nonzero scaling** — that is, for each $\mathbf{a}_i$ there exists $\mathbf{a}_{\sigma(i)}'$ such that $\mathbf{a}_i = \lambda_i \mathbf{a}_{\sigma(i)}'$. This is exactly the equivalence relation measured in the ICA literature by the Mean Correlation Coefficient (MCC).

### MCC, PW-MCC, and GT-MCC

For two dictionaries $\mathbf{A}$, $\mathbf{B}$ (columns are feature vectors), let $n=\min(d_A,d_B)$; MCC is defined as the maximum, over all one-to-one matchings $M$, of the average absolute cosine similarity:

$$
\text{MCC}(\mathbf{A}, \mathbf{B}) = \frac{1}{n} \max_{M} \sum_{(i,j) \in M} \frac{|\langle \mathbf{a}_{i}, \mathbf{b}_{j} \rangle|}{\|\mathbf{a}_{i}\|_2 \|\mathbf{b}_{j}\|_2}
$$

where the optimal matching $M^*$ is solved with the **Hungarian algorithm**. Two metrics follow: **PW-MCC** (Pairwise Dictionary MCC) compares two learned dictionaries each containing $d_{\text{sae}}$ features, measuring run-to-run consistency; **GT-MCC** (Ground-Truth MCC) measures recovery quality in a synthetic setting where a known ground-truth dictionary $\mathbf{A}_{\text{gt}}$ exists. The core empirical argument is: GT-MCC requires ground truth and is unobtainable in practice, whereas PW-MCC requires no ground truth yet can serve as its reliable proxy.

### A Hand-Computable PW-MCC Example

Take two small dictionaries with $m=2$, each with 2 features, to see how the matching dominates the result:

```text
Features of A :  a1 = (1, 0),      a2 = (0, 1)
Features of A':  b1 = (0.1, 1.0),  b2 = (1.0, 0.2)

The four absolute cosine similarities (|cos|):
  |cos(a1,b1)| = |1*0.1 + 0*1.0| / (1 * sqrt(0.01+1.0)) = 0.1 / 1.005 = 0.0995
  |cos(a1,b2)| = |1*1.0 + 0*0.2| / (1 * sqrt(1.0+0.04)) = 1.0 / 1.020 = 0.9806
  |cos(a2,b1)| = |0*0.1 + 1*1.0| / (1 * 1.005)          = 1.0 / 1.005 = 0.9950
  |cos(a2,b2)| = |0*1.0 + 1*0.2| / (1 * 1.020)          = 0.2 / 1.020 = 0.1961

The two one-to-one matchings:
  identity matching (a1-b1, a2-b2): 0.0995 + 0.1961 = 0.2956  → average 0.148
  swapped matching  (a1-b2, a2-b1): 0.9806 + 0.9950 = 1.9756  → average 0.988

Hungarian takes the "maximum total" = swapped matching.
PW-MCC = 1.9756 / 2 = 0.988
```

The key is the last step: if we do no matching and compare index by index, this pair of dictionaries looks only 0.148 similar — almost inconsistent; only after the Hungarian algorithm finds the $\sigma$ that aligns b2↔a1 and b1↔a2 does it become clear that they are in fact the same set of features, merely permuted, with PW-MCC as high as **0.988**. This shows that "consistency" measures equivalence **up to permutation/scaling**, not column-wise equality.

### Why TopK Can Achieve High Consistency: From the Spark Condition to Identifiability

The paper's theoretical backbone is the **spark condition** of sparse dictionary learning: if a dictionary $\mathbf{A}$ is injective on $k$-sparse vectors, then the sparse representation is unique. Invoking the result of Hillar & Sommer, any two dictionaries that both satisfy the spark condition and achieve zero reconstruction error on sufficiently covering data must be consistent up to permutation and scaling ($\mathbf{A}'=\mathbf{A}\mathbf{P}\mathbf{D}$). Because a TopK SAE enforces **exactly $k$-sparse** via $\operatorname{TopK}_k$, approaches zero reconstruction, and induces the spark condition through a round-trip property, when all three hold simultaneously any two TopK SAEs learn dictionaries that are **identical up to permutation and scaling**. This is the theoretical source of "TopK is more consistent than ReLU/Standard."

This also explains why choosing the right $k$ is crucial: sweeping $k$ in the matched regime ($d_{\text{gt}}=d_{\text{sae}}=40$, ground-truth sparsity $s=8$), dictionary recovery quality peaks exactly at $k=s=8$, and is **asymmetric** with respect to misspecification — underestimating $k$ hurts more than overestimating it.

![In the matched regime ($d_\text{gt}=d_\text{sae}=40$, ground-truth sparsity $s=8$), the final dictionary-recovery MCC of TopK as a function of the training parameter $k$ (averaged over the last 100 steps). The red dashed line marks $k=s=8$, where MCC peaks (about 0.81). The underestimation region ($k<8$) is lower overall, dropping as deep as about 0.77; the overestimation region ($k>8$) decays gently to about 0.80 — the asymmetry means that when choosing $k$ it is better to err slightly high than low.](imgs/mcc_vs_k.png)

### Synthetic Validation: A Ladder from the Ideal to the Real

In the **matched** (matched, $d_{\text{sae}}=d_{\text{gt}}$) ideal synthetic setting ($m=8,\ d_{\text{gt}}=16,\ k=3,\ n=5\text{e}4$), **TopK's GT-MCC reaches 0.97 vs 0.63** (Standard), and the PW-MCC curve hugs the GT-MCC curve — this is the cleanest evidence that "PW-MCC can serve as a proxy for GT-MCC." Once capacity mismatch is allowed, a gap appears: in the **redundant** (redundant, $d_{\text{sae}}=160>d_{\text{gt}}=80$) regime GT-MCC is still high (0.95) but **PW-MCC drops to 0.77**, because the excess capacity causes selection ambiguity (there are multiple equally good learned vectors to choose for the same ground-truth feature); in the **compressive** (compressive, $d_{\text{sae}}=80<d_{\text{gt}}=800$) regime GT-MCC and PW-MCC decline together (0.75 / 0.60).

![The MCC convergence curves for the redundant regime (redundant, $d_\text{gt}=80$, $d_\text{sae}=160$, $k=8$). Dark blue is GT-MCC, converging quickly to about 0.95; light blue is PW-MCC, converging only to about 0.77. The gap between the two lines is exactly the selection ambiguity caused by excess capacity — the dictionary is recovered well, yet the selection is unstable. The shaded band is the max–min range over 5 seeds.](imgs/combined_mcc_redundant.png)

![The MCC convergence curves for the compressive regime (compressive, $d_\text{gt}=800$, $d_\text{sae}=80$, $k=8$). When capacity is severely insufficient, dark-blue GT-MCC (about 0.75) and light-blue PW-MCC (about 0.60) are pushed down together and always retain a gap. The shaded band is the max–min range over 5 seeds.](imgs/combined_mcc_compressive.png)

Real language data has a Zipfian long-tailed feature-frequency distribution; the paper folds this into a model organism and finds that a TopK SAE does **not** allocate dictionary capacity evenly across the frequency groups, but approximately follows a power law: $C_c \approx d_{\text{sae}} \cdot p_c^{\alpha} / \sum_j p_j^{\alpha}$, with empirically $\alpha \approx 1.4$. Thus high-frequency concepts get more capacity and higher GT-MCC, while low-frequency concepts are squeezed into a locally compressive regime with low cross-run similarity — consistency is therefore **frequency-dependent**.

![The rank–frequency log-log plot for 1M tokens of the Pile (`monology/pile-uncopyrighted`): blue dots are the measured token frequencies, the red line is the theoretical Zipf law. A few high-frequency tokens take up the vast majority of occurrences, while the rest drag out an extremely long sparse tail; this power-law long tail is exactly the data justification for the synthetic model organism adopting a Zipfian frequency distribution.](imgs/token_zipfian_c4_train_camera_ready.png)

![The synthetic Zipfian model organism (β=1, 10 groups, d_gt=800, d_SAE=80): in the left panel the red line is each group's GT-MCC, which declines as the group's frequency drops (about 0.66→0.55); in the right panel the green line is the number of SAE features allocated, which likewise decreases with frequency. The light-blue bars shared by both panels are each group's occurrence frequency.](imgs/cluster-rank-mcc.png)

![The two-phase Zipfian model organism (d_gt=5000, d_SAE=1000): the horizontal axis is the minimum activation frequency of a matched feature pair, the vertical axis is the cross-seed dictionary-vector similarity. The red mean line rises monotonically from about 0.66 (extremely rare features) to about 0.84 (high-frequency features), quantifying "the rarer, the less consistent."](imgs/freq-vs-similarity.png)

### Real-World Results on Actual LLMs

Training the SAE on the activations of **Pythia-160M** (residual stream, layer 8), with dictionary width $2^{14}$, using 500M tokens of `monology/pile-uncopyrighted`, running 3 independent runs per architecture (`random_seeds = [42, 43, 44]`) and selecting the hyperparameters with the highest PW-MCC. The conclusion is: high consistency is **achievable** on real data — **the PW-MCC of a TopK SAE ≈ 0.80**, the empirical fulcrum for the paper's entire normative claim. The frequency-dependence phenomenon also reappears: binning matched feature pairs by activation frequency, the rarest bin has a mean similarity of only 0.514, while the highest-frequency bin reaches 0.964.

| Act freq / 1M tokens | Features | Matched-pair similarity (mean ± sd) |
|-:|-:|-:|
| 0.1–2.4 | 127 | 0.514 ± 0.280 |
| 2.4–54.1 | 2,542 | 0.742 ± 0.295 |
| 54.1–1.2k | 10,013 | 0.837 ± 0.209 |
| 1.2k–26.7k | 3,548 | 0.888 ± 0.157 |
| 26.7k–592.2k | 33 | 0.964 ± 0.087 |

Going further, the authors use an automated interpretation pipeline to generate a textual description for each feature, then use an LLM to rate the semantic similarity of "the two descriptions on either side of the same matched pair" (GPT Score, 1–10). The higher the dictionary-vector cosine similarity, the higher the semantic similarity: the lowest-similarity bin has a GPT Score of only 1.71, while the highest-similarity bin (0.5795–0.9999, 13,640 pairs in total) reaches 8.28. This is the crucial step that connects "geometric consistency" to "semantic consistency."

| Dictionary-vector similarity bin | Feature pairs | GPT Score (1–10) |
|-:|-:|-:|
| 0.0654–0.1128 | 34 | 1.71 |
| 0.1128–0.1947 | 311 | 2.19 |
| 0.1947–0.3359 | 975 | 3.27 |
| 0.3359–0.5795 | 1,423 | 4.12 |
| 0.5795–0.9999 | 13,640 | 8.28 |

![The frequency-contribution decomposition of overall PW-MCC for the four architectures TopK, Standard, Gated, JumpReLU: the horizontal axis is the minimum activation frequency of a matched pair (%), the bars (left axis) are each bin's contribution, the solid lines are the cumulative contribution, and the dashed lines (right axis) are the distribution of features across bins. TopK has the highest consistency, with its contribution distribution spanning all frequencies and few dead features; Standard and JumpReLU are markedly lower, with large numbers of features crowded into the lowest-frequency bin and contributing almost nothing to the cumulative PW-MCC (the final values read off the legend are approximately TopK 0.82, Gated 0.74, Standard 0.47, JumpReLU 0.50).](imgs/mcc-contributions.png)

The cross-architecture comparison echoes the theory: TopK and BatchTopK attain the highest PW-MCC, while Standard and JumpReLU clearly lag. The appendix's precise sweep quantifies this ordering — on Pythia-160M layer 8, TopK with target $k=20$ takes the top PW-MCC of 0.8181, followed by BatchTopK at 0.7656 and Gated at 0.7370, while JumpReLU (target $L_0=40$) reaches only 0.4947 and Standard (sparsity penalty 0.06) sits at the bottom with 0.4739, with the best results all appearing at training step 244,140. From this the authors argue that PW-MCC is not merely "yet another metric," but can provide a decisive model/hyperparameter selection signal when reconstruction loss cannot distinguish between candidates (for example, TopK's reconstruction loss decreases monotonically with $k$ and cannot indicate the optimal sparsity).

## 🧪 Critical Assessment

### The Problem Is Real, but "Should Be Prioritized" Is a Value Judgment, Not a Conclusion

Feature inconsistency is indeed a real problem: Standard SAE overlap is as low as 30%, and downstream tasks — circuits, steering, unlearning — all presuppose that features are stable entities, so this motivating chain holds. The paper transplants the scientific commonplace of "reproducibility" onto SAEs, and the appeal is legitimate. But one must distinguish: "consistency can be measured, and TopK can reach 0.80" is an empirical fact; "the field **should** elevate consistency to a first-class evaluation criterion" is a normative claim, and the latter does not automatically follow from the former being true — it must still compete with other objectives such as "reconstruction fidelity," "concept coverage," and "downstream task utility" for limited research attention. The paper itself, in its Alternative Views, admits that over-pursuing stability may stifle exploration, but then waves this away with "rigorous science needs measurable baselines," never truly weighing the opportunity cost. Readers should read it as a **data-backed advocacy piece**, not as a settled conclusion that has been justified.

### "Is 0.80 High?": It Is Relatively High, and It Is a High Diluted by Frequency

Interpreting the 0.80 requires three reservations. First, it is **architecture-relative** high: relative to Standard/JumpReLU (whose consistency is clearly lower in the same batch of experiments) it stands out, but the absolute meaning of 0.80 is "the average matched pair still has about twenty percent of its direction misaligned," still far from the "near-equality" required for canonical features. Second, it is a number **averaged over frequency**: within the same TopK SAE, the matched similarity of the rarest features is only 0.514 — and rare features are often exactly the "specific concepts" that interpretability most wants to capture; the place where consistency is worst is exactly where it is most needed. Third, the ideal synthetic setting can reach 0.97 while real data only reaches 0.80, and this 0.17 gap is precisely the price of the Zipfian long tail and capacity compression; the paper presents this honestly, but the headline-level "high consistency is achievable" makes it easy to skip over this layer of conditions.

The redundant-regime synthetic diagnosis makes the mechanism of "high recovery yet low consistency" clearest: sorting each run's learned features by their cosine similarity to ground truth from high to low, the similarity still decays very slowly after crossing the ground-truth dimensionality (rank=40), meaning that each ground-truth concept in fact has a whole batch of equally good candidate vectors. Thus which one the Hungarian algorithm should pick becomes highly sensitive between the two runs, forming selection ambiguity — GT-MCC is very high, yet PW-MCC is clearly lower. This shows that part of the "not high" portion within the 0.80 is not because the features were poorly learned, but because there are too many good features and the selection is unstable.

![In the redundant regime ($d_\text{gt}=40$, $d_\text{sae}=160$, $k=8$), each run's learned features are sorted by "cosine similarity to ground truth" from high to low: red is Seed 0, blue is Seed 1. The similarity stays near 0.9 even after crossing the ground-truth dimensionality (black dashed line, rank=40) and slides down only very slowly, meaning each ground-truth concept has multiple equally good candidate vectors. The in-figure annotation notes that in this setting GT-MCC is about 0.965 while PW-MCC is only 0.815 — high recovery yet low consistency is exactly the direct evidence of selection ambiguity.](imgs/cosine_decay_160.png)

### Between the Metric Definition and the Official Implementation Hides a Sign Discrepancy

The paper's MCC definition explicitly uses the **absolute value** $|\langle \mathbf{a}_i,\mathbf{b}_j\rangle|$, on the grounds that "features pointing in opposite directions should still be regarded as directionally consistent" (sign ambiguity). But a static inspection of the official repo reveals that the two pieces of code that actually compute MCC (`examples/dictionary_learning/utils.py` and `synthetic/utils.py`) both use `cost = 1 - A_norm @ A_est_norm.T` and then `mcc = 1 - cost`, which amounts to **signed cosine similarity, with no absolute value taken**. If there really were a matched feature pair with opposite directions ($\lambda_i<0$), the paper's formula gives +1, while this code gives −1 and treats it in the Hungarian algorithm as a bad match. This is a verifiable "paper definition vs released implementation" inconsistency: it could be that in the unit-norm, non-negative dictionary setting sign almost never arises and causes no harm, or it could systematically bias the reported PW-MCC downward. Either way, it reminds the reader: before reproducing these numbers, first confirm which version of MCC you are using. (This point comes from a static reading of the official repo; no code was executed.)

### Circularity Risk: Using a Self-Defined Metric to Argue That This Self-Defined Metric Should Be Prioritized

There is a methodological tension worth naming. The validity chain of the whole paper is "PW-MCC tracks GT-MCC on synthetic data → therefore PW-MCC is a good proxy → therefore everyone should use PW-MCC." But GT-MCC is only defined inside the **author-built** linear generative model organism ($\mathbf{X}=\mathbf{A}_{\text{gt}}\mathbf{F}_{\text{gt}}$, Gaussian dictionary, artificial Zipfian); whether real LLM features truly possess a "ground-truth dictionary" in the sense of permutation/scaling is exactly the core of the opposing side's (Paulo & Belrose et al.) challenge. The paper patches in real-side evidence with "PW-MCC strongly correlates with semantic similarity," but semantic similarity is itself rated by an LLM (GPT Score), and the sample size of the high-similarity bin (13,640 pairs) is far larger than that of the low-similarity bin (34 pairs), so the correlation may be dominated by high-frequency, easily interpretable features. In other words, this validation is internally coherent within the equivalence relation it defines for itself, but it does not — and can hardly — answer the more fundamental question of "whether a canonical feature set exists"; if it does not exist, pursuing high PW-MCC may amount to stably fitting an artifact. The paper's response to this counter-position is "a pragmatic decomposition has scientific value as long as it is stable," which is a reasonable retreat, but it also quietly swaps the objective from "finding true features" to "finding stable features," and the two are not equivalent.

### Baselines and Ablations Are Broadly Adequate, but the Real-Side Architecture Coverage Is Thin

On the positive side, the synthetic end sweeps all three capacity states — matched / redundant / compressive — as well as $k$ misspecification, and the local-identifiability-regime explanation is self-consistent and supported by figures; the gain of PW-MCC relative to reconstruction loss also has a concrete scenario (TopK's $k$ selection). The shortcoming is on the real side: the main text presents complete results only on a single model (Pythia-160M) and a single layer (layer 8), with Gemma-2-2B pushed to the appendix; yet "how consistency varies across different layers and model scales" is exactly the key to judging whether this claim can generalize. Moreover, "selecting the hyperparameters with the highest PW-MCC" for each architecture itself carries a selection bias favorable to the paper's own metric — using PW-MCC to select hyperparameters and then claiming PW-MCC is high, the fairness of the comparison needs the reader to discount on their own.

## One-Minute Version

- **The inconsistency problem**: training an SAE with identical data and identical architecture, simply changing one random initialization, produces a markedly different learned feature dictionary — a Standard SAE run twice independently sometimes has feature overlap as low as only 30%.
- **How PW-MCC is measured**: first use the Hungarian algorithm to match the features of the two dictionaries one-to-one, then compute the post-matching cosine similarity. The same pair of dictionaries, compared rigidly by index, is only 0.148 and looks entirely unrelated; only after matching does it turn out to be merely a permutation, with PW-MCC actually as high as 0.988.
- **Consistency depends on frequency**: a TopK SAE's overall PW-MCC is about 0.80, but broken down, the rarest features have a mean similarity of only 0.514 while the highest-frequency ones reach 0.964 — the rarer, the less stable.
- **The pain point is in the rare regime**: the place where consistency is worst (rare features) is exactly the "specific concepts" that interpretability most wants to capture; the demand and the reliability are exactly inverted.
- **Paper vs code**: the paper's formula explicitly requires taking the absolute value (so that oppositely-directed features still count as consistent), but the two pieces of code in the official repo that compute MCC both use signed cosine without the absolute value — before reproducing these numbers, first confirm which version you are using.

## 🔗 Related notes

<!-- No safely resolvable related notes yet. -->
