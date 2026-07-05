# MaskLLM — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models |
| Venue | NeurIPS 2024 |
| Year | 2024 |
| Authors | Gongfan Fang, Hongxu Yin, Saurav Muralidharan, Greg Heinrich, Jeff Pool, Jan Kautz, Pavlo Molchanov, Xinchao Wang (NVIDIA, National University of Singapore) |
| Official Code | https://github.com/NVlabs/MaskLLM |
| Venue Kind | paper |

> Note: The full-text evidence for this note is taken from the camera-ready source of the arXiv e-print (`maskllm_camera_ready.tex`, the NeurIPS 2024 final layout); all numbers and formulas are based on that `.tex`.

## First Principles

The problem MaskLLM aims to solve is **N:M sparsity in semi-structured pruning**: within every consecutive M weights, keep at most N nonzero values. The reason for locking onto N:M rather than arbitrary unstructured sparsity is that the regular arrangement of N:M is friendly to accelerators such as GPUs, obtaining both "structured acceleration" and "the flexibility of fine-grained sparsity" at the same time. This paper's main setting is 2:4, that is, keeping two of every four consecutive parameters and clearing two.

Once sparsification is written as a **mask selection** problem, the size of the candidate set is a combinatorial number. For a block $\mathcal{W} \in \mathbb{R}^{1\times4}$ made of four consecutive parameters, a 2:4 binary mask must contain exactly two zeros, so the candidate set has only the 6 options below:

$$
\mathbf{S}^{2:4} = \{\mathcal{M} \in \mathbb{B}^{1\times4} \mid \textstyle\sum \mathcal{M} = 2\} = \{[1,1,0,0], [1,0,1,0], [1,0,0,1],[0,1,0,1],[0,1,1,0],[0,0,1,1]\}
$$

The generalized candidate-set size is $|\mathbf{S}|=\binom{M}{N} = \frac{M!}{N!(M-N)!}$, and for 2:4 it is $\binom{4}{2}=6$. A single block having only 6 choices seems simple; the real difficulty lies in **scale**: the paper points out that in the dense layers of a fully 2:4-sparsified LLaMA-2 7B there are 1.6 billion (1.6 billion) 2:4 masks to decide, and the whole thing is an astronomically large combinatorial optimization problem.

$$
\{\mathcal{M}_i^{*}\} = \operatorname{argmin}_{\{\mathcal{M}_i \in \mathbf{S}^{2:4}\} } \mathbb{E}_{x\sim p(x)} \left[ \mathcal{L}_{LM}(x; \{\mathcal{W}_i \odot \mathcal{M}_i\}) \right]
$$

The equation above is the ideal objective: on observed data, find a set of masks that minimizes the pruned language-model loss $\mathcal{L}_{LM}$; $\odot$ is element-wise multiplication. The problem is that the selection of masks is itself discrete and non-differentiable, and cannot be directly backpropagated through. The way existing approaches (such as SparseGPT and Wanda) bypass this is to use a small calibration set together with hand-designed importance criteria to estimate which weights can be removed, but this brings two structural weaknesses: first, a small calibration set is insufficient to represent the knowledge an LLM learns from a vast and diverse corpus — the paper observes that after enlarging the calibration set beyond 256 samples the result no longer improves; second, using a hand-crafted criterion as a proxy for the "true pruning error" itself accumulates estimation error.

MaskLLM's core technique is to rewrite "selecting a mask" as "**sampling**": for each parameter block, define a categorical distribution with class probabilities $p_1, p_2, \ldots, p_{|\mathbf{S}|}$ and $\sum_j p_j = 1$. During training, if a sampled mask yields good quality after pruning, its probability is increased; after repeated sampling and updating, the high-probability masks are those that still maintain quality after pruning. The objective thus turns from optimization over discrete masks into optimization over a probability distribution:

$$
\{p^{*}(\mathcal{M}_i)\} = \operatorname{argmin}_{\{p(\mathcal{M}_i)\}} \mathbb{E}_{x\sim p(x),\, \mathcal{M}_i \sim p(\mathcal{M}_i)} \left[ \mathcal{L}_{LM}(x; \{\mathcal{W}_i \odot \mathcal{M}_i\}) \right]
$$

But sampling from a categorical distribution is likewise non-differentiable. The authors borrow the reparameterization trick **Gumbel Softmax** to resolve this: first use Gumbel Max to outsource the randomness of sampling to an independent noise variable $g_i=-\log(-\log \epsilon_i),\ \epsilon_i \sim U(0,1)$, obtaining a hard one-hot index $y=\text{onehot}(\operatorname{argmax}_i [\log(p_i) + g_i])$; then use Softmax to replace the non-differentiable argmax, obtaining a soft index controlled by a temperature $\tau$:

$$
\tilde{y}_i = \frac{\exp((\log(p_i) + g_i) / \tau)}{\sum_j \exp( (\log(p_j) + g_j) / \tau )}
$$

As the temperature $\tau \rightarrow 0$, the soft index approaches one-hot. With the soft index $\tilde{\mathbf{y}}$, this row vector, and by stacking the 6 candidate masks into a matrix $\mathbf{S}$ (the $i$-th row is candidate $\hat{\mathcal{M}}_i$), a single matrix multiplication yields a differentiable "soft mask" — which is in fact the average of the candidate masks weighted by the soft index:

$$
\tilde{\mathcal{M}} = \tilde{\mathbf{y}} \times \mathbf{S}=\sum_{i=1}^{|\mathbf{S}|} \tilde{y}_i \cdot \hat{\mathcal{M}}_i
$$

In practice the authors do not learn the probabilities directly, but learn the logits $\pi_i$, together with a scaling factor $\kappa$ to obtain $p_i = \frac{\exp(\pi_i \cdot \kappa)}{\sum_j \exp( \pi_j \cdot \kappa ) }$. This $\kappa$ is the key knob controlling "exploration vs. convergence": when $\kappa$ is too large, the logits overwhelm the Gumbel noise, sampling is almost fixed and exploration is lost; when $\kappa$ is too small, noise dominates, the mask keeps changing, and convergence is very slow. Throughout, the paper linearly ramps $\kappa$ from 1e2 to 5e2.

The authors additionally found a practical pitfall: pruning zeroes out some parameters, causing gradient vanishing and harming subsequent downstream transfer and finetuning. To address this they add **Sparse Weight Regularization**, encouraging the remaining weights to maintain a large enough magnitude, forming the final learning objective:

$$
\min_{\{p_{\pi}(\mathcal{M}_i)\}} \mathbb{E}_{x, \tilde{\mathcal{M}}_i \sim p_{\pi}(\mathcal{M}_i)} \left[ \mathcal{L}_{LM}(x; \{\mathcal{W}_i \odot \tilde{\mathcal{M}}_i\}) \right] - \lambda \sum_i \|\mathcal{W}_i \odot \tilde{\mathcal{M}}_i\|^2_2
$$

The second term is weighted by $\lambda$, and the paper uses $\lambda=$ 1e-5; Table 8 (appendix) shows that the average gradient norm of GPT-3 2B over the first 500 steps rises from 0.219 without regularization to 0.542 (1e-5) and 0.559 (1e-4), corroborating that it indeed maintains the gradient.

Finally there is the **transfer learning** part. Since what is learned is a probability distribution, one can use precomputed masks to initialize the logits and accelerate convergence. The authors propose a Mask Prior: given a prior mask $\mathcal{M}_0$ (which may come from Magnitude, SparseGPT, or Wanda), compute its similarity to each candidate mask and re-center it, then adjust the initial logits according to the similarity:

$$
\pi_i^{\prime} = \pi_i + \sigma(\pi)* \text{sim}(\mathcal{M}_0, \hat{\mathcal{M}}_i) * \alpha
$$

where $\sigma(\pi)$ is the standard deviation of the logits and $\alpha$ controls the prior strength; when $\alpha=0$ it is equivalent to using no prior at all, learning purely from scratch. The whole training process can be condensed into the algorithm below:

```text
Algorithm 1  MaskLLM: Learnable 2:4 Semi-Structured Sparsity
  S = { [1,1,0,0], [1,0,1,0], ... , [0,0,1,1] }          # 6 candidate 2:4 masks
  # Executed in parallel over all parameter blocks W:
  Initialize logits  π_i ~ N(0, σ)
  Update with prior mask M0    π'_i = π_i + σ(π) * sim(M0, M̂_i) * α
  while training not finished:
      # Differentiable sampling
      ỹ_i = softmax((π_i·κ + g_i)/τ),  g_i = -log(-log ε_i),  ε_i ~ U(0,1)
      M̃   = ỹ × S = Σ_i ỹ_i · M̂_i
      Update logits by gradient:  ∇_π [ L_LM(x; W ⊙ M̃) - λ‖W ⊙ M̃‖² ]
  k  = argmax(π)
  M* = M̂_k                                                # final hard mask at inference time
```

**A concrete forward example (numbers taken from the paper; the illustrative weight values are defined by this note and marked as ours).** Take a 2:4 block and set its four weights to $\mathcal{W}=[0.8, -0.1, 0.5, 0.05]$ (ours). The candidate set is the 6 masks above. If initialized with a Magnitude prior, the two positions of largest magnitude are slots 1 and 3 ($|0.8|,|0.5|$), corresponding to candidate $\hat{\mathcal{M}}_2=[1,0,1,0]$, so the Mask Prior raises $\pi_2$. During training the Gumbel noise still occasionally lets the model sample other candidates (e.g., $[1,1,0,0]$) to explore, but as long as those choices make $\mathcal{L}_{LM}$ worse, their logits are suppressed. After 2,000 steps, $\operatorname{argmax}(\pi)$ lands on $[1,0,1,0]$, and the final hard mask prunes the block into $\mathcal{W}\odot\mathcal{M}^*=[0.8, 0, 0.5, 0]$. Applying this mechanism in parallel to LLaMA-2 7B's 1.6 billion blocks, freezing the weights and learning only the masks, after running 2,000 steps the Wikitext PPL drops from SparseGPT's 10.42 to 6.72, close to the dense model's 5.12.

The table below shows the paper's main results (freezing weights, learning only masks; the SparseGPT column does weight updates), giving a clear view of MaskLLM's position relative to three 2:4 baselines (Wikitext PPL, lower is better; alongside the parentheses is the average accuracy Avg. over seven zero-shot tasks):

| Method (LLaMA-2 7B) | Wikitext PPL | Avg. (7 tasks) |
|-|-|-|
| Dense (dense upper bound) | 5.12 | 57.16 |
| Magnitude | 54.71 | 46.19 |
| SparseGPT (with weight update) | 10.42 | 47.16 |
| Wanda | 11.29 | 45.98 |
| MaskLLM (frozen weights) | 6.72 | 52.09 |

On LLaMA-2 13B, Nemotron-4 15B, and GPT-3 2B/843M there is a consistent trend: MaskLLM's PPL is comprehensively better than the three baselines. The effect of the prior is also clear — LLaMA-2 7B can only learn to 9.12 PPL with "no prior," dropping to 6.77 after switching to a Magnitude prior and to 6.72 with a SparseGPT prior, showing that "prior initialization + end-to-end refinement" is better than either alone.

![Overview of MaskLLM's learnable N:M sparsity: the learned general mask can be further transferred to downstream tasks](imgs/teaser_learnable_nm_sparsity.png)

![The MaskLLM framework: modeling mask selection as distribution learning, and after end-to-end training it can be transferred to downstream tasks to achieve lossless compression](imgs/framework_overview.png)

![Sampling a stochastic mask from a learnable distribution with Gumbel Softmax: each M consecutive parameters is associated with a distribution over candidate masks, and both sampling and weighted averaging are differentiable throughout](imgs/gumbel_mask_sampling.png)

On the downstream-application side, the paper argues that the learned mask can "losslessly" adapt a frozen LLM to a specific domain: for GPT-3 2B, directly reusing the general mask degrades to an average PPL of 10.61, training an expert mask from scratch gives 7.51, whereas using the general mask as a prior and then transferring reaches 7.39, even slightly better than the dense 7.42. On cost, each task only needs to store a mask and shares the same set of weights: with simple arithmetic coding, each parameter is only 0.65 bits ($\log_2(6)/4$), about a 25× storage saving relative to storing a full set of 16-bit finetuned weights; running batch size 1 with TensorRT-LLM on an A6000, 2:4 sparsity brings about a 1.4× (measured 1.36–1.41×) throughput speedup and about 27% memory saving.

## 🧪 Critical Assessment

### The hardware reality of 2:4 sparsity and the structural limitation of small calibration sets
N:M sparsity is a real, hardware-backed problem rather than a manufactured need: NVIDIA GPUs after Ampere have native acceleration for 2:4, and the paper also includes measured TensorRT-LLM throughput (LLaMA-2 7B about 1.36–1.41×, 13B about 1.50–1.57×), meaning the PPL improvement genuinely has a landing path and does not stop at paper metrics. The authors' diagnosis of the weaknesses of existing methods also holds up: a small calibration set (no improvement beyond 256 samples) and a hand-crafted importance criterion as a proxy for the true pruning error are indeed structural limitations of the SparseGPT/Wanda line.

### Evaluation falls on PPL and zero-shot, lacking direct verification of generation quality
The baselines cover the three main lines of Magnitude, SparseGPT, and Wanda, with the appendix adding SOTA methods such as ADMM-Iter, GBLM, RIA, and Pruner-Zero, and it clearly labels SparseGPT as "does weight updates" while MaskLLM is "frozen weights," so the comparison conditions are disclosed relatively honestly. The ablations are also fairly complete: prior type, scaling factor $\kappa$, Gumbel temperature $\tau$, prior strength $\alpha$, sparse regularization, and layer sensitivity are all swept. A questionable point is that the metrics lean heavily on Wikitext PPL and zero-shot accuracy — PPL is sensitive to sparse perturbations, but does not correspond linearly to true generation quality (long-text coherence, instruction following), and the paper does not provide such end-to-end downstream generation evaluation.

### Cost is underestimated: the real expense hides on the training side
"Freezing weights and learning only the masks" sounds very light, but learning the mask itself is not cheap: LLaMA-2 7B requires 64 A100s, 8-way tensor parallel, running 2,000 steps for a total of about 1,280 GPU hours, and 13B is even about 2,304 GPU hours. By contrast SparseGPT/Wanda are "one-time, small calibration, almost no training." Therefore MaskLLM's PPL advantage is bought with "several orders of magnitude more compute"; the paper's narrative places emphasis on quality improvement, but this order-of-magnitude compute difference is in fact very critical to the practical decision of "who should use which method," and deserves a more prominent side-by-side presentation.

### Scaling a known technique to billion-scale frozen LLMs, and the author-defined "lossless" threshold
The core components (Gumbel Softmax, learnable sparse masks, prior initialization) already exist in vision models and earlier learnable-sparsity literature, and the paper itself admits it is the "first" to bring this set to a frozen, billion-parameter-scale LLM. Therefore its contribution is closer to "successfully scaling a known technique to a new scale and solving the problems that only surface at scale (gradient vanishing → sparse weight regularization)," rather than an entirely new mechanism. This is a solid engineering contribution, but reading it as a major methodological breakthrough would overestimate its originality. In addition, the term "lossless" is used somewhat loosely: a downstream PPL of 7.39 vs. the dense 7.42 is only a tie on that metric, and calling it "lossless compression" is a target threshold defined by the authors with their own metric, and does not mean no drop under arbitrary tasks and arbitrary metrics.

### Is the claimed problem really solved, and its real-world relevance
On the narrowed proposition of "using more compute to learn a better 2:4 mask over frozen weights," the evidence is sufficient: multiple model families, consistently better than baselines, and with deployment data. But the larger narrative — "N:M sparsity can achieve lossless compression on LLMs" — is supported by the evidence to a degree weaker than the literal claim: it holds for the PPL of a specific downstream domain, and is premised on expensive per-task mask training. Real-world relevance is therefore two-sided: for scenarios that already repeatedly deploy the same frozen model in large volume and can afford the one-time mask-training cost (e.g., a cloud provider permanently serving one LLaMA-2), the 25× storage saving and 1.4× speedup are quite attractive; for users with limited compute who only want to quickly compress once, the cost-effectiveness of a one-time method may still win out.

## 🔗 Related notes

- [AttentionIsAllYouNeed](../AttentionIsAllYouNeed/)
- [Lora](../Lora/)
