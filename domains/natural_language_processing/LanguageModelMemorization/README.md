# How much do language models memorize? — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | How much do language models memorize? |
| Venue | arXiv preprint (2505.24832v3, cs.CL) |
| Year | 2025 |
| Authors | John X. Morris, Chawin Sitawarin, Chuan Guo, Narine Kokhlikyan, G. Edward Suh, Alexander M. Rush, Kamalika Chaudhuri, Saeed Mahloujifar |
| Official Code | unknown |
| Venue Kind | paper |

> This note is written from the arXiv preprint `2505.24832v3` (cs.CL); this version is a preprint, and the numbers in the official published version (if any) may differ. The authors are affiliated with FAIR at Meta, Google DeepMind, Cornell University, and NVIDIA.

## Introduction

Over the past few years the amount of training data for language models has exploded, yet parameter counts have stayed at the scale of a few billion: the example the paper gives is an 8B-parameter model (about 32 GB on disk) trained on 15 trillion tokens (about 7 TB on disk). When the amount of data is far larger than the information the weights can hold, "how much of the training data does the model actually memorize" becomes a core question for privacy and copyright. Existing approaches mainly take two routes: extraction (can training samples be reproduced from the weights) and membership inference (deciding whether a given sample was in the training set), but neither can cleanly separate "memorization" from "generalization" — a model being induced to emit a string may simply be because it learned a regularity, not because it memorized that particular piece of data.

This paper proposes a compression-based (bits) definition of memorization and uses it to measure the capacity of modern language models. The core approach splits total memorization into two parts: unintended memorization (the information a model holds about a specific dataset) and generalization (the knowledge a model learns about the true data-generating process, which the paper equates with intended memorization). What remains after subtracting generalization is unintended memorization; because it has super-additivity, summing per-sample unintended memorization gives only a **lower bound** estimate of "dataset-level unintended memorization," not the dataset's total memorization (the latter also includes the generalization part). When the data is large enough and the model saturates its memory, this lower bound approaches a plateau, whose height is the estimate of the model's capacity.

The measurement proceeds in three steps. The first step uses uniformly random sampled bitstrings as training data: because every token is independently uniform, there is no generalizable regularity, so unintended memorization is almost equal to total memorization, and capacity can be measured on its own — this gives the capacity estimate of **about 3.6 bits-per-parameter for the GPT family**. The second step switches to real text (FineWeb), observing how the model transitions between memorization and generalization and connecting this transition point to double descent. The third step uses capacity and data size to derive a scaling law for membership inference and validates it on models from 500K to 1.5B parameters. The experiments train hundreds of GPT-2-architecture Transformers in total, measuring metrics including per-sample memorization (bits), the F1 of loss-based membership inference, and the extraction rate under greedy decoding.

## First Principles

### Why extraction is not enough to define memorization

The paper's starting point is that a model being able to generate a string does not mean it memorized that string. Prior work has shown that language models can be induced to output almost any string, so "it was output" is not by itself evidence of memorization. Even if the prompt length is limited or the prompt is required to align with a prefix, it still cannot distinguish whether the model relies on memorization or on generalization — a model asked to add two numbers can compute the answer without ever having seen that equation. The paper uses a concrete sample to point out the core difficulty: the training sample `Q: What is 2^100? A: 1267650600228229401496703205376` would be judged highly memorized by almost every extraction-style definition, but being able to do exponentiation is exactly a capability a language model should have, and that part should not count as memorization of that particular piece of data.

### A statistical view: writing memorization as mutual information

First, use Shannon information theory to build the skeleton. Denote the data distribution as a random variable $X$, and let the training algorithm $L$ map samples to the trained model $\hat{\Theta}$. The total information the model holds about $X$ is the mutual information of the two:

$$\text{mem}(X, \hat{\Theta}) = I(X, \hat{\Theta}) = H(X) - H(X \mid \hat{\Theta})$$

This quantity counts all information, including generalization. To keep only the "unintended" part, the paper first uses the prior $\Theta$ of a true model to fix the generalizable information, defining unintended memorization as:

$$\text{mem}_U(X, \hat{\Theta}, \Theta) = I([X \mid \Theta], \hat{\Theta}) = H(X \mid \Theta) - H(X \mid (\Theta,\hat{\Theta}))$$

Generalization (intended memorization) is then total memorization minus the unintended part, $\text{mem}_I = \text{mem}(X,\hat{\Theta}) - \text{mem}_U(X,\hat{\Theta},\Theta)$. This split inherits the idea from Brown et al. of defining memorization with conditional mutual information, but the paper's key difference is doing this at the **single-instance level**.

A property that supports the later measurement is the super-additivity of unintended memorization: for $n$ i.i.d. samples, the sum of per-sample memorization is a lower bound on the dataset's memorization, while the model's own entropy is its upper bound.

$$\sum_{i\in[n]} \text{mem}_U(X_i, \hat{\Theta}, \Theta)\leq \text{mem}_U(X, \hat{\Theta}, \Theta) \leq H(\hat{\Theta})$$

This equation has two practical implications: to estimate a lower bound on dataset-level memorization, one can simply sum the per-sample memorization; and unintended memorization grows with data size but never exceeds the model capacity $H(\hat{\Theta})$ — this is exactly the theoretical basis for the "plateau" phenomenon later.

### Switching to Kolmogorov complexity: handling the "only one sample" predicament

The definitions above are all built on the entropy of random variables, but in a real setting we only have one trained model $\hat{\theta}$, one dataset $x$, one reference model $\theta$ — all singletons, so entropy cannot be estimated for a single sample. The paper therefore switches to a compression-based Kolmogorov complexity: the information content $H^K(x)$ of a string $x$ is defined as the length of the shortest program that produces $x$ under some computational model.

$$H^K(x) = \min_{f(p)=x} |p|$$

The relative version $H^K(x \mid \theta)$ is "the shortest length to describe $x$ when $\theta$ is on hand as a reference." So the Kolmogorov version of unintended memorization is written as:

$$\text{mem}^K_U(x,\theta,\hat{\theta}) = H^K(x\mid \theta) - H^K(x\mid (\theta, \hat{\theta}))$$

The intuition is: given a general reference model $\theta$, how many extra bits does $x$ still need to be described; if adding $\hat{\theta}$ can shorten this length further, the bits saved are $\hat{\theta}$'s unintended memorization of $x$. The paper also proves that in expectation the Kolmogorov version differs from the Shannon version by a constant independent of $n,\ell,\ell'$, so the two are compatible.

### Making the uncomputable definition measurable: arithmetic coding and likelihood

Kolmogorov complexity itself is **uncomputable**, so the paper approximates it with an off-the-shelf compression algorithm, choosing arithmetic coding — because it is effective for text compression, and its code length can be computed directly from the model's likelihood. The Kolmogorov definition of unintended memorization $\text{mem}^K_U = H^K(x\mid\theta) - H^K(x\mid(\theta,\hat{\theta}))$ is the difference of two terms: the first is the shortest length to describe $x$ when "holding only the reference model $\theta$," and the second is the shortest length when "holding both the reference and the target model." The paper estimates each term once using negative log-likelihood (and when holding only the target model there is also $H^K(x\mid\hat{\theta})\approx -\log p(x\mid\hat{\theta})$, used to estimate total memorization):

$$H^K(x \mid \theta) \approx -\log p(x \mid \theta), \qquad H^K(x \mid \theta,\hat{\theta}) \approx -\log \max\{p(x \mid \hat{\theta}),\, p(x \mid \theta)\}$$

The latter term takes the larger of the two models' likelihoods, because during compression one is free to pick whichever model compresses shorter. Subtracting the two terms, unintended memorization reduces to a clean form (the following simplification is a derivation of this note):

$$\text{mem}^K_U(x,\theta,\hat{\theta}) = H^K(x \mid \theta) - H^K(x \mid (\theta,\hat{\theta})) \approx \max\left\{\log \frac{p(x \mid \hat{\theta})}{p(x \mid \theta)},\; 0\right\}$$

This equation makes it clearest **how the reference model subtracts out generalization**: unintended memorization is positive only when the target model $\hat{\theta}$ assigns $x$ a higher likelihood than the reference model $\theta$, and its value is exactly the extra bits $\hat{\theta}$ compresses out relative to $\theta$; if the $\theta$ that represents generalizable knowledge already explains $x$ equally well or better, this term goes to zero — the part explainable by generalization is subtracted out entirely. Therefore **the reference model's likelihood $p(x\mid\theta)$ is the baseline in the estimator**, and its choice directly determines how much generalization is subtracted. The paper uses two kinds of reference model: in the synthetic random-string experiment the data-generating distribution is known, so the true distribution is used directly as $\theta$; in the text experiment the main reference model is one with **the same parameter count, trained on the largest amount of data (the entire dataset)**, plus an oracle reference model that pursues the lowest evaluation loss and may have many more parameters. It is worth noting that the paper starts from likelihood, detours through Kolmogorov, and finally returns to likelihood for estimation, but stresses that the likelihood here depends on decoding parameters (such as temperature, top-k), and is not the same as the original likelihood notion.

### Measuring capacity with synthetic random strings

Because uniformly sampled data has no generalizable structure, its Shannon information content can be computed exactly. Given dataset size $N$, $S$ tokens per sequence, and vocabulary size $V$, the entropy of the entire dataset is $H(x) = N S \log_2 V$; then use $\hat{\theta}$'s arithmetic-coding code length to estimate $H^K(x \mid \hat{\theta})$, and subtracting gives the memorization $\text{mem} = H(x) - H^K(x \mid \hat{\theta})$, with model capacity taken as the maximum memorization across all dataset sizes. The experiments train GPT-2-architecture models from scratch, 1 to 8 layers, hidden from 32 to 512, parameters from 100K to 20M, trained for $10^6$ steps, batch 2048, Adam, A100, bfloat16, with defaults $V=2048$, $S=64$, running 5 random seeds per setting.

![Unintended memorization (bits) of models of different sizes on uniformly random data vs the number of training samples. Each line is one model size; small datasets are fully memorized (hugging the grey Dataset size diagonal), and once the data is large enough each model's memorization hits a horizontal plateau proportional to its parameter count, no longer growing with data size.](imgs/synth_plateau.png)

The plateau height shows an extremely smooth linear relationship with model parameter count: plotting each model's maximum memorization against parameter count, the slope is bits-per-parameter. The paper's headline number is the abstract's "about 3.6 bits-per-parameter for the GPT family," corresponding to the fit value $\alpha = 3.64$ in the main figure under bfloat16; the body also summarizes it as "3.5 to 4 bits per parameter, depending on architecture and precision." Note that 3.5–4 is a rounded summary interval, not the actual range of individual models: the per-configuration $\alpha$ listed one by one in Table 5 actually spans from a low of 2.86 (8 layers, $d_\text{model}=32$, bfloat16) all the way to a high of 4.23 (1 layer, $d_\text{model}=32$, fp32), with the average landing at 3.51 for bfloat16 and 3.83 for fp32. This average is slightly larger than the roughly 2 bits-per-parameter estimated via quantization by Allen-Zhu et al., but consistent with the earlier finding that "fact storage grows linearly with capacity."

![Plotting each model's measured maximum memorization (Total memorization, bits, x-axis) against model parameter count (y-axis), points for different $d_\text{model}$ (32/64/128/256) fall on the same line with slope 3.64 bits-per-parameter (labeled in the top-right corner). This is the most central capacity estimate of the whole paper.](imgs/capacity_bpp.png)

A concrete sanity check: fixing $N=4096$ samples, $S=64$, $V=2048$, the dataset entropy $H(x)=4096 \times 64 \times \log_2 2048 = 4096 \times 64 \times 11 \approx 2.88 \times 10^6$ bits; this model has about $6.67 \times 10^5$ parameters, so with $\alpha=3.642$ the capacity is estimated at about $2.43 \times 10^6$ bits. Since the capacity is smaller than the data entropy, the predicted memorization takes the capacity end, and the paper (after correcting for embedding size) gives an expected value of $2.36 \times 10^6$ bits, with a measured $2.29 \times 10^6$ bits, an error of 2.97%. The prediction formula the paper uses is:

$$\text{mem}(X,L(X)) \approx \min(\text{capacity}(L),\, H(X))$$

When sweeping across sequence length and vocabulary size, the average error of this linear capacity prediction is only 1.7% (sweeping $S$) and 1.8% (sweeping $V$).

The effect of precision is surprisingly small: switching bfloat16 to fp32 doubles the number of bits in $\theta$, but the average $\alpha$ only rises from 3.51 to 3.83 — far short of 2×, meaning most of the extra bits from increased precision are not used for raw storage.

| Precision | Average $\alpha$ (bits-per-parameter) |
|-|-|
| bfloat16 | 3.51 |
| float32 | 3.83 |

### Text: separating unintended memorization from generalization

Switching to real text, learning now mixes sample-level unintended memorization with population-level generalization. The paper switches to the FineWeb dataset (because it applies state-of-the-art deduplication) and additionally performs a strict deduplication (otherwise, after truncating to 64 tokens, about 1–2% of sequences become duplicates), because deduplication is extremely important for faithfully measuring the extraction rate.

![Measuring unintended memorization of text using a large oracle reference model. Each line is one model size (3.6M/8M/19.2M). Memorization first rises with data size — small models learn more than the oracle on small training sets — then turns to decline after reaching capacity, because the model starts to generalize and its average performance actually loses to the high-capacity oracle. This "rise then fall" curve is direct evidence of memorization giving way to generalization.](imgs/text_oracle_mem.png)

This rise-then-fall curve visualizes the transition from memorization to generalization: the model first fills its capacity with sample-level details, and once it can no longer fit them, it starts to replace the memorization of individual samples with reusable general regularities. The paper connects this transition point to double descent. Note that the cleanest double-descent plot the paper uses to draw "dataset-to-capacity ratio vs test loss" is actually on **synthetic bitstrings** (Figure 1, which the body uses to visualize the ratio of data size to capacity), not the text experiment of the previous paragraph — because synthetic data allows exact computation of the dataset size (via the reference model's compression rate) and the model capacity (via the estimated $\alpha$), so the ratio can be plotted accurately. On this synthetic plot, the peak of the test loss falls exactly at the position where the ratio equals 1.

![Double descent of the **synthetic bitstring experiment** (the paper's Figure 1): test loss (y-axis, values as high as $10^3$–$10^4$, the loss magnitude of uniformly random data rather than text) vs dataset-to-capacity ratio (x-axis). The curves of multiple models simultaneously peak at ratio = 1 (the dashed line "Model capacity = Dataset size"), and after crossing it the loss drops sharply and converges to a low point. On this basis the paper argues that double descent begins exactly when data capacity exceeds model capacity. This figure measures synthetic data; the corresponding verification on text is in the figure below — the text experiment instead plots train/test loss vs "number of training samples" (rather than the ratio), which is the direct evidence on the text side.](imgs/double_descent.png)

![Plotting directly the raw training and testing loss of the text experiment, one can see the full shape of the same double-descent curve. On FineWeb text, train (solid) and test (dashed) loss of four models (1.7M/3.6M/8.0M/19.2M parameters) vs number of training samples (not ratio). With little data, train loss is extremely low (nearly memorized), while test loss rises first; each model's test loss peak shifts right as parameter count grows — the yellow 19.2M peak falls at about $6\times10^4$ samples, after which test loss drops sharply and converges together with train loss to a floor of about 4–5. The peak position corresponds exactly to where the model's capacity is filled by data.](imgs/text_train_val_loss.png)

The behavior of extraction also supports the same story. The paper measures the extraction rate on the full training set and 10,000 non-overlapping test samples: a 32-token prefix has 100% extractability on very small training sets, decreasing as the training set grows; but when the (deduplicated) dataset is large enough, the extraction rate does not go to zero, but converges to almost equal the extraction rate of the test set. In other words, when the data is large enough, all successful training-data extraction can be attributed to generalization, not to memorization of specific samples.

![On deduplicated FineWeb, extraction rate of 8/16/32-token prefixes vs number of training samples, solid lines for the training set and dashed lines for the test set. When the training set is very small, all three prefixes have nearly 100% extractability; once the training set crosses $\sim10^5$ the extraction rate plunges, and at $>10^6$ samples the training set (solid) and test set (dashed) almost coincide — the 32-token (yellow) lines both converge to about $3\times10^{-4}$. The convergence of training and test extraction rates shows precisely that successful extraction under large data comes from generalization, not sample memorization.](imgs/text_extraction_rates.png)

![For a 20M-parameter model trained beyond capacity, plotting per training sample the Kolmogorov memorization (bits, x-axis) against TF-IDF (y-axis). Among samples with positive memorization, higher TF-IDF (more rare words) means more memorized; the red points at the far top-right are the small group of samples with the highest memorization.](imgs/tfidf_memorization.png)

Which samples are most memorized? The paper's distribution analysis shows that a model trained on a large amount of text can memorize only a "few" samples. Using TF-IDF to measure how many rare words each document has, it finds that among samples with positive memorization, TF-IDF is strongly correlated with memorization: the more rare words a sample contains, the more it is memorized. The single sample with the highest TF-IDF in the entire training set (a passage of Japanese) ranks third in memorization, and the model can even regurgitate the whole passage from a single token (囚) — even though it is just one of 260,000 training samples; among the top twenty most-memorized samples, all but three contain tokens from other languages (Japanese, Chinese, Hebrew).

![After a 20M-parameter model is trained beyond capacity, the top 20 samples with the highest TF-IDF in the whole training set (sorted by TF-IDF from high to low), listing the text, TF-IDF, Memorization (bits), and language. The top three by TF-IDF (index 0–2) are all Japanese; index 0 has TF-IDF 78553 and memorization 2.98 bits, the highest in the table. The entire table is non-English (Japanese, Chinese, Hebrew, Greek) long-tail text, intuitively showing that rare-word samples are the most likely to be memorized verbatim.](imgs/table03_examples.png)

### The scaling law of membership inference

The paper uses the most standard loss-based membership inference: set a loss threshold to decide whether a sample is a member of the training set. For a model of fixed capacity, the F1 of membership inference is roughly sigmoid-shaped vs data size — when a large model overfits a small dataset, F1 starts at 1, and as the data grows it becomes harder and harder to distinguish training from test samples, finally decaying to 0.5 (random guessing).

![The loss-based membership inference F1 of four model sizes (1.71M/3.62M/8.03M/19.20M parameters) vs number of training samples. All four lines start at F1≈1.0 when data is scarce, and fall along a sigmoid down to the floor of 0.5 (random guessing) as data grows; larger-capacity models (yellow 19.2M) have their falling point further to the right — they need a larger dataset before overfitting breaks down. This sigmoid is exactly the target of the scaling law fit below.](imgs/membership_f1_curves.png)

The paper fits it as:

$$\text{Membership}_{F_1}(\theta, \mathcal{D}) = \frac{1}{2}\left(1 + c_1\,\sigma\!\left(c_2\left(\frac{\text{Capacity}(\theta)}{|\mathcal{D}|} + c_3\right)\right)\right)$$

Nonlinear least-squares fitting gives $c_1 = 1.34$, $c_2 = -0.034$, $c_3 = -33.14$, with the fit values within the observed 1–2%. The paper then argues that its limiting behavior is "very clean": as $|\mathcal{D}| \to \infty$ the attack performance approaches 0.5, so for a model trained on infinitely large data, both membership inference and extraction become impossible.

There is an internal inconsistency in the paper itself here worth laying out. Substituting the printed coefficients above directly back into the printed formula above does not yield 0.5: as $|\mathcal{D}| \to \infty$, $\text{Capacity}(\theta)/|\mathcal{D}| \to 0$, the inside of the sigmoid approaches $c_2 c_3 = (-0.034)\times(-33.14) \approx 1.13$, $\sigma(1.13)\approx 0.755$, and substituting back gives $F_1 \approx \tfrac{1}{2}(1 + 1.34\times0.755) \approx 1.01$, not 0.5. That is, the "closed-form fit + coefficients" the paper printed contradict its own claimed limit of 0.5 — the original paper (the formula at source/main.tex line 248, the limit statement at line 252, the coefficients at line 254) carries the same conflict. What must be distinguished is: the **measured** F1 curve of the previous figure does indeed drop to 0.5 as data grows (that is a data observation), but the printed analytic expression itself does not reproduce this limit, so any derivation that "substitutes infinite data size into the formula to get 0.5" does not hold, and the contemporary-model conclusion below cannot be supported by the numerical values of this formula either.

To validate, the paper uses the scaling law to solve for the data size needed to reach target F1 values (0.55, 0.75, 0.95), and actually trains models to measure. There is a naming inconsistency in the paper itself worth noting here: the body writes "GPT-2 small (125M params)" and "GPT-2 XL (1.5B params)," but the Table 4 used for validation labels the row with 123,702,528 parameters as GPT2-Medium — the same model is called two names in the body and the table, and the table below follows the table's labels. The predictions land roughly within 1.5 percentage points of the true F1, with the least accurate being near F1≈0.75 (the steepest part of the sigmoid). Both F1 columns of the table below are shown in percentage points (a target value of 55.00 means F1=0.55), to avoid misreading the predicted decimals and the measured percentages as an order-of-magnitude difference:

| Model (per Table 4 label) | Parameters | Data size $\lvert D\rvert$ | Predicted F1 | Measured F1 |
|-|-|-|-|-|
| GPT2-XL | 1,556,075,200 | 170,654,583 | 55.00 | 54.61 ± 1.3 |
| GPT2-XL | 1,556,075,200 | 18,851,574 | 95.00 | 95.85 ± 0.8 |
| GPT2-Medium | 123,702,528 | 13,566,442 | 55.00 | 53.44 ± 1.1 |
| GPT2-Medium | 123,702,528 | 1,498,634 | 95.00 | 97.98 ± 0.3 |

![The scaling law of membership inference: x-axis data size (samples), y-axis model capacity (bits), color is F1 (50–100), contours are the fitted curves, and dots are measurements. Top-left (large model, small data) has high F1 (yellow), bottom-right (small model, large data) approaches F1 50 (blue).](imgs/mi_scaling_law.png)

On this basis the paper points out: the tokens-per-parameter ratio of all contemporary language models is above $10^2$, and argues that substituting this ratio into the scaling law yields F1≈0.5, so under this framework statistically significant loss-based membership inference on average samples is not feasible. But this step lands exactly on the internal inconsistency above: contemporary models are precisely in the region where $\text{Capacity}(\theta)/|\mathcal{D}|$ is very small (approaching 0), and substituting the printed coefficients into the printed formula, this region gives $F_1 \approx 1.01$, not 0.5. So "substituting into the formula gives 0.5" cannot be derived from the paper's printed analytic expression; what can support "membership inference degrades to random under large data" is actually the observation from the earlier **measured** F1 curve dropping to 0.5 as data grows, not the numerical extrapolation of this fit formula. This contemporary-model conclusion should be read as a qualitative claim the paper makes from the measured trend, not a quantitative result strictly derived from the scaling law.

## 🧪 Critical Assessment

### Is the problem real and important

The problem itself is real: memorization directly bears on privacy, copyright, and training-data leakage, and the two mainstream routes of extraction and membership inference indeed cannot distinguish "memorizing a piece of data" from "learning a regularity" — the paper's $2^{100}$ counterexample points out this gap very clearly. Defining memorization in bits and forcibly subtracting the generalizable part is a conceptually clean and operational contribution, not a repackaging.

### The sufficiency of capacity, baseline, data, and metric

The experimental design is quite solid on the "synthetic" side: uniformly random data makes the Shannon information content exactly computable, the linear relationship of capacity to parameter count holds across depth, width, and precision, and there is cross-sequence-length/vocabulary validation with 1.7–1.8% error and 5 seeds. But membership inference throughout uses only a single loss-based threshold attack, and does not include stronger attacks like LiRA, reference-model calibration, or shadow-model types; the conclusion "membership inference on average samples is not feasible" therefore holds for **this one** attack, and does not necessarily hold for stronger attacks or the most vulnerable samples (worst-case) — and privacy risk is often driven by worst-case samples. More critically, the scaling law that supports this conclusion has an internal inconsistency itself: as shown above, substituting the paper's printed coefficients back into the printed formula gives $F_1\approx1.01$ rather than the 0.5 it claims as the data size goes to infinity, so this quantitative "not feasible" statement currently rests only on the observation that the measured curve drops to 0.5, with no self-consistent analytic expression numerically holding it up. This is where I think the strength of the evidence is overstated by the paper's wording.

### 3.6 bits-per-parameter: an empirical plateau, or an information-theoretic upper bound?

Two things need to be carefully distinguished. 3.6 bpp is an **empirical plateau** observed on uniformly random sequences, trained with gradient descent, and the paper itself points out that because SGD does not guarantee finding the global optimum, what is measured is actually a **lower bound** on capacity. Therefore reading it as some universal information-theoretic upper bound is inappropriate: it depends on the GPT-2 architecture family, a specific training budget, and bfloat16 precision, and the precision experiment (3.51→3.83) also shows this number is sensitive to implementation details. The paper is largely honest about this point (using wording like "approximate" and "lower bound"), but in the abstract "3.6 bits-per-parameter" is presented as a single clean constant, easily over-generalized by readers.

### The gap in extrapolating from synthetic uniform strings to natural text

The largest extrapolation risk is here. The clean linear relationship of capacity was measured on **uniformly random** data — no structure at all, each sample independent, so unintended memorization is almost equal to total memorization. Real text is highly structured, with a long tail and duplication, and memorization and generalization are entangled; the paper's own oracle curve and TF-IDF analysis also show that memorization of text concentrates on a few rare samples, a distribution very different from the synthetic case. Therefore the extent to which the constant "the model has 3.6 bpp capacity," derived from synthetic data, can be used to infer how much a model memorizes of **natural text** remains open; the alignment of double descent is a beautiful observation, but what the paper establishes is an **alignment and prediction** around the dataset-to-capacity ratio, not a strict causal proof — "once it can't fit, it is forced to share information and generalize" is currently a plausible but causally unverified hypothesis.

### The gap among memorization, extractability, membership, and privacy

The paper clearly defines memorization as information content (bits), but there is still a distance between this and actual privacy leakage: a piece of data, even with very low unintended memorization, may still leak because it can be extracted; conversely the paper also observes cases where membership F1 reaches 0.97 while the extraction rate is 0, showing the three are not equivalent.

![Membership inference F1 (y-axis) vs suffix extraction rate of a 32-token prefix (x-axis), showing a clear L shape. Along the entire vertical line where extraction rate is exactly 0, membership F1 goes all the way from 0.5 up to about 0.93 (measured points around 0.50, 0.53, 0.56, 0.71, 0.88, 0.93); as soon as extraction is slightly greater than 0, F1 almost hugs 1.0. This shows that "unable to extract any original text" does not mean there is no privacy risk — the success threshold of membership inference is clearly lower than extraction, and the two cannot substitute for each other.](imgs/membership_vs_extraction.png)

Moreover the entire method depends on access to the model's likelihood (white-box, or at least being able to compute $p(x\mid\theta)$), making it hard to apply directly to closed-source models with only API access; and although the scaling law is validated up to 1.5B parameters, modern frontier models are two to three orders of magnitude larger, and extrapolating there remains an unverified inference. None of this negates the paper's core contribution, but it is a reminder that "average-case membership inference is not feasible" should not be read as "these models have no privacy risk."

## One-minute version

- **Why "memorization" is so hard to measure**: in the past, memorization was judged by "whether the model can be made to emit a string," but being able to emit it does not equal having memorized it — a model that can add two numbers may just have learned arithmetic, and need not have seen that equation during training.
- **How to separate rote memorization from generalization**: the paper instead defines memorization by information content (bits), subtracting the "understanding of true regularities," and what remains counts as unintended memorization of a particular piece of data. The model first fills its capacity with sample-level details, and once it can no longer fit them, it replaces the memorization of individual samples with general regularities.
- **How much a model can memorize**: the GPT family is about 3.6 bits per parameter. Converting for a model of about 660,000 parameters, the capacity is about 2.43 million bits, and this estimate is quite close to the measurement.
- **Don't misapply this number**: 3.6 bits-per-parameter was measured on uniformly random strings, and should not be taken directly as a universal upper bound for natural text — real text is highly structured, with a long tail and duplication, and memorization and generalization are entangled, a situation very different from synthetic strings.
- **Don't relax on privacy**: the paper argues that "loss-based membership inference on average samples tends to become infeasible," but this quantitative conclusion relies on a fit formula with an internal inconsistency (using its printed coefficients to reverse-solve the infinite-data limit gives F1≈1, not 0.5), and what actually holds up is only the observation that "the measured F1 drops to 0.5 as data grows"; and even if it holds, it does not mean the model has no privacy risk — real leakage is often driven by the most vulnerable extreme samples, and membership F1 can still be as high as above 0.9 when extraction is 0.

## 🔗 Related notes

- [Scaling Laws for Neural Language Models](../ScalingLaws/) — this paper's capacity and membership scaling law follow the scaling-law methodology and GPT-2 training setup of Kaplan et al.
- [Attention is all you need](../AttentionIsAllYouNeed/) — all experiments in the paper measure the GPT-2 (Transformer decoder) architecture.
