# Refusal in Language Models Is Mediated by a Single Direction — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Refusal in Language Models Is Mediated by a Single Direction |
| Venue | NeurIPS 2024 (peer-reviewed, main conference) |
| Year | 2024 |
| Authors | Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda |
| Official Code | https://github.com/andyrdt/refusal_direction |
| Venue Kind | paper |

This note is written from the full text of arXiv v3 (arXiv id `2406.11717v3`, 2024-10-30) together with the official code; that version corresponds to the NeurIPS 2024 camera-ready (DBLP `conf/nips/ArditiOSPPGN24`), though minor details may still differ in the official proceedings version.

## Introduction

Today's chat models go through two kinds of fine-tuning — instruction-following and safety — so they are simultaneously "obedient" and "well-behaved": they comply with harmless requests and refuse harmful ones. This refusal behavior is ubiquitous across chat models, but what mechanism triggers it "inside" the model was previously unclear. The specific question this paper asks is: how complex is the structure that mediates the high-level behavior of refusal within the representation space of the residual stream? The authors' answer is surprisingly simple — a single one-dimensional subspace (a single direction) is enough to mediate it.

This question matters because it bears on both "interpretability" and "safety" at once. Open-weight models have fully public weights, so an attacker holds white-box access; if refusal is held up by only a single direction, then the robustness of safety fine-tuning becomes questionable. The authors indeed turn this understanding into an attack: a rank-one weight edit can make a 70B model almost never refuse, at a cost of under $5.

The paper's high-level approach has three steps. First, using a small batch of contrastive pairs of harmful and harmless instructions, it computes candidate directions via difference-in-means and then picks the single most effective direction `r`. Second, it performs two kinds of causal intervention to validate this direction: erasing the direction from the residual stream (directional ablation) makes the model stop refusing harmful instructions; adding the direction back into the activation (activation addition) makes the model refuse even harmless instructions. Third, it equivalently implements "erasing the direction" as a direct weight edit (weight orthogonalization), yielding a white-box jailbreak that is both simple and free of gradient optimization.

On measurement, the authors span 13 open-source chat models (1.8B to 72B), covering five families — Qwen, Yi, Gemma, Llama-2, Llama-3. Bypassing refusal is evaluated on JailbreakBench's 100 harmful instructions, and inducing refusal on Alpaca's 100 harmless instructions. Each generation is given two scores at once: a refusal score that uses string matching to judge whether it "looks like a refusal," and a safety score that uses Meta Llama Guard 2 to judge whether the content is actually harmful. The strength of the white-box jailbreak is compared against other jailbreak methods via HarmBench's attack success rate (ASR); capability preservation is measured with the four standard benchmarks MMLU, ARC, GSM8K, TruthfulQA, comparing the relative change before and after the modification.

## First Principles

### The residual stream and the linear representation hypothesis

A decoder-only transformer maintains the representation of each token on a single residual stream: the activation at layer `l`, position `i` is written $\mathbf{x}_i^{(l)} \in \mathbb{R}^{d_{\mathrm{model}}}$, and both attention and MLP write into this stream "additively." The core hypothesis behind this research is the "linear representation hypothesis": the model encodes concepts (features) as linear directions in activation space, and these directions are causal mediators of behavior that can be used for fine-grained steering. The authors focus only on the post-instruction token positions in the chat template — the positions "after the instruction" (denoted the set `I`) — because it is at these positions that the model decides how to respond.

### Extracting the direction with difference-in-means

To find the "refusal direction," for each layer `l` and each post-instruction position `i`, the authors compute the mean activation of harmful instructions $\mu_i^{(l)}$ and the mean activation of harmless instructions $\nu_i^{(l)}$, and take their difference to get a candidate vector:

$$
\mathbf{r}_i^{(l)} = \mu_i^{(l)} - \nu_i^{(l)}.
$$

This difference-in-means vector carries two meanings at once: the direction itself represents the difference orientation between the harmful and harmless mean activations, and the length quantifies the distance between the two group means. The official code implements it very literally — `get_mean_diff` is simply `mean_activations_harmful - mean_activations_harmless`, accumulating the mean in high-precision float64 to avoid numerical error. The training sets $\mathcal{D}_{\mathrm{harmful}}$ and $\mathcal{D}_{\mathrm{harmless}}$ each contain only 128 instructions; harmful ones are drawn from AdvBench, MaliciousInstruct, TDC2023, and harmless ones from Alpaca.

### Selecting the "single direction" from a pool of candidates

Computing an $\mathbf{r}_i^{(l)}$ for every layer and position yields $|I| \times L$ candidates. The authors use 32 validation instructions and compute three scores for each candidate: `bypass_score` (the mean refusal metric on the harmful validation set after ablating that direction — the lower, the better it bypasses refusal), `induce_score` (the mean refusal metric on the harmless validation set after adding that direction — the higher, the better it induces refusal), and `kl_score` (the mean KL divergence of the last-token distribution on the harmless set before vs. after ablation — the lower, the less it disturbs normal behavior). The selection rule is: among candidates satisfying `induce_score > 0` (the paper states it as strictly greater than zero), `kl_score < 0.1`, and source layer `l < 0.8L` (avoiding being too close to the unembedding and degenerating into merely blocking specific output tokens), pick the one with the lowest `bypass_score`. The official `select_direction.py`'s `filter_fn` uses the same set of threshold parameters `kl_threshold=0.1`, `induce_refusal_threshold=0.0`, `prune_layer_percentage=0.2` (dropping directions from the last 20% of layers), but in implementation it excludes candidates with `steering_score < induce_refusal_threshold` (i.e. `< 0`); so strictly speaking, a boundary candidate whose `induce_score` is exactly 0 is kept by the code but excluded by the paper's strict inequality. This is a tiny difference at the threshold boundary and in practice has almost no effect on the selected direction (the probability that a floating-point score lands exactly on 0 is extremely low).

Take Llama-3 8B Instruct as a walkthrough: the model has 32 layers, $d_{\mathrm{model}}$ corresponds to its architecture, and the direction the authors finally select comes from the 5th-to-last token position ($i^*=-5$) and layer 12 ($l^*=12/32$), corresponding to `bypass_score = -9.715`, `induce_score = 7.681`, `kl_score = 0.064`. The `direction_metadata.json` shipped with the official repo records exactly `{"pos": -5, "layer": 12}`, consistent with the paper's table. The table below lists the selection results for three representative models:

| Chat model | $i^*$ | $l^*/L$ | bypass_score | induce_score | kl_score |
|-|-|-|-|-|-|
| Llama-3 8B | -5 | 12/32 | -9.715 | 7.681 | 0.064 |
| Qwen 72B | -1 | 62/80 | -4.246 | 1.885 | 0.034 |
| Gemma 2B | -2 | 10/18 | -14.435 | 6.709 | 0.067 |

Plotting out the entire candidate space makes it visually clear why pos -5, layer 12 gets chosen. In appendix Figure 11 the authors scan `bypass_score` (left) and `induce_score` (right) layer-by-layer and position-by-position for Llama-3 8B: the horizontal axis of both subplots is the source layer (0–31), and each line corresponds to a post-instruction position (pos -5 to -1). The `bypass_score` reaches its deepest, about -10, at layer 12, pos -5 (blue line) — the lower, the better it bypasses refusal — while the `induce_score` at the same point is also near its peak of about 8 (the higher, the better it induces refusal). The two conditions meet at the same (position, layer), landing exactly on the selected candidate.

![Figure 11 (left): the layer-by-layer bypass_score curves of each candidate direction $\mathbf{r}_i^{(l)}$ for Llama-3 8B Instruct, each line a post-instruction position. pos -5 (blue) drops to its minimum (about -10) near layer 12, corresponding to the candidate best able to bypass refusal after ablation.](imgs/fig11a_bypass_scan.png)

![Figure 11 (right): the layer-by-layer induce_score curves for the same model. Multiple positions spike to about 8–10 between layers 10–14; pos -5 is still near its peak at layer 12, coinciding with the best bypass point in the left plot, so the selected direction has both necessity and sufficiency.](imgs/fig11b_induce_scan.png)

It is worth noting that this extraction procedure is "not" an overall optimization of the whole model, but rather picks a single candidate under a set of heuristic thresholds; the authors themselves regard this as an existence proof that "such a direction exists," rather than a definitive account of "how best to extract it."

### Two inference-time causal interventions

Having a single direction, the authors use two symmetric interventions to test its "necessity" and "sufficiency."

The first is **directional ablation**: projecting out the unit vector $\hat{\mathbf{r}}$ from every residual stream activation,

$$
\mathbf{x}' \leftarrow \mathbf{x} - \hat{\mathbf{r}}\,\hat{\mathbf{r}}^{\top}\mathbf{x}.
$$

This operation is applied "at all layers and all token positions," equivalent to making the model henceforth unable to represent this direction in the residual stream. The official hook is a single line, `activation -= (activation @ direction).unsqueeze(-1) * direction`, first normalizing `direction` and then projecting-and-subtracting. This tests "necessity": if erasing it makes the model stop refusing, this direction is a necessary component of refusal.

The second is **activation addition**: adding the (unnormalized, original-length) $\mathbf{r}^{(l)}$ back into the layer-`l` activation,

$$
\mathbf{x}^{(l)\prime} \leftarrow \mathbf{x}^{(l)} + \mathbf{r}^{(l)}.
$$

This is applied only at the layer where the direction was extracted, but at all token positions. It tests "sufficiency": if adding this direction to a harmless input can induce refusal, this direction is enough to trigger refusal.

Why must these two interventions be divided this way, rather than uniformly using "adding the negative direction" to bypass refusal? In appendix Figure 22 the authors use Gemma 7B IT to plot the cosine similarity between the last token at each layer and the refusal direction to explain. With no intervention, harmful (red solid line) pulls the direction expression up to about 0.42 near layer 14, while harmless (green solid line) stays near 0. If one instead uses "adding the negative direction" (act add, dashed) to jailbreak: the direction expression of harmful is indeed pushed back near 0, but harmless is pushed to about -0.45 — far from its original distribution, becoming an out-of-distribution state never seen during training, which in turn raises the CE loss on harmless inputs. By contrast, directional ablation (black dotted line) projects both groups to 0 and keeps them within the distribution, which is why the authors choose it as the main bypass surgery.

![Figure 22: the cosine similarity between the last token at each layer and the refusal direction for Gemma 7B IT (128 harmful/harmless prompts). Solid lines are no intervention, dashed lines are activation addition in the negative direction, black dotted line is directional ablation. act add pushes harmless (green dashed) to the out-of-distribution region around -0.45, while directional ablation pulls both groups back to 0.](imgs/fig22_act_add_ood.png)

### What it means to equivalently bake "erasing the direction" into the weights

Directional ablation is an inference-time intervention, but in the appendix the authors give an algebraic derivation proving it can be equivalently baked into the weights. Conceptually, the approach is to orthogonally remove the refusal direction from the output of every "output matrix that writes into the residual stream" (the embedding, the attention output projection, the MLP down projection, etc.), yielding a purely statically modified model.

As long as all preceding writes have been treated this way, this weight version is fully equivalent to the layer-by-layer, position-by-position inference-time ablation. This equivalence first has a scientific benefit: since both give the same result, the numbers measured on the residual stream for the earlier "bypass refusal" set also precisely characterize the effect of the weight version, without needing to run a separate evaluation for it.

For safety, this is the sharpest consequence of the "single direction" conclusion. Mathematically, this modification is merely a rank-one low-rank edit of the output matrices along a single direction, requiring no gradient optimization, and the authors accordingly estimate that the compute cost of modifying a 70B open-weight model so that it almost never refuses is under $5; in other words, safety fine-tuning may be held up by only a very thin layer of linear representation, and the bar for a white-box attacker holding the weights to remove it is extremely low — a robustness gap worth being wary of. This note explains only at the mechanistic level "why the equivalence holds and what risk it reveals," and deliberately does not restate the official implementation's weight-modification function or reproducible operational steps; to understand reproducibility details, one should return to the source code of the official repository and the paper's appendix, rather than treating this section as an operating manual for removing a model's safety mechanisms.

### How "refusal" and "harmful" are judged

Evaluation uses two complementary metrics. The **refusal score** is string matching: it records 1 as long as the completion contains any "refusal substring" (such as `"I'm sorry"`, `"I cannot"`, `"As an AI"`, and 12 such phrases, case-insensitive, appearing at any position). The **safety score** uses Meta Llama Guard 2, an open-source model specialized in detecting harmful content, recording 1 for `safe` and 0 for `unsafe`. The two are complementary because looking at strings alone misjudges: the appendix gives three mismatch cases — the model gives no refusal phrase but also no harmful content (refusal=0, safety=1), the model first says "I cannot" but then provides attack steps (refusal=1, safety=0), and so on. All generations used for evaluation use greedy decoding with a maximum length of 512 tokens (following HarmBench's setting), making the numbers reproducible. In addition, the authors define a `refusal_metric` proxy that does not require actual generation and only looks at the refusal-token probability of the last token, used to quickly sweep the validation set for direction selection. Take Gemma 2B IT as an example: setting the refusal token set to $\mathcal{R}_{\mathrm{Gemma}} = \{235285\}$ (the paper's Table 4 maps this token ID to the refusal-opening string `"I cannot"`), and looking at the log-odds of that token relative to other tokens: the scores of harmful instructions almost all fall above 0 (most near +5), while harmless instructions concentrate below -10, the two groups barely overlapping — proving this proxy metric is enough to judge whether the model will refuse without generating a full response. (Here we adopt the official implementation `pipeline/model_utils/gemma_model.py`'s `GEMMA_REFUSAL_TOKS = [235285]`, consistent with the paper's Table 4 mapping `235285 → "I cannot"`. Note that the caption of the paper's Figure 10 additionally cites a different token ID `234285` and maps it to the single character `I`, which does not match Table 4 / the code — an internal inconsistency in the original text; this note consistently follows the executable implementation value `235285` and Table 4's string mapping, and does not adopt Figure 10's `I` claim.)

![Figure 10: the refusal_metric distribution of harmful (red) and harmless (blue) instructions on Gemma 2B IT. Computed as the log-odds of the refusal-opening token, harmful concentrates near +5 and harmless near -15, showing a clear bimodal separation.](imgs/fig10_refusal_metric_gemma.png)

### Evidence: necessity, sufficiency, attack strength, and capability preservation

**Necessity (Figure 1)**: ablating the direction across 13 models greatly lowers the refusal rate, and the safety score collapses along with it. Take Llama-3 8B as an example: with no intervention the refusal score is 0.95 and the safety score 0.97; after directional ablation the two drop to 0.01 and 0.15 respectively, meaning it is not merely "not saying refusal phrases" but actually emitting unsafe content.

![Figure 1: after erasing the refusal direction, the refusal score (orange) and safety score (blue) of 13 models on JailbreakBench's 100 harmful instructions drop greatly (hatched bars are directional ablation, solid are no intervention).](imgs/fig1_bypass_refusal.png)

**Sufficiency (Figure 3)**: the reverse operation — adding the direction to 100 Alpaca harmless instructions — pulls almost every model's refusal rate close to 1 (Llama-3 70B is a clear exception, reaching only about 0.3). This is consistent with the particularly low `induce_score` (0.126) of its selected direction.

![Figure 3: after activation addition on harmless instructions, most models' refusal score is pulled close to 1, with Llama-3 70B as a notable exception.](imgs/fig3_induce_refusal.png)

**Attack strength (HarmBench ASR)**: weight orthogonalization (labeled Ortho) achieves an ASR on HarmBench's 159 standard behaviors comparable to or even better than other general jailbreaks; on the Qwen family, this "general" method even approaches the per-prompt-optimized, prompt-specific method GCG. The table below excerpts a few rows (the parenthesized value is the ASR without the system prompt):

| Chat model | Ortho | GCG-M | GCG-T | DR |
|-|-|-|-|-|
| Llama-2 7B | 22.6 (79.9) | 20.0 | 16.8 | 0.0 |
| Qwen 7B | 79.2 (74.8) | 73.3 | 48.4 | 7.0 |
| Qwen 14B | 84.3 (74.8) | 75.5 | 46.0 | 9.5 |

Llama-2 is very sensitive to the system prompt (22.6% with prompt vs. 79.9% without), while Qwen is almost unaffected; the authors' supplementary analysis notes this difference need not stem only from the prompt content and may reflect different families' responses to system-level instructions.

**Capability preservation**: running the models before and after modification on MMLU, ARC, GSM8K, TruthfulQA, the orthogonalized model is almost identical to baseline on the first three (most falling within the 99% confidence interval), with only TruthfulQA consistently declining. Take Llama-3 70B as an example: MMLU 79.8 vs. 79.9, GSM8K 90.8 vs. 91.2, but TruthfulQA 59.5 vs. 61.8 (a drop of 2.3). The authors speculate that TruthfulQA contains subject matter close to the "refusal boundary" such as misinformation and conspiracy theories, so behavior naturally changes once the safety guardrail is removed.

| Chat model | MMLU | ARC | GSM8K | TruthfulQA |
|-|-|-|-|-|
| Llama-3 70B | 79.8 / 79.9 (-0.1) | 71.5 / 71.8 (-0.3) | 90.8 / 91.2 (-0.4) | 59.5 / 61.8 (-2.3) |
| Qwen 72B | 76.5 / 77.2 (-0.7) | 67.2 / 67.6 (-0.4) | 76.3 / 75.5 (+0.8) | 55.0 / 56.4 (-1.4) |

### How the adversarial suffix suppresses this direction

Finally the authors use this same direction to "mechanistically" explain the adversarial suffix (GCG-style attacks that append a string of garbage at the end of the instruction to jailbreak). They find a universal suffix for Qwen 1.8B Chat and run each of 128 harmful instructions three times: the original instruction, with the adversarial suffix appended, and with an equal-length random suffix appended. Measuring the cosine similarity between the last token's activation and the refusal direction (Figure 5): the direction expression of harmful instructions is very high, still high with the random suffix appended, but greatly lowered once the adversarial suffix is appended, almost indistinguishable from harmless instructions. Further using direct feature attribution (DFA) to analyze the top eight attention heads that write the most into this direction, they find that the adversarial suffix "hijacks" the attention of these heads — originally attending to the harmful instruction region, the attention shifts to the suffix region after the suffix is appended, leaving the harmful content.

![Figure 5: the cosine similarity between the last token at each layer and the refusal direction for Qwen 1.8B Chat. The adversarial suffix (red) suppresses the direction expression close to harmless (green), while the random suffix (orange) remains as high as harmful (blue).](imgs/fig5_adv_suffix_cosine.png)

DFA decomposes this suppression down to the head level: the top eight attention heads that write the most into the refusal direction — the sum of their outputs projected onto that direction — is about 4.9 with no suffix, still about 3.8 with a random suffix, but suppressed to about 1.35 once the adversarial suffix is appended — the "write source" of the direction is indeed shut off.

![Figure 6a: the top eight attention heads with the highest DFA contribution in Qwen 1.8B Chat, and the value of their projection into the refusal direction at the last token position (stacked by head). no_suffix is about 4.9, random_suffix about 3.8, adv_suffix suppressed to about 1.35.](imgs/fig6a_head_dfa_suppression.png)

Looking further at where these heads' attention goes: with a random suffix, they still focus attention on the instruction region (total weight about 4.6), with only about 1.4 landing on the suffix; once switched to an adversarial suffix, the attention is almost entirely "hijacked" to the suffix region (about 3.7), with only about 0.55 left on the instruction region. The heads no longer read the harmful instruction, so the direction naturally cannot be written in.

![Figure 6b: the attention of the top eight key heads from the last token position toward the instruction region and the suffix region (stacked by head). With the random suffix (left) attention is mainly on the instruction (about 4.6), while with the adversarial suffix (right) it flips to the suffix (about 3.7).](imgs/fig6b_attention_hijack.png)

## 🧪 Critical Assessment

### Is the problem real, and is the contribution as advertised?

"Understanding the internal mechanism of refusal" is a real problem, and it does have safety implications: the authors turn an interpretability insight into a concrete attack that jailbreaks a 70B model for under $5, a beautiful demonstration that "model internals have practical value." But we should be honest about the novelty: difference-in-means, activation steering, directional ablation, and baking steering into the weights — these individual techniques all existed before (the authors cite them extensively too). The real contribution of this paper is not inventing a new mechanism, but rather (1) systematically validating the single direction across 13 models with bidirectional causal interventions of "erasing" and "injecting," and (2) equivalently reducing ablation to an extremely simple rank-one weight surgery. This is a strong integration and existence proof, not the discovery of an entirely new principle — the authors also concede in the limitations that it is more like an existence proof.

### To what degree is "mediation by a single direction" proven?

Both necessity (no refusal after ablation) and sufficiency (refusal after addition) are backed by causal interventions, stronger than pure correlation. But the title "single direction" is easily over-interpreted. The direction is the "single most effective" candidate picked from $|I|\times L$ candidates using 32 validation instructions under three heuristic thresholds, and the authors state plainly that the extraction method is "likely not optimal and relies on several heuristics." What this supports is that "there exists a one-dimensional direction sufficient to mediate refusal," not that "refusal geometrically occupies only one dimension with no other redundant or alternative directions." Moreover the semantics of the direction remain undetermined: in the limitations the authors admit it may represent "harm" or "danger," or may not even be intuitively semanticizable, and "refusal direction" is merely a functional name.

### Are the baselines, metrics, and data strong enough?

The evaluation design has clear self-defined components. The refusal score is substring matching, and the authors themselves demonstrate three misclassification cases in the appendix; the safety score hands the judging over to another model, Llama Guard 2, whose biases enter the conclusions directly; direction selection also relies on a self-defined `refusal_metric` proxy. These all have the flavor of "self-designed evaluation": not fabrication, but the metrics and the validated direction come from the same design, carrying a certain circularity risk. The validation set is only 32 items and is a "pick the best candidate" selective procedure, naturally biased toward finding a direction that looks effective. HarmBench ASR is highly sensitive to the system prompt (Llama-2 7B 22.6% vs. 79.9%), meaning "how strong the attack is" depends to a large extent on the choice of evaluation setup. The consistent decline of TruthfulQA is also a reminder: the so-called "capability is almost unaffected" has exceptions, and that exception happens to fall on the subject matter most related to safety.

### The boundary of generalization scope and real-world significance

The authors state it very clearly in the limitations, and it is worth transcribing faithfully: the conclusions may not generalize to untested models, especially larger-scale, currently strongest closed-source models or future models; the mechanistic analysis of the adversarial suffix was moreover done only on "a single model (Qwen 1.8B), a single suffix," and the authors concede it is hard to find a suffix universal across prompts and models. Therefore this paper's attack surface mainly falls in the open-weight scenario — requiring white-box weight access. A boundary especially critical for this task is: this result cannot be used in reverse as a "weight cause" for "any quantized model or deployed model not internally examined." To determine whether the refusal of a specific model (e.g. a Muse-Glimmer-type system with only dialogue outputs) is also mediated by some direction, one must perform activation-level internal-access experiments on "that model itself"; inferring its weights or refusal direction from dialogue outputs is an extrapolation not supported by this paper's evidence.

## One-Minute Wrap-Up

- **Refusal is mediated by a single direction**: a model's refusal of harmful instructions is internally mediated by only a one-dimensional direction in the residual stream. Example: after erasing this direction from Llama-3 8B, the safety score on harmful instructions drops from 0.97 to 0.15.
- **White-box jailbreak is very cheap**: equivalently writing "erase the direction" into the weights (weight orthogonalization) becomes an attack requiring no gradient optimization. Example: with just a batch of harmful "instructions," the authors estimate the compute cost of modifying a 70B model at under $5.
- **"Single direction" is existence, not geometric uniqueness**: what this paper proves is that "there exists a direction sufficient to mediate refusal," not that "refusal geometrically occupies only one dimension." Example: this direction is the "single most effective" candidate picked from $|I| \times L$ candidates using 32 validation instructions under three heuristic thresholds, and the authors concede the extraction method is not necessarily optimal.
- **Capability preservation comes at a cost**: after removing the safety guardrail, regular capabilities barely change, but subject matter close to the "refusal boundary" regresses. Example: after modifying Llama-3 70B, MMLU barely moves (79.8 vs. 79.9), but TruthfulQA drops from 61.8 to 59.5.

## 🔗 Related notes

- [SAE-Feature-Consistency](../SAE-Feature-Consistency/) — likewise built on the assumption that "features are represented as linear directions," but instead uses a sparse autoencoder to extract features unsupervised, which can be contrasted with this paper's supervised difference-in-means extraction method.
