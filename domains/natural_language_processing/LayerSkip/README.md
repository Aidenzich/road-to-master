# LayerSkip — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding |
| Venue | ACL 2024 (Volume 1: Long Papers) |
| Year | 2024 |
| Authors | Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agrawal, Ahmed Roman, Ahmed A Aly, Beidi Chen, Carole-Jean Wu |
| Official Code | https://github.com/facebookresearch/LayerSkip |
| Venue Kind | paper |

This note uses the arXiv full text `2404.16710` corresponding to the official ACL 2024 version (aclanthology `2024.acl-long.681`) as its evidence source; both are the same long paper from the Meta FAIR team. All numbers and citations are taken from the LaTeX source to avoid PDF text-extraction errors.

## First Principles

### What this paper tries to solve: depth is latency

Every time an autoregressive LLM generates a token, it must run through all $L$ transformer layers. The authors' core observation is that most tokens do not actually need to traverse every layer before their prediction is finalized. They fed Llama1 7B one HumanEval prompt and, using the same single LM head to unembed the output of every layer, found that on average a token only needs 23.45 layers (the model has 32 layers in total) to converge to its final prediction; in other words, even with a zero-overhead perfect "early-exit predictor," at most 26% of the computation could be saved. This shows that the model inherently spreads its compute evenly across layers and lacks any incentive to "decide early," so the authors aim to force the model to rely more on shallow layers from the training side.

![LayerSkip's end-to-end three-stage scheme: training (layer dropout + early-exit loss), early-exit inference, and self-speculative decoding](imgs/overview.png)

LayerSkip decomposes "acceleration" into three mutually reinforcing stages: a training recipe that makes the same set of weights equivalent to "an ensemble of submodels of different depths"; an early-exit inference that attaches the LM head directly at layer $E$ to emit tokens; and a self-speculative decoding scheme that uses shallow layers to draft and deep layers to verify and correct, recovering the accuracy lost to early exit. All three stages share the same model and the same single LM head, adding no auxiliary layer or extra draft model at all — this is its key difference from most early-exit / speculative-decoding work.

### Training recipe part one: layer dropout (make the model rely less on deep layers)

The first modification is to apply dropout to "entire layers" rather than to weights. At layer $l$ and iteration $t$, the transformer operation becomes:

$$x_{l+1,t} = x_{l,t} + M(p_{l,t}) f_l(x_{l,t})$$

where $M(p)$ is a Bernoulli mask that returns 0 with probability $p_{l,t}$ (the entire layer is skipped) and 1 otherwise. The key is that the dropout rate is not constant but increases exponentially with layer index — shallow layers are almost never dropped, deep layers are almost always dropped:

$$D(l) = e^{\frac{l\text{ln}2}{L-1}} - 1$$

The actual per-layer rate is $p_{l,t} = S(t)\,D(l)\,p_{max}$, where $p_{max}$ is the overall upper bound and $S(t)$ is a curriculum along the time axis. The authors found that when continuing from an existing pretrained model via continual pretraining or finetuning, the time axis needs no scaling ($S(t)=1$); but when pretraining from scratch, an exponential time curriculum $S_{exp}(t)=e^{\frac{t\ln 2}{T-1}}-1$ is required to obtain the best accuracy. This "low for shallow, high for deep" asymmetric design is precisely what forces the shallow layers to shoulder the prediction responsibility themselves.

### Training recipe part two: early-exit loss (make the same single LM head understand shallow layers)

The LM head is originally trained only to unembed the representation of the last layer, and cannot "understand" the embeddings of shallow layers. LayerSkip therefore, in the total loss, connects the output of every layer to the same single LM head, computes cross-entropy, and takes a weighted sum:

$$J(X,Y,t) = \sum_{l=0}^{l=L-1} \tilde{e}(t,l)J_{\text{CE}}(g(x_{l+1}),Y)$$

$\tilde{e}(t,l)$ is the normalized per-layer weight (the layers sum to 1), where $e(l)$ gives quadratically growing weight to deeper layers (deep-layer prediction is easier, so it is penalized more heavily), and $C(t,l)$ is the curriculum that decides "whether to enable early exit at layer $l$ at iteration $t$." The authors experiment with two curricula: rotational (open one early exit every $R$ layers, rotating iteration by iteration, doing only $\lceil L/R \rceil$ unembeddings per step) and gradual (from the last layer backward, opening one more layer every $T/2L$ iterations). They deliberately do not enable early exit at all layers simultaneously, because that would slow training and hurt the accuracy of the last layer.

![A single training run yields a family of submodels of different depths that share weights](imgs/motivation.png)

It is worth emphasizing that, unlike early-exit work such as DepthAdaptive and CALM, LayerSkip does not add a dedicated LM head to each layer, nor does it add any early-exit module; instead, all layers share the same single LM head. This makes training faster, saves both training and inference memory, and makes deployment and maintenance simpler.

### Inference stage: early exit and self-speculative decoding

Early-exit inference itself is straightforward: when generating each token autoregressively, only run the first $E$ layers, take $g(x_E)$ directly as the model output, and skip the remaining layers. But plain early exit loses accuracy, so the authors use speculative decoding to recover that accuracy — this is exactly the most engineering-ingenious part of the paper.

![A comparison of autoregressive decoding, traditional speculative decoding, and this paper's self-speculative decoding](imgs/self_speculative_decoding.png)

Self-speculative decoding proceeds in two steps: Self-Drafting uses early exit (the first $E$ layers) to autoregressively draft $d$ tokens; Self-Verification uses "the remaining $L-E$ layers" to run one parallel forward pass over this batch of draft tokens, finds the first point of divergence between draft and verification, adopts the draft tokens before the divergence point together with the verification token at that point, and then continues from the drafting stage. Because drafting and verification traverse "the same model, the same layers in the same order," the computation of the first $E$ layers can be fully reused: the authors only need to maintain a single KV cache, plus introduce one exit query cache — storing only the query vector of layer $E-1$, so that the verification stage can continue directly from layer $E$ to the last layer. The authors call the union of the KV cache and the exit query the KVQ cache. Compared with Draft & Verify (Zhang et al. 2023), which skips middle layers and cannot reuse the drafting stage's activations and KV, this paper — because the two stages share the front-segment layers — can save this recomputation, and this is the source of its dual win in memory and latency.

### A concrete walk-through: Llama 1.5B (24 layers) on TOPv2

Finetuning Llama 1.5B (24 layers) on the TOPv2 semantic parsing data with LayerSkip ($p_{max}=0.2$, $e_{scale}=1.0$, gradual curriculum), the table below shows measured results of the same model under three decoding methods (8 speculations, greedy decoding, at most 80 tokens per instance):

| Generation | $E$ (early-exit layer) | EM | Token Acc. | Time/Token (ms) | Speedup |
|-|-|-|-|-|-|
| Autoregressive | – | 85.9% | – | 36 | 1.00× |
| Early Exit | 18 | 83.3% | – | 28 | – |
| Early Exit | 12 | 79.4% | – | 19 | – |
| Early Exit | 6 | 62.9% | – | 10 | – |
| Self Speculative | 18 | 82.9% | 98.9% | 29 | 1.24× |
| Self Speculative | 12 | 82.9% | 97.6% | 22 | 1.64× |
| Self Speculative | 6 | 82.9% | 76.0% | 18 | 2.0× |

Read it as follows: exiting directly at layer 6 takes only 10 ms per token (much faster than 36 ms), but EM collapses from 85.9% to 62.9% — this is the price of "exiting too early." Switching to self-speculative decoding and drafting at layer 6 as well, Token Acceptance is only 76.0% (about three out of every four draft tokens are accepted), but because the rejected tokens are verified and corrected by the remaining 18 layers, the final EM returns to 82.9%, while the average time per token drops to 18 ms, translating to 2.0× speedup. Contrast with layer 18: drafts are almost all accepted (98.9%), quality holds (82.9%) but there is only 1.24× speedup — because the drafting stage itself already runs 18/24 layers, saving little. This trade-off curve of "the shallower $E$, the greater the speedup but the lower the acceptance" is precisely the operating knob of the whole method.

### Early-exit accuracy and cross-task speedup

In the continual pretraining of Llama2 7B/13B, the early-exit quality of the middle layers (layer 16 / 20) improves substantially relative to baseline, especially on open-ended generation tasks: for example, NaturalQuestions on the middle layer of Llama2 7B rises from the baseline's 0% to 4%. Classification-type tasks (multiple-choice / yes-no) are inherently more tolerant of early exit than generation-type tasks, because only one token is scored and the candidates are only 2–4 rather than tens of thousands of dictionary entries; on a hard problem like MMLU, from the last layer to the middle layer in the Llama2 13B baseline it only drops from 55.2% to 49.2%. On end-to-end speedup, the authors report that self-speculative decoding achieves 1.34×–2.16×: Llama2 7B pretrained from scratch reaches 2.16× on CNN/DM summarization, code-finetuned Llama1 7B reaches 1.82× on HumanEval and 2.0× on TOPv2, and compared with Draft & Verify on a common setting, CNN/DM is clearly faster (1.81× vs. 1.5×) and XSUM slightly slower (1.34× vs. 1.48×).

## 🧪 Critical Assessment

### The 23.45/32-layer motivation: why intervening at the training side is unavoidable
LLM inference latency and memory cost are genuine deployment pain points, and "saving compute at the granularity of layers" — compared with quantization / sparsification — requires no custom kernel or hardware, so this angle does have value. The authors' own motivation analysis also honestly quantifies the upper bound: an existing model needs on average 23.45/32 layers, so a perfect predictor saves at most 26%; therefore the ceiling of the "add a predictor without changing training" early-exit route is very low, and one must reform from the training side — this argument makes the necessity of the method stand up, rather than being novel for novelty's sake.

### Coverage of four training regimes, and quality held by the verification stage
The evidence is relatively solid: it covers four training regimes — continual pretraining, pretraining from scratch, domain finetuning, and task finetuning — across multiple sizes of Llama1/2/3, and it directly compares against the similar Draft & Verify on common models and tasks. The quality of self-speculative decoding is verifiable — because the verification stage corrects with the full model, ROUGE-2 / EM is nearly on par with autoregressive (e.g., TOPv2's 82.9% vs. 85.9%), so the speedup numbers are not obtained by sacrificing quality. This is the most credible part of the paper.

### Last-layer degradation: a downplayed cost
The method is not without a bill to pay. After LayerSkip continual pretraining on Llama3 8B, the last-layer accuracy shows a clear decline (e.g., MMLU 66.5%→60.5%, HumanEval 37.8%→28.7%, GSM8K 54.2%→45.0%), even with more training tokens. The authors attribute the cause to "Llama3's shallow-layer perplexity being two to three orders of magnitude higher than Llama2's to begin with, and thus harder to compress," and instead recommend "just adopting this recipe when pretraining from scratch in the future." This explanation is reasonable but not directly verified, and rephrasing a regression as "motivation for future work" reads as somewhat downplaying an existing shortcoming: for a user who already has a Llama3 checkpoint and only wants to do continual pretraining, losing a few points on the last layer is a real cost; the paper's table honestly lists it, but the narrative clearly leans toward the gains on the early-exit layers.

### Context-dependence and potential cherry-picking of the speedup numbers
The 2.16× headline number comes from a Llama2 7B that was "pretrained from scratch on only 26B tokens," a relatively controlled and weaker model setting; whereas on the more practice-relevant continual pretraining, Llama2 13B on XSUM achieves only 1.34×, even losing to Draft & Verify's 1.48×. Speedup depends heavily on token acceptance, and acceptance in turn depends on task predictability and the chosen $E$ / $d$. The paper mostly reports the better $E$ under each setting, and readers can hardly judge whether these exit layers were selected in a targeted way; the absence of "a full sensitivity curve of acceptance versus $E$ and the worst case" leaves room for the headline numbers to be amplified by the most favorable setting. Moreover, everything is evaluated with greedy decoding and CNN/DM as 1-shot (deliberately aligned with Draft & Verify), so performance under sampling decoding or multi-turn dialogue remains unknown.

### The exit query cache stitches three old components into a self-consistent chain
Layer dropout, early exit, and speculative decoding are none of them firsts. LayerSkip's real contribution is to stitch them into a self-consistent chain: asymmetric layer dropout and the shared-LM-head early-exit loss make "the same model as a family of submodels" a reality, while the exit query cache / single KVQ cache lets drafting and verification genuinely share the front-segment computation — this point is a substantive mechanistic difference relative to Draft & Verify, which skips middle layers, not just a change of terminology. And it is compared head-to-head against existing public tasks and similar methods, rather than erecting a benchmark favorable only to itself around its own method's strengths; therefore it is more like an integration with clear engineering novelty, but it should also be honestly acknowledged that no single component alone could be called a breakthrough.

### Retraining cost: the applicability boundary for off-the-shelf-checkpoint users
Under the definition of "obtaining 1.3×–2× speedup with no obvious quality loss, and requiring no second model or extra memory," the paper largely achieves its claim, and it has open-sourced code and checkpoints, so practical usability is high. But "the problem is solved" comes with caveats: first, most of the strong speedups depend on training with LayerSkip from scratch or large-scale continual pretraining, so the barrier is not low for those who want to directly apply off-the-shelf weights; second, last-layer degradation truly exists on some models, and must be traded off against retraining cost. Overall the conclusion is credible but not universal — it is an acceleration scheme that holds "for those willing to pay the training cost."

## 🔗 Related notes

<!-- No safely resolvable related notes at present; the heading is retained, to be linked once related topics (e.g. speculative decoding, LoRA) are added. -->
