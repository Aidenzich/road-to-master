# Layer-Condensed KV Cache — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Layer-Condensed KV Cache for Efficient Inference of Large Language Models |
| Venue | ACL 2024 (Volume 1: Long Papers) |
| Year | 2024 |
| Authors | Haoyi Wu, Kewei Tu |
| Official Code | https://github.com/whyNLP/LCKV |
| Venue Kind | paper |

This note is written using the arXiv full text (`2405.10637`) as its evidence source; the formally published version is an ACL 2024 main-conference long paper (pp. 11175–11188), and where the two differ in detail, the formal version prevails. The authors are from ShanghaiTech University.

## First Principles

### Why the KV cache is a deployment bottleneck

During autoregressive generation, the Transformer stores the key/value of each layer and each already-generated token into the KV cache to avoid recomputation. Its memory footprint is simultaneously proportional to sequence length and to the number of layers: the paper cites existing measurements pointing out that the KV cache can occupy over 30% of GPU memory at deployment. For deep models, the multiplier of "number of layers" is especially deadly—LLaMA-7B has 32 layers, 30B has 60 layers, and each additional layer stores one whole layer's worth of K and V more.

Writing the KV cache's memory as a rough proportionality relation makes clear the two axes one can act on:

$$
\text{KV cache memory} \;\propto\; L \times n \times d_{\text{kv}}
$$

where $L$ is the number of layers, $n$ is the sequence length, and $d_{\text{kv}}$ is the KV dimension per token per layer (this expression is an arrangement written by this note for illustration, not the paper's original formula). The vast majority of past work has been compressing the $n$ axis (compressing the prompt, keeping only the initial and recent tokens, evicting tokens by attention score). This paper's angle is orthogonal to these methods: directly cutting away the $L$ multiplier.

### Core mechanism: the queries of all layers pair only with the top layer's KV

This paper proposes a new Transformer decoder variant: having **the queries of all layers pair with the key/value of "only the top layer,"** rather than each layer pairing with the KV of its own layer. This way, layers other than the top layer do not need to cache—or even compute—KV at all; correspondingly, the weights $W_K, W_V$ that map the hidden representation to K, V can also be discarded in these layers, so memory, compute, and model parameter count all decrease simultaneously. The authors' intuition comes from interpreting the Transformer's stacked structure as an iterative process of "refining the token representation layer by layer"—the top-layer representation has the highest information content, so having all layers attend to the top layer is reasonable; this also echoes the encoder–decoder design in which all decoder layers do cross-attention over the top-layer encoder output.

![LCKV's computation graph: non-top layers no longer compute or cache their own KV (the gray nodes in the figure), and the queries of all layers point along the diagonal edges to the top layer's KV; the diagonal is masked, and each token does not attend to itself.](imgs/architecture.png)

### Circular dependency and the diagonal mask

This design has a chicken-and-egg problem: since each token's attention at the lower layers also uses "its own top-layer KV," but the top layer can only be computed after all lower layers finish, a circular dependency forms. The authors' solution is straightforward—mask the diagonal of the attention matrix so that each token no longer attends to itself; the first token of the sequence thus has nothing to attend to, and a zero vector is used as its dummy KV. Because the residual connection still exists, the token's own information can still be carried in during the bottom-up computation, and the paper's experiments find that this diagonal mask has almost no effect on performance.

### Warmup layers and the sandwich configuration

Using only the above design, the performance of the language model and downstream tasks would lag behind the standard Transformer. The authors observe that the Transformer's lower layers lean toward syntax and higher layers toward semantics, and feeding the same layer's KV to all layers may break this division of labor; therefore they keep a few layers with standard attention, called **warmup layers**, and only apply the "share the top-layer KV" approach to the remaining layers. They further propose a sandwich configuration that splits the warmup layers into "the top $w/2$ layers + the bottom $w/2$ layers," and experiments show this arrangement outperforms putting all warmup layers at the bottom or all at the top, with almost no performance loss relative to the standard Transformer.

### Making the circular dependency trainable in parallel

At inference, this method is almost identical to a standard Transformer (decoding one token at a time from left to right), but training is much more troublesome: since each token depends on the top-layer KV of preceding tokens, it cannot be trained in parallel over the whole sequence the way a standard Transformer is. The authors rewrite the original computation graph of "doing $n$ bottom-up computations sequentially over $n$ tokens" into an equivalent computation graph of "doing $n$ iterations simultaneously over all tokens, where each iteration pairs 'the top-layer KV from the previous iteration' with this iteration's queries of all layers," and prove by induction that the two are equivalent for training (from the $i$-th iteration onward, the $i$-th token is exactly consistent with the sequential version). This rewrite swaps the original graph's horizontal dependency for a vertical one, and the longest dependency chain length is unchanged, so $n$ iterations are still needed—the real speedup comes from the two-cut reduction of the iteration count that follows.

### Backpropagation only propagates through the last $b$ iterations

The loss is computed only at the last iteration, and if it were backpropagated through all $n$ iterations, the computation graph would be too large to fit on the GPU. The authors emulate Transformer-XL's gradient stopping, letting the loss backpropagate only through the last $b$ iterations ($b \ll n$). There is a detail here: the KV used in the last iteration comes from the second-to-last iteration, so if $b=1$ the gradient cannot reach the parameters that compute the KV at all, and performance degrades substantially; experiments show that $b \geq 2$ is enough to match the standard Transformer, so the default is $b=2$.

### Forward propagation relies on the KV's fast convergence for another cut

After gradient stopping, the first $n-b$ iterations are used only for forward computation to feed the KV to the last $b$ iterations. The authors' key observation is: the KV converges very fast across iterations, and one need not actually run the full $n-b$ iterations. Using a randomly initialized model with the same configuration as TinyLlama and an input of 2048 tokens, they measure the mean squared error of the KV between adjacent iterations, and find that the KV converges within just dozens of iterations, and the more warmup layers, the faster the convergence; therefore they switch to approximating with $m$ iterations ($m \ll n$), and experiments show $m=7$ is already sufficient, with no further improvement from increasing it.

![Adjacent-iteration MSE of the KV on a randomly initialized model: the more warmup layers, the faster the convergence, dropping to very small within dozens of iterations.](imgs/kv-convergence.png)

### Handling the prompt

Generation itself is straightforward, but this method cannot encode the prompt in parallel the way a standard Transformer does (for the same reason it cannot train in parallel). Fortunately the KV converges fast, and it suffices to iterate over the prompt $m+b$ times; since $m+b$ is usually far smaller than the number of tokens to be generated, the extra overhead of encoding the prompt is negligible.

### A walkthrough with real numbers

Take the paper's 1.1B model (with configuration aligned to TinyLlama: 22 layers, hidden size 2048, 32 attention heads, 4 KV heads, vocabulary 32000, training length 2048) and $w=10$ as an example: the sandwich configuration keeps the topmost 5 layers and the bottommost 5 layers as standard attention, and the middle 12 layers no longer compute or cache their own KV, all switching to the shared top-layer KV; training and prompt encoding each use $m+b=7+2=9$ iterations. The table below excerpts the maximum batch size and throughput on the A100 (80GB) and RTX 3090 (24GB) ($x+y$ denotes prompt length $x$, generation length $y$, and the multiplier is relative to the standard Llama):

| GPU | Model Size | Seq. Length | Llama batch | Ours $w{=}2$ batch | Ours $w{=}10$ batch | Llama throughput | Ours $w{=}2$ throughput | Ours $w{=}10$ throughput |
|-|-|-|-|-|-|-|-|-|
| A100 | 30B | 2048+2048 | 1 | 32 (32×) | 8 (8×) | 14.10 | 108.29 (7.7×) | 77.65 (5.5×) |
| RTX 3090 | 30B (CPU-offload) | 512+1024 | 4 | 83 (20.8×) | 23 (5.8×) | 0.23 | 5.99 (26.0×) | 1.63 (7.1×) |
| RTX 3090 | 7B | 5+2043 | 5 | 64 (12.8×) | 16 (3.2×) | 140.88 | 534.02 (3.8×) | 315.38 (2.2×) |

The abstract's claim of "up to 26× throughput, up to 32× batch" corresponds exactly to the corners of this table: the 26.0× throughput comes from the 512+1024 setting of 30B on the RTX 3090 relying on CPU-offload ($w=2$, 0.23→5.99 token/s), and the 32× batch comes from the 2048+2048 setting of 30B on the A100 ($w=2$, batch 1→32). In the same row, the batch is scaled up 32× but the throughput is scaled up only 7.7×, showing that throughput does not grow linearly with batch.

On the quality side, the from-scratch pre-trained 1.1B model (a 100B-token subset of SlimPajama) is compared against TinyLlama:

| Model | Dev ppl. | HellaSwag | WinoGrande | ARC-e | BoolQ | PIQA | Avg |
|-|-|-|-|-|-|-|-|
| TinyLlama | 9.219 | 44.58 | 50.99 | 46.38 | 60.46 | 68.93 | 46.65 |
| Ours ($w=2$) | 9.746 | 42.22 | 49.64 | 43.10 | 61.38 | 66.49 | 45.45 |
| Ours ($w=10$) | 9.265 | 44.74 | 51.70 | 46.38 | 61.38 | 67.90 | 46.84 |

One can see that $w=10$ is almost lossless (the average 46.84 is even slightly higher than TinyLlama's 46.65), while $w=2$ regresses slightly on most tasks but trades for higher throughput. If the KV cache memory is approximated as the number of layers cached, $w=2$ needs to cache only about 3 of the 22 layers (saving about 86%), and $w=10$ needs to cache about 11 layers (saving about 50%)—this conversion is derived by this note itself according to the sandwich configuration, to illustrate that the trade-off between "saving memory" and "preserving quality" is continuously controlled by $w$.

## 🧪 Critical Assessment

### Whether the problem is real and important

The proportion of deployment memory occupied by the KV cache is indeed considerable, and its memory growing linearly with the number of layers is a hard bottleneck for deep models, so "cutting the number of layers"—an axis orthogonal to compressing sequence length—is a valuable angle. More crucially, this method not only saves cache but also simultaneously saves the KV computation and the $W_K, W_V$ parameters of the non-top layers, which lets it, beyond batch scaling, also lower latency at the same batch size (the appendix's latency experiments support this point). The problem setting itself holds up.

### Whether baselines, ablations, data, and metrics are sufficient

Evidence strength is where this paper most needs discounting. The only scale that is truly "pre-trained from scratch and compared on quality" is a single 1.1B, and the baseline is only the single line of TinyLlama; 7B and 30B measured only throughput and latency, with no quality numbers at all, so "equally lossless on large models" has no empirical evidence and is only extrapolation. The downstream evaluation is zero-shot accuracy on seven commonsense-reasoning tasks, whose number and difficulty are both light, and absolute scores like HellaSwag's roughly 44 also show the model itself is weak, making it hard to assert from this that the method remains lossless on strong models. The ablations (sandwich arrangement, number of warmup layers, $m$, $b$, KV loss) are relatively solid and are a plus of this paper.

### Is it a renaming or genuinely something new

The authors themselves point out the high similarity to the Feedback Transformer—the latter likewise aggregates the representations of all layers (including using only the top layer) as memory, the difference being that Feedback's sequential training is impractical for large models. Therefore this paper's real new contribution is not in the idea of "sharing the top-layer KV" itself, but in two pieces of engineering that make it scalable: the parallelizable iterative training (with a proof of equivalence) and the approximation of "the KV converges fast → run only $m=7$ iterations." Positioning the contribution as "making an old idea trainable and scalable" is honest, but it also means the method's novelty is mainly at the systems/training level rather than the representation level.

### Whether the throughput numbers are measured in a way biased toward itself

The way the headline multipliers are obtained warrants caution. Several throughput settings (such as 5+8187, 512+1024) have sequence lengths exceeding the model's actual training length (the 1.1B's training length is only 2048), and the paper admits that "some sequence lengths exceed the training limit, but this does not affect the batch and throughput measurements"—the problem is that these corner settings are simultaneously used to report the most eye-catching batch/throughput multipliers, while the generation quality under those lengths is not presented alongside. The 26× headline also relies on the CPU-offload scenario of 30B (where the standard baseline throughput is pressed down to 0.23 token/s, and only such a tiny denominator sustains the large multiplier). This is a way of defining the comparison baseline using an operating point favorable to itself: the multipliers themselves are not fake, but the chosen operating points amplify the advantage and also dodge the quality check under the same setting.

### Is the problem really solved, and how much practical significance does it have

"Almost lossless" holds only at $w=10$, and $w=10$ already halves the saved memory (saving about 50% rather than $w=2$'s about 86%); to obtain the maximum memory saving one must accept the visible regression of $w=2$, so the trade-off has not disappeared, it is merely parameterized by $w$. The cost side is not cheap either: iterative training makes the pre-training time about 2.7–2.8× that of TinyLlama, and the authors defend this with "training is one-time, inference happens repeatedly," which is still reasonable but requires the user to weigh it themselves. What truly limits the practical applicability is the shortcoming the paper itself states: because the prompt also has to be encoded iteratively, when the prompt is much longer than the generation length (such as document summarization) the throughput degrades, and the method is better suited to scenarios with large generation lengths (translation, dialogue, CoT). Overall this is a clearly positioned method that does have value for "long-generation, deployment-heavy" scenarios, but reading it as "cutting away the KV cache at no cost" would be overly optimistic.

## 🔗 Related notes

- [Attention is all you need](../AttentionIsAllYouNeed/)
