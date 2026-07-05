# Byte Latent Transformer — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Byte Latent Transformer: Patches Scale Better Than Tokens |
| Venue | arXiv preprint (2412.09871) |
| Year | 2024 |
| Authors | Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer (FAIR at Meta) |
| Official Code | https://github.com/facebookresearch/blt |
| Venue Kind | tech-report |

> Note: This note is based on the arXiv preprint `2412.09871` (FAIR at Meta technical report, `fairmeta` typesetting); the camera-ready version of the official conference may differ from this; all numbers and quotations in this note are taken from the preprint's LaTeX source. The venue tier is marked `unknown` here, because there is no citable ranking source in the ledger.

## First Principles

### The Problem: Tokenization Is the Only Stage Not Learned End-to-End

Modern large language models (LLMs) are trained almost entirely end-to-end, with the sole exception of tokenization—a preprocessing step that compresses a byte sequence into a fixed vocabulary using heuristic rules. This static vocabulary brings several structural side effects: sensitivity to domain/modality, fragility to input noise, lack of orthographic knowledge, and multilingual inequity. The reason people still could not do without tokenization is that training an LLM directly on raw bytes produces sequences that are too long, with prohibitive compute cost at scale.

The Byte Latent Transformer (BLT) argues: rather than making every token consume the same amount of compute, it is better to let the model "deploy compute only where it is needed." It dynamically splits bytes into variable-size **patches**, and the patch is the primary unit of computation; the split is based on the prediction entropy of the next byte, allocating more compute and model capacity only where the data is more complex. The authors claim this is the first work to conduct a FLOP-controlled scaling study of byte-level models, pushing all the way to 8B parameters and 4T training bytes, and claim to match tokenization-based models at scale for the first time.

### The Fundamental Difference Between Patch and Token

A "token" refers to a group of bytes selected from a finite vocabulary before training; a "patch" is a group of bytes split out dynamically, without a fixed vocabulary. The key distinction is: when using tokens, the model cannot directly access the underlying byte features; whereas patches retain access to byte information. More importantly, it redefines the "vocabulary size vs. compute" trade-off: in a standard LLM, enlarging the vocabulary means longer average tokens and fewer model steps, but the dimension of the final output projection layer also explodes. The paper gives the example that, to raise the average token from 3.7 bytes to 4.4 bytes, Llama 3 paid the cost of enlarging the embedding table 4× relative to Llama 2. Because BLT has no fixed vocabulary, it can freely enlarge the patch size without being constrained by this embedding inflation.

The paper compares several schemes for grouping bytes into patches (figure below): strided patching that cuts every k bytes (such as MegaByte), the BPE tokenizer, splitting by space-like bytes (SpaceByte), and this paper's entropy patching. Because each patch requires one expensive global transformer step, **the number of patches directly determines the primary FLOP overhead**, so the "average patch size" is the main factor determining training and inference cost.

![Comparison of various patching schemes](imgs/patching_types.png)

### Entropy Patching: Splitting by the Next-Byte Entropy of a Small Byte LM

BLT does not use a rule like "split whenever you hit a space," but instead finds "high-uncertainty" next-byte positions in a data-driven way. The authors first train a small byte-level autoregressive language model on BLT's training data, compute its distribution $p_e$ over the next byte on the byte vocabulary $\mathcal{V}$, and take its entropy:

$$H(x_i) = \sum_{v \in \mathcal{V}} p_{e}(x_i=v \mid \pmb{x}_{<i}) \log p_{e}(x_i=v \mid \pmb{x}_{<i})$$

With the entropy of each byte, there are two criteria for finding patch boundaries: one is entropy exceeding a global threshold $\theta_g$, and the other is the jump in entropy relative to the previous byte exceeding a threshold $\theta_r$ (an approximate monotonic constraint):

$$\text{Global:}\quad H(x_t) > \theta_g \qquad\qquad \text{Monotonic:}\quad H(x_t) - H(x_{t-1}) > \theta_r$$

This small entropy model in the experiments is a 100M-parameter, 14-layer transformer with hidden 512 and a sliding window of 512 bytes; when the receptive field is small enough, it can even be encoded as an efficient lookup table. The threshold $\theta_g$ is derived backward from the "desired average patch size," so patch size is a knob that can be freely chosen in BLT.

A key correctness property is **incremental patching**: at generation time the model must decide whether the current position is a patch boundary without having seen the subsequent bytes (because this determines whether to invoke the global transformer). Formally, the splitting function $f_p$ must satisfy the following, that is, the splitting of a prefix cannot be changed by future bytes:

$$f_p(\pmb{x}_{<i}) = f_p(\pmb{x})_{<i}$$

BPE does not satisfy this property—the same prefix will be split differently depending on the subsequent content—which is precisely entropy patching's advantage over tokenization in inference consistency.

### A Three-Stage Architecture: Two Lightweight Local Models Sandwiching One Heavyweight Global Model

![BLT consists of three modules: Local Encoder, Latent Transformer, and Local Decoder](imgs/blt_architecture.png)

BLT consists of three transformer blocks: a large global **Latent Transformer** (autoregressive, running on patch representations, using block-causal attention), and two lightweight byte-level local models ($l_{\mathcal{E}} \ll l_{\mathcal{G}}$, $l_{\mathcal{D}} \ll l_{\mathcal{G}}$). The Local Encoder encodes the byte sequence into patch representations, and the Local Decoder decodes the patch representations back into raw bytes. The global model consumes the vast majority of the FLOPs, so "when to invoke it" is the knob that controls compute allocation.

The Local Encoder also adds **hash n-gram embeddings** on top of each byte embedding: for each position $i$ it takes byte-grams of $n \in \{3,4,5,6,7,8\}$, maps them via a rolling polynomial hash into a fixed-size embedding table, and sums them. Each BLT model uses 500,000 hashes and a single hash function. The augmented embedding is:

$$e_i = x_i + \sum_{n=3}^{8} E_{n}^{hash}(\text{Hash}(g_{i,n}))$$

The information flow between bytes and patches is bridged by **cross-attention**: on the Encoder side, patch representations serve as the query and byte representations as the key/value to pool bytes into patches; on the Decoder side the roles are swapped, with byte representations as the query and patch representations as the key/value to "unpatch" patches back into bytes. Each query patch only attends to the bytes belonging to its own patch.

### The Evaluation Metric: Bits-Per-Byte

Because perplexity is only meaningful under a fixed tokenizer, to fairly compare byte-level and token-level models the paper follows prior work in instead reporting **Bits-Per-Byte (BPB)**, that is, normalizing the cross-entropy loss over the entire data by the total byte count and a constant, yielding a tokenizer-independent perplexity:

$$\text{BPB}(x) = \frac{\mathcal{L}_{CE}(\pmb{x})}{\ln(2)\cdot n_{\text{bytes}}}$$

For FLOP estimation, the paper follows Chinchilla's transformer FLOP formula, but treats the input embedding layer as an efficient lookup, counting it as a 0-FLOP operation, and assumes the FLOPs of backpropagation are twice those of the forward pass.

### A Concrete Forward Pass (Using the Paper's Real Numbers)

Taking the example sentence from the paper's Figure 4, `Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R.R. Martin.`, walk through entropy patching: the small entropy model computes $H(x_i)$ byte by byte, and any byte whose $H(x_i)$ exceeds the red line $\theta_g$ opens a new patch. Within the named entity `George R.R. Martin`, the entropy of `G` and `e` exceeds $\theta_g$, so `G` becomes a single-byte patch of its own and `e` opens a larger patch—because the entropy then falls all the way down (the second half of the name is easy to guess), the rest of the entity produces no new patches. Intuitively, a hard-to-predict prefix gets one expensive global step, while an easy-to-guess suffix is packed cheaply.

![The byte-by-byte entropy and patch boundaries of the example sentence](imgs/entropy_patching.png)

Now look at the compute ledger at the architecture level (8B setting, from Appendix Table 8): the Local Encoder has only 1 layer, $h_{\mathcal{E}}=1280$, about 20M parameters; the global Latent Transformer has 32 layers, $h_{\mathcal{G}}=4096$, about 6.4B parameters; the Local Decoder has 6 layers, $h_{\mathcal{D}}=1280$, about 120M parameters; cross-attention uses 20 heads and $k=4$. On BLT-1T the average patch size is about 4.5 bytes, so for a 16k-byte context the expensive global model runs only about $16000/4.5 \approx 3556$ steps, rather than one step per byte. The real efficiency lever is: pulling the patch size from 4.5 to 8 nearly halves the global steps, buying the near-50% inference FLOP savings the paper claims; and when the local models grow from 400M to 8B they only roughly double, so enlarging the patch affects almost only the global transformer's FLOPs, not the byte-level modules.

### Headline Results

The comparison of three 8B models trained on BLT-1T data under the same FLOP budget is as follows (Table 1, higher accuracy is better):

| Task | Llama 3 (BPE) | BLT-Space | BLT-Entropy |
|-|-|-|-|
| Arc-E | 77.6 | 75.4 | **79.6** |
| Arc-C | **53.3** | 49.8 | 52.1 |
| HellaSwag | 79.1 | 79.6 | **80.6** |
| PIQA | 80.7 | **81.1** | 80.6 |
| MMLU | **58.1** | 54.8 | 57.4 |
| MBPP | 40.2 | 37.6 | **41.8** |
| HumanEval | 31.1 | 27.4 | **35.4** |
| Average | 60.0 | 58.0 | **61.1** |
| Bytes/Patch | 4.4 | **6.1** | 4.5 |

BLT-Entropy beats the same-data-volume Llama 3 on 4 of 7 tasks, averaging 61.1 vs 60.0; and although BLT-Space is slightly behind on average, it trades its larger 6.1-byte patch for significant inference FLOP savings. The authors attribute BLT-Entropy's advantage to (1) dynamic patching using training compute more effectively and (2) directly modeling byte-level information.

On character-level tasks, the advantage of byte modeling is even more pronounced (excerpt from Table 2, 8B models):

| Task | Llama 3 (1T) | Llama 3.1 (16T) | BLT (1T) |
|-|-|-|-|
| HellaSwag noised average | 56.9 | 64.3 | **64.3** |
| CUTE (total) | 27.5 | 20.0 | **54.1** |
| CUTE - Spelling | 1.1 | – | **99.9** |

BLT is over 25 points higher than the two BPE Llama 3 models on the CUTE character-understanding benchmark, and the spelling task even reaches 99.9%—and it uses only 1/16 of Llama 3.1's data volume. On this basis the authors argue that character-level information is "hard to learn for BPE models by more data alone." In addition, initializing the global transformer with pretrained Llama 3.1 weights and fine-tuning at 1/10 the learning rate (the paper calls it "byte-ify" distillation) can, with only 220B tokens, raise MMLU from the 25.2 of BLT trained from scratch to 63.7, approaching native Llama 3.1.

### Patches Scale Better Than Tokens: A New Dimension Under a Fixed Inference Budget

BLT's most central claim is that it unlocks a new scaling axis: **simultaneously enlarging the model and the patch size under a fixed inference FLOP budget**. The table below (Table 3) is the model pairing used in the fixed-inference scaling study—each row has the same inference FLOPs/byte:

| Llama 2 | Llama 3 | Entropy ps=6 | Entropy ps=8 | Inference FLOPs | Crossover (Bytes) |
|-|-|-|-|-|-|
| 470m | 450m | 610m (1.2x) | 760m (1.6x) | 3.1E8 | 150B |
| 3.6B | 3.9B | 5.2B (1.3x) | 6.6B (1.7x) | 2.1E9 | 1T |

Because longer patches save compute on average, this compute can be used to grow the global latent transformer larger (because it runs fewer times). BPE models are better when the training budget is very small, but are soon overtaken by BLT—the crossover point falls just slightly after the compute-optimal point (converging from 3x down to about 2.5x at large FLOP scales). The authors emphasize that 8B-scale models trained far beyond compute-optimal (for example, Llama 3.1 trained on two orders of magnitude more data) are exactly the ideal scenario for this strategy of "paying a one-time pretraining cost in exchange for a better model under a fixed inference budget."

## 🧪 Critical Assessment

### Is the Problem Real, or a Pseudo-Issue Manufactured by the Tokenizer

The pain points of tokenization—multilingual inequity, fragility to noise, orthographic ignorance—are real problems supported by the literature, not contrived. BLT's gap of 54.1 vs 27.5 on CUTE, and its roughly 8-point average advantage on noised HellaSwag, do quantify the value of "direct byte access." But we must honestly point out: these character tasks are inherently the structural blind spots of the tokenizer, and a byte model winning here is almost a definitional inevitability; it is more like "proving the byte model does not have this innate defect" than "the byte model is stronger across the board." On the genuinely hard knowledge and reasoning tasks (MMLU, Arc-C), BLT actually loses slightly or ties.

### Do the Baselines, Ablations, and Data Support the Claim of "Matching Llama 3"

The experimental design is quite restrained and self-disadvantaging: the authors deliberately make the byte count of each batch equal in expectation, shortening the sequence length of large-patch models, avoiding BLT gaining an advantage from a longer context, which is commendable. The ablations also cover the main design choices such as entropy-model size, cross-attention position, n-gram hash vocabulary size, and number of local layers. But "matching Llama 3" is built on a borrowed assumption: BLT directly reuses the compute-optimal (parameter/data ratio) and optimal step count that Llama 3 computed for a BPE transformer. The paper itself admits in Limitations that this scaling law was computed for a BPE transformer and may lead to a suboptimal configuration for BLT—that is, the current BLT is very likely not yet placed at its own optimal operating point, which is an undigested confounder when comparing the two.

### Is It a New Architecture, or an Engineering Recombination of Existing Parts

BLT's individual parts mostly have predecessors: static patching comes from MegaByte, space patching from SpaceByte, entropy/boundary-predictor-style dynamic splitting had early prototypes in Nawrot et al., and cross-attention pooling is explicitly stated to follow Perceiver. The real novelty lies in "assembling entropy patching + hash n-gram + bidirectional cross-attention, and for the first time matching SOTA tokenizer models at a FLOP-controlled, 8B/4T scale." This is a solid scaling contribution, but one should be careful not to misread "first to achieve it at this scale" as "the method itself is wholly new."

### Is the "Patches Scale Better" Evaluation Designed Around Its Own Strengths

The conclusion of fixed-inference scaling relies heavily on FLOP as a theoretical proxy metric, rather than real wall-clock. The paper itself admits in Limitations that existing libraries are highly optimized for tokenizer transformers, and that BLT uses non-standard layers such as FlexAttention, so the actual wall-clock time "may not yet be on par with tokenizer models." Therefore "patch size 8 saves near-50% inference FLOP" is a theoretical upper bound, and the reader should not directly treat it as a measured 1.9× speedup. In addition, the crossover point, the choice of patch size, and temporarily adjusting the entropy threshold from 0.6 to 0.1 at inference time to gain CUTE scores all carry a flavor of "the benchmark being defined by the authors around their own method's strengths"—part of BLT-Entropy's advantage in Table 1 comes from this inference-time threshold adjustment, rather than a pure architectural win.

### Is the Claimed Problem Really Solved, and How Much Does It Matter in the Real World

On the proposition of "scaling to 8B/4T and matching Llama 3 without a fixed vocabulary," the paper provides the strongest evidence to date, which is substantial progress. But "solved" must be discounted: it requires additionally training and keeping an entropy model resident at inference time, relies on a pile of hyperparameters not yet validated by a BLT-specific scaling law, and lacks evidence of wall-clock efficiency. The most practical immediate real-world significance is actually byte-ify distillation—converting the already expensively trained Llama 3.1 into BLT, raising MMLU to 63.7 with just 220B tokens; this route avoids the cost of byte training from scratch and may reach deployment faster than a BLT trained from zero.

## 🔗 Related notes

- [Byte-Level BPE (BBPE)](../tokenizer/ByteLevelBPE/) — A representative of the fixed-vocabulary tokenization line that BLT aims to replace.
