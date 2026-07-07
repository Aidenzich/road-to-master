# Gemma4 — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Gemma 4 Technical Report |
| Venue | arXiv:2607.02770 (technical report, not peer-reviewed) |
| Year | 2026 |
| Authors | Gemma Team, Google DeepMind |
| Official Code | unknown |
| Venue Kind | tech-report |

> This note is written from the arXiv preprint `2607.02770v1` (submitted 2026-07-02, cs.CL), with evidence drawn primarily from the e-print's LaTeX source. This is a technical report from Google DeepMind rather than a peer-reviewed paper; numbers in the camera-ready or later versions may change. The weights are released under Apache 2.0; the paper body gives no code repository URL, so Official Code is recorded as `unknown`.

## First Principles

Gemma 4 is an open-weight, natively multimodal (text, image, audio) model family. Its core selling point is not a single new algorithm but a set of engineering trade-offs made in service of "on-device inference efficiency × reasoning capability." The following walks through the model family, long-context KV cache efficiency, the encoder-free architecture, thinking mode and the MTP drafter, quantization, and finally works out image tiling concretely with a real forward-pass example.

### Model family and parameter configuration

Gemma 4 offers both dense and Mixture-of-Experts (MoE) architectures, ranging from effective 2.3B (E2B) up to 31B, plus a MoE variant with 3.8B active / 26B total (26B-A4B). The supported modality and encoder configuration is not uniform across sizes: E2B/E4B have a 305M audio + 150M vision encoder, 26B-A4B/31B have only a 550M vision encoder (no audio), while 12B goes with an encoder-free projection and carries no separate encoder. The tokenizer reuses Gemini's SentencePiece (split digits, byte-level) with a 262k vocabulary.

| Model | Type | Scale | Einsums | Drafter |
|-|-|-|-|-|
| E2B | Dense | 2.3B effective | 1,870M | 76M |
| E4B | Dense | 4.5B effective | 3,940M | 77M |
| 12B | Dense (encoder-free) | 12B | 10,890M | 400M |
| 26B-A4B | MoE | 3.8B active / 26B total | 24,500M | 430M |
| 31B | Dense | 31B | 29,290M | 500M |

Here E2B and E4B reuse Gemma 3n's per-layer embeddings, which lets them reach 2.3B and 4.5B effective at 5B and 8B total parameters respectively; this "large total parameters, small effective parameters" design is a comparison caliber to watch out for in the critique below.

### Long context and KV cache efficiency

Extending context makes KV cache memory explode. Gemma 4's countermeasure is to set most attention layers to cheap local sliding-window, keeping only a few expensive global self-attention layers: the local-to-global ratio is 5:1 (the 2.3B E2B uses 4:1), and in the global layers it directly reuses keys as values (except for E2B, E4B), saving half of the value storage.

$\text{values} = \text{keys}$

For positional encoding, the global layers use $p\text{-RoPE}$ ($p=0.25$, rotating only a quarter of the dimensions) and the local layers use ordinary RoPE, with RoPE frequencies set to 1M and 10k in the global/local layers respectively. The paper claims these combinations reduce global KV cache occupancy by "up to" 37.5%, and additionally applies 20/35 and 18/42 KV cache sharing ratios to E2B and E4B.

### Encoder-free unified architecture (12B only)

12B takes a distinctive path: trained from scratch, discarding separate encoders. On the vision side, it ingests 48×48×3 RGB patches and replaces the 550M vision encoder with a single large 35M-parameter matmul, then adds 2D coordinate-style positional embeddings to the patch representations and passes them through a LayerNorm to preserve spatial sense.

The audio side is even more radical: the 305M USM conformer encoder is "completely discarded"; raw audio is sliced at 16kHz into 40ms chunks (640 dimensions per chunk) and then projected directly into the LLM embedding space. Because audio is itself a temporal sequence, no additional positional encoding is needed.

### Thinking mode and the MTP drafter

Gemma 4 adds thinking mode in post-training: the model can output a reasoning trace before answering, improving performance in reasoning-intensive domains such as math and code (toggled via a `<|think|>` control token).

To speed up decoding, Gemma 4 trains a small autoregressive multi-token prediction (MTP) drafter head for speculative decoding: it feeds the main model's previous-step last-layer activations and token embeddings into a separate embedder plus a 4-layer Transformer, and cross-attends the main model's KV, so it needs no MTP prefill and supports arbitrary draft lengths. The drafter's model dim is 256 for E2B/E4B and 1024 for 26B-A4B/31B, with three local and one global attention layers. The E2B/E4B drafter also uses "cluster top-k" to replace the final projection over the whole vocabulary with a top-k operation over token clusters:

$d \times 262{,}000 \rightarrow d \times 4096$

The paper's Figure 1 illustrates this "no prefill needed" mechanism clearly: the MTP decoder (the blue block on the right) uses cross-attention to directly read the main model's feature layers at different depths — including the last prefilled local layer, the last prefilled global layer, and the final Layer N — then autoregressively emits multiple draft tokens step by step, thereby avoiding the overhead of running another prefill over the draft sequence.

![The attention structure of the MTP drafter: MTP Layer 1–4 on the right use cross-attention (red dotted lines) to read the main model's last prefilled local layer, last prefilled global layer, and Layer N respectively; the main model's last-layer activation (blue dashed line) is fed together with the previous token embedding into Concat + down-proj, supporting multi-token speculative decoding without prefill.](imgs/mtp_gemma_4.png)

### Quantization-aware training (QAT)

Gemma 4 releases two quantization formats alongside the raw checkpoint: mobile (weights mixing int2/int4, activations int8) and blockwise Q4_0. It also quantizes the encoders: the 150M vision encoder goes to W8A8 for a 2× forward-pass memory reduction (400 MB → 200 MB) and a 44% on-device latency reduction relative to Gemma 3n, while the audio encoder compresses on-disk footprint from Gemma 3n's 390 MB down to 87 MB (78% reduction).

### One concrete forward pass: variable-resolution image tiling

The vision encoder supports variable aspect ratios, with the max token count $N_\text{max}$ taking only 70, 140, 280, 560, or 1120. The following aspect-ratio-preserving resize algorithm (the paper's Algorithm 1) determines how many soft tokens each image is compressed into:

```text
Input: image I ∈ R^(H×W×C), patch size p, max token count N_max, pooling kernel size k
m ← k · p                       # pooled patch side length
T ← N_max · m^2
f ← sqrt(T / (H · W))           # ideal scaling factor
H_ideal ← f · H ; W_ideal ← f · W
H_target ← floor(H_ideal / m) · m   # round down to a multiple of m
W_target ← floor(W_ideal / m) · m
I_resized ← BicubicResize(I, H_target, W_target)
```

Walking through it with the parameters `patch_size=16`, `pooling_kernel_size=3`, `max_soft_tokens=10` from the paper's appendix figure: the pooled patch side length is

$m = k \cdot p = 3 \times 16 = 48$

and the token budget converted to pixel area is

$T = N_{\max} \cdot m^2 = 10 \times 48^2 = 23040$

so the image is resized to 2×4 pooled patches (each pooled patch has a 48 px side, i.e. 48×48 px, for 8 patches total ≤ the budget of 10). Converted to 16px patches, that is 6×12 = 72 patches entering the vision encoder, then 3×3 pooling, and finally only 72 / 9 = 8 soft tokens are handed to the LLM backbone — this shows that in Gemma 4 "resolution" is derived backward from the token budget and pooling, rather than being a fixed size.

The paper's Figure 2 demonstrates the same pipeline with a $572 \times 1024$ (aspect ratio 1:1.79) space-otter image: under `k=3`, `max_soft_tokens=10`, `patch_size=16`, the algorithm computes the target size $96 \times 192$ (1:2) that is closest to the original ratio while staying within the token budget, corresponding to 2×4 pooled patches and 72 16px patches, finally compressed into 8 soft tokens.

![Figure 2's image-scaling example: the left image is the original 572×1024 (1:1.79) input; under k=3, max_soft_tokens=10, patch_size=16 it is resized with aspect ratio preserved into the right image's 96×192 (1:2), corresponding to 2×4 pooled patches, 72 16px patches, and finally 8 soft tokens.](imgs/varasp_varres_resize.png)

### Performance: the leap over Gemma 3 27B

On static benchmarks, Gemma 4 across all sizes (thinking mode) leads its predecessor Gemma 3 27B (non-thinking) by a wide margin, most notably on reasoning tasks:

| Benchmark (thinking) | Gemma 4 31B | Gemma 4 E2B | Gemma 3 27B (non-thinking) |
|-|-|-|-|
| MMLU Pro | 85.2 | 60.0 | 67.6 |
| AIME 2026 (no tools) | 89.2 | 37.5 | 20.8 |
| LiveCodeBench v6 | 80.0 | 44.0 | 29.1 |
| GPQA Diamond | 84.3 | 43.4 | 42.4 |
| Codeforces (Elo) | 2150 | 633 | 110 |

On the human blind-test Arena Text (2026-06-19 snapshot), Gemma 4 31B scores 1451 Elo (43rd overall, first among dense open models) and 26B-A4B scores 1438, far above Gemma 3 27B's 1366, standing on the leaderboard alongside MoE open-source models more than ten times its parameter count (such as DeepSeek V4, Kimi K2.6, GLM 5). On long context, RULER 128k has 31B at 96.4 versus Gemma 3 27B's 66.0; on audio, E4B beats Gemma 3n E4B on CoVoST translation average, 38.2 vs 34.7 (this is AVG CorpusBLEU, higher is better); in the same audio evaluation set, FLEURS ASR average WER also drops from Gemma 3n E4B's 0.085 to Gemma 4 E4B's 0.075 (lower is better).

## 🧪 Critical Assessment

### The provenance of the 37.5% efficiency number is internally inconsistent

The most eye-catching efficiency claim — global KV cache reduced by "up to" 37.5% — has two mutually conflicting attributions in the paper: the Introduction credits it to the combination of "KV cache sharing + reusing keys as values in the global layers," while the Model Architecture section writes that it is $p\text{-RoPE}$ ($p=0.25$) "effectively reducing the global KV cache by 37.5%." Two mechanisms, one number, no derivation or ablation, plus the "up to" wording — making it impossible to tell where this 37.5% actually comes from or at which sizes it holds. This is a textbook "conclusion first, derivation missing" efficiency claim.

### The main comparison group is only the in-house Gemma 3 27B, and thinking vs non-thinking

Aside from the Arena leaderboard, which blind-tests against many external models (relatively fair), the only longitudinal comparison in the three main tables — static, vision, and long context — is the in-house previous-generation Gemma 3 27B, with no same-size contemporary open-source competitor (such as the Qwen or Llama families) listed item by item. More critically, the caliber: the tables note that all Gemma 4 runs are in thinking mode while Gemma 3 27B is non-thinking, so part of the "across-the-board large lead" is actually the thinking / non-thinking gap rather than pure generational progress — turning off thinking on the same Gemma 4 and re-comparing would be the fair generational comparison, but the paper does not provide it.

### open-weight is not open-recipe: missing key details make it hard to reproduce

As a technical report, it accounts for the architecture and the dimensions of each size, yet stays almost silent on the things that actually determine success or failure: pre-training only says "similar to Gemma 3," and the data composition and mixing ratios, total training token count, actual training compute (only the TPU chip count and sharding are given), and the post-training data and RL settings are all missing. This makes Gemma 4 a model whose weights are downloadable but whose training is not reproducible; external researchers cannot re-run or verify any efficiency or capability claim from it.

### The safety claims are self-reported, lacking comparable external benchmark numbers

The safety section claims "major improvements in every category of content safety" and "minimal policy violations," but the entire passage contains no benchmark name, number, or table, nor any comparable external red-team results; moreover the testing is self-evaluated "without safety filters." For an open model marketed for enterprise and on-device deployment, this safety narrative — adjectives only, no auditable numbers — carries clearly less weight than its detailed treatment of capability benchmarks.

### The effective-param caliber and selective reporting amplify the "small models can compete too" narrative

Claims like "E2B roughly matches Gemma 3 27B with 10× fewer parameters" rest on the effective-parameter caliber: E2B's total parameters are in fact 5B (only counted as 2.3B effective thanks to per-layer embeddings), and on hard reasoning (AIME 2026 only 37.5, GPQA 43.4) it still lags large models by a clear margin, so "roughly matches" mainly holds on the easier composite tasks. The tables also contain many `-` entries (e.g. HLE with search reports only 31B/26B, MTOB 256k Full-book reports only the three larger models 31B/26B/12B and leaves E4B/E2B blank), which is selective reporting; readers cannot easily see the true ceiling of the small models on difficult tasks.

## One-minute wrap-up

- **KV cache slimming**: the biggest pain of extending context is KV cache memory. Gemma 4 changes most attention layers to cheap local sliding-window and directly reuses keys as values in the global layers — the paper claims this combination can reduce global KV cache occupancy by up to 37.5% (but the attribution of this number is internally inconsistent in the text; see the critique below).
- **Encoder-free 12B**: 12B simply discards the separate vision encoder and projects image patches directly into the language model — a single 35M-parameter matmul replaces the original 550M vision encoder.
- **Thinking mode**: the model can output a reasoning trace before answering, sharply raising scores in reasoning-intensive domains such as math and code — 31B scores 89.2 on AIME 2026, far above the previous-generation Gemma 3 27B's 20.8.
- **Uneven comparison baseline**: that "across-the-board large lead" above is partly "Gemma 4 with thinking on" beating "the previous-generation Gemma 3 27B with thinking off"; the paper does not provide a fair generational comparison of the same Gemma 4 with thinking turned off.
- **open-weight is not open-recipe**: the report opens the model weights for download, yet stays silent on the training data composition, mixing ratios, and total training token count, so external researchers cannot reproduce the training from it.

## 🔗 Related notes

- [Scaling Laws for Neural Language Models](../ScalingLaws/) — the scaling trade-offs among parameters/data/compute, the backdrop for Gemma 4's "effective parameters vs performance" narrative.
- [Byte Latent Transformer](../ByteLatentTransformer/) — takes the same route of "discarding the fixed tokenizer/encoder and ingesting raw patches directly," a good contrast for Gemma 4's encoder-free design.
- [LayerSkip](../LayerSkip/) — self-speculative decoding, in the same decoding-acceleration family as Gemma 4's MTP drafter.
- [Scaling Test-Time Compute](../ScalingTestTimeCompute/) — trading test-time compute for performance, corresponding to Gemma 4's thinking mode.
