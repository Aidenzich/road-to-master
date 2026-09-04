# Qwen3-VL-Embedding and Qwen3-VL-Reranker — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking |
| Venue | arXiv (technical report) |
| Year | 2026 |
| Authors | Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai et al., Tongyi Lab, Alibaba Group |
| Official Code | https://github.com/QwenLM/Qwen3-VL-Embedding |
| Venue Kind | tech-report |

> This note takes the full technical report of arXiv:2601.04720 (the LaTeX source) as its primary evidence, and supplements deployment details with the official repository (pinned commit `393e297`) and the model card. The technical report is not peer-reviewed, and the official version may differ from this one.

## Introduction

Modern web content simultaneously contains natural images, document screenshots, infographics, and video; text search over a single modality can no longer cover these needs. What Qwen3-VL-Embedding and Qwen3-VL-Reranker aim to solve is mapping text, images, document images, and video into "the same representation space," so that queries and documents can retrieve one another regardless of which modality they belong to. This is a real retrieval problem: for example, using the text "urban architecture" to fetch the corresponding photos of urban buildings or a clip of street scenery.

![Illustration of the unified multimodal representation space (paper Figure 1): text, images, visual documents, and video mapped onto the same semantic manifold](imgs/demonstration.png)

The paper's Figure 1 draws this goal as a schematic: instances of four modalities—text (descriptions such as "urban architecture," "user interface design," "expressive movement"), natural images (an urban skyline), visual documents (a dashboard screenshot), and video (a clip of a person moving)—are projected onto the same hemispherical "Unified Multimodal Representation Space" manifold; in the legend, green, blue, light blue, and pink respectively denote the representation vectors of Text, Image, Visual Document, and Video. As long as the semantics match—for example a passage of text and an urban photo both about "urban architecture"—they fall at nearby positions on the manifold, and can therefore retrieve one another across modalities.

The high-level solution the paper proposes is a two-stage retrieval pipeline of "an embedding model + a reranker model," both built on top of the Qwen3-VL backbone. The embedding model adopts a bi-encoder, compressing each instance into a single dense vector and using cosine similarity as the relevance metric, responsible for large-scale recall; the reranker model adopts a cross-encoder, performing cross-attention over query–document pairs and outputting a fine-grained relevance score, responsible for reranking. Both inherit Qwen3-VL's multilingual ability (claimed to support 30+ languages), and each is released in 2B and 8B sizes.

How does the paper measure whether the solution works? Primarily it evaluates on MMEB-V2, a multimodal benchmark spanning the three domains of Image, Video, and Visual Document, 9 task categories, and 78 datasets in total, supplemented by text-only MMTEB and visual-document retrieval on JinaVDR and ViDoRe v3. The flagship model Qwen3-VL-Embedding-8B achieves an overall score of 77.8 on MMEB-V2, and the paper claims it ranked first at the time of evaluation (January 2026).

This note focuses on the "video embedding" path: how video is sampled, tokenized, and pooled into a vector, how quality/resolution/duration/FPS/frame-count compete against one another within a fixed pixel/token budget, and on what evidence the video capability the paper demonstrates actually rests.

## First Principles

### Two models: one bi-encoder, one cross-encoder

![Architecture overview: the Embedding model and the Reranker model share a Vision Encoder and the Qwen3 LM Dense Decoder](imgs/arch.png)

Both models use Qwen3-VL as the backbone, use causal attention, and are initialized from Qwen3-VL-Instruct. The embedding model's input follows Qwen3-VL's conversational structure: the instruction goes in the system message (the default instruction is "Represent the user's input."), the multimodal instance to be represented goes in the user message, and finally a `<|endoftext|>` PAD token is appended; the last-layer hidden state corresponding to this token is taken as the dense vector for the whole instance. The reranker instead places both the "instruction defining relevance" and "the pair of instances to compare" into the user message, and estimates relevance by the model's predicted probability that the next token is "yes" or "no."

These two architectures each have trade-offs, which also explains why chaining "embedding + reranker" beats either alone. The bi-encoder encodes query and document separately, so vectors can be indexed offline and only cosine is computed online—suitable for fast recall over corpora from millions to billions—but query and document cannot see each other during encoding; the cross-encoder concatenates query and document into the same sequence for deep cross-attention, which can capture fine-grained correspondence and judge more accurately, but every pair must run a full forward pass and cannot be pre-indexed. The paper's approach is to use Embedding-2B to first recall the top-100 candidates, then use the reranker to rerank those 100, balancing speed and precision.

The four models the paper releases have the following specs: the Embedding model 2B has 28 layers and outputs 2048 dimensions, the 8B has 36 layers and outputs 4096 dimensions; both list a sequence length of 32K and support quantization and Matryoshka Representation Learning. The Reranker 2B/8B likewise have 28/36 layers and a 32K sequence length, but because they output a relevance score rather than a vector, they have no embedding dimension and do not support MRL or quantization.

| Model | Size | Layers | Sequence length | Embedding dim | MRL | Instruction-aware |
|-|-|-|-|-|-|-|
| Qwen3-VL-Embedding | 2B | 28 | 32K | 2048 | Yes | Yes |
| Qwen3-VL-Embedding | 8B | 36 | 32K | 4096 | Yes | Yes |
| Qwen3-VL-Reranker | 2B | 28 | 32K | - | - | Yes |
| Qwen3-VL-Reranker | 8B | 36 | 32K | - | - | Yes |

### From video file to vector: the complete preprocessing path

This is the core of this note. The paper's main text covers video preprocessing in a single sentence (during training, "sample at 1 FPS, at most 64 frames, all frames together with a token budget of 4,500, roughly 9.2×10⁶ pixels"); the complete path can only be reconstructed from the official repository's `src/models/qwen3_vl_embedding.py`. Each step below is marked as to whether it is stated in the paper, is official-code behavior, or is our inference.

**(1) Decoding and temporal sampling (the two input paths are not identical).** The official embedder's defaults are `FPS = 1` and `MAX_FRAMES = 64`, but video has two input forms whose sampling behavior differs, and static reading of the code can clearly distinguish them. When the input is a **video-file path** (a string), `format_model_input` only hands `{'fps': 1, 'max_frames': 64}` to the Qwen3-VL processor, which is responsible for decoding and sampling—`sample_frames` is not executed locally. When the input is a **list of frames** (a frame list), the code then calls `sample_frames` locally, uniformly taking 64 frames from the existing frames via `np.linspace(0, len(frames) - 1, 64, dtype=int)`. The common consequence of both paths is: whenever the number of valid frames exceeds 64 (for example a long video producing hundreds of frames at 1 FPS, or a frame list of hundreds of images), it is compressed down to exactly 64 frames—temporal resolution grows coarser as the video grows longer, and the interval between frames stretches wider and wider.

**(2) Spatial resolution and the pixel budget.** Each frame keeps its original aspect ratio, but the whole video is limited by a total budget. In the code the default `total_pixels` value for video comes from `MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS = 7,864,320`, where `FRAME_MAX_PIXELS = 768 * 32 * 32 = 786,432` is merely the coefficient used to derive this "total"—static reading of the code confirms it is never passed to the processor as a per-frame cap (it appears in the entire codebase only on the line defining `MAX_TOTAL_PIXELS`). The README further explains that this `total_pixels` is "multiplied by 2" inside the model (an effective total of roughly 15,728,640 pixels), and gives the example "for a 16-frame video, each frame can go up to 983,040 pixels (1280×768 resolution)"—note that 983,040 already exceeds 786,432, which precisely shows that how many pixels each frame gets is determined by "effective total ÷ actual frame count," not by some fixed per-frame cap; in other words, this repository does not expose a verifiable per-frame video pixel cap. That is to say, frame count and per-frame resolution share the same budget: the more frames, the fewer pixels each frame can get. But note: this `total_pixels` is only carried in the `video_kwargs` of the **frame-list path**; under the **video-file path**, `video_kwargs` is overwritten to a dict containing only `fps` and `max_frames`, `total_pixels` is simply discarded, and the per-frame pixels are instead determined by the processor's own default. Therefore all the pixel-conversion examples below that use `total_pixels` strictly apply only to the frame-list path.

**(3) Visual tokenization and spatiotemporal merging.** Qwen3-VL follows a 16×16 patch, and with 2×2 spatial merging maps one merged patch to roughly 32×32=1024 pixels per visual token; adjacent frames are further merged along the temporal dimension. This step is not itemized in the paper itself; it is inferred from the Qwen3-VL backbone and the pixel↔token conversion (the paper's image-side "1,280 tokens ≈ 1.3×10⁶ pixels" is exactly the ratio of 1024 pixels/token).

**(4) Fusion with text/instruction, truncation, pooling.** The visual tokens together with the system instruction and user text are strung into a single sequence and fed into the Qwen3 LM decoder. The official embedder's default context length is `MAX_LENGTH = 8192` (below the 32K capacity stated on the model card). Note that the implementation differs from intuition: the truncation that actually takes effect happens when `_preprocess_inputs` calls the processor with `truncation=True, max_length=self.max_length`, delegating to the processor, which decides which tokens to drop. Although the file also defines a `_truncate_tokens` (whose logic is to keep all special tokens and only truncate the surplus non-special tokens), static reading of the code confirms it has no call site in this released wrapper, so the rule "keep special tokens" is not the truncation path that actually takes effect—it is just an unused helper. Finally `_pooling_last` uses the `attention_mask` to locate and take the hidden state of the last valid (mask=1) position of each sequence as the vector. Here too the paper's template and the released code must be told apart: the wrapper builds text only via `apply_chat_template(..., add_generation_prompt=True)`, and `PAD_TOKEN` (`<|endoftext|>`) is only defined but not used in this file, so the static evidence can only prove that what it pools is the "last valid position," and cannot prove that position is the PAD token—the paper's template does indeed contain a PAD, but that is the paper's narrative and cannot be taken directly as the behavior of the released implementation.

**(5) Projection, normalization, and similarity.** The extracted vector is the 4096 dimensions of the 8B (or 2048 dimensions of the 2B), then L2-normalized via `F.normalize(embeddings, p=2, dim=-1)`, after which the relevance of any two instances is the inner product of their normalized vectors (cosine similarity). The reranker takes a different route: it takes the hidden state of the last position of the sequence through the LM head and computes the "yes"/"no" logits.

### Three-stage training, distillation, and model merging

![Multi-stage training pipeline: starting from Qwen3-VL-Instruct, through contrastive pretraining, multi-task contrastive learning, distillation, and model merging, producing s0–s3](imgs/pipeline.png)

Training uses LoRA, initializes from Qwen3-VL-Instruct, and is divided into three stages, aimed at reconciling the imbalance between "large amounts of weakly-supervised data" and "scarce high-quality data." Stage 1 does contrastive pretraining on large-scale synthetic data, obtaining the initial version s0 (the pipeline figure marks roughly 300M synthetic examples); Stage 2 mixes public and proprietary data, supplemented by synthetic data, for multi-task contrastive learning to obtain s1 (roughly 40M examples), while training Qwen3-VL-Reranker on the retrieval subset; Stage 3 uses this reranker to score a smaller dataset (roughly 4M examples), distilling into the embedding model to obtain s2, and finally merges s2 with s1 to obtain the final s3.

There is a clear motivation behind this order: although s2 improves greatly on retrieval-type tasks, it regresses slightly on classification and QA, so model merging brings back s1's generality. The training data itself has also been through video-oriented cleaning—first coarse-grained quality filtering to remove low-resolution and abnormal-aspect-ratio material, then scene cut detection to detect shot changes and remove static or corrupted segments to "preserve the temporal-dynamic integrity of the video," then Qwen3-VL-32B to produce fine-grained labels and GME embedding similarity to filter out samples with poor cross-modal alignment. On the video side four tasks are synthesized: Video Classification, Video Question Answering, Video Retrieval, and Moment Retrieval for fine-grained temporal localization.

![The category distribution of the data-synthesis seed pool (paper Figure 4): (a) image, (b) video](imgs/data_distribution.png)

The paper's Figure 4 presents the composition of these seed pools with two two-ring donut charts. On the video side (right), the inner ring has the three major categories Human-Centric, Nature/Scenery, and Media/Entertainment, and the outer ring subdivides into Daily Activity, Sports, Interview, Performance (human-centric), Wildlife, Time-lapse, Phenomenon (nature/scenery), and News, Animation, Gaming, Movie Clip (media/entertainment). One can see that people and daily activity make up the largest share—which also foreshadows the reservation below: such videos are mostly composed of recognizable people and static scenes, and may not require continuous motion to be retrieved.

The performance of each stage on MMEB-V2 (2B) bears out the trade-off above. The table below gives the raw numbers from the paper's "Performance across training stages" table; one can see Video Overall rising from s0's 57.5 all the way to s3's 61.9, while the distillation stage s2 (59.5) is indeed inferior to the pure multi-task s1 and needs merging to be recovered:

| Stage | Image Overall | Video Overall | VisDoc Overall | All |
|-|-|-|-|-|
| s0 | 65.8 | 57.5 | 74.8 | 66.6 |
| s1 | 74.8 | 60.3 | 77.1 | 72.1 |
| s2 | 71.3 | 59.5 | 80.9 | 71.5 |
| s3 | 75.0 | 61.9 | 79.2 | 73.2 |

### Training objectives, MRL, and quantization

The embedding model's retrieval data uses standard InfoNCE in Stage 1, gathering negatives from five sources into the denominator, and using a mask $m_{ij}$ to filter out suspected false negatives (masking a negative when its similarity exceeds the positive similarity + 0.1):

$$
\mathcal{L}_{\mathrm{retrieval}} = - \frac{1}{N} \sum_{i}^{N} \log\frac{e^{s(q_i, d_i^+)/\tau}}{Z_i}
$$

where $s(\cdot,\cdot)$ is cosine similarity and $\tau$ is the temperature. Stage 2 further removes the two kinds of in-batch negatives, query–query and document–document, from $Z_i$; the paper says this empirically performs better on high-quality multimodal data. Classification data switches to a contrastive form that "only treats clearly-wrong labels as negatives"; STS data uses CoSent loss to keep cosine ordered with respect to the ground-truth scores.

The third-stage distillation transfers the reranker's judgment to the embedding model: for each query, the reranker offline computes the relevance logits of the positive and $k$ negatives, the embedding's cosine score is used online, and the cross-entropy of the two distributions is minimized:

$$
\mathcal{L}_{\mathrm{distill}} = -\sum_{i=1}^{k+1} P_{\mathrm{reranker}}(d_i \mid q)\, \log P_{\mathrm{embedding}}(d_i \mid q)
$$

The reranker itself treats reranking as binary classification, trained with $-\log p(l \mid I, q, d)$ ($l$ being yes/no), and at inference the score is $s = \mathrm{sigmoid}(\mathrm{logit}(\text{yes}) - \mathrm{logit}(\text{no}))$.

![The accuracy–storage–latency trade-off of MRL and quantization (left: MS MARCO text retrieval; right: VL3-Syn image-text retrieval, MRR@10). The blue line is Float32, orange is INT8, green is BINARY. At 1024 dimensions on MS MARCO, INT8 and Float32 both have MRR@10 of 0.360, but latency drops from 43ms to 12ms and index memory from 32,539MB to 8,135MB (about 75% memory and 72% latency saved); on VL3-Syn, 1024-dim Float32 is 0.497 and INT8 is 0.487, nearly on par. BINARY clearly loses points, e.g. MS MARCO 128-dim is only 0.188.](imgs/mrl_qat.png)

For deployment efficiency, training additionally computes a loss on the "truncated low-dimensional prefix" (MRL), and uses LSQ Quantization-Aware Training to keep vectors robust under int8/binary. The analysis shows these trade-offs are within an acceptable range: on the 2B for text retrieval, reducing dimensions from 1024 to 512 loses only 1.4% of retrieval performance while yielding 50% storage savings and twice the retrieval speed; int8 is nearly lossless, but binary significantly hurts retrieval, and the lower the dimension the greater the harm.

### Evaluation design and the "real video" results

MMEB-V2's video domain subdivides into the four task categories CLS, QA, RET, and MRET (moment retrieval), which, per the "# of Datasets" row at the top of the paper's table, contain 5, 5, 5, and 3 datasets respectively, 18 in total. At evaluation the context is limited to 16,384 tokens, and video tasks set a total token cap of 15,000 and a frame cap of 64. Here the boundary of the evidence must be made clear to the reader: the paper provides only the above "summed by task category" scores—it neither states the evaluation metrics these scores use (e.g. Recall, accuracy, or a weighted average), nor lists one-by-one the names of these 18 video datasets, their retrieval direction (text→video or video→text), or their respective sample sizes—these fields are defined by the external MMEB-V2 benchmark (i.e. VLM2Vec-V2, Meng et al. 2025, arXiv 2507.04590), which this paper only cites and does not reproduce in the main text. The only datasets specifically named on the video side in the main text appear in the appendix's similarity-demonstration table (one example each of UCF101, NExTQA, MSR-VTT), which is a qualitative illustration rather than a complete list. Therefore every cell in the table below should be understood as "the summed score of multiple datasets under that task category," not a number traceable to a single dataset, single metric, or single retrieval direction. The table below extracts MMEB-V2's five Video columns, placing Qwen3-VL-Embedding side by side with the strongest open-source and closed-source baselines:

| Model | Size | Video CLS | Video QA | Video RET | Video MRET | Video Overall |
|-|-|-|-|-|-|-|
| RzenEmbed | 8B | 58.8 | 63.5 | 51.0 | 45.5 | 55.7 |
| Ops-MM-embedding-v1 | 8B | 59.7 | 62.2 | 45.7 | 43.2 | 53.8 |
| Seed-1.6-embedding-1215 | - | 85.2 | 66.7 | 59.1 | 54.8 | 67.7 |
| Qwen3-VL-Embedding-2B | 2B | 71.9 | 64.9 | 53.9 | 53.3 | 61.9 |
| Qwen3-VL-Embedding-8B | 8B | 78.4 | 71.0 | 58.7 | 56.1 | 67.1 |

Among open-source models Qwen3-VL-Embedding-8B's video overall of 67.1 clearly leads (RzenEmbed-8B is only 55.7); but it is worth noting that the closed-source Seed-1.6-embedding-1215's video overall of 67.7 is actually slightly higher than Qwen's 67.1, and its Video CLS as high as 85.2 is an outlier. In other words, the "MMEB-V2 rank #1" that the paper's abstract claims is built on the summed All=77.8 across the three domains; looking at the video domain alone, it is not unrivaled.

![The cross-domain benchmark comparison on the paper's front page: MMEBimage / MMEBvideo / MMEBvisdoc / MMTEB](imgs/performance_comparison.png)

The comparison bar chart on the paper's front page (below the abstract) draws this "summed first, video not necessarily first" structure very clearly: the dark-purple bar is Qwen3-VL-Embedding-8B, the light-purple bar is 2B, and the remaining blue-family bars are visual baselines such as Seed-1.6-Embedding-1215, IFM-TTE-7B, RzenEmbed-7B, GME-7B (the rightmost MMTEB column additionally includes text-only baselines like Qwen3-Embedding-8B, Gemini, OpenAI). The 8B clearly leads on MMEBimage with 80.1 (the runner-up is only 78.0, a 2.1-point gap), but although its MMEBvisdoc of 82.4 is the highest on the field, it only narrowly beats the closed-source Seed-1.6-Embedding-1215's 82.2 by 0.2 points, a marginal advantage rather than a "large lead"; and in the MMEBvideo column Seed-1.6-Embedding-1215's 67.7 is slightly higher than the 8B's 67.1—in other words, the only domain where the 8B truly opens up a gap is images; document images and video are both neck-and-neck or even behind.

The reranker's video evidence must also be unpacked. The paper uses Embedding-2B to recall the top-100 and then reranks, and Reranker-8B pushes the video retrieval score to 61.0 (a clear improvement over Embedding-2B's 53.6), but Reranker-2B's video score of 53.2 is actually slightly below the same-size Embedding-2B's 53.6—that is to say, on the video modality, the reranking a small reranker brings is not necessarily worthwhile; the gains come mainly from the 8B.

### A concrete video embedding example

Running once through a 120-second, native-30-FPS video with the official defaults (doing text-to-video retrieval for a text query like "baseball player hits ball"), and explicitly taking the **frame-list path** (first using an external tool to extract one frame per second, getting about 120 frames, then handing this stream of frames to the embedder): `sample_frames` uniformly takes 64 frames from these 120 with `np.linspace(0, 119, 64)`, equivalent to keeping one frame about every 1.9 seconds. Then `video_kwargs` carries `total_pixels=7,864,320` (15,728,640 pixels after the internal ×2) distributed across the 64 frames, roughly 245,760 pixels per frame (about 512×480, 0.25 MP). By 1024 pixels/visual token and further temporal merging of adjacent frames, we estimate the whole video produces roughly 7–8 thousand visual tokens, falling within the evaluation's 15,000 video cap and the official 8192 context; finally the 4096-dim hidden state of the last valid position of the sequence (the paper's design being an appended PAD token) is taken, L2-normalized, and cosine is computed against the query text's vector. (If instead the whole video file is thrown directly at the embedder via the video-file path, sampling and per-frame pixels are decided by the processor and `total_pixels` does not take effect, so the pixel and token counts above are only for order-of-magnitude reference.) In the paper's appendix demonstration examples, MSR-VTT's "baseball player hits ball" has a similarity of 0.80 to the correct video, UCF101 action classification 0.66, and NExTQA video question answering 0.64.

This example also directly exposes the tension of the input conditions: for the same 120-second video, if it is cut into short segments (each < 64 seconds), 1 FPS temporal resolution can be preserved and each frame can also get more pixels; once the whole thing is thrown in, both time and space are compressed simultaneously by the two caps of 64 frames and total_pixels.

### The input-condition matrix: setting, mechanistic consequence, and whether measured

The table below decomposes each dimension of video input into three statements of different nature: (a) the official supported default setting, (b) the purely mechanistic consequence, and (c) whether the paper actually measured the impact on embedding quality. Every column where the paper does no controlled ablation is explicitly marked "not measured," with no speculation.

| Dimension | Official setting (paper/code) | Mechanistic consequence | Controlled measurement? |
|-|-|-|-|
| FPS | Training 1 FPS; code default `fps=1` (only effective for video files) | For >64s video the effective sampling rate drops below 1 FPS | Not measured (no FPS ablation) |
| Max frames | `max_frames=64` (64 in both training and evaluation) | Long video uniformly compressed to 64 frames, temporal detail lost | Yes: Video (Scaling Frames) curve |
| Per-frame pixels | No public per-frame cap; per-frame pixels = effective total (≈15,728,640) ÷ actual frame count (16-frame example can reach 983,040 ≈ 1280×768) | More frames ⇒ lower per-frame resolution | Per-frame resolution not directly measured |
| Total token/pixel budget | Training 4,500 tokens; evaluation 15,000 tokens; code `total_pixels=7,864,320` (internal ×2, only carried on the frame-list path) | Space and time share one budget | Yes: Video (Scaling Tokens) curve |
| Input path | frame-list: local `sample_frames` + carries `total_pixels`; video file: only hands `fps`, `max_frames` to the processor (`total_pixels` discarded) | The two paths differ in sampling and per-frame pixel budget | Not measured (no cross-path comparison) |
| Context length | model card 32K; evaluation 16,384; code default 8192 | Overly long sequences are truncated | Indirect: slight regression at extremely high budget |
| Duration cap | No explicit hard cap, implied by frame count × budget | The longer the duration, the more severe the compression | Not measured (no duration ablation) |
| Aspect ratio | Each frame keeps its original aspect ratio | No forced cropping | Not measured |
| Compression/quality/low-light | No dedicated setting | — | Not measured (no compression/noise ablation) |

![The impact of visual granularity on performance across domains: video scaled respectively by token budget and frame count, both showing diminishing returns](imgs/granularity.png)

The paper's only granularity-focused measurement is "Impact of Spatial and Temporal Granularity": it splits video into the two axes of "frame count" and "total token budget" and scales each separately. From the curves one can read that video rises with frame count from ~40% at 2 frames to ~55% at 16 frames and ~57.5% at 64 frames, and with token budget from ~42% at ~600 tokens to ~57.5% above 3,000, both axes showing clear diminishing returns and even a slight fallback at the highest spending, which the paper attributes to "the model's own degradation when processing overly long context." It must be stressed: these are scaling curves treating tokens and frames as resources; the paper provides no controlled ablation of resolution, compression rate, FPS, or video duration, nor any experiment that separates "temporal understanding" from "sparse static frames."

## 🧪 Critical Assessment

### Is the problem real and important

Cross-modal unified retrieval is a real problem of economic value: e-commerce product search, scientific-literature exploration, and community navigation all need "retrieval regardless of which modality the query or document is." The route of building unified embeddings on a VLM backbone (E5-V, GME, VLM2Vec, etc.) also already has a community foundation, and the paper's positioning is clear and does not manufacture a false need. What truly needs examining are the following points—the strength of the video-capability evidence, the degree to which the evaluation is self-defined, and the gap between the paper's claims and the deployable defaults.

### Video capability: temporal understanding, or sparse static frames?

This is my biggest reservation about this paper. All video evaluation is done under the budget of "at most 64 frames, a video-task total token cap of 15,000, and a context of 16,384 tokens" (1 FPS is the sampling setting stated by the paper's training-stage implementation; the main text's MMEB-V2 evaluation section only states the frame and token caps and does not state the FPS used for evaluation), and the data examples the appendix demonstrates even use a single thumbnail (`.jpg`) to represent an entire video; the granularity curve shows 16 frames already reaches about 96% of the 64-frame performance (55.0 / 57.5 = 95.65%; this curve only scales frame count and does not bind each point to video duration or FPS), meaning most of the gain comes from "seeing a few more sparse frames" rather than continuous motion. Among the four video tasks the paper synthesizes there is indeed Moment Retrieval, which requires temporal localization, and MRET is also the lowest-scoring of the four video sub-items (the 8B is only 56.1), but the paper does no "shuffled frame order vs preserved order" or "dense sampling vs sparse sampling" comparison whatsoever, so it cannot rule out the competing hypothesis that "the video score can actually be explained by static keyframes + object/scene cues." Calling it "video embedding" holds at the recall level, but "temporal/event understanding" is not yet supported by the evidence.

### The adequacy of baselines, ablations, and the self-defined evaluation

The main results are almost entirely tied to the single leaderboard of MMEB-V2, and the SOTA claim is also measured by it. This brings two risks: first, the summed All score is diluted by the number of domains and each domain's difficulty weighting, and looking at the video domain alone the closed-source Seed-1.6-1215 (67.7) already exceeds Qwen-8B (67.1), so "rank #1" is a statement at the summed level rather than the video level. Second, the paper's data synthesis heavily uses Qwen3-VL-32B to produce labels and queries, and if the evaluation benchmark is homologous to this synthetic distribution there is a concern of a self-defined evaluation aiming at one's own data, and the paper provides no contamination/leakage analysis to rule this out. On ablations, apart from the granularity curve and the stage table, there is a lack of controlled experiments on instruction sensitivity, multilingual video, and—most crucially—"resolution/FPS/compression."

### Novelty: integration engineering or methodological innovation

In terms of individual technical components, this work is mostly an integration of existing methods: bi-encoder + last-token pooling, InfoNCE, cross-encoder reranker, MRL, QAT/LSQ, LoRA, and model merging are none of them firsts, and the training objective is explicitly stated to "extend Qwen3-Embedding." The real contribution is at the system level—integrating video-oriented data cleaning (scene cut, removing static segments), synthesis of four video tasks, reranker→embedding distillation, and s1/s2 merging into a pipeline that can run to SOTA, along with practical MRL/quantization deployment properties. This is solid engineering integration, but reading it as "a methodological breakthrough in video representation learning" would overestimate its novelty.

### Reproducibility and the deployment gap

The technical report has front-to-back inconsistencies on key numbers that will affect reproduction: the video total token budget is 4,500 in training, 15,000 in evaluation, and the code's `total_pixels` conversion is yet another order of magnitude (7,864,320 pixels, internal ×2), and the paper gives no conversion or explanation among the three, while this `total_pixels` is also only effective on the frame-list path and is decided by the processor default on the video-file path, making "how much pixel budget a video actually consumes" even harder to pin down; the context length is 32K on the model card, 16,384 at evaluation, and only 8192 by default in the released embedder. The training data is mostly proprietary and only a selection of synthetic prompts is attached, so complete external reproduction is infeasible. Hyperparameters such as hardware requirements, batch size, and temperature $\tau$ are also not fully disclosed. None of this negates the results, but it leaves a gap between "the claimed general video support" and "the operating conditions actually tested and deployable" that users must fill in themselves.

## One-minute version

- **Cross-modal unified retrieval**: mapping text, images, document images, and video into the same representation space to retrieve one another. For example, inputting the text "urban architecture" can directly fetch the corresponding photos of urban buildings or a clip of street scenery.
- **Two-stage pipeline**: the vector model (bi-encoder) first indexes offline over a million-scale corpus and quickly recalls the top-100, then the cross-attention model (cross-encoder reranker) deeply reranks those 100, balancing speed and precision.
- **The video budget competition**: frame count and per-frame resolution share the same pixel budget. On the frame-list path, a 120-second video is uniformly compressed to exactly 64 frames (one about every 1.9 seconds), each frame about 512×480, and the whole thing produces roughly 7–8 thousand visual tokens.
- **Open-source overall first, video not necessarily first**: Qwen3-VL-Embedding-8B achieves 77.8 on the MMEB-V2 summed All and claims rank #1, but looking at Video Overall alone it is 67.1, still slightly below closed-source Seed-1.6's 67.7.
- **Temporal understanding remains in doubt (important reservation)**: the high video scores mostly come from recognizing static cues in sparse frames rather than understanding continuous motion—sampling 16 frames already reaches about 96% of the 64-frame performance, and the paper does no "shuffled frame order vs preserved order" comparison at all that could prove temporal understanding.
- **A small reranker is not worthwhile for video**: Reranker-8B pushes the video retrieval score from 53.6 to 61.0, but Reranker-2B is only 53.2, actually slightly below Embedding-2B's original 53.6; the gains come mainly from the 8B.

## 🔗 Related notes

- [Video-MME](../Video-MME/)
- [VideoLLM-online](../VideoLLM-online/)
