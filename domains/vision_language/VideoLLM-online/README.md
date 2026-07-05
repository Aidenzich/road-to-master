# VideoLLM-online — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | VideoLLM-online: Online Video Large Language Model for Streaming Video |
| Venue | CVPR 2024 |
| Year | 2024 |
| Authors | Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, Mike Zheng Shou |
| Official Code | https://github.com/showlab/videollm-online |
| Venue Kind | paper |

> This note is written from the arXiv version `2406.11816` (the Llama-3-upgraded version of the CVPR
> camera-ready); numbers and quotations follow the paper's LaTeX source, and if the official
> conference version differs, the official CVPR version prevails. The citation count is marked as
> unavailable at the time of writing because the Semantic Scholar API returned HTTP 429, not 0.

## First Principles

### Problem setup: from "offline video QA" to "streaming video dialogue"

Most video large multimodal models (VideoLLMs) treat a video as a "pre-selected short clip" to understand:
the user uploads a clip, and the model reads the whole segment before emitting a single answer. This paper
argues that this offline paradigm cannot support an "always-on, timeline-hugging" AI assistant (e.g. AR
glasses that remind you to flip while you cook). The authors name the new problem video streaming dialogue:
the model continuously receives constantly refreshing frames, needs to speak proactively at the right moment,
and stay silent the rest of the time.

The authors decompose the difficulty into three mutually competing dimensions. The first is temporally aligned:
a query like "remind me when it's time to flip the steak" requires the model to scan frame by frame and cannot
just give a video-level overall answer. The second is long-context: to answer summary and planning questions,
the model must retain a large amount of historical visual and language tokens, which quickly overruns the LLM's
context window, slows down causal decoding, and eats up GPU memory. The third is real-time: the response must
keep up with the video's stream rate to achieve always-on.

Formally, given the context `[Ctx]` before $t_1$ and the frames `[Frame]` continuously arriving between $t_1$
and $t_2$, the model has a two-level objective: first decide "whether the current moment $t_2$ is suitable for
doing language modeling," and only if it decides to speak, do standard next-token language modeling at that
moment. This two-stage structure of "first decide whether to speak, then decide what to say" is the starting
point of all subsequent designs.

### Why per-frame dialogue doesn't work: Streaming EOS prediction

An intuitive solution is to raise the interaction frequency to every frame: treat each frame as a query, do
language modeling frame by frame, and on frames that need no answer output a very short utterance (e.g. "it's
not the time to answer yet"). But this forces the model to run a lengthy, recurrent billion-scale next-token
decoding on every frame, which cannot possibly be real-time in speed, and also repeatedly consumes dialogue
template tokens like `[INST]`, `[/INST]`, making the context of long videos balloon quickly. The authors also
empirically test GPT-4V's per-frame prompting: it tends to output long content on every frame, causing obvious
latency, and is not suitable for real-time streaming.

The core contribution of this paper is a new training objective, streaming EOS prediction. For a moment $t_2$
that "should answer," do language modeling as usual:

$$\max P(\texttt{[Txt}^{t_2}_{i+1}\texttt{]} \mid \texttt{[Ctx}^{<t_2}\texttt{]},\ \texttt{[Frame}^{t_2}\texttt{]},\ \texttt{[Txt}^{t_2}_{\le i}\texttt{]})$$

while for the "redundant, no-answer-needed" frames with $t_1 \le t < t_2$, directly teach the model to predict
EOS on that frame's token:

$$\max P(\text{EOS} \mid \texttt{[Ctx}^{<t}\texttt{]},\ \texttt{[Frame}^{t}\texttt{]}),\quad \text{where } t_1 \le t < t_2$$

The key ingenuity is that this EOS serves only as a supervision signal and **is not appended into the
input/output sequence**, so it does not raise perplexity the way actually inserting a large number of EOS
tokens would, nor does it occupy context. It is essentially not next-token prediction (EOS never appears in the
sequence), yet it can coexist with the autoregressive loss and together train a streaming model that "knows when
to shut up." The authors also emphasize that the EOS here need not be the language model's native `</s>`, and can
be any token agreed upon in the system prompt.

### Training loss: language modeling loss + streaming loss

Combining the two objectives, the training loss is the sum of per-token cross-entropy (the notation follows the
paper's Eq. 5):

$$L = \frac{1}{N}\sum_{j=1}^{N}\left(-\log l_{j+1} P_j^{\texttt{[Txt}_{j+1}\texttt{]}} -\ w\log f_j P_j^{\texttt{[EOS]}}\right)$$

where $l_j$ is the language-token indicator (1 only if the $j$-th token is a language response), and $f_j$ is the
streaming indicator: it is 1 only when the $j$-th token is the **last** token of a certain frame and the next
position is not a language token ($l_{j+1}=0$), i.e. applying the EOS supervision only on frames "about to fall
silent." $w$ is a balancing coefficient, defaulting to $w=1$. When each frame uses only 1 token, the ranges over
which the language loss and streaming loss act on the sequence are marked in the figure below.

![The LIVE training method: frame tokens and language tokens interleave over time, the Streaming Loss supervises silent frames to output EOS, and the LM Loss supervises the language tokens at the moment to answer](imgs/fig4.png)

### Data engine: turning offline annotations into streaming dialogue

The above training needs "user queries and assistant responses within the video stream" data, but mainstream
video datasets mostly only have offline temporal-segment annotations. The authors propose two data-generation
routes. For data that is itself annotated in a streaming way, like Ego4D narration (annotators describe in real
time while watching a 5-minute video), they directly reuse the instructions given to human annotators as the
training prompt. For offline data with only temporal-segment annotations (e.g. COIN), they synthesize dialogue
with an LLM: first prepare a question template bank covering past/present/future tenses — the authors prepare 50
questions per category, $N=150$ queries in total; then organize the timeline annotations into language prompts
like "time $t_a \sim t_b$: boiling the water," treat all key timestamps of state transitions as ideal response
moments, and let the LLM generate a response at each key timestamp. During training, one query is randomly
sampled, randomly inserted at some timestamp $t_r$, and responses before $t_r$ are dropped; each sample inserts
at most 3 queries.

![The LIVE data generation method: randomly inserting question templates into the video timeline, and "exposing" the timestamped ground-truth annotations to the LLM to generate time-segmented responses](imgs/fig3.png)

### Model architecture and inference pipeline

The architecture follows LLaVA's three parts: an image encoder, an MLP projector, and a language model. The
image encoder uses a CLIP ViT-L pretrained on DataComp-1B, extracting frame embeddings at 2 FPS, of shape
$(1+h_p\times w_p)\times c$ (1 CLS token plus average-pooled spatial tokens). The main-text experiments
deliberately set $h_p=w_p=0$, i.e. only 1 CLS token per frame, which is the most economical setting and can
process nearly half an hour of video within a 4096 context window; the released demo model uses $1+3\times3=10$
tokens/frame to trade for finer dialogue detail. Frame tokens are projected by the MLP and interleaved with
language tokens as input to Llama-2-7B-Chat or Llama-3-8B-Instruct, with LoRA added on every linear layer for
efficient fine-tuning (rank 128, scaling 256).

The inference side has three engineering designs. The first is probability correction: because EOS is so common
it biases the model toward silence, the authors introduce a threshold $\theta$, only taking EOS as the next token
when $P_j^{\texttt{[EOS]}} \ge \theta$; in practice $\theta$ set at $0.5 \sim 0.8$ is clearly better than no
threshold. The second is a continuous key-value cache: the video streams in frame by frame, and the KV cache
avoids recomputing every frame, which combined with the "tend to be silent" training makes continuous inference
very efficient. The third is encoding/decoding parallelization: CLIP ViT-L (307M) is far smaller than the 7B/8B
LLM, and the speed gap between them would cause frame skipping; the authors use a FIFO queue to let the fast
encoder keep encoding without waiting for the slow LLM to finish decoding, avoiding the bottleneck.

![The LIVE inference pipeline: frames stream in, a continuous KV cache is maintained, and the fast encoder is parallelized with the slow LLM to avoid frame skipping](imgs/fig5.png)

### A walkthrough with real numbers

Take the paper's default VideoLLM-online-7B-v1 (CLIP + Llama-2-7B-Chat, 1 CLS token per frame) running on a
5-minute Ego4D narration as an example: sampling @2 FPS produces about 600 frames, 1 token per frame, so the whole
stream is only about 600 frame tokens, far below the 4096 context limit. Suppose the query "Remind me when X
appears" is inserted at $t_r$; on every frame before X appears, the model is supervised to predict EOS on that
frame's last token (stay silent); when X appears at $t_2$, the model is supervised to output the "appeared"
language tokens. Precisely because silent frames do not occupy the sequence, the streaming method's average
training token length is only 1694, about 4× less than per-frame streaming's 6737, and the training time also
drops from 22h to 12h. At inference, setting $\theta \in [0.5, 0.8]$ for EOS threshold correction, a 5-minute
stream can run on a single A100 at 18.2 GB memory and an average 13.5 FPS.

The table below is the ablation of streaming learning methods (Ego4D Narration Stream validation set, 7B-v1). No
Training has an LM-PPL as high as 498.5 and near-zero Fluency; the interleaved dialogue does compress PPL to 2.45
but still has a TimeDiff of 6.47 and almost never stays silent; per-frame streaming lowers TimeDiff to 2.52 but
sacrifices PPL (3.34). This paper's streaming method is the best on all three metrics, with the same training
tokens and cost as the most economical interleaved.

| Method | Objective | LM-PPL↓ | TimeDiff↓ | Fluency↑ | #Train Token↓ | Cost |
|-|-|-|-|-|-|-|
| No Training | n/a | 498.5 | 6.50 | 0.1% | n/a | n/a |
| Interleaved Dialogue | Language Modeling | 2.45 | 6.47 | 11.1% | 1694 | 12h |
| Per-frame for Streaming | LM (w/ EOS turns) | 3.34 | 2.52 | 37.7% | 6737 | 22h |
| Streaming Dialogue (Ours) | LM + Streaming EOS | 2.43 | 2.32 | 42.6% | 1694 | 12h |

The efficiency ablation similarly supports the streaming design: interleaved consumes 34.4 GB and only 1.5 FPS
because it outputs language on every frame; per-frame streaming improves to 24.9 GB / 7.5 FPS; this paper's
streaming, because it spends no tokens on redundant frames and has a smaller KV cache, reaches 18.2 GB / 13.5 FPS.

| Method | Mem↓ | FPS↑ |
|-|-|-|
| Interleaved | 34.4G | 1.5 |
| Per-frame Streaming | 24.9G | 7.5 |
| Streaming | 18.2G | 13.5 |

On offline benchmarks, the authors claim SOTA among end-to-end models. Among COIN's six Top-1 Accuracy metrics,
7B-v1's step recognition is 59.8, and 8B-v1+ reaches 63.1, both higher than the previous best VideoTaskGraph
(57.2); on Ego4D LTA's ED@Z=20 (lower is better) Action column, 8B-v1+ is 0.884, better than the equally
end-to-end VideoLLM (0.921), but still slightly behind the non-end-to-end AntGPT (0.877), which uses egocentric
pretraining features and cascades multiple complex methods.

| Method | COIN Step↑ | COIN Task↑ | Ego4D LTA Action ED↓ |
|-|-|-|-|
| VideoTaskGraph | 57.2 | 90.5 | n/a |
| VideoLLM | n/a | n/a | 0.921 |
| VideoLLM-online-7B-v1 | 59.8 | 92.1 | 0.897 |
| VideoLLM-online-8B-v1+ | 63.1 | 92.7 | 0.884 |

## 🧪 Critical Assessment

### Is the problem a real need, or a setting tailored to the method

The motivation for an "always-on video assistant" is credible: GPT-4o's multimodal interaction at the time still
needed a human voice to trigger, and frame-by-frame reminders/summaries/predictions are indeed an unmet
capability. The idea of streaming EOS — "a supervision signal that never enters the sequence, teaching the model
when to be silent" — is also clean and effective, a substantive innovation rather than a rebrand. But note that
the main evaluation the paper relies on to make its case — Ego4D Narration Stream — is highly aligned with the
method's strengths: the authors themselves admit narration text is "relatively simple, mainly consisting of
subject, verb, object," which is why LM-PPL and LG-Match are barely applicable. In other words, this streaming
benchmark is defined around the simple language scenario the method can excel at, and for more complex, free-form
online dialogue ability, the paper explicitly says existing metrics are "not effective" and leaves it to future
work. This is a self-drawn evaluation boundary that the reader should be alert to.

### Are the baselines, ablations, and metrics sufficient

On the positive side, the ablation is quite complete: the learning method (interleaved / per-frame / streaming),
the streaming loss function (CE vs OHEM vs Focal), the loss weight $\tau$, and memory/speed each have a table,
and the default choice of CE and $\tau=1.0$ is data-supported. But there are several gaps worth questioning.
First, the three core metrics TimeDiff, Fluency, and LG-Match are all defined by the authors, with no comparable
baseline from an external dataset or existing literature, so the reader can hardly judge whether a TimeDiff of
2.32 seconds is good or ordinary in absolute terms. Second, the main streaming experiment is only validated on a
single dataset (Ego4D narration), and the free-form setting of COIN+Ego4D is only shown qualitatively, lacking
cross-dataset quantitative extrapolation. Third, none of the numbers show variance or confidence intervals across
multiple random seeds; for a gap like Fluency differing by only 0.2 percentage points among $\tau=0.5/1.0/2.0$,
whether it is statistically significant is unclear.

### The offline SOTA claim should be read with a discount

The paper's title and narrative make "offline benchmark SOTA" a selling point, but the comparison conditions in
the tables are not entirely on par. The authors' SOTA claim is restricted to the subset of "end-to-end models":
on Ego4D LTA, the truly best AntGPT (0.877) and Palm are listed in gray below the end-to-end dividing line, on
the grounds that they use egocentric pretraining features and cascaded methods. This delineation itself is
reasonable and disclosed, but reading it as "unconditional SOTA" would distort it — this paper's 0.884 actually
still loses to existing non-end-to-end methods. On the COIN side, a "Not use HT100M" column is added to highlight
that the authors used no extra pretraining, which, while a fairness argument, is also a way of shrinking the
comparable set to make itself stand out more.

### Does it really solve the claimed problem, and its real-world relevance

For the engineering goal of "can it run a 5-minute stream at >10 FPS on a single GPU and respond in a timely
manner," the paper's evidence (13.5 FPS, 18.2 GB) is convincing, and the method is indeed usable on its defined
task. But the larger narrative of "solving the always-on video assistant" still has a distance: the main-text
experiments deliberately use the most economical 1-token-per-frame setting, and the authors admit this sacrifices
spatial detail and that more spatial tokens are needed to do zero-shot downstream spatial understanding well,
which is precisely listed as future work; the demo's 10-token/frame setting does not enter the quantitative
evaluation. Moreover, the data engine relies on Llama-2/3 to "hallucinate" dialogue responses for COIN
annotations, and the correctness and bias of these synthetic responses are not systematically evaluated, possibly
introducing the LLM's existing bias into the supervision signal. Therefore the more measured conclusion is: this
paper gives a beautiful and efficient solution on a clearly defined but relatively narrow streaming-narration
task, but relative to its vision of a general, free-form real-time assistant, it remains a promising starting
point rather than an endpoint.

## 🔗 Related notes

<!-- There are currently no parseable existing notes under domains/vision_language directly related to streaming video dialogue, so the heading is kept empty. -->
