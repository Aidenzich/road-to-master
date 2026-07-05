# Video-MME — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis |
| Venue | CVPR 2025 |
| Year | 2025 |
| Authors | Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, ..., Ran He, Xing Sun (21 authors total, from NJU / XMU / HKU / PKU / CUHK / ECNU / CASIA, etc.) |
| Official Code | https://video-mme.github.io |
| Venue Kind | paper |

## First Principles

### What exactly does this paper try to solve

Past evaluations of multimodal large language models (MLLMs) almost all concentrated on **static image** understanding, lacking a high-quality benchmark that comprehensively and finely measures a model's ability to "watch video." Video-MME was born to fill this gap: it manually collects 900 YouTube videos and annotates 2,700 four-way multiple-choice questions (3 per video), spanning 6 major domains (Knowledge, Film & Television, Sports Competition, Artistic Performance, Life Record, Multilingual) and 30 fine-grained categories, with video lengths ranging from 11 seconds to 1 hour. It also incorporates subtitles and audios into the evaluation, so that the evaluation is not only about the visuals. (This note is based on the arXiv:2405.21075 version, i.e. the CVPR 2025 camera-ready source; the official conference version may differ slightly.)

![Video-MME's statistical analysis: left is the distribution of 6 domains and 30 subcategories, right is the distribution of video duration and question types](imgs/Video-MME.png)

The authors deliberately use a three-step procedure to squeeze out data quality: first they build a domain hierarchy based on trending YouTube topics and collect short (< 2 minutes), medium (4–15 minutes), and long (30–60 minutes) videos, obtaining 900 videos with 744 subtitles and 900 audio files; then annotators with vision-language research experience watch each video in full before writing questions; finally there is a manual review, and questions that give "only the text stem" are fed to Gemini 1.5 Pro, and any question answerable from text alone is removed. Statistics show that Gemini 1.5 Pro achieves less than 15% accuracy in the text-only setup, which is used to counter-prove that the questions indeed require watching the video to be answered.

### Using certificate length to quantify "how hard is this question"

How to objectively measure how long a question requires "watching the video" to answer? Video-MME follows EgoSchema's certificate length: the certificate of a QA pair is "the minimal necessary set of sub-clips sufficient for a human verifier to confirm the annotated answer is correct," and the certificate length is the total duration of these sub-clips. Below we write this definition as a formula using our own notation (the notation is added by this note):

$$\mathrm{CL}(q) = \sum_{c \in \mathcal{C}(q)} \lvert c \rvert$$

where $\mathcal{C}(q)$ is the minimal sufficient set of sub-clips for question $q$, and $\lvert c \rvert$ is the duration of sub-clip $c$. The authors randomly sample 3 videos per category to estimate the distribution, obtaining median certificate lengths for short, medium, and long videos of 26s, 164.7s, and 890.7s respectively — the long-video subset requires digesting far more content than EgoSchema (whose videos cap at just 180 seconds), which is the main basis for the authors' claim that Video-MME is the "most challenging" video QA dataset.

### The main results at a glance

The table below excerpts a few representative rows from the "Performance of MLLMs on Video-MME" table (overall is the overall accuracy across all durations, %):

| Model | LLM Params | Short w/o subs | Long w/o subs | Overall w/o subs | Overall w/ subs |
|-|-|-|-|-|-|
| Gemini 1.5 Pro | - | 81.7 | 67.4 | 75.0 | 81.3 |
| GPT-4o | - | 80.0 | 65.3 | 71.9 | 77.2 |
| GPT-4V | - | 70.5 | 53.5 | 59.9 | 63.3 |
| VILA-1.5 | 34B | 68.1 | 50.8 | 59.0 | 59.4 |
| VITA-1.5 | 7B | 67.0 | 47.1 | 56.1 | 58.7 |
| InternVL-Chat-V1.5 | 20B | 60.2 | 45.6 | 50.7 | 52.4 |

Using frame input alone, Gemini 1.5 Pro attains an accuracy of 75%, ahead of GPT-4o's 71.9% and GPT-4V; the strongest open-source model VILA-1.5 (34B) only reaches 59.0% overall, still with a clear gap from commercial models. Notably, the pure-image model InternVL-Chat-V1.5 also reaches 50.7% by multi-frame input, comparable to the video-specialized LLaVA-NeXT-Video, which the authors use to argue that image understanding is the foundation of video understanding, and to show that the benchmark is applicable to both image and video models.

### A concrete modality-ablation example

![Radar charts of four representative models across question types, with counting as the common bottleneck](imgs/radar_performance.png)

Focusing on Gemini 1.5 Pro's per-category ablation (the "Performance of Gemini 1.5 Pro across six major categories" table) makes the value of multimodality clear. Take the long videos in the Multilingual category as an example: with only frames the accuracy is 70.8%, and after adding subtitles it jumps to 87.5% (+16.7), while adding audio instead reaches 83.3% (+12.5). Zooming out to the entire long-video subset, adding subtitles raises overall from 67.4% to 77.4% (+10.1), whereas on short videos adding subtitles only brings +2.8, showing that the marginal benefit of subtitles is far greater for long videos than short ones. Another axis is duration: Gemini 1.5 Pro's accuracy declines by −14.3% from short to long videos, exposing the model's weakness in long-range temporal relationships; the authors attribute the main causes to the rising proportion of reasoning questions in long videos, the fixed number of frames leading to over-sparse sampling, and the inherent difficulty of long contexts.

## 🧪 Critical Assessment

### The problem is real, but the "first-ever" framing is debatable

The insufficiency of video-understanding evaluation is a real problem: putting open-domain videos from 11 seconds to 1 hour, with subtitles and audios, into the same manually annotated benchmark indeed fills a gap in existing benchmarks. However, the positioning of "the first-ever comprehensive" carries a marketing element — contemporaneous benchmarks such as MVBench, TempCompass, and EgoSchema already existed for video MLLMs, and Video-MME's differentiation lies mainly in the combination of "long video + multimodal + manual annotation" rather than originality on any single dimension; reading it as "an engineering integration that assembles several existing dimensions at once" is closer to the fact than "a brand-new problem."

### Several gaps in metric and baseline design

The scale is actually not large: 2,700 questions, with only 3 per video, which works out to an average of only about 90 questions per 30-subcategory, and the long-video subset has only 300 questions, so statistically a difference of a few percentage points between models likely falls within noise, yet the paper does not report any confidence interval or significance test. Certificate length is estimated from only 3 videos per category (about 90 videos across the whole table), an extremely small sample, yet it is used to support a global claim like "most challenging," which is unconvincing. Moreover, the main modality-gain conclusions heavily depend on a single model, Gemini 1.5 Pro — the per-category modality ablation is done only on it, and almost no open-source model can consume audio, so the extrapolability of conclusions like "subtitles are more effective than audio" is questionable.

### The circular risk from "setting your own questions and grading them yourself"

The question filtering uses Gemini 1.5 Pro (filter out QA pairs answerable from text alone), while Gemini 1.5 Pro is also the top-ranked tested model on the leaderboard, which constitutes a subtle circularity: using a model to filter out the questions it itself thinks can be answered from text alone may systematically retain question types friendly to that model's visual pipeline, letting it dominate the final leaderboard. The paper does not cross-use multiple filters to rule out this bias. The domain list itself also has a minor flaw: when the main text lists the 6 domains it only writes out 5 (omitting Artistic Performance), showing that the description of the annotation process is not entirely rigorous.

### Does this benchmark actually push the problem toward being solved

As a "diagnostic tool," Video-MME is successful: it clearly quantifies the two phenomena of "the longer the duration, the lower the accuracy" and "multimodality can supplement information," which helps the community define the next step. But it measures model capability and itself provides no solution; and because the videos all come from public YouTube content, newer commercial models may well have seen related material during pretraining, causing potential data contamination, which would inflate the leaderboard's absolute numbers over time. As a snapshot leaderboard it has a shelf life, but as evidence that "long-video understanding is solved" it falls far short.

## 🔗 Related notes

<!-- There are currently no safely parseable related notes under vision_language; the heading is kept and left empty for now. -->
