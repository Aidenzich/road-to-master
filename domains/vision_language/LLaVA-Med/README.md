# LLaVA-Med — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day |
| Venue | NeurIPS 2023 (Datasets and Benchmarks Track) |
| Year | 2023 |
| Authors | Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, Jianfeng Gao |
| Official Code | https://github.com/microsoft/LLaVA-Med |
| Venue Kind | paper |

## First Principles

General-domain large multimodal models (LMMs) such as LLaVA are trained on generic web image-text pairs, and when facing biomedical images they often behave like a layperson — refusing to answer, giving wrong responses, or even hallucinating entire passages. LLaVA-Med's core thesis is exactly this: to adapt an off-the-shelf general LMM into a conversational assistant capable of open-ended question answering over biomedical images at an affordable cost, rather than training from scratch or doing traditional classification-style VQA.

The key of the whole method is not a new network but a data engine. The authors sample from PMC-15M — a large-scale figure-caption dataset (15 million biomedical image-text pairs) extracted from PubMed Central — and then use language-only GPT-4 to self-instruct instruction-following data based solely on the text (the figure caption plus the sentences in the source article that mention the figure, i.e. citances). GPT-4 never sees the image throughout, and is only asked to produce multi-turn Q&A in a tone "as if it could see the figure," so the entire pipeline requires zero human annotation.

The concept-alignment data used in the first stage turns 600K image-text pairs sampled from PMC-15M into instructions using the most naive expansion: the instruction only asks to "describe this image," and the target output is the original caption. It switches between a "concise description" and a "detailed description" set of prompts by caption length, using 30 words as the cutoff (about 25% of PMC-15M captions have fewer than 30 words). This stage covers only a single task (image description), aiming to blanket the coverage of biomedical concepts.

The instruction-tuning data used in the second stage first filters out multi-subfigure images and keeps only single-subfigure ones, then samples 60K pairs from the five most common modalities — CXR (chest X-ray), CT, MRI, histopathology, gross pathology — uses GPT-4 to generate multi-turn Q&A, and treats the sentences that mention the figure in the source PubMed article (inline mentions, IM) as additional context. The authors deliberately make three versions — 10K, 60K, 60K-IM — to ablate the impact of the data generation strategy on the downstream model.

Each alignment sample is organized into a single-turn instruction-following format, where $\Xmat_{\texttt{q}}$ is the sampled description instruction, $\Xmat_{\texttt{v}}$ is the image, and $\Xmat_{\texttt{c}}$ is the caption used as the target output:

```
Human : X_q  X_v  <STOP>\n
Assistant : X_c  <STOP>\n
```

The model architecture directly follows LLaVA: a vision encoder, a linear projection layer, and a language model. Training uses a two-stage curriculum learning. Stage 1 (biomedical concept feature alignment) freezes both the vision encoder and the language model and updates only the projection matrix, aligning a large number of new biomedical visual concepts to the language model's existing word embeddings; Stage 2 (end-to-end instruction tuning) freezes only the vision encoder and updates both the projection layer and the language model weights, letting the model learn open-ended conversational semantics. The authors analogize this process to a layperson gradually being trained into a professional assistant.

![Figure 3: schematic of LLaVA-Med's two-stage curriculum-learning training pipeline](imgs/training_pipeline.png)

This recipe emphasizes "affordability": Stage 1 and Stage 2 take about 7 and 8 hours respectively, totaling under 15 hours, running on 8× 40G A100 GPUs, which is the origin of the title "in One Day." The authors provide the actual wall-clock time for each stage and epoch, letting users make their own cost-quality trade-offs:

| Stage / epoch | Stage 1 (1 ep) | Stage 1 (3 ep) | Stage 2 10K (1 ep) | Stage 2 10K (3 ep) | Stage 2 60K (1 ep) | Stage 2 60K (3 ep) |
|-|-|-|-|-|-|-|
| Time (hours) | 6.8 | 19.4 | 0.6 | 1.8 | 2.6 | 8.0 |

Evaluation is split into two axes. The first is open-ended visual conversation: the authors construct 193 brand-new questions (143 conversation plus 50 detailed description), use language-only GPT-4 as a judge to score candidate models and the GPT-4 reference answers on helpfulness, relevance, accuracy, and level of detail, and then compute a relative score normalized by GPT-4's reference score. The table below gives the overall relative scores on these 193 questions for each setting:

| Model setting | Conversation | Description | Overall |
|-|-|-|-|
| LLaVA | 39.4 | 26.2 | 36.1 |
| LLaVA-Med Stage 1 | 22.6 | 25.2 | 23.3 |
| LLaVA-Med 10K | 42.4 | 32.5 | 39.9 |
| LLaVA-Med 60K | 53.7 | 36.9 | 49.4 |
| LLaVA-Med 60K-IM | 55.1 | 36.4 | 50.2 |

Walking through the actual numbers makes the effect of curriculum learning clearer: general LLaVA has an overall relative score of 36.1; doing only Stage 1 actually drops to 23.3, because the single image-description instruction makes the model lose the ability to follow diverse instructions; after adding the Stage 2 instruction data, 10K→60K→60K-IM rise back to 39.9, 49.4, 50.2 respectively, with the best 60K-IM version reaching 50.2% of the GPT-4 reference ceiling. This curve illustrates two things at once: Stage 1 alone is insufficient to be a chatbot, and both instruction-data volume and inline mentions contribute positively to quality.

The second axis is three existing biomedical VQA benchmarks: VQA-RAD, SLAKE, PathVQA. Closed-set questions report accuracy and open-set questions report recall (because LLaVA-Med answers by free-text generation rather than selecting from a candidate set). The table below excerpts the comparison against prior supervised SoTA after downstream fine-tuning (closed-set accuracy):

| Method | VQA-RAD Closed | SLAKE Closed | PathVQA Closed |
|-|-|-|-|
| LLaVA | 65.07 | 63.22 | 63.20 |
| LLaVA-Med (From LLaVA) | 84.19 | 85.34 | 91.21 |
| LLaVA-Med (BioMed CLIP) | 83.09 | 86.78 | 91.09 |
| M2I2 | 83.50 | 91.10 | 88.00 |

After downstream fine-tuning, LLaVA-Med sets new supervised SoTA on the closed-set questions of VQA-RAD and PathVQA (VQA-RAD closed 84.19, PathVQA closed 91.21), verifying that as long as the instruction is precise enough (e.g. a yes/no question), the model can reliably complete biomedical tasks by following the instruction.

But on open-set questions, LLaVA-Med only achieves SoTA on SLAKE and is limited or even lags existing methods on the other datasets; the authors attribute the reason to open-ended biomedical questions being inherently semantically ambiguous when the candidate answers are not restricted. This honest self-assessment marks precisely the boundary of this line of approach.

## 🧪 Critical Assessment

### Hallucination on medical images is a safety risk, not just a quality flaw
The problem itself is real: general LMMs hallucinate like a layperson on medical images, which in a clinical context is not just a quality flaw but a safety risk. The authors also admit in the conclusion that LLaVA-Med is still limited by hallucination and weak in-depth reasoning. Taking "domain adaptation" rather than "a larger model" as the core thesis is a reasonable and pragmatic entry point for high-value but data-scarce vertical domains.

### GPT-4 is both examiner and grader: the risk of circular evaluation
The ablation is quite solid: Table 3(b) systematically sweeps the number of epochs for Stage 1/2 and downstream fine-tuning, 7B vs 13B language models, and CLIP vs BioMedCLIP vision encoders, and attaches running time for cost-quality trade-offs. The real concern lies in the metric: the main metric for chat quality is a GPT-4 self-assessed relative score, while the training data is also generated by GPT-4 — the same model family is both examiner and grader, which carries the bias risk of circular evaluation, and a pretty number does not necessarily equal independent clinical validity.

### The novelty lies in the "caption→instruction" data recipe and the two-stage curriculum, not the architecture
In terms of components, LLaVA-Med almost entirely uses existing building blocks: the LLaVA architecture, GPT-4 self-instruct, PMC-15M data, and optional Vicuna or BioMedCLIP initialization. The real novelty concentrates in the "caption→instruction" data generation recipe and the two-stage curriculum, closer to successfully transferring a mature recipe to a high-value vertical domain than to an architecture-level innovation. In fairness, open-sourcing its dataset and pipeline, together with the training cost being as low as "within one day," is itself a contribution of practical value and should not be dismissed as mere "recombination."

### Closed-set sets new SoTA, open-set still lags: the capability boundary and the self-coupling of the 193-question benchmark
"Is it solved" must be examined separately. Closed-set VQA sets new SoTA, which in essence is answering yes/no and multiple-choice questions correctly with a generative model; whereas on open-set questions, which are closer to real clinical inquiries, LLaVA-Med mostly lags existing methods, showing that open-ended biomedical understanding is far from solved. Moreover, the 193-question visual conversation benchmark is generated by the authors with the same self-instruct pipeline as the training data, so the benchmark's definition is highly coupled to the model's strengths, and the GPT-4 reference answers consume the golden caption and inline mentions rather than a true understanding of the image, making the chat scores more like internal consistency than clinical validity. The evaluation also does not touch real deployment, annotator agreement, or safety audits. Therefore this paper is better understood as an affordable domain-adaptation "recipe and data asset" than a clinically usable finished system.

## 🔗 Related notes

- [Stable Diffusion](../stable-diffusion/)
