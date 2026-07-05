# LLM-Training-Types — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | The distinct training types/stages of modern LLMs and what each stage changes |
| Venue | unknown |
| Year | unknown |
| Authors | unknown |
| Official Code | unknown |
| Venue Kind | survey |

## 📚 Sources

This note draws on the following four papers as its primary evidence sources; the full texts of all of them were obtained and stored in the local cache. The table below records only the preprint versions actually fetched, while the formal conference versions are annotated by inference in the main text.

| # | Title | Venue (fetched) | Year | arXiv id | Access |
|-|-|-|-|-|-|
| 1 | Training language models to follow instructions with human feedback (InstructGPT) | arXiv preprint | 2022 | arXiv:2203.02155 | fetched (arXiv full text) |
| 2 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | arXiv preprint | 2023 | arXiv:2305.18290 | fetched (arXiv full text) |
| 3 | LoRA: Low-Rank Adaptation of Large Language Models | arXiv preprint | 2021 | arXiv:2106.09685 | fetched (arXiv full text) |
| 4 | Instruction Tuning for Large Language Models: A Survey | arXiv preprint | 2023 | arXiv:2308.10792 | fetched (arXiv full text) |

Based on the author's prior knowledge (not verified word-for-word against the fetched arXiv full texts, and thus a reasonable inference): #1 is NeurIPS 2022, #3 is ICLR 2022, #2 is NeurIPS 2023, and #4 is a survey-style preprint on the instruction tuning topic; the numbers in the formal camera-ready versions may differ slightly from the preprints.

## Why the seed's four-column table needs to be broken apart

The issue's seed uses the four columns "Pre-training / Post-Pretraining / Fine-tuning / Instruct-tuning," treating each cell as a parallel "stage," and attaches order-of-magnitude guesses for data volume and compute. The actual literature does not support this flat four-column carving: in the survey, instruction tuning is almost synonymous with SFT, preference optimization (RLHF/DPO) is another independent alignment stage after SFT, and LoRA is not a stage at all but a method that can be applied to any fine-tuning stage. The survey defines instruction tuning as supervised re-training on (instruction, output) pairs, whose purpose is that it bridges the gap between the next-word prediction objective of LLMs and the user's goal of having the model obey instructions—this single sentence shows that what it tunes is the behavioral interface of "outputting according to instructions," not a re-injection of knowledge.

The table below is the seed re-calibrated against the four papers; the order-of-magnitude guesses are replaced with the real numbers from the papers, and "method" is separated from "stage":

| Training component | Attribute | Objective function / mechanism | Typical data volume (real numbers from this note's literature) | What it mainly changes |
|-|-|-|-|-|
| Pre-training | stage | self-supervised next-token prediction | hundreds of billions to trillions of tokens (e.g., using the survey's citation of T5's pre-training stage as a baseline) | the primary source of knowledge and general capability |
| Continued / continual pre-training | stage (optional) | same objective as pre-training, swapping in domain/time-period corpora | additional corpus, smaller in magnitude than the initial pre-training | supplements domain/recency corpora, still the same objective |
| SFT ＝ instruction tuning | stage | supervised fine-tuning on (instruction, output) | InstructGPT's SFT dataset contains about 13k training prompts; the survey states fine-tuning costs 0.2% of T5's pre-training compute | makes the model obey instructions, with output format and style aligned to human expectations |
| Preference optimization (RLHF / DPO) | stage | RLHF: RM + PPO; DPO: single binary cross-entropy loss | InstructGPT: RM 33k, PPO 31k prompts, 6B RM | reorders the output distribution according to human preference rankings |
| LoRA / PEFT | method (orthogonal to stages) | freeze weights, inject low-rank BA updates | on GPT-3 175B, 10,000 times fewer trainable parameters possible | does not change stage semantics, only changes "how to do fine-tuning cheaply" |

The two key corrections in this table are: first, the Fine-tuning and Instruct-tuning that the seed lumped together are converged into the same SFT stage, because the survey plainly states that SFT allows for a more controllable and predictable model behavior compared to standard LLMs, so both tune the same thing; second, LoRA is pulled out of the "stage column" and stands alone as a "method," because it can be applied on top of any fine-tuning stage.

## Pre-training provides the knowledge, and the alignment stages mostly only change the interface

To judge the folk claim of "whether instruct-tuning adds new knowledge," one must first look at where the knowledge comes from. The superficial alignment hypothesis compiled in the survey argues that the knowledge and capabilities of a model are almost acquired in the pre-training stage, while the subsequent alignment training (including instruction tuning) teaches models to generate responses under user-preferred formalizations. If this hypothesis holds, the seed's sentence that "instruct-tuning does not add new factual knowledge, it only improves parsing and response capability" is broadly defensible—but note that in the paper it is a hypothesis (LIMA validated it with about 1k examples), not a widely re-verified law.

## SFT / instruction tuning changes behavior, but the survey itself also has reservations

The survey does not uncritically endorse instruction tuning wholesale: it plainly records a strong skepticism, holding that SFT captures surface-level patterns and styles (e.g., the output format) rather than comprehending and learning—that is, the model may only learn the surface format of the output, not truly understand the task. This point both supports and inversely cautions the seed's claim: it supports the "no new knowledge" part, yet also hints that "improving parsing and response capability" may be overstated as "only learning the format."

InstructGPT provides the cleanest evidence of behavioral change: under human evaluation, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters. These two models have identical architectures and differ only in whether they were fine-tuned with human data, so the preference gap comes from alignment rather than scale or new knowledge. At the same time, InstructGPT models show improvements in truthfulness and reductions in toxic output generation, showing that alignment did move behavioral dimensions like "truthfulness/toxicity," not just format wrapping.

## Preference optimization: from RLHF's three steps to DPO's one step

InstructGPT's alignment is three steps: (1) supervised fine-tuning (SFT), (2) reward model~(RM) training, and (3) reinforcement learning via proximal policy optimization (PPO). Its data scale is written very concretely in the paper—the SFT dataset contains about 13k training prompts; RM and PPO are 33k and 31k prompts respectively, and In this paper we only use 6B RMs, using 6B rather than 175B as the reward model to save compute and stabilize RL.

DPO collapses steps (2) and (3) above into one step: it optimize a policy using a simple binary cross entropy objective, and does so without learning an explicit, standalone reward model or sampling from the policy during training. Its loss function writes the "classification over preference data" in closed form:

$$
\mathcal{L}_\text{DPO}(\pi_{\theta}; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}\left[\log \sigma \left(\beta \log \frac{\pi_{\theta}(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_{\theta}(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

DPO's theoretical selling point is that it implicitly optimizes the same objective as existing RLHF algorithms (reward maximization with a KL-divergence constraint)—this is also the meaning of the title's phrase Your Language Model Is Secretly a Reward Model: the policy model itself implicitly encodes the reward. Experimentally, on TL;DR summarization, DPO has a win rate of approximately 61 % (temperature 0.0), slightly beating PPO's roughly 57%, showing that removing explicit RL did not sacrifice quality.

## LoRA / PEFT is an orthogonal "method," not a "stage"

Putting LoRA in the stage column is the seed's biggest category error. LoRA's approach freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, approximating full fine-tuning with a low-rank update:

$$
h = W_0 x + \Delta W x = W_0 x + BA x, \qquad W_0+\Delta W=W_0+BA
$$

Its benefit is on the resource side and does not change the semantics of any stage: on GPT-3 175B, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times, with the checkpoint going from 350GB to 35MB, and LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3. At deployment, merging BA back into the weights means we do not introduce any additional latency during inference. Precisely because LoRA can be applied on top of the SFT or preference stage, it and "which stage" are two orthogonal axes.

## An end-to-end quantitative comparison

Walking through InstructGPT's signature numbers is the clearest way to see "what the alignment stage actually changed, and what it did not." Same GPT-3 architecture: the 175B version is unaligned; the 1.3B version first does SFT with the SFT dataset contains about 13k training prompts, then aligns with 33k/31k preference and PPO data, and finally the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3. On the surface this looks like "a small model beating a large model," but it only proves alignment's effect on the preference metric of "answering according to instructions," and does not prove that 1.3B has 175B's breadth of knowledge; on the contrary, the paper admits the alignment process comes at the cost of lower performance on certain tasks that we may care about (SQuAD, DROP, HellaSwag, WMT all regress), and requires mixing PPO updates with updates that increase the log likelihood of the pretraining distribution to keep it in check. Converting to compute, the survey's order-of-magnitude figure is fine-tuning costs 0.2 % (relative to T5 pre-training)—so "alignment is cheap, only changing the interface" is broadly right in direction, but the half-sentence "with no cost at all" is directly overturned by the alignment tax.

## 🧪 Critical Assessment

### Comparability of the four sources

The four papers span different generations and tasks, so directly comparing them side-by-side is dangerous. InstructGPT's 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3 is a 2022, GPT-3 family, human preference on the OpenAI API prompt distribution; LoRA is 2021, with task-style metrics on RoBERTa/GPT-2/GPT-3; DPO is 2023, measured by win rate. None of these is a head-to-head against "contemporary frontier models," so the stage map this note assembles from them is more like "separate cross-sections of each generation" than a single ruler, and this must be preserved when citing their numbers.

### Whether the benchmarks and metrics are sufficient to support the conclusions

Preference-type metrics themselves carry the risk of a self-defined bullseye. InstructGPT's preference is evaluated on its own API's prompt distribution by its own hired annotators; DPO's win rate of approximately 61 % also relies on automatic evaluation and reference completions, and the paper itself acknowledges that automatic evaluation metrics such as ROUGE can be poorly correlated with human preferences. That is, "alignment makes outputs better" is to a large extent relative to "the kind of good that the evaluators prefer," and reading it as a general capability improvement over-extrapolates.

### Novelty vs. repackaging: method and stage are conflated

The essence of the seed's classification error is treating a method as a stage. LoRA's contribution is on the resource side—we do not introduce any additional latency during inference, 10,000 times fewer parameters—it neither adds nor removes the semantics of any training stage; likewise, instruction tuning and SFT are two names for the same thing in the survey. Listing these as four equivalent columns leads readers to mistakenly think they are four mutually exclusive steps in a pipeline, which is a repackaging of naming and classification rather than a real division of stages.

### Whether the claimed problem is really solved, and whether it matters for practice

The sentence the seed most wants to validate—"instruct-tuning adds no new knowledge, only changes parsing and response"—has split evidence. On one hand, the superficial alignment hypothesis says the knowledge and capabilities of a model are almost acquired in the pre-training stage, supporting the first half; on the other hand, InstructGPT's alignment tax shows that alignment comes at the cost of lower performance on certain tasks that we may care about, directly disproving the implicit conclusion of "only changing the interface, zero cost." Moreover, superficial alignment is only a hypothesis validated with about 1k examples, not yet widely re-verified on contemporary large models, so the correct reading for practice is: alignment mainly changes behavior and format at low cost, but there will be a measurable capability degradation, and "all knowledge comes from pre-training" remains to be proven rather than settled.

## 🔗 Related notes

- [InstructioinTuningWithGPT4](../InstructioinTuningWithGPT4/)
- [Lora](../Lora/)
- [ChatGPT](../ChatGPT/)
