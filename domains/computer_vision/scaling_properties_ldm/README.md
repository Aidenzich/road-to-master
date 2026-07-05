# Bigger is not Always Better: Scaling Properties of Latent Diffusion Models — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Bigger is not Always Better: Scaling Properties of Latent Diffusion Models |
| Venue | TMLR (Transactions on Machine Learning Research) |
| Year | 2024 |
| Authors | Kangfu Mei, Zhengzhong Tu, Mauricio Delbracio, Hossein Talebi, Vishal M. Patel, Peyman Milanfar |
| Official Code | unknown |
| Venue Kind | paper |

> Note: this note covers the full text of arXiv 2404.01367 (the TMLR-accepted version — the LaTeX source carries `\usepackage[accepted]{tmlr}`, `\def\year{2024}`, and OpenReview `forum?id=0u7pWfjri5`). The authors span Johns Hopkins, Texas A&M, and Google, with the main experiments run on Google-internal data and TPUv5.

## First Principles

### What is this paper actually asking

The Achilles' heel of the diffusion model is **sampling efficiency**: an image must be run through many denoising steps before it becomes sharp, so the total cost of sampling is the product of sampling steps and the cost of each step. The two past lines of acceleration targeted these two factors respectively — designing faster network architectures to drive down the "cost per step", and designing better samplers to drive down the "number of steps". This paper points to a third, overlooked variable: **model size itself**. The question it asks is a very practical one: **under a fixed inference budget, should I use a large model or a small one?**

The authors' answer is counterintuitive, and it is exactly the title "Bigger is not Always Better": under a constrained sampling budget, smaller models frequently outperform their larger equivalents.

### Experimental setup: scaling Stable Diffusion downward as a yardstick

The paper takes the `866M` Stable Diffusion v1.5 as its baseline, changing only the base channel count $c$ of the denoising U-Net residual blocks while keeping the overall $[c, 2c, 4c, 4c]$ ratio and leaving every other architectural component untouched, thereby obtaining a family of 12 models spanning `39M` to `5B`. All models are trained from scratch on roughly 600M aesthetically-filtered internal text-to-image data (WebLI), with batch size 2048, learning rate 1e-4, for 500K steps. This "scale the channels only, keep everything else fixed" design makes the comparison across sizes a clean controlled scaling.

Table 1 is the backbone of the whole paper: it gives both the architecture specifications and the pretraining quality measured on the COCO-2014 validation set with 30k samples under 50-step DDIM (CFG=7.5):

| Params | 39M | 83M | 145M | 223M | 318M | 430M | 558M | 704M | 866M | 2B | 5B |
|-|-|-|-|-|-|-|-|-|-|-|-|
| Filters $c$ | 64 | 96 | 128 | 160 | 192 | 224 | 256 | 288 | 320 | 512 | 768 |
| GFLOPS | 25.3 | 102.7 | 161.5 | 233.5 | 318.5 | 416.6 | 527.8 | 652.0 | 789.3 | 1887.5 | 4082.6 |
| Norm. Cost | 0.07 | 0.13 | 0.20 | 0.30 | 0.40 | 0.53 | 0.67 | 0.83 | 1.00 | 2.39 | 5.17 |
| FID ↓ | 25.30 | 24.30 | 24.18 | 23.76 | 22.83 | 22.35 | 22.15 | 21.82 | 21.55 | 20.98 | 20.14 |
| CLIP ↑ | 0.305 | 0.308 | 0.310 | 0.310 | 0.311 | 0.312 | 0.312 | 0.312 | 0.312 | 0.312 | 0.314 |

There is one easily-misread point worth clearing up first: the GFLOPS, cost, and parameter counts here count **only the denoising U-Net in latent space**, and exclude the `1.4B` text encoder and the `250M` latent encoder/decoder. So the "39M model" refers to a 39M denoising network; at actual deployment it still carries a fixed encoder/decoder overhead — this must be kept in mind when interpreting "how much a small model saves".

Reading Table 1 for pretraining alone, the conclusion is in fact "bigger is better": at the full 50-step budget, FID drops monotonically from 25.30 at 39M to 20.14 at 5B (39M being the only outlier exception), while CLIP saturates early at ~0.312. The three 50-step DDIM text-to-image results below likewise confirm that bigger means better detail:

| 39M | 866M | 2B |
|-|-|-|
| ![39M model 50-step result](imgs/t2i_39M.jpg) | ![866M model 50-step result](imgs/t2i_866M.jpg) | ![2B model 50-step result](imgs/t2i_2B.jpg) |

So how does "the smaller model is better" emerge? The key lies in swapping the x-axis from "parameter count" to "**sampling cost (normalized cost $\times$ sampling steps)**".

### Core mechanism: converting sampling cost under a fixed budget (worked example)

Define the sampling cost $\text{cost} = (\text{Norm. Cost}) \times (\text{steps})$ — that is, the sampling cost (normalized cost $\times$ sampling steps). Given a fixed budget, each model can afford a different number of steps: a small model is cheap per step, so it can run several more steps; a large model is expensive per step, so under the same budget it can only run very few steps.

Taking the **fixed budget cost = 3** that the paper's Fig. 7 (`analyze_inference_costs`) explicitly names as an example, using the Norm. Cost from Table 1 to convert one by one the number of steps each model can run under this budget:

- `866M` (Norm. Cost 1.00): $3 / 1.00 = 3$ steps — only 3 steps, denoising is severely insufficient, the image is blurry.
- `318M` (Norm. Cost 0.40): $3 / 0.40 \approx 7$ steps.
- `145M` (Norm. Cost 0.20): $3 / 0.20 = 15$ steps.
- `83M` (Norm. Cost 0.13): $3 / 0.13 \approx 23$ steps — the steps are ample, enough to run the denoising trajectory to near convergence.

The paper's observation is: at cost = 3, the `83M` model achieves the best FID among all models. Intuitively, sampling quality is the product effect of "model capacity" and "whether the denoising steps are sufficient"; in the small-budget regime, the harm of insufficient steps to a large model far outweighs the harm of insufficient capacity to a small model, so the small model wins. Conversely, when the budget is loosened (steps become sufficient for the large model too), the capacity advantage surfaces, and the large model regains the lead and overtakes on detail generation. This is what the title means by bigger is not *always* better — it is "under a constrained budget", not otherwise.

The paper further validates the robustness of this scaling sampling-efficiency along a series of axes:
1. **Sampler-agnostic**: replacing DDIM with the stochastic DDPM or the higher-order DPM-Solver++, the trend that smaller models are more economical under the same sampling cost holds in every case (DPM-Solver++ is tested only at ≤20 steps because its design is unsuitable beyond 20 steps).
2. **Holds for downstream tasks (at low step counts)**: on 4× real-world super-resolution, when steps ≤ 20 the small model is still more economical; but past 20 steps, the large model is instead more efficient.
3. **Still holds after distillation**: using conditional consistency distillation to distill each model into a 4-step sampler, distillation brings roughly a $5\times$ consistent speedup to every model and improves FID across the board; but at a sampling cost ≈ 8, **the undistilled 83M small model can still match the distilled 866M large model** — showing that distillation does not overturn the scaling trend.

At the same time, there is a finding pointing in the opposite direction from the efficiency conclusion but equally important: **downstream quality is determined by pretraining**. On super-resolution, FID is driven mainly by model size rather than the amount of finetuning training; a small model, even with more training, cannot make up the quality gap that a large model's pretraining brings (Fig. 4 shows that a large SR model wins over a small one even with only a brief finetune). So this paper is not advocating "always use a small model", but conditions the choice on "which segment your inference budget falls in" and "whether you want low-step fast sampling or ultimate final quality".

## 🧪 Critical Assessment

### Sampling latency is a real pain point, and the scarce data of 12 controlled models is itself a contribution

Sampling efficiency is indeed a real pain point for deploying diffusion models, and the paper's motivation (50-step DDIM latency being too high on mobile devices) stands up. And the interaction of "model size × sampling budget" has indeed been rarely addressed head-on by past acceleration research (changing architecture, changing sampler, distillation), so the entry point is genuine. More rare still, the authors trained 12 models from scratch spanning 39M–5B; controlled experiments of this scale are beyond most teams (requiring TPU clusters, weeks, and hundreds of thousands of dollars of cost), and the scarcity of the data points is itself a contribution.

### FID is the sole judge, and cost counts only the U-Net while omitting the 1.4B text encoder and 250M autoencoder

Controlled scaling (changing only the channel count) is a clean design, and the two robustness axes of samplers (DDIM/DDPM/DPM-Solver++) and distillation are supplemented honestly. But there are three structural weaknesses to name. First, **the metric relies almost entirely on FID**: the authors themselves admit in the limitations that because there were over 1000 variants they gave up human evaluation, and candidly acknowledge that FID may diverge from visual quality. The whole "small models are more economical" conclusion is essentially "small models are more economical on FID", and the way FID is sensitive to sampling diversity/blurriness may just so happen to systematically favor the smoother small-model outputs with ample steps; this potential bias is not cross-checked by an independent metric. Second, **the cost definition counts only the denoising U-Net**, excluding the fixed 1.4B text encoder and 250M autoencoder overhead from the normalized cost; in real end-to-end latency, this fixed overhead dilutes the relative "cheapness" advantage of small models, so the paper's cost axis is optimistic for small models. Third, the data and models are all Google-internal with no released code, so outsiders cannot reproduce these 12 models or verify the numbers in Table 1.

### "Bigger is better" is an existing pretraining conclusion; the reversal on the cost axis is the real novelty

One must stay clear-headed: Nichol et al. long ago pointed out that diffusion models get "bigger is better", and the LLM scaling-law literature (Kaplan, Hoffmann, etc.) has long discussed the compute-optimal tradeoff. The pretraining part of this paper (Table 1, Fig. 3) is basically restating "bigger means lower FID". The real novelty concentrates in the reversal phenomenon that emerges once the x-axis is swapped for "sampling cost", and its consistency across the three axes of sampler/downstream/distillation. This is a valuable but relatively limited observation — it is closer to "a robust empirical regularity" than to an extrapolable quantitative scaling law (the paper gives no closed-form relation that can predict the optimal model size).

### The reversal point holds only under the custom normalized-cost and FID-only coordinates

There is a point to be wary of: the most striking conclusions (83M best at cost=3, 83M matching the distilled 866M at cost≈8) all hold under the authors' custom normalized-cost coordinate and with FID as the sole judge. Change the cost definition (include encoder/decoder), change the quality metric (human evaluation or a diversity metric), and the reversal point may well shift or even vanish. The paper also honestly narrows the generality to "holds only for the SD v1.5 U-Net family studied in this work", explicitly stating that transformer backbones (DiT/SiT/MM-DiT) and cascaded models (Imagen3, Stable Cascade) are unverified — this is an appropriate self-limitation, but it also means the applicable scope of the conclusion is narrower than the universal tone of the title.

### Usable as a selection guide for low-budget deployment, but lacking end-to-end latency and quantitative-selection validation

As a "decision guide" it is useful: if your deployment is stuck in the low-step/low-budget regime, prioritizing a smaller model is a reasonable strategy with empirical support, and this conclusion is robust to both sampler and distillation. But it does not solve the quantitative problem of "how to automatically select the optimal model size under a given hardware and quality threshold", nor does it validate the magnitude of the advantage under end-to-end real latency and real metrics. So my judgment is: this is a solid, honest, but conclusion-conditional and metric-single empirical study, whose value lies in proposing and robustifying a counterintuitive regularity, rather than providing an extrapolable scaling law. It does not moan without cause, but neither should it be read as "small models are better across the board".

## 🔗 Related notes

- [DDPM, Denoising Diffusion Probabilistic Models](../diffusion/) — the denoising-diffusion foundation and the DDPM/DDIM samplers on which this paper's sampling-efficiency discussion depends.
