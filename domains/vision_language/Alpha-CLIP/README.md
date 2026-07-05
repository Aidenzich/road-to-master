# Alpha-CLIP — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Alpha-CLIP: A CLIP Model Focusing on Wherever You Want |
| Venue | CVPR |
| Year | 2024 |
| Authors | Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, Jiaqi Wang |
| Official Code | https://aleafy.github.io/alpha-clip |
| Venue Kind | paper |

> This note is written from the arXiv version (arXiv:2312.03818, whose LaTeX source is the CVPR 2024 submission template); the official camera-ready version may differ slightly. All numbers and quotations follow the paper's original text.

## First Principles

### Starting from CLIP's blind spot: why an alpha channel is needed

The original CLIP obtains aligned features through "whole image ↔ whole caption" contrastive learning, so it inherently mixes the semantics of every object in the frame together and cannot answer with respect to only a user-specified local region. Past workarounds fall into two families: crop the region of interest into a standalone patch, or mask the irrelevant pixels/features to a background color — the former cuts off context, while the latter directly alters the original image content. Another line (e.g. MaskCLIP) instead uses an attention mask to make the global `[CLS]` token attend only to a local region, but it "can only emit a `[CLS]` token," and therefore cannot feed downstream models that need the entire feature map (BLIP-2, LLaVA, Point-E). Alpha-CLIP's motivation is exactly this: to inject the "where to look" hint into the model without corrupting the original image and while preserving the whole feature map.

### Data engine: turning image-text into RGBA region-text

To fine-tune a CLIP that can consume an alpha channel, the key bottleneck is data. The authors design a dual-branch data pipeline. The first, a grounding data pipeline, directly reuses Kosmos-2's GRIT dataset (GLIP and CLIP automatically label box-level region-text pairs), then uses SAM to produce high-quality pseudo-masks for each box, upgrading them into mask-level pairs; the classification experiments scale this up to GRIT-20m. This branch keeps the full background of natural images, teaching the model to "focus when there is context."

The second, a classification data pipeline, targets object-centric scenarios: it first uses SAM to generate several masks for each ImageNet image, crops, centers, and enlarges the foreground, then scores each image against its class label with CLIP and takes the high-scoring masks; on the text side, it places the foreground on a pure white background and hands it to BLIP-2 to generate a caption, finally merging the ImageNet fine-grained class label with the BLIP-2 caption. From this the authors ultimately select about 460k RGBA region-text pairs, used to train tasks such as REC, OVD, region captioning, and 2D/3D generation.

### Model architecture: a zero-initialized Alpha Conv parallel to the RGB convolution

The architectural change is deliberately kept minimal to preserve the CLIP prior. The first layer of the ViT image encoder is originally a large-kernel RGB convolution; the authors add an Alpha Conv "side by side" next to it, dedicated to consuming the extra alpha channel, whose input range is $[0,1]$ (1 for foreground, 0 for background). The most crucial trick is to initialize this Alpha Conv's weights to all zeros, so that at the start of training Alpha-CLIP completely ignores the alpha channel and behaves identically to the original CLIP, after which gradients gradually learn to exploit this new channel. The training objective reuses CLIP's image-text contrastive loss (InfoNCE), where $\mathcal{V}$, $\mathcal{T}$ are the image and text encoders, $\langle\cdot,\cdot\rangle$ is cosine similarity, and $\tau$ is the temperature; the standard form is given below (the notation is added by this note):

$$
\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B}\sum_{m=1}^{B}\log\frac{\exp\!\big(\langle\mathcal{V}(x^{(m)}),\mathcal{T}(s^{(m)})\rangle/\tau\big)}{\sum_{n=1}^{B}\exp\!\big(\langle\mathcal{V}(x^{(m)}),\mathcal{T}(s^{(n)})\rangle/\tau\big)}
$$

### Training recipe: freeze the text side, fully unfreeze the image side

During training the text encoder is frozen throughout and only the image encoder is trained, with a lower learning rate for the subsequent transformer blocks than for the first-layer Alpha Conv. To keep the model from forgetting how to "look at the whole image," the authors use a sampling strategy: with a ratio of $r_s=0.1$ they swap the RGBA-text back to the original image-text pair and set alpha to all ones. The appendix ablation shows that unfreezing all 12 transformer blocks works best (LoRA is worse), and $r_s=0.1$ is the sweet spot (neither no sampling nor too much sampling is as good). Concrete hyperparameters include batch size 4096, AdamW (weight decay 2e-2), Alpha Conv learning rate 2e-4, other layers 2e-6, a cosine scheduler; GRIT-1m is trained for 6–8 epochs and GRIT-20m for 2 epochs.

![Alpha-CLIP's data generation pipeline and model architecture: (a) generating millions of RGBA region-text pairs, (b) adding an Alpha Conv side by side in the first layer](imgs/method_overview.png)

### One concrete forward pass: ImageNet-S zero-shot classification

Take ViT-L/14 and walk through it once. Given a 224×224 input image, the first-layer convolution has kernel/stride both 14, so it is split into $224/14=16$, i.e. $16\times16=256$ patch tokens plus one `[CLS]` token. Besides the three RGB channels, we also split the alpha map aligned with the foreground mask into the same grid, which after the zero-initialized Alpha Conv is added into the patch embedding. The evaluation dataset ImageNet-S has 919 classes with semantic segmentation annotations, and the authors use "the mean of per-class accuracy" as the metric. When the alpha given is the whole image (all ones, i.e. no region hint), ViT-L/14 reaches a top-1 of 73.37, almost on par with the original CLIP's 73.48; switching to a rectangular-box alpha raises it to 75.62; switching to a fine mask alpha raises it further to 77.41. This increasing curve of "no region → box → mask" is direct evidence that the alpha channel indeed steers attention toward the foreground.

| Alpha Map (ViT-L/14) | Top-1 | Top-5 |
|-|-|-|
| Original CLIP (no alpha) | 73.48 | 91.60 |
| whole image (all ones) | 73.37 | 91.75 |
| rectangular box | 75.62 | 93.34 |
| mask | 77.41 | 94.45 |

### From recognition to downstream: REC, OVD, and MLLM/generation

The same Alpha-CLIP can plug-and-play replace CLIP in various tasks. On zero-shot REC (RefCOCO/+/g), replacing the CLIP in the ReCLIP pipeline with Alpha-CLIP beats ReCLIP and Red Circle by 6.8% and 3.0% on average, respectively. On open-vocabulary detection (OV-LVIS), the authors replace the ImageNet used in Detic's second-stage pseudo-labeling with their own MaskImageNet: using only 460K images, they push mAP_novel from Detic's baseline 24.6 to 28.6 (original-CLIP labeling gives 27.9), whereas Detic originally used 1.2M images — more accurate and more data-efficient at once.

On the generation and multimodal side, replacing the visual backbone of BLIP-2 / LLaVA-1.5 with Alpha-CLIP lets one specify the object to describe with a single stroke or mask; quantitatively, Alpha-CLIP+LLaVA-1.5 reaches a Visual Genome CIDEr of 160.3 on region-level captioning, surpassing dedicated models such as GPT4RoI and GLaMM. 2D (BLIP-Diffusion) and 3D (Point-E, PureCLIPNeRF) generation are mainly shown through qualitative examples of controllable focus and completing missing parts.

## 🧪 Critical Assessment

### The headline 77.41 gain rests on the dataset's own perfect masks

That CLIP "cannot look accurately" at a specified region is a real and widely encountered pain point, which is beyond doubt. But note the conditions under which the headline number holds: the 77.41 on ImageNet-S is obtained by feeding the **dataset's own ground-truth segmentation masks** as alpha; in other words, the most striking gain is built on the premise that "an almost perfect foreground mask is already available," whereas in real deployment masks mostly come from models like SAM and are of uneven quality. Moreover, the introduction and abstract repeatedly claim a "4.1% top-1 improvement," but in the final main table ViT-L/14 with masks is 77.41 and the original CLIP is 73.48, an actual difference of only 3.93 percentage points; the 4.1% corresponds to an outdated number that was later revised (77.58), an internal inconsistency worth the reader verifying. Thus "solved when a good mask is available" is the more defensible reading, while "solved automatically in the wild" is more weakly supported.

### The self-modified MaskCLIP baseline and the asymmetric retrain-vs-training-free comparison

The ablation is quite solid: the sampling ratio $r_s$, number of unfrozen layers, data volume, and LoRA vs full fine-tuning are all systematically swept, and the control that "fully fine-tuning CLIP on the original image-text alone yields almost no gain" shows the gain comes from the alpha channel rather than from simply seeing more data — a key and well-placed control experiment. But there are two reservations about the fairness of the comparison: first, the main competitor on recognition, MaskCLIP, is a self-modified baseline the authors only used after "making necessary modifications," and those modification details affect its strength; second, Alpha-CLIP requires full-parameter fine-tuning at the GRIT-20m scale on dozens of A100s, whereas competitors like MaskCLIP are training-free methods — this is an asymmetric "retrain vs training-free" contest, and the absolute accuracy gap must be read together with its compute cost.

### A simple channel injection supported by the data engine and breadth of transfer

The core technique — adding one input channel, zero-initializing it, and fine-tuning on region-text — is itself fairly simple, conceptually closer to an extension of the vision-prompt / MaskCLIP line than a brand-new mechanism. What truly carries the weight of the contribution is two pieces of engineering: one is the data engine that uses SAM + GRIT + BLIP-2 to automatically turn ordinary image-text into millions of RGBA region-text pairs, and the other is plug-and-play validating the same model across recognition, REC, OVD, MLLM, and 2D/3D generation — a breadth spanning perception and generation. It is fairer to appraise it as "a simple idea that is thoroughly implemented and transfers extremely widely" than as a theoretical breakthrough.

### The generation claims rely mostly on qualitative examples, with thin quantitative evidence

Recognition, REC, OVD, and region captioning all have hard metrics behind them and are highly credible; but the advantages on 2D/3D generation are overwhelmingly shown through cherry-picked examples, with thin quantitative evidence — for 3D there is only a small appendix comparison against PureCLIPNeRF using R-Precision, and the 2D image variant has almost no quantitative metric. Add that the method at inference still depends on a good-enough mask, and the paper's own admission that it loses focus on very small objects, and these all limit the real-world generality of "just draw a stroke and it gets better." Overall, treating Alpha-CLIP as a practical component that "reliably improves regional focus within pipelines where reliable masks can be obtained" is reasonable, but one should stay cautious about its generation claims.

## 🔗 Related notes

- [Stable Diffusion](../stable-diffusion/)
