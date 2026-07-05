# CoDi — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | CoDi: Conditional Diffusion Distillation for Higher-Fidelity and Faster Image Generation |
| Venue | CVPR 2024 |
| Year | 2024 |
| Authors | Kangfu Mei, Mauricio Delbracio, Hossein Talebi, Zhengzhong Tu, Vishal M. Patel, Peyman Milanfar |
| Official Code | unknown |
| Venue Kind | paper |

*This note is written based on the arXiv full-text version `2310.01407v2` (updated 2024-02-17); the paper was formally published at CVPR 2024, and where the camera-ready version differs from this preprint, the formal version prevails. The authors have not released a public code repository; the project page is https://fast-codi.github.io.*

## First Principles

### The Problem Setting in One Sentence

Diffusion models can produce high-quality images, but at inference they require 20–200 sampling steps (function evaluations), even when using advanced samplers such as DPM-Solver, which makes them hard to apply in real time. CoDi aims to solve two things at once: to adapt a pretrained latent diffusion model (LDM) that "only takes a text condition" into a conditional model that "can take an additional image condition", and to compress the sampling steps down to 1–4, all done within a **single stage** and without requiring the original text–image data.

### Why Existing Approaches Fall Short

Before CoDi, combining "adding a new condition" with "acceleration" followed two mainstream routes, which the paper organizes as CM-X and GD-X. The first is distill-first (corresponding to CM-I): first distill the unconditional text–image model into a few-step student, then finetune it with conditional data. The second is finetune-first (corresponding to GD-II / CM-II): first finetune the diffusion model with the new conditional data, then conducting distillation on this already-finetuned conditional model.

Each of these routes has its own hard limitation. distill-first needs the original large-scale text–image data (e.g. LAION) to complete the first-stage unconditional distillation, which may not be obtainable in practice; finetune-first may sacrifice the generative prior (diffusion prior) brought by pretraining during the first-stage finetuning. CoDi's claim is that merging "conditioning" and "distillation" into a single stage sidesteps both drawbacks at once — it eliminates the need for the original text-to-image data, and it no longer requires a prior-damaging finetune first.

### Step 1: Adapting the Unconditional Model into a Conditional Model (Zero Initialization)

CoDi follows a ControlNet-style idea, duplicating the encoder layers of the pretrained network (the U-Net encoder) to serve as a "conditional encoder", and uses a learnable scalar $\mu$ to fuse the backbone features $\boldsymbol{h}_\theta(\mathbf{z}_t)$ with the conditional features $\boldsymbol{h}_\eta(c)$:

$$\boldsymbol{h}_\theta(\mathbf{z}_t)' = (1 - \mu)\,\boldsymbol{h}_\theta(\mathbf{z}_t) + \mu\,\boldsymbol{h}_\eta(c)$$

The key is that $\mu$ is initialized to $0$. When $\mu=0$, the fused features are exactly equal to the original unconditional features — that is, at the start of training the adapted model is **exactly equivalent** to the pretrained model, and the prior is preserved intact; as training proceeds, $\mu$ gradually grows and conditional information is progressively injected. This is the so-called zero initialization: Starting from this zero initialization, a "harmless-at-the-start" path adapts the unconditional backbone $\hat{\mathbf{v}}_\theta(\mathbf{z}_t,t)$ into the conditional model $\hat{\mathbf{w}}_\theta(\mathbf{z}_t,c,t)$.

### Step 2: Conditional Diffusion Consistency

The core of a consistency model is self-consistency: along a probability-flow ODE (PF-ODE) trajectory, the model's predictions of the clean signal at any two time points should agree. CoDi carries this property over to the conditional model, enforcing a self-consistency in the predicted signal space by requiring

$$\hat{\mathbf{w}}_\theta(\mathbf{z}_t, c, t) = \hat{\mathbf{w}}_\theta(\hat{\mathbf{z}}_s, c, s), \quad \forall\, t, s \in [0, T]$$

where $\hat{\mathbf{z}}_s$ is drawn from the PF-ODE of **the adapted model itself**. Here, this consistency property differs from the one in consistency models in two ways: first, the ODE used to sample $\hat{\mathbf{z}}_s$ is the model currently being trained, not a frozen teacher; second, the space in which the consistency loss is applied (noise space vs. signal space). The paper explains via a remark: as long as the model satisfies self-consistency on the noise prediction $\hat{\epsilon}_\theta = \alpha_t \hat{\mathbf{v}}_\theta + \sigma_t \mathbf{z}_t$, then through a change of variables it simultaneously satisfies consistency on the signal prediction $\hat{\mathbf{x}}_\theta = \alpha_t \mathbf{z}_t - \sigma_t \hat{\mathbf{v}}_\theta$. This lets "self-consistency in the noise space" and "conditional generation in the signal space" be learned together by the same set of parameters.

### Step 3: Take One Step with the Model's Own ODE (Instead of a Frozen Teacher)

Previous distillation approaches differ in how they obtain the sample at the next time point: consistency models solve the ODE using the Euler solver, while progressive distillation (progressive / guided distillation) uses a frozen pretrained teacher to run two DDIM steps. CoDi instead takes one step directly with the **diffusion model being adapted**:

$$\hat{\mathbf{z}}_s = \alpha_s\,\hat{\mathbf{x}}_\theta(\mathbf{z}_t, c, t) + \sigma_s\,\epsilon, \quad \text{where}\ \mathbf{z}_t = \alpha_t \mathbf{x} + \sigma_t \epsilon,\ \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

The paper argues that this substitution "harmonizes the conflicting optimization directions": the consistency distillation coming from the pretraining data, and the conditional guidance coming from the conditional data. Because $\hat{\mathbf{z}}_s$ is now a trajectory generated by the conditional model itself, the distillation's target trajectory gets "pulled" toward the correct direction by the conditional data, rather than being locked to the frozen teacher's unconditional trajectory.

### Loss: Noise Consistency + Signal Guidance

Combining the two things above, the training loss is (with $\theta^-$ the EMA target network, used to stabilize training):

$$\mathcal{L}(\theta) := \mathbb{E}\big[\, d_\epsilon\big(\hat{\epsilon}_{\theta^-}(\hat{\mathbf{z}}_s, s, c),\ \hat{\epsilon}_\theta(\mathbf{z}_t, t, c)\big) + d_\mathbf{x}\big(\mathbf{x},\ \hat{\mathbf{x}}_\theta(\mathbf{z}_t, t, c)\big) \,\big]$$

The first term $d_\epsilon$ is the self-consistency in the noise space (making predictions at different step counts converge to the same trajectory — this is the source of "acceleration"); the second term $d_\mathbf{x}$ is the conditional guidance in the signal space (making the output match the real image $\mathbf{x}$ corresponding to the condition — this is the source of "conditioning"). The boundary conditions use the same skip-connection parameterization as the consistency model / EDM, $c_\mathrm{skip}(t) = \sigma_\mathrm{data}/(t^2 + \sigma_\mathrm{data}^2)$, $c_\mathrm{out}(t) = \sigma_\mathrm{data}\,t/\sqrt{t^2 + \sigma_\mathrm{data}^2}$, and takes $\sigma_\mathrm{data}=0.5$.

Regarding the choice of $d_\mathbf{x}$, the paper runs an ablation: using $\ell_2$ in pixel or encoding space makes multi-step sampling smooth and blurry; using the LPIPS perceptual distance oversaturates; ultimately it adopted the $\ell_2$ distance in the latent space by default, because it gives the best visual quality and FID at 4-step sampling.

### The Full Training Algorithm (CDD)

The paper's appendix gives the Conditional Diffusion Distillation (CDD) as pseudocode. The core loop is as follows (using a velocity-parameterized DDIM update):

```text
輸入: 條件資料 (x, c) ~ p_data, 適配模型 ŵ_θ(z_t, c, t), 學習率 η,
      距離函數 d_ε 與 d_x, EMA 係數 γ
θ⁻ ← θ                                  # 目標網路初始化
repeat
    抽樣 (x, c) ~ p_data,  t ~ [Δt, T]    # 經驗上 Δt = 1
    抽樣 ε ~ N(0, I);  s ← t − Δt
    z_t ← α_t x + σ_t ε
    x̂_t ← α_t z_t − σ_t ŵ_θ(z_t, c, t)     # 訊號預測
    ε̂_t ← α_t ŵ_θ(z_t, c, t) + σ_t z_t     # 噪聲預測
    ẑ_s ← α_s x̂_t + σ_s ε̂_t                # DDIM (v 參數化) 走一步
    ε̂_s ← α_s ŵ_{θ⁻}(ẑ_s, c, t) + σ_s ẑ_s   # 用 EMA 目標網路
    L(θ, θ⁻) ← d_ε(ε̂_t, ε̂_s) + d_x(x, x̂_t)
    θ  ← θ − η ∇_θ L(θ, θ⁻)
    θ⁻ ← stopgrad(γ θ⁻ + (1−γ) θ)          # 指數移動平均
until 收斂
```

Note that $\hat{\mathbf{z}}_s$ is obtained by taking one step with the **current** $\theta$ (the model's own PF-ODE), while the consistency target $\hat{\epsilon}_s$ is computed with the EMA (exponential moving average) $\theta^-$; the pairing of the two is exactly the mechanism that accomplishes "acceleration + conditioning" in a single stage.

### Parameter-Efficient Variant: PE-CoDi

The same loss can be used to selectively update parameters pertinent to distillation / conditional finetuning and freeze the rest. Applying it to an interface like ControlNet that only copies the encoder yields PE-CoDi: the backbone is fully frozen and only the copied-out encoder is trained. This brings a practically nice property — different tasks can share most parameters, and each new conditional task only needs a small amount of extra parameters.

### A Walkthrough with Real Numbers (real-world super-resolution)

Take the paper's real-world super-resolution experiment (DF2K, the Real-ESRGAN degradation pipeline, 3,000 randomly degraded image pairs as the test set) as an example, and string one forward pass together end to end:

1. Take a degraded low-resolution image as the condition $c$; run the corresponding high-resolution latent $\mathbf{x}$ through the forward process to get $\mathbf{z}_t = \alpha_t \mathbf{x} + \sigma_t \epsilon$; within the same batch use **the same $t$** (the ablation shows this performs better for 1–4 steps, contrary to progressive distillation which draws a different $t$ per sample).
2. The condition $c$ passes through the copied encoder to get $\boldsymbol{h}_\eta(c)$, fused into the backbone via $\mu$; the model outputs velocity $\hat{\mathbf{w}}_\theta$, which is converted into $\hat{\mathbf{x}}_t$ and $\hat{\epsilon}_t$.
3. Take one step with the model itself to get $\hat{\mathbf{z}}_s$; the EMA network computes $\hat{\epsilon}_s$ at $s$, forming the loss $d_\epsilon(\hat{\epsilon}_t, \hat{\epsilon}_s) + d_\mathbf{x}(\mathbf{x}, \hat{\mathbf{x}}_t)$.
4. After distillation is complete, inference runs in only **4 steps**.

The results are compared in the table below (lower is better for both FID and LPIPS):

| Sampling Steps | Method | FID ↓ | LPIPS ↓ |
|-|-|-|-|
| 250 steps | LDMs | 19.200 | 0.2639 |
| 4 steps | LDMs | 29.266 | 0.3014 |
| 4 steps | ControlNet | 34.56 | 0.3381 |
| — | GD-II | 23.675 | 0.2796 |
| — | CM-II | 27.810 | 0.3172 |
| 4 steps | **PE-CoDi (Ours)** | 25.214 | 0.2941 |
| 4 steps | **CoDi (Ours)** | 19.637 | 0.2656 |

In plain terms: CoDi's 4-step FID of 19.637 nearly matches the teacher LDM's 19.200 at 250 steps, and clearly beats the also-4-step GD-II (23.675) and CM-II (27.810). The appendix measures CoDi's 4-step latency on TPUv5 at $107 \pm 3$ ms, versus LDM's $977 \pm 1$ ms at 50 steps — comparable quality but roughly one-ninth the latency. As for training cost, each compared method (including the text–image pretraining) was each trained on 64 TPU-v4 pods for 8 days.

### How the Ablation Decomposes the Contributions

The paper decomposes the contribution of each design on the SR task (all with 4-step sampling):

| Method | Params | FID | LPIPS |
|-|-|-|-|
| LDMs | 865M | 29.266 | 0.3014 |
| + ControlNet | 1.22B | 28.951 | 0.3049 |
| PE-CoDi (Ours) | 364M | 25.214 | 0.2941 |
| CoDi (Ours) | 1.22B | 19.637 | 0.2656 |
| − distilling PF-ODE | 1.22B | 20.307 | 0.2733 |
| − noise-consistency | 1.22B | 25.728 | 0.3252 |

Two points are worth noting: simply adding the ControlNet module (without CoDi's distillation) barely improves anything (29.266 → 28.951), indicating the gain does not come from an architectural change; whereas removing the noise-consistency term drops FID from 19.637 to 25.728, the largest single-item degradation, showing that self-consistency in the noise space is the main source of the contribution. PE-CoDi, with only 364M parameters (fewer than the 865M backbone before freezing, because only trainable parameters are counted), compresses FID from 29.266 to 25.214.

### The Closer the Condition Is to the Target, the Easier Distillation Becomes

On instruction-based image editing (InstructPix2Pix data), CoDi's single-step sampling already reaches visual quality close to IP2P's 200 steps; that is, our single-step sampling result can achieve comparable visual quality to 200 steps. From this the paper conjectures that the benefit of conditional guidance to distillation is related to "the similarity between the condition and the target data" — when the condition itself is already very close to the answer (e.g. an edit that only fine-tunes lighting), a single step suffices.

The figure below is a qualitative comparison on the depth-to-image task: the depth condition (left), the result from ControlNet sampled in 4 steps (middle), and ours from the unconditional pretraining sampled in 4 steps (right) (all 4-step sampling).

![Depth condition input](imgs/depth_condition.png)
![ControlNet 4-step result](imgs/depth_controlnet_4step.png)
![CoDi 4-step result](imgs/depth_codi_4step.png)

## 🧪 Critical Assessment

### Is the Problem Real and Important (problem realness)

The slow inference of diffusion models is a widely acknowledged real problem, and both acceleration and conditioning are active directions with practical value. CoDi's angle — merging "conditioning" and "distillation" into a single stage — does correspond to a practical pain point: two-stage approaches either depend on large-scale text–image data that may be unavailable, or damage the prior during the first-stage finetuning. In terms of problem setup, this is defensible. A more reserved point: the paper makes 1–4 steps its headline, but for some tasks (such as super-resolution) parallel work already indicates that few steps are needed, so the marginal benefit of "extremely few steps" is not equivalent across tasks — something the paper itself acknowledges in its related work.

### Are the Baselines, Ablations, Datasets, and Metrics Sufficient?

The ablation is the more persuasive part of this paper: ControlNet-only yields almost no gain, and removing noise-consistency causes the largest degradation; these two comparisons effectively attribute the credit to the consistency term rather than the architecture. The baselines cover both arrangements of CM and GD as well as the DPM-Solver family, which is fairly comprehensive. But several weaknesses are worth pointing out. First, key columns of the main results lack step-count annotations: rows such as GD-II and CM-II in the performance table do not explicitly mark the sampling steps, and the reader can only infer them from the "4 steps" title, so the rigor of cross-row comparison is discounted. Second, the metrics lean on distribution-level or perceptual proxies such as FID / LPIPS / CLIP and lack human-preference evaluation, while the strong claim that IP2P is "single-step ≈ 200 steps" is supported mainly by qualitative figures and the authors' own statement of "only minor visual differences", so the evidence is soft.

### Is It a Self-Defined Benchmark or a Fair Comparison?

One point that warrants caution: CoDi uses a private base model "trained on internal text-to-image data, claimed to be comparable to SD v1.4", while baselines such as CM / GD are implemented by the authors themselves with the same architecture and the same parameter count (e.g. CM implemented with an unfrozen ControlNet). This means the strength of the baselines depends heavily on the authors' implementation and tuning, and cannot be independently reproduced externally. In other words, the comparison is carried out within a scenario the authors built themselves and designed around the strengths of their own method; although the authors deliberately aligned the architecture and parameter count for fairness, the combination of "private base model + self-implemented baselines" lowers the external verifiability of the SOTA claim. For inpainting, the authors themselves also state that the resolution differs from Palette and the numbers are "for reference only", which counts as honest disclosure.

### Is the Problem Really Solved, and Does It Have Real-World Significance?

Within the setting the paper defines, CoDi's evidence chain is complete: 4-step FID 19.637 matching the teacher at 250 steps, latency reduced to about one-ninth, and clear ablation attribution all support the conclusion that "single-stage conditional distillation is feasible and effective". Real-world significance also exists — conditional generation at the 107 ms level is a perceptible speedup for consumer-grade image applications (super-resolution, editing). But "solved" needs to be discounted: first, the method relies on an adapter architecture, and the authors admit in the limitations section that this introduces additional computation and has not yet achieved a lightweight form; second, the most striking claims (IP2P single-step ≈ 200 steps, matching the 250-step teacher), lacking human evaluation and externally reproducible baselines, look more like "holding within a controlled setting" than "holding generally". Overall this is a cleanly designed, ablation-solid work with limited external verifiability; its conclusions should be regarded as credible within its experimental protocol, while cross-setting generality still awaits third-party reproduction.

## 🔗 Related notes

- [DDPM — Denoising Diffusion Probabilistic Models](../diffusion/)
