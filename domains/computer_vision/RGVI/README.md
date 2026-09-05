# Elevating Flow-Guided Video Inpainting with Reference Generation — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Elevating Flow-Guided Video Inpainting with Reference Generation |
| Venue | AAAI 2025 |
| Year | 2025 |
| Authors | Suhwan Cho, Seoung Wug Oh, Sangyoun Lee, Joon-Young Lee |
| Affiliations | Yonsei University; Adobe Research |
| Paper Version | arXiv 2412.08975v1 (submitted 2024-12-12) |
| DOI | https://doi.org/10.1609/aaai.v39i3.32255 |
| Publication Status | Published in AAAI 2025 proceedings |
| Official Code | https://github.com/suhwan-cho/RGVI |
| Venue Kind | paper |

## Introduction

Video inpainting has two competing tasks: if an occluded background appeared in another frame, the system should bring back the real pixels; if that region is never revealed throughout the video, it can only generate new content. The former determines texture and temporal consistency, while the latter determines whether large holes can be filled plausibly. Propagation-only methods have no pixels to carry when facing backgrounds that were never visible, whereas methods relying only on short-window generation tend to change appearance from frame to frame.

RGVI separates the two tasks. It first uses RAFT to estimate optical flow between adjacent frames, masks the flow inside the object to be removed, and completes it with ProPainter's recurrent flow completion; it then establishes correspondence between any two frames and transports known pixels from the video through one-shot pixel pulling. If holes remain, it uses Stable Diffusion to generate a reference only in the key frame with the greatest reach, propagates the reference across the full video, and finally uses a lightweight per-frame network to clean up the remaining holes.

The paper evaluates on HQVI, DAVI, and YTVI. HQVI uses PSNR, SSIM, LPIPS, and VFID, while also reporting peak memory and per-clip time on a single TITAN RTX; DAVI/YTVI use PSNR and SSIM. Compared methods include STTN, FGVC, FuseFormer, E$^2$FGVI, and ProPainter, with propagation ablation, occlusion mask ablation, and mean rankings from 10 people over 29 DAVIS videos providing additional evidence for interpretation.

![RGVI connects internal pixel propagation, single-frame reference generation, reference propagation, and per-frame completion into four stages; flow completion serves both propagation stages.](imgs/framework.png)

## First Principles

### Move real pixels first, then generate content that does not exist

The inputs are masked frames $X$, binary masks $M$, and flow estimated between adjacent frames. RGVI's ordering is not arbitrary: internal propagation first uses backgrounds observable within the video itself to shrink the holes; reference generation then handles regions that still have no source; reference propagation lets the same generated content be shared across frames; finally, per-frame completion takes over pixels deemed unreliable by flow verification or still left unfilled. Using a single reference prevents each frame from generating independently and conflicting with the others, but it also turns an error generated in that frame into an error shared by the entire video.

### Flow tracing resamples only the coordinate field

For any source frame $j$ and target frame $i$, the method chains adjacent flows into a global correspondence. Let $w(A,B)$ denote sub-pixel grid warping of $A$ with flow $B$; when $i<j$, the accumulation is

$$
f_{i\rightarrow j}=f_{i\rightarrow j-1}+w(f_{j-1\rightarrow j},f_{i\rightarrow j-1}).
$$

Traditional recurrent pixel warping samples RGB once for every frame crossed; for example, propagation from frame 1 to frame 4 sequentially produces three intermediate images, each resampling the previous sampling result. RGVI still repeatedly warps the smoother flow to obtain $f_{1\rightarrow4}$, but samples color only once at the end from the original frame 4 with $w(X_4,f_{1\rightarrow4})$, thereby replacing repeated RGB resampling with a single sample. The close-ups of the metal railing and red-tiled roof in Figure 6 show recurrent warping to be blurrier, while one-shot pulling retains more edges.

![Controlled propagation comparison (Figure 6): recurrent warping in the middle column is blurrier after multi-step resampling; one-shot pulling in the right column preserves clearer metal railing bars and the ridgelines of red-tiled roofs in the mountain village.](imgs/prop.png)

### Bidirectional collection does not blindly trust flow

For every target frame, the algorithm searches sources once in each of the forward and backward directions, prioritizing the nearest unoccluded source pixel. If both directions find corresponding colors, it compares their normalized-RGB L1 distance: when it is below the empirical threshold of 1, the two are averaged; when it exceeds the threshold, the location is marked as an invalid propagation area $V$. The process stops when the hole is filled or the source frames are exhausted. This can reject correspondences where the two directions contradict each other, but cannot prove that agreement between both sides means the background is correct.

### A single key frame turns generation into a propagatable reference

After internal propagation, the connection count is the number of unknown pixels in other frames linked to each frame. The paper defines

$$
C_i=\sum_{j=1}^{L}\left\{\sum_p\left(w(\hat{M}_j,f_{i\rightarrow j})\odot\hat{M}_i\right)\right\},\qquad
k=\underset{i}{\arg\max}\ C_i.
$$

Thus, $k$ is not the frame with the "best-looking image," but the frame whose generated pixels are expected to cover the most holes across frames. Removal mode fixes the prompt to `Empty background, high resolution`; generation mode crops the image around the hole so that text can control newly added material. After generation,

$$
\tilde{X}_i=\hat{X}_i+\hat{M}_i\odot w(\hat{X}_k,f_{i\rightarrow k})
$$

propagates the key-frame colors to the remaining frames. For the few cases where one frame is insufficient, the paper only proposes using multiple key frames sequentially, without providing an automatic stopping criterion or quantitative results.

![Generation mode (Figure 3): the first row contains three input frames with green removal regions; the two prompts, "Standing Minions" and "Sleeping cat," each generate an object and maintain roughly the same appearance and positional relationships across the three frames. This figure demonstrates text control and cross-frame propagation, but three frames alone cannot establish whether long-video dynamics are natural.](imgs/generation.png)

### How one concrete pixel travels through the entire pipeline

Suppose a hole pixel in frame 1 has a visible source in frame 4. RGVI first composes three completed-flow segments into $f_{1\rightarrow4}$, then samples color once, directly from the sub-pixel coordinate in $X_4$. If the L1 distance between that RGB value and the reverse-direction source is 0.8, the two are averaged and this mask is cleared because $0.8<1$. If the distance is 1.2, that point enters $V$; the final network actually receives $\Psi(\tilde{X}\odot(1-V),\tilde{M}+V)$, meaning that untrusted pixels are masked again before per-frame completion. The values 0.8 and 1.2 are illustrative examples for explaining the threshold, not paper measurements; the threshold of 1 and the computational path do come from the methods section.

### Occluding objects require both positive and negative masks

When the object to be removed is occluded by another object that should remain, the latter's motion contaminates the background flow. RGVI labels the object to remove as a negative mask and the occluder as a positive mask, temporarily merges both before inference, and overlays the original positive-mask pixels after completion; the green and red annotations in HQVI show the two exchanging front-back order over time. On the subset containing occluders, adding this mask increases PSNR from 35.13 to 37.31 and reduces LPIPS from 0.0137 to 0.0102, but this requires additional fine-grained annotation and is not fully automatic object removal.

![An occlusion example from HQVI: green is the negative mask to be removed, while red is the positive mask to be retained and overlaid at the end.](imgs/hqvi.png)

![Additional-mask ablation (Figure 7): the left column marks the person to remove in green and the foreground person to retain in red; without the additional mask, the enlarged region in the middle column shows more noticeable red bleeding beside the red skirt, while the right column is cleaner after adding the mask. This is one qualitative case and does not imply that artifacts can be eliminated for every occlusion.](imgs/mask.png)

### HQVI combines scorable synthesis with occlusions closer to editing scenarios

HQVI uses alpha compositing to place foreground objects from VideoMatte240K over Pexels background videos, with each clip at a resolution of $1200\times2160$; fine alpha mattes avoid hard-edged pasting, and the foreground-free backgrounds are retained as ground truth. It includes cases with large holes that require generation, as well as cases where the target is occluded by other objects as described by negative/positive masks. The main text does not state the number of videos, the train/validation/test split, or the distributions of foreground or mask sizes, so readers cannot assess sample diversity or data-leakage risk from the paper.

This setup is well suited to testing object removal with "moving foregrounds and clean backgrounds available as answers," and it also permits fair PSNR calculation; however, shadows, reflections, motion blur, transparent objects, and interreflection in real footage are not necessarily reproduced by alpha compositing. For example, there is no single pixel-level answer to whether a shadow should also disappear after the foreground is removed, so HQVI's high scores cannot be directly extrapolated to such edits.

### The numbers show that reference improves perceptual metrics, but not always pixel-wise error

| HQVI setting | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | VFID ↓ | Mem. | Time per clip |
|-|-|-:|-:|-:|-:|-:|-:|
| 240×432 | RGVI w/o Ref. | **31.60** | **0.9559** | 0.0390 | 0.1868 | 8.3G | 55s |
| 240×432 | RGVI | 30.66 | 0.9527 | **0.0335** | **0.1825** | 8.3G | 58s |
| 480×864 | RGVI w/o Ref. | **31.19** | **0.9534** | 0.0403 | 0.0404 | 8.3G | 1m 38s |
| 480×864 | RGVI | 30.90 | 0.9513 | **0.0342** | **0.0311** | 8.3G | 1m 41s |
| 1200×2160 | RGVI w/o Ref. | 29.81 | **0.9501** | 0.0403 | 0.0101 | 17.2G | 7m 56s |
| 1200×2160 | RGVI | **30.10** | 0.9489 | **0.0357** | **0.0058** | 17.2G | 7m 59s |

At 240p, reference reduces PSNR by 0.94 dB while improving LPIPS by 0.0055; 480p follows the same direction. Only at $1200\times2160$ does using reference raise PSNR, from 29.81 to 30.10. This supports the interpretation that generated textures may be sharper and have better perceptual distance while failing to match the unique ground truth pixel by pixel; it does not support superiority over the non-generation version at every resolution and on every metric.

In the same table's 240p comparison with external methods, RGVI w/o Ref. has a PSNR of 31.60, higher than E$^2$FGVI at 30.63 and ProPainter at 30.62; RGVI with reference is best in LPIPS at 0.0335 and VFID at 0.1825. At 480p, only FGVC, ProPainter, and the RGVI variants remain in the table; at 2K there is no external baseline at all. Thus, "can complete a run at 2K" is supported by the measured 17.2G and 7m59s, whereas "outperforms existing methods at 2K" lacks support from a matched comparison.

### Public benchmarks, ablations, and human preferences answer different questions

DAVI consists of 50 videos from the DAVIS 2016 train+validation sets, while YTVI uses 508 videos from the YouTube-VOS 2018 test set; both are corrupted with random free-form masks, evaluated at 240p, and do not use reference generation. RGVI w/o Ref. obtains 29.75 PSNR / 0.9186 SSIM on DAVI and 31.70 / 0.9335 on YTVI, where its PSNR ties ProPainter at 31.70 while its SSIM is higher. This demonstrates that the propagation/restoration pipeline is competitive, but does not test text generation or large holes.

![Qualitative video restoration results (Figure 5): rows 1 and 3 are DAVI/YTVI inputs with random free-form masks, and rows 2 and 4 are RGVI outputs without reference generation. The six cases cover roads, beaches, swings, traffic, surfing, and swimming-pool scenes; most holes in the figure reconnect to surrounding content, but these selected cases cannot replace full-dataset statistics.](imgs/restoration.png)

Propagation ablation on HQVI at 240p shows that, with internal propagation only, recurrent warping reaches 31.43 PSNR / 0.0595 LPIPS and one-shot reaches 31.60 / 0.0390; when both internal and reference propagation are enabled, they respectively reach 30.17 / 0.0558 and 30.66 / 0.0335. The comparison simultaneously changes recurrent sequential distribution to one-shot bidirectional collection, so it supports the entire propagation protocol, but cannot attribute the entire gain solely to "sampling only once."

The user study asks 10 participants to rank FuseFormer, ProPainter, and RGVI on 29 DAVIS videos, yielding mean ranks of 2.52, 1.90, and 1.59 respectively; however, input resolutions are inconsistent: FuseFormer uses $240\times432$, while the latter two use $480\times864$. The paper does not report blinding, randomization, confidence intervals, or significance, so the best rank of 1.59 can only be treated as a limited-sample preference signal, not a conclusion that has ruled out resolution and procedural confounds.

![DAVIS 2016 qualitative comparison (Figure 1): the four rows are Input, FuseFormer, ProPainter, and RGVI; in the enlarged areas around the wooden fence, rocks, and curved railroad tracks, RGVI's edges and textures are more continuous. This is visual evidence from author-selected cases and cannot replace a quantitative comparison under matched conditions.](imgs/intro.png)

## 🧪 Critical Assessment

### The problem is decomposed correctly, but errors are also amplified along the pipeline

Separating "real pixels that can be moved" from "unknown content that must be generated" is a meaningful decomposition of practical failure modes; one-shot pulling also directly targets repeated RGB resampling. Yet the entire pipeline still depends on completed flow: if large displacements, long occlusions, or non-rigid boundaries produce incorrect correspondence, colors in the two directions can happen to be similar and still pass the threshold; if the error falls in the key frame, reference propagation then carries the structural displacement to multiple frames. The paper itself acknowledges that inaccurate flow can cause obvious structural misalignment.

### Evidence for reference controllability is stronger than evidence for stability

The paper shows that text prompts can replace material, and the default removal prompt is only `Empty background, high resolution`; it provides no prompt sweep, seed variance, or identity consistency across key-frame choices, and does not quantify the incidence of unnatural generation. Therefore, "can accept text control" is supported by Figure 3, while "is insensitive to prompts, maintains identity over long videos, and rarely hallucinates" remain unproven; the authors also explicitly list occasionally unnatural generated references.

### HQVI raises the resolution but does not provide a sufficient data-audit surface

The $1200\times2160$ resolution and fine alpha mattes are indeed closer to post-production than low-resolution random masks, but the main text omits the dataset size, splits, scene/subject distributions, and the form of release after licensing. More fundamentally, placing VideoMatte240K foregrounds over Pexels backgrounds is still a synthetic distribution: occlusion contours can be precise, while shadows, reflections, transparency, and contact relationships may not obey real-world physics. This benchmark can test the specified pattern, but is not sufficient to represent general object removal.

### Comparison conditions limit the strength of the "high-resolution lead"

The 240p table has five external baselines, 480p retains only two, and 2K has no external method at all; fixed-resolution methods are marked with $\dagger$, but memory and time are still affected by implementation, video length, and generation steps. RGVI's 2K record on a single TITAN RTX proves that it can run, not that it is faster or more memory-efficient than competitors not measured on the same hardware and at the same resolution; the authors' claim of handling it "with ease" also leaves a practical-criterion gap against roughly 8 minutes per clip.

### System integration is valuable, while causal attribution remains incomplete

RGVI combines RAFT, ProPainter flow completion, Stable Diffusion, flow tracing/grid warping, and a lightweight per-frame network; its main novelty lies in one-shot correspondence/verification and the single-key-frame reference design. Existing ablations do not isolate the effects of flow tracing, nearest source in both directions, L1 verification, key-frame selection, and Stable Diffusion, nor provide a cost-quality curve that holds one-shot propagation constant while swapping generators. The tested HQVI ablation supports reference improving LPIPS at three resolutions, but PSNR decreases at 240p and 480p; therefore, it only establishes a perceptual-metric gain for this protocol, not a consistent improvement overall or from every component.

### Release information is insufficient to close the reproducibility gap

Training information for the per-frame completion network is relatively specific: YouTube-VOS 2018 train images are resized to $240\times432$, random free-form and random object masks are used, and the network is trained with L1 plus adversarial loss and Adam at a fixed learning rate of $10^{-4}$. However, although the paper provides a code URL, the main text still does not record the exact Stable Diffusion checkpoint, sampler, steps, guidance scale, random seed, flow-completion checkpoint, or HQVI size and splits. No code_repo_url was provided for this task, so this note does not use repository static analysis to fill those fields; from the paper alone, an external team would struggle to reproduce the 58-second and 7-minute-59-second results or determine the variance caused by prompt/seed changes.

## One-Minute Wrap-Up

- Video inpainting must handle two things at once: backgrounds that appeared before should be moved back from other frames, while content that never appears throughout the video needs to be generated; RGVI therefore propagates first and generates afterward.
- One-shot pulling repeatedly composes flow coordinates, while RGB is sampled only once from the original source frame. For propagation from frame 1 to frame 4, for example, it does not produce three repeatedly interpolated intermediate images; LPIPS in the 240p internal-only ablation also falls from 0.0595 to 0.0390.
- A single key frame is selected by connection count. Stable Diffusion fills the reference only in that frame, then propagates it across the video with flow, preventing each frame from generating independently and conflicting with the others.
- This reference has a metric tradeoff: at 240p, LPIPS falls from 0.0390 to 0.0335, but PSNR also falls from 31.60 to 30.66, a reduction of 0.94 dB.
- The greatest risk also comes from being "shared across the entire video": flow deviation or unnatural generation in the key frame is propagated to multiple frames; the paper does not quantify prompt, seed, or long-video consistency.
- HQVI's 2K test uses 17.2G and 7 minutes 59 seconds per clip on a single TITAN RTX; there is no external baseline at that resolution, so it only proves that the run completes, not that it leads under matched conditions.

## 🔗 Related notes

- [ProPainter](../ProPainter/) — RGVI uses its recurrent flow completion and responds to the resampling problem of recurrent pixel warping with one-shot pulling.
- [DiffuEraser](../DiffuEraser/) — An alternative route that uses diffusion to strengthen the generative capability of video inpainting.
