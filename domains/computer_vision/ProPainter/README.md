# ProPainter — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | ProPainter: Improving Propagation and Transformer for Video Inpainting |
| Venue | ICCV 2023 |
| Year | 2023 |
| Authors | Shangchen Zhou, Chongyi Li, Kelvin C.K. Chan, Chen Change Loy (S-Lab, Nanyang Technological University) |
| Official Code | https://github.com/sczhou/ProPainter |
| Venue Kind | paper |

## Introduction

Video inpainting (VI) fills the occluded or missing regions of a video with content while keeping the result spatially plausible and temporally coherent. Its practical uses are very concrete: removing objects from a scene, erasing watermarks and logos, and restoring damaged frames. The hard part is "establishing correct correspondences across frames that are far apart" — the same occluded background patch may have been exposed only once, dozens of frames away, and bringing it back requires reliable long-range propagation.

Before ProPainter, the mainstream approaches followed two routes, each with a clear pain point. The first is "image propagation": complete the optical flow, use it to bidirectionally warp known pixels on the RGB image to fill holes, and then attach a separate inpainting network to complete whatever holes remain. The problem is that this two-stage pipeline is decoupled — once the flow is inaccurate it leaves misaligned textures and artifacts, and the later network has no way to correct the earlier stage's errors. The second route is "feature propagation" and video Transformers: E$^2$FGVI puts flow completion and content hallucination into an end-to-end framework, but it warps in the downsampled feature domain, where spatial precision is limited and results are prone to blur; more critically, both feature propagation and spatiotemporal attention are constrained by memory and computation, so they can only operate over a very short temporal range and cannot reach the textures in distant frames.

ProPainter's high-level solution merges the strengths of the two routes into "dual-domain propagation," paired with a "mask-guided sparse video Transformer" (MSVT) tailored for VI. The whole thing has three components: first an efficient recurrent flow completion network (RFC) completes the corrupted flow; then global propagation is done in the image domain and local propagation in the feature domain; finally several MSVT blocks refine and decode the complete video. In the figure below, (a)(b) are the two core designs, (c) is a scatter plot of PSNR against runtime (bubble size represents memory usage, and ProPainter sits in the top-left corner — fast, accurate, and memory-light), and (d–h) are a qualitative car-removal comparison — (f) FGT leaves black haze and distortion inside the box, while (h) ProPainter fills it cleanly.

![ProPainter's two core designs and efficiency/qualitative comparison: in (c), ProPainter sits in the top-left corner with high PSNR, low runtime, and a small memory bubble](imgs/teaser.png)

The paper tests whether this solution holds up with concrete experiments: on the two datasets YouTube-VOS (508 clips in the test set) and DAVIS (50 of its 90 clips), it computes PSNR, SSIM, VFID and the temporal-consistency metric $E_{warp}$ under a fixed stationary mask, and compares accuracy and efficiency (FLOPs, seconds per frame) against nine existing methods (DFVI, CPNet, FGVC, STTN, TSAM, FuseFormer, ISVI, FGT, E$^2$FGVI). All videos are uniformly resized to $432\times 240$ for training and evaluation. Below we first take the mechanism apart and rebuild it, then turn back to question whether the evidence supports the conclusions.

## First Principles

### From an occluded video to a complete output: the data flow

Given a masked video $X=\{X_t\in\mathbb{R}^{H\times W\times 3}\}_{t=1}^{T}$ and corresponding binary masks $M=\{M_t\in\mathbb{R}^{H\times W\times 1}\}_{t=1}^{T}$ (a value of 1 marks the region to fill), RAFT first extracts the forward and backward optical flows $F^f, F^b$. The order of the whole pipeline is fixed: RFC completes the flow → image-domain global propagation → feature-domain local propagation → refinement by several MSVT blocks → the decoder reconstructs the output $\hat{Y}$. The figure below is the official overview: the "Masked Flows→Recurrent Flow Completion→Completed Flows" branch computes the flow first, then feeds it to the inpainting backbone on the right, "Image Prop. (global)→Encoder→Feature Prop. (local)→MSVT Blocks×N→Decoder."

![ProPainter overview: recurrent flow completion, dual-domain propagation (image-domain global + feature-domain local), MSVT refinement and decoding](imgs/overview.png)

### Recurrent flow completion (RFC): complete the flow first, and do it fast

The paper's position is: directly completing RGB content is hard, completing flow is comparatively easy, and using the completed flow to warp pixels preserves temporal coherence better — so a "separately trained" flow-completion module is needed. If flow completion were learned jointly with the inpainting loss, you would get suboptimal, less accurate flow. RFC first encodes the flow $F_t$ into a feature $f_t$ downsampled by a factor of 8, then uses deformable-convolution (DCN)-based deformable alignment to bidirectionally propagate information from neighboring frames to fill the holes. Taking backward propagation as an example, the alignment can be written as

$$
\hat{f_t} = \mathcal{R}\big(\mathcal{D}(\hat{f}_{t+1}; o_{t\rightarrow t+1}, m_{t\rightarrow t+1}), f_t\big),
$$

where $\mathcal{D}$ is the deformable convolution and $\mathcal{R}$ is the convolution that fuses the aligned feature with the current feature. It replaces the sliding window of past methods with a recurrent network, avoiding repeated inference over overlapping frames. The effect is both fast and accurate: the flow endpoint error (EPE) is 0.020 on YouTube-VOS and 0.051 on DAVIS, on par with the best methods, but it takes only 0.005 s per frame — the paper claims about 40× faster than SOTA (about 192 fps on a single V100).

### Image-domain global propagation: no learning, a reliability check, running on the GPU

Image-domain propagation deliberately contains no learnable operations and only does "warp by flow + reliability check." The key is to judge whether the flow is trustworthy using the forward-backward consistency error:

$$
\mathcal{E}_{t\rightarrow t+1}(p) = \Big\| \hat{F}_{t\rightarrow t+1}(p) + \hat{F}_{t+1\rightarrow t}\big(p+\hat{F}_{t\rightarrow t+1}(p)\big) \Big\|_2^2 ,
$$

only when the consistency error is small enough ($C_1:\mathcal{E}<\epsilon$, with the threshold $\epsilon$ set to 5), the point in the current frame is indeed occluded ($C_2:M_t(p)=1$), and the corresponding point in the source neighbor frame is not occluded ($C_3$), is it treated as a reliable propagation region $A_r$. The propagation itself is

$$
\hat{X}_t = \mathcal{W}(X_{t+1}, \hat{F}_{t\rightarrow t+1}) * A_r + X_t * (1-A_r),
$$

and right after warping the mask is updated to $\hat{M}_t = M_t - A_r$, so that subsequent frames can continue to fill in relay. Because pixels are warped only at positions that pass all three checks, the misalignment caused by wrong flow is blocked rather than forced into the picture. This step runs on the GPU, replacing the time-consuming CPU-side flow-trajectory indexing and Poisson blending of past methods such as FGVC; more importantly, it is trained together with the whole network, so the later modules can correct its residual errors.

The actual power of this step is directly visible: in the figure below, the top two rows are a car-removal clip and the bottom two rows a pedestrian-removal clip; the first and third rows are the input (green is the mask), and the second and fourth rows are the result of "only finishing image-domain propagation" — most of the mask, sometimes the entire block, is already filled, and the residual green region shrinks dramatically. In other words, the later modules mostly only need to refine and complete a small amount of residual holes, rather than learning the whole inpainting from scratch.

![Intermediate result after only image-domain global propagation: the green mask is mostly filled by pixels from neighbor frames, leaving only a few residual holes for the later modules](imgs/img_prop.png)

### Feature-domain local propagation: use flow as the baseline, feed in extra mask conditions

After image-domain propagation has filled the large regions, an encoder with the same structure as FuseFormer/E$^2$FGVI extracts a local sequence into $\frac{H}{4}\times\frac{W}{4}\times C$ features, then performs "flow-guided deformable alignment." This differs from RFC's version that directly learns the DCN offsets: here the completed flow is used as the DCN's baseline offset, and only the residual offset relative to the flow is learned. ProPainter's difference from E$^2$FGVI lies in feeding in richer conditions — besides the current feature, the warped propagated feature, and the downsampled flow, it additionally adds the flow validity map $V$ obtained from the consistency check, the original mask $M^{\downarrow}$, and the updated mask $\hat{M}^{\downarrow}$ after image propagation. With these conditions, this step can focus attention on the truly hard-to-fill regions "where the flow is invalid and the earlier image propagation was unreliable." The figure below is the module's internal structure: the condition pool in the gray dashed box at the top concatenates the current feature $e_t$, the flow validity map $V_{t+1\rightarrow t}$, the original downsampled mask $M^{\downarrow}_t$, the updated mask after image propagation $\hat{M}^{\downarrow}_t$, the downsampled completed flow $\hat{F}^{\downarrow}_{t+1\rightarrow t}$, and the warped neighbor feature $\mathcal{W}(\hat{e}_{t+1})$; the convolution layer accordingly outputs the DCN's modulation mask and residual offset, the residual offset is added at the $\oplus$ in the figure to the flow baseline offset to become the final offset, the neighbor feature $\hat{e}_{t+1}$ is deformably aligned, and it is then concatenated with the current feature and fused by convolution into $\hat{e}_t$.

![Feature-domain flow-guided deformable alignment: the condition pool concatenates the flow validity map and the dual masks before and after updating, the convolution layer predicts the DCN mask and residual offset, and the residual offset is added to the flow baseline offset before aligning the neighbor feature](imgs/dcn_align.png)

### Mask-guided sparse video Transformer (MSVT): prune independently in two directions

The cost of a classical spatiotemporal Transformer grows quadratically with the number of tokens; the paper notes that FuseFormer and FGT cannot even process 480p video on a 32G GPU. ProPainter's observation is: the mask usually covers only a small local region (on DAVIS the object region averages just 13.6%), and the textures of adjacent frames are highly redundant. So it prunes separately in the two spaces of query and key/value. The features first go through soft split to obtain patch embeddings $Z\in\mathbb{R}^{T_l\times M\times N\times C_z}$, then are cut into $m\times n$ non-overlapping windows (the experiments use small $5\times 9$ windows).

On the query side, attention is only computed for windows that "touch the mask." Downsample the mask to the window grid $M^{\downarrow}$, sum over the temporal dimension and clip to 1, to get the sparse mask

$$
S_Q = \mathrm{Clip}\Big(\sum\nolimits_{t=1}^{T_l} M^{\downarrow}_t,\ 1\Big),
$$

if a window has never touched the mask in all past frames, $S_Q(i,j)=0$, and the spatiotemporal attention for that window can be skipped entirely. The key/value side exploits the redundancy of adjacent frames: it uses a temporal stride of 2 to sample alternately — odd blocks let only odd frames and even blocks let only even frames participate, halving the key/value space directly; in addition, window expand and pooled global tokens restore a larger range of spatial association. The figure below clearly shows these two paths: the $S_Q$ in the top row (a window grid made of 0/1) decides which query windows are kept, and the temporal sparse in the bottom row cuts the number of frames from $T$ to $T/2$, then adds local (expand window) and global key/value tokens.

![MSVT: the 0/1 sparse mask S_Q decides which query windows are kept; key/value is halved with a temporal stride of 2 and supplemented with local tokens from the expand window and global tokens from pooling](imgs/msvt.png)

### One concrete forward pass and the headline numbers

Chaining the above together for one pass: a DAVIS object-removal video is resized to $432\times 240$, RAFT (running only 5 iterations at inference) computes the bidirectional flow; RFC recurrently completes the flow on features downsampled by 8×, at 0.005 s per frame; image-domain propagation uses the $\epsilon=5$ consistency check to fill most of the mask and update the mask; the encoder downsamples the picture to $108\times 60$ features for flow-guided alignment; 8 MSVT blocks refine on $5\times 9$ windows, skipping unmasked windows via $S_Q$ and halving the key/value; the decoder outputs. Because it is efficient enough, the temporal length used at inference can be stretched to 20 frames, while the local sequence length at training is 10.

The headline accuracy and efficiency results are as follows (excerpted from the paper's Table 1, FLOPs and seconds per frame for 10 frames):

| Model | YT-VOS PSNR↑ | DAVIS PSNR↑ | DAVIS VFID↓ | DAVIS E*warp↓ | FLOPs | Runtime |
|-|-|-|-|-|-|-|
| FuseFormer | 33.32 | 32.59 | 0.137 | 1.349 | 1025G | 0.114 |
| E$^2$FGVI | 33.71 | 33.01 | 0.116 | 1.289 | 986G | 0.085 |
| ProPainter (ours) | 34.43 | 34.47 | 0.098 | 1.187 | 808G | 0.083 |

ProPainter leads on all metrics, while its FLOPs (808G) is lower than the runner-up E$^2$FGVI (986G). The compute-saving advantage of the sparse Transformer is even more evident as the sequence lengthens and the resolution grows: for the same Transformer block, the FLOPs at temporal length 10 is 25.77G (E$^2$FGVI 37.65G, FuseFormer 75.1G), and at temporal length 60 it is 253G (E$^2$FGVI 690G, FGT 824G); computed with a missing ratio of 1/6.

| Temporal length | FuseFormer | FGT | E$^2$FGVI | ProPainter |
|-|-|-|-|-|
| 10 | 75.1 | 70 | 37.65 | 25.77 |
| 30 | 544 | 292 | 206 | 97 |
| 60 | — | 824 | 690 | 253 |

The compute-saving advantage is even more exaggerated as resolution grows. In the figure below, the left is the FLOPs curve versus temporal length (FuseFormer shoots up to 937G at length 40 and cannot even plot longer points), and the right is the curve versus spatial resolution: FGT explodes to 1880G at 720p, while ProPainter needs only 374G even at 960p (E$^2$FGVI is 602G). This figure is the main quantitative evidence that MSVT can hold up at high resolution; looking only at the temporal-length table in the main text would miss the resolution axis.

![Growth of FLOPs versus temporal length (left) and spatial resolution (right): ProPainter (red) has the flattest slope, FGT reaches 1880G at 720p, and ProPainter needs only 374G at 960p](imgs/flops_compare.png)

### Ablation: which component is actually doing the work

The paper's ablation (Table 2, PSNR/SSIM) shows: removing image-domain propagation drops from 34.15 to 33.05, the largest single decline; replacing image-domain propagation with FGVC's version (without retraining) is actually worse (32.91), because FGVC is easily led astray by wrong flow, causing texture distortions the later stages cannot correct. Feature-domain propagation contributes less: removing it drops to 33.17, and replacing it with the E$^2$FGVI version gives 33.94. The sparse Transformer barely loses any quality — the full-token version is 34.18 and the sparse version 34.15, on which the paper bases its claim that "pruning removes only redundant and unnecessary tokens without hurting performance."

The reason "removing image-domain propagation drops 1.10 dB" is visible at a glance in the qualitative figure. The figure below has only two settings: the middle column removes image propagation (Exp. a) and relies only on feature-domain alignment and Transformer inpainting, and the right column is the full model. In the top-row racing-car scene, the "GOODYEAR" lettering inside the green box, and in the bottom-row off-road-cycling scene, the wire mesh inside the green box, are blurred into a mess or distorted in the middle column, while the right column, because it warps directly on the original pixels, cleanly restores the text edges and the grid — the feature domain operates on downsampled features and cannot recover such high-frequency detail.

![Qualitative ablation with and without image-domain propagation: in the middle column (w/o Img Prop.) the GOODYEAR lettering and wire mesh inside the green box are blurred and distorted, while the right column (w/ Img Prop.) warps directly on the original pixels for a clean restoration](imgs/flow_prop_ablation.png)

And "replacing image propagation with FGVC is actually worse (32.91)" comes from the presence or absence of the reliability check. In the figure below, column 2 is FGVC's image-propagation intermediate result: without a forward-backward consistency gate, it is led by wrong flow to force deformed content into the mask, so FGVC's final output in column 3 leaves ghosting inside the red box that the later stages cannot remove. ProPainter does the opposite — column 4 is its image-propagation intermediate result, where the consistency check actively rejects unreliable warping and leaves uncertain regions as green holes, and column 5 completes them on this clean basis. This is exactly the reason the earlier forward-backward consistency error threshold ($\mathcal{E}<\epsilon$) exists.

![Qualitative comparison of FGVC's and ProPainter's image propagation: FGVC (columns 2–3) lacks the reliability check and forces wrong-flow content in, leaving ghosting; ProPainter (columns 4–5) uses the consistency check to leave unreliable regions blank and then completes them](imgs/img_prop_comparison.png)

### Checking against the official implementation: what the code confirms, and what it says beyond the paper

The stage order of the official repo's inference entry point `inference_propainter.py` matches the paper: first RAFT extracts the flow in chunks, then the recurrent network completes the flow, then image-domain propagation, then feature-domain propagation plus the Transformer, and finally the fused output; the process loads the three weights raft-things.pth, recurrent_flow_completion.pth, and ProPainter.pth. The code also confirms the paper's key settings one by one: the Transformer has 8 blocks (depths = 8), the window size is 5×9 (window_size = (5, 9)), image-domain propagation has no learnable parameters (learnable=False) while feature-domain propagation is learnable, and the temporal stride of key/value is 2 (t_dilation=2).

But the code also exposes two details the paper does not spell out yet are crucial for reproduction. First, the long-video inference the paper describes is, in the released version, handled by `--subvideo_length` (sub-clip length, default 80), which splits the video into sub-clips processed block by block, and which the official description says "decouples GPU memory cost from video length." The official README lists both `--subvideo_length` and `--fp16` as tunable memory options, rather than as hard prerequisites for high resolution: at 1280×720, for example, the default 80-frame sub-clip OOMs under fp32 and needs about 25G with fp16, but shrinking the sub-clip length to 50 frames brings fp32 down to just 28G and fp16 to 19G; at 720×480, whether 50 or 80 frames, fp32 or fp16, it all stays within 13G. In other words, a longer sub-clip trades fewer seams for higher memory, `--fp16` further lowers memory, and both are trade-off knobs. Second, the released version's RAFT iteration default is 20 (`--raft_iter` defaults to 20), while the paper specifically declared only 5 iterations when measuring efficiency; to reproduce the paper's seconds-per-frame, it must be turned back down to 5.

## 🧪 Critical Assessment

### The problem is real, but the evaluation resolution is misaligned with the use case the paper cares about most

Video inpainting and object removal are problems with real demand — that much is beyond doubt. But the headline comparisons are all done at a very low resolution like $432\times 240$, whereas the real applications (watermark removal, object removal) almost all happen at 720p and above. The paper adds only one 480p table in the appendix (actual size $864\times 480$), and that table leaves only two opponents, STTN and E$^2$FGVI — because TSAM, FuseFormer, and FGT already run out of memory or become too slow on a 32G GPU. On this table with only two opponents left, ProPainter's PSNR is 33.81 versus E$^2$FGVI's 32.98, and 0.249 s per frame versus 0.332 s, so it does still lead; but this is exactly the point — in other words, the high-resolution scenario that would best highlight ProPainter's efficiency claim is precisely where opponents are fewest and fair comparison is hardest, while the headline "large margin" was obtained under the low-resolution setting with a full field of opponents.

### The headline numbers have one internal mismatch of their own

The abstract and Table 1 point to about a 1.46 dB PSNR lead on DAVIS ($34.47-33.01=1.46$), yet the main-text Comparisons paragraph writes "surpasses SOTA methods by 1.14 dB on DAVIS." These two numbers are inconsistent, and 1.14 cannot be derived from the table. This kind of internal inconsistency does not change the direction of the conclusion, but it reminds the reader: the paper's proofreading of numbers is not rigorous, and when citing, one should use the 1.46 dB that is consistent between the table and the abstract, not the 1.14 dB in that main-text sentence.

### The gain depends heavily on "the scene having enough motion"

The entire value of dual-domain propagation is built on "there being reliable flow to warp." The paper itself admits in the appendix: the gain on DAVIS is clearly larger than on YouTube-VOS, because YouTube-VOS has many nearly stationary scenes with no motion, which limits the effect of the propagation module, and it attaches the motion-magnitude distributions of the two datasets as evidence. This is actually an honest but non-trivial limitation: for nearly static shots, the occluded region is never exposed in another frame, propagation has nothing to work with, and ProPainter's advantage over Transformer-only methods converges. The mere 0.72 dB lead on YouTube-VOS ($34.43-33.71$) is exactly the quantification of this effect. The motion-magnitude distribution histogram from the appendix below states this limitation very plainly: the green YouTube-VOS is almost entirely bunched into the peak below 1 pixel of motion magnitude, representing a large number of nearly static shots; the blue DAVIS extends smoothly all the way past 14 pixels. For an occluded background to be warped back, it must have been exposed in another frame due to motion — YouTube-VOS's static distribution is exactly what leaves dual-domain propagation with nothing to work with.

![Motion-magnitude distributions of YouTube-VOS (green) and DAVIS (blue): YouTube-VOS concentrates in the static peak at <1 pixel, DAVIS extends smoothly past 14 pixels, explaining the gap in propagation gain between the two datasets](imgs/motion_hist.png)

### The sparsity assumption loosens under large masks and at cuts

Both premises of the efficiency claim are data-dependent. Sparse query relies on "the mask being only 13.6%," but this is the number for DAVIS object removal; faced with large-area watermarks, subtitle bars, or objects occupying much of the frame, $S_Q$ is almost all 1, and the savings on the query side evaporate (the paper's FLOPs curve also deliberately computes with a missing ratio of 1/6). Sparse key/value uses an alternating temporal stride of 2, which implicitly assumes "adjacent frames are highly redundant"; once fast motion or a shot cut is encountered, the skipped half of the frames may be exactly the ones carrying the needed content, and the paper does not test this assumption on long shots containing cuts. In addition, the claim that "sparsity does not hurt performance" actually has a measurable small decline of 0.03 dB (34.18→34.15) and 0.0001 SSIM, only that the magnitude is tiny.

### The novelty leans toward systems integration, lacking quantification of the "removal" use case and failure cases

The paper positions itself as a "systematic study," which is honest: flow-guided deformable alignment comes from BasicVSR++, soft split from FuseFormer, global tokens and windowing from FGT/E$^2$FGVI, and the consistency check from DFVI/FGVC. The genuinely new parts are two — the trainable GPU image-domain propagation with a reliability check, and the mask-guided sparse query plus temporal-stride KV. This is solid engineering integration, but treating it as a methodological breakthrough would overstate its novelty. The more practical gap is: all quantitative metrics are built on the proxy task of "reconstructing randomly placed stationary masks" (where ground truth exists to compute PSNR), while the "object removal" that users actually care about has only qualitative figures — no quantification or user study — and no systematic failure-case analysis (for example, what the collapse looks like when the flow is severely wrong). Therefore "leading across the board on benchmarks" should not be read as "universally best on real object removal."

The paper's only evidence for the "removal" use case is the qualitative figure below (the first two rows are video completion, the last two rows are object removal). ProPainter (rightmost column) does clean up the large patches of black haze and noise that FuseFormer and FGT leave inside the white box, reconstructing more continuous road surfaces and track fences; but look closely at row 3 — the green car being removed has its body disappear, yet the shadow it originally cast on the ground remains in the outputs of all methods, including ProPainter. This precisely hits the proxy-task problem mentioned earlier: the evaluation mask only frames the object body, not its shadow, so the "removal" is visually incomplete, and this kind of gap will not be caught by a reconstruction metric like PSNR.

![Qualitative comparison of video completion (rows 1–2) and object removal (rows 3–4): ProPainter (rightmost) has far less black haze than FuseFormer and FGT, but in the row-3 car removal the ground shadow remains in all methods (including ProPainter)](imgs/visual_comparison.png)

## One-minute version

- The core difficulty is not "creating something from nothing," but reaching across frames that are far apart to bring the clean background that was once exposed back to the occluded position. Guessing from only a few neighboring frames (e.g. FGT) leaves black haze and distortion inside the car-removal box; only reliable long-range propagation fills it cleanly.
- Image-domain propagation warps neighbor-frame pixels directly into the holes on the original, un-downsampled image, using only forward-backward-consistent flow. In the car and pedestrian removals, the large masks are almost entirely filled by this single step; removing it in the ablation drops PSNR from 34.15 to 33.05, the largest single decline.
- The sparse Transformer computes spatially only the windows that "touch the mask," and temporally halves the key/value frames with a stride of 2. When processing a 60-frame-long video, the compute of a single Transformer block is compressed from FGT's 824G and E$^2$FGVI's 690G down to 253G.
- The headline win is quality and efficiency at the same time: PSNR 34.47 dB on DAVIS, with only 808G FLOPs and 0.083 s per frame for 10 frames — both more accurate and more efficient than the runner-up E$^2$FGVI.
- The most critical flaw is that the gain depends heavily on the shot having motion: on the nearly static YouTube-VOS the lead is only 0.72 dB; and all objective scores are built on the reconstruction proxy task of randomly placed stationary masks, with no quantitative evaluation or user study at all for real object removal.
- There are also implementation gaps: the official code defaults RAFT to 20 iterations rather than the 5 declared when measuring speed; memory is a tunable trade-off — at 1280×720, for example, the default 80-frame sub-video OOMs under fp32 and needs about 25G with fp16, while shrinking the sub-video to 50 frames brings fp32 down to just 28G, and both `--subvideo_length` and `--fp16` are knobs rather than hard prerequisites.

## 🔗 Related notes

- [DiffuEraser](../DiffuEraser/) — a follow-up work that directly uses ProPainter as its prior model (prior) and argues for switching to a diffusion model to make up for its insufficient generative power under large masks.
