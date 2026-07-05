# Wan 2.2 Architecture Technical Note
> **English** | [繁體中文](./README.zh-TW.md)

### 1. Core Idea: The Generality of the Transformer and Dimensional Magic

- **Essence:** The Transformer's Attention Kernel only recognizes `(B, L, D)`. It does not understand what "space" or "time" is.
- **Strategy:** Through **View-based Parallelism (dimensional transformation)**, Wan 2.2 dynamically maps the "physical dimensions ($T, H, W$)" in turn onto the "computational dimensions ($B, L$)", thereby achieving Spatiotemporal Decoupling.

### 2. Input Layer: From 5D to 3D (Tokenization)

- **Raw input:** $X \in \mathbb{R}^{B \times C \times T \times H \times W}$
- **3D VAE & Patching:**
  - It does not simply cut planar patches, but cuts **Volumes (voxels)**.
  - For example, Patch Size $(t_p, h_p, w_p) = (4, 8, 8)$.
- **Flatten:**
  - Straighten out all patches to form the initial sequence $L_{total} = \frac{T}{4} \times \frac{H}{8} \times \frac{W}{8}$.
  - **Transformer Input:** `(B, L_total, D)`. At this point $L$ mixes spatiotemporal information.

### 3. Attention Mechanism: Factorized Attention (Sequential Processing)

To avoid the computational blow-up of $O(L_{total}^2)$, Attention is decomposed into two steps, controlling the information flow through `einops`-style Reshapes:

#### A. Spatial Attention (handling composition and texture)

- **Logic:** Let each frame be computed independently, without interfering with one another.
- **Tensor transformation:**
  $$(B, T \cdot H \cdot W, D) \xrightarrow{\text{Reshape}} (B \cdot T, H \cdot W, D)$$
- **For the Transformer:**
  - **Effective Batch:** $B' = B \times T$ (the Batch grows larger)
  - **Effective Length:** $L' = H \times W$ (the Sequence becomes shorter)
- **Physical meaning:** The time axis is isolated in the Batch dimension, and Attention cannot cross from frame to frame.

#### B. Temporal Attention (handling motion and coherence)

- **Logic:** Let each spatial position $(h, w)$ independently compute its temporal change.
- **Tensor transformation:**
  $$(B, T \cdot H \cdot W, D) \xrightarrow{\text{Reshape}} (B \cdot H \cdot W, T, D)$$
- **For the Transformer:**
  - **Effective Batch:** $B'' = B \times H \times W$ (the Batch is maximized)
  - **Effective Length:** $L'' = T$ (the Sequence is extremely short)
- **Physical meaning:** Spatial neighbors are isolated in the Batch dimension, and a pixel can only see its own past and future.

### 4. Cross-Attention (Conditioning)

- **Position:** Inserted between (or after) the Spatial and Temporal Blocks.
- **Purpose:** ID Preservation (identity locking).
- **Mechanism:**
  - $Q$ = the currently generated Video Latents `(B, L, D)`
  - $K, V$ = the Embeddings of the Reference Image
- **Difference from GPT:** GPT (Decoder-only) has no Cross-Attn; Wan 2.2 must have this layer to "pull" the generated pixels back to the features of the reference image.

### 5. Generation and Positional Encoding

- **Positional Embedding (3D RoPE):**
  Because the Transformer has Permutation Invariance, and the Reshape scrambles the order, it must rely on **3D Rotary Embedding** to mark the $(x, y, t)$ coordinates of each Token.
- **Flow Matching:**
  Not Autoregressive ($t \to t+1$), but global denoising (ODE Solver). This means that when Temporal Attention is computed it is **Bidirectional**, able to see the context before and after throughout the entire video.

---

### Summary (Engineering Takeaway)

The core innovation of Wan 2.2 does not lie in inventing a new Transformer, but in designing an efficient **Data Pipeline**:

1.  **Physical level:** Compress information through the 3D VAE.
2.  **Logical level:** Through Reshape operations, trick the Transformer into "mistakenly believing", at different steps, that it is only processing images or only processing a time series.
3.  **Control level:** Through Cross-Attention, forcibly inject reference-image features to ensure the generated content does not drift.
