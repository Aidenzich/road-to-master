# Wan 2.2 Architecture Technical Note

### 1. 核心觀念：Transformer 的通用性與維度魔術

- **本質：** Transformer 的 Attention Kernel 只認 `(B, L, D)`。它不理解什麼是「空間」或「時間」。
- **策略：** Wan 2.2 透過 **View-based Parallelism (維度變換)**，動態地將「物理維度 ($T, H, W$)」輪流映射到「計算維度 ($B, L$)」，以此實現時空解耦（Spatiotemporal Decoupling）。

### 2. 輸入層：從 5D 到 3D (Tokenization)

- **原始輸入：** $X \in \mathbb{R}^{B \times C \times T \times H \times W}$
- **3D VAE & Patching：**
  - 不單純切平面 Patch，而是切 **Volume (體素)**。
  - 例如 Patch Size $(t_p, h_p, w_p) = (4, 8, 8)$。
- **展平 (Flatten)：**
  - 將所有 Patch 拉直，形成初始序列 $L_{total} = \frac{T}{4} \times \frac{H}{8} \times \frac{W}{8}$。
  - **Transformer Input:** `(B, L_total, D)`。此時的 $L$ 混合了時空資訊。

### 3. Attention 機制：Factorized Attention (串聯處理)

為了避免 $O(L_{total}^2)$ 的計算量爆炸，將 Attention 拆解為兩個步驟，透過 `einops` 風格的 Reshape 來控制資訊流：

#### A. Spatial Attention (處理構圖與紋理)

- **邏輯：** 讓每一幀獨立計算，互不干擾。
- **Tensor 變換：**
  $$(B, T \cdot H \cdot W, D) \xrightarrow{\text{Reshape}} (B \cdot T, H \cdot W, D)$$
- **對 Transformer 來說：**
  - **Effective Batch:** $B' = B \times T$ (Batch 變大了)
  - **Effective Length:** $L' = H \times W$ (Sequence 變短了)
- **物理意義：** 時間軸被隔離在 Batch 維度中，Attention 無法跨越幀與幀。

#### B. Temporal Attention (處理動作與連貫性)

- **邏輯：** 讓每一個空間位置 $(h, w)$ 獨立計算其時間變化。
- **Tensor 變換：**
  $$(B, T \cdot H \cdot W, D) \xrightarrow{\text{Reshape}} (B \cdot H \cdot W, T, D)$$
- **對 Transformer 來說：**
  - **Effective Batch:** $B'' = B \times H \times W$ (Batch 極大化)
  - **Effective Length:** $L'' = T$ (Sequence 極短)
- **物理意義：** 空間鄰居被隔離在 Batch 維度中，像素只能看自己的過去與未來。

### 4. Cross-Attention (Conditioning)

- **位置：** 插入在 Spatial 與 Temporal Block 之間（或之後）。
- **目的：** ID Preservation (身份鎖定)。
- **機制：**
  - $Q$ = 當前生成的 Video Latents `(B, L, D)`
  - $K, V$ = 參考圖 (Reference Image) 的 Embeddings
- **與 GPT 的差異：** GPT (Decoder-only) 無 Cross-Attn；Wan 2.2 必須有此層才能將生成的像素「拉回」參考圖的特徵。

### 5. 生成與位置編碼

- **Positional Embedding (3D RoPE)：**
  由於 Transformer 具有 Permutation Invariance (排列不變性)，且 Reshape 打亂了順序，必須依賴 **3D Rotary Embedding** 來標記每個 Token 的 $(x, y, t)$ 座標。
- **Flow Matching：**
  非 Autoregressive ($t \to t+1$)，而是全局去噪 (ODE Solver)。這意味著 Temporal Attention 在計算時是 **Bidirectional (雙向)** 的，可以看到整部影片的前後文。

---

### 總結 (Engineering Takeaway)

Wan 2.2 的核心創新不在於發明了新的 Transformer，而在於設計了一套高效的 **Data Pipeline**：

1.  **物理層面：** 透過 3D VAE 壓縮資訊。
2.  **邏輯層面：** 透過 Reshape 操作，欺騙 Transformer 讓它在不同的步驟「誤以為」自己只在處理圖片或是只在處理時間序列。
3.  **控制層面：** 透過 Cross-Attention 強制注入參考圖特徵，確保生成內容不走樣。
