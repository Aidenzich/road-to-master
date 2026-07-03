# DiT：以 Transformer 為骨幹的可擴展擴散模型 — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Scalable Diffusion Models with Transformers |
| Venue | ICCV |
| Year | 2023 |
| Authors | William Peebles, Saining Xie |
| Official Code | https://github.com/facebookresearch/DiT |
| Venue Kind | paper |

本文（下稱 DiT）處理一個具體且長期被忽略的架構問題：自 DDPM 以來，影像擴散模型幾乎一律沿用卷積式的 U-Net 作為去噪骨幹，而 Transformer 雖已橫掃語言與視覺辨識，卻遲遲未進入擴散模型的主幹。作者（UC Berkeley 的 William Peebles 與 NYU 的 Saining Xie，工作於 Meta AI FAIR 期間完成）主張 U-Net 的歸納偏好（inductive bias）並非擴散模型效能的關鍵，並以純 Transformer 骨幹在 class-conditional ImageNet 256×256 上取得 2.27 的 state-of-the-art FID。本文以 ICCV 2023 oral 發表，官方程式碼與模型權重由 Meta（facebookresearch/DiT）釋出。

## First Principles

### 從 U-Net 到 Transformer：問題設定與潛空間策略

DiT 的核心主張很直接：把擴散模型常用的 U-Net 骨幹換成一個直接在潛空間 patch 上運作的 Transformer。作者刻意盡量忠於標準 Transformer / ViT 設計，以便繼承其可擴展性（scaling）性質，並把「網路複雜度 vs 樣本品質」當作研究主軸——複雜度以理論 Gflops 量測，品質以 FID 量測。

為了讓運算可負擔，DiT 建立在 Latent Diffusion Models（LDM）框架上：先用一個凍結的 VAE 編碼器 $E$ 把影像壓成較小的空間表徵 $z = E(x)$，擴散過程在 $z$ 上進行，取樣完再用解碼器 $x = D(z)$ 還原。本文所用的 VAE 直接取自 Stable Diffusion，其編碼器有 8 倍下採樣（a downsample factor of 8）：一張 $256\times256\times3$ 的影像會被壓成 $32\times32\times4$ 的潛表徵。這一步把像素空間的高解析度負擔轉嫁給輕量的 VAE，讓 Transformer 只需處理 $32\times32$ 這種小網格。

擴散本身沿用 DDPM 的標準公式。前向加噪過程對真實資料 $x_0$ 逐步加入高斯噪聲：

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

網路 $\epsilon_\theta$ 被訓練來預測所加的噪聲，主損失是預測噪聲與真實噪聲之間的均方誤差：

$$
\mathcal{L}_{simple}(\theta) = ||\epsilon_\theta(x_t) - \epsilon_t||_2^2
$$

DiT 同時沿用 Nichol & Dhariwal 的做法，額外以完整的變分下界訓練可學習的協方差 $\Sigma_\theta$。條件生成則靠 classifier-free guidance：訓練時隨機把類別 $c$ 換成一個學到的 null 嵌入 $\emptyset$，取樣時把輸出往「有條件」方向外推

$$
\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t,\emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))
$$

其中 $s>1$ 為 guidance 尺度，$s=1$ 即回到一般取樣。這條公式在後面的 SOTA 數字裡是關鍵——沒有 guidance 時 DiT-XL/2 的 FID 只有 9.62，加上 $s=1.5$ 的 guidance 後才降到 2.27。

### Patchify：把潛表徵切成 token 序列

![DiT 的輸入規格：patchify](imgs/patchify.png)

DiT 的第一層是 patchify：把形狀 $I\times I \times C$ 的潛表徵，用 $p\times p$ 的 patch 線性嵌入成一段長度為 $T = (I/p)^2$、每個 token 維度為 $d$ 的序列，再加上標準 ViT 的 sine-cosine 頻率位置編碼。patch 大小 $p$ 是決定序列長度的關鍵超參數：把 $p$ 減半會讓 $T$ 變成四倍，因此至少讓 Transformer 的總 Gflops 變成四倍；但 $p$ 幾乎不影響參數量。作者把 $p \in \{2,4,8\}$ 納入設計空間，這正是「同樣參數量、不同運算量」得以被獨立研究的機制。

### 四種條件注入方式：為何 adaLN-Zero 勝出

![DiT 區塊設計與條件注入](imgs/architecture.png)

擴散模型需要把噪聲時間步 $t$ 與類別 $c$ 注入骨幹。作者比較了四種 Transformer 區塊：（1）in-context——把 $t$、$c$ 當成兩個額外 token 接在序列後；（2）cross-attention——在自注意力後多加一層對 $t$、$c$ 的交叉注意力，約增加 15% Gflops；（3）adaptive layer norm（adaLN）——不直接學縮放與平移，而是從 $t$、$c$ 的嵌入和回歸出 layer norm 的 $\gamma$、$\beta$；（4）adaLN-Zero——在 adaLN 之上，再回歸一組作用於殘差連接前的縮放參數 $\alpha$，並把該 MLP 初始化為輸出零向量，使整個 DiT 區塊初始為恆等函數（identity function）。

![四種條件策略的 FID 比較](imgs/conditioning.png)

實驗結果很明確：adaLN-Zero 在所有訓練階段都優於 cross-attention 與 in-context，且是最省算力的一種（在 XL/2 上 adaLN-Zero 只要 118.6 Gflops，cross-attention 卻要 137.6 Gflops）。在 400K 訓練步時，adaLN-Zero 的 FID 幾乎是 in-context 的一半，而恆等初始化本身也很重要——adaLN-Zero 明顯勝過未做零初始化的 vanilla adaLN。因此全文其餘部分一律採用 adaLN-Zero。這是本文少數真正的架構性發現：條件注入方式的選擇，對最終品質的影響大到足以左右結論。

### 模型尺寸與解碼頭

作者沿用 ViT 的配置慣例，聯合縮放層數 $N$、隱藏維度 $d$ 與注意力頭數，定義四個規模：

| Model | Layers $N$ | Hidden size $d$ | Heads | Gflops ($I$=32, $p$=4) |
|-|-|-|-|-|
| DiT-S | 12 | 384 | 6 | 1.4 |
| DiT-B | 12 | 768 | 12 | 5.6 |
| DiT-L | 24 | 1024 | 16 | 19.7 |
| DiT-XL | 28 | 1152 | 16 | 29.1 |

四個配置涵蓋 0.3 到 118.6 Gflops 的範圍。經過最後一個 DiT 區塊後，解碼頭是一個標準線性層：先做（adaLN 版的）最後 layer norm，再把每個 token 線性解碼成 $p \times p \times 2C$ 的張量——一半通道是預測噪聲、一半是預測對角協方差，最後重排回原始空間佈局。

### 一次具體的前向傳遞（DiT-XL/2，256×256）

以下用論文的真實數字走一遍最強模型的前向路徑：

```text
輸入影像            256 × 256 × 3
  │  VAE 編碼器（下採樣 8，凍結）
潛表徵 z            32 × 32 × 4          （I=32, C=4）
  │  patchify，patch 大小 p=2
token 序列          T = (32/2)^2 = 256 個 token，每個維度 d=1152
  │  + sine-cosine 位置編碼；注入 t、c 用 adaLN-Zero
  │  28 個 DiT 區塊（N=28, heads=16）      → 118.6 Gflops
線性解碼頭          每個 token → 2 × 2 × (2·4) = 2×2×8
  │  重排回空間佈局
輸出                噪聲預測 32×32×4 + 對角協方差 32×32×4
  │  DDPM 取樣 250 步；classifier-free guidance s=1.5
還原                VAE 解碼器 → 256 × 256 × 3
結果                FID-50K = 2.27（ImageNet 256×256，SOTA）
```

同一個 XL/2 架構搬到 512×512 時，輸入潛表徵變成 $64\times64\times4$，patch 大小仍為 2，於是序列長度變成 1024 個 token、單次前向 524.6 Gflops；即便如此，它仍遠比像素空間的 U-Net 省算力（ADM 用 1983 Gflops、ADM-U 用 2813 Gflops）。

### 可擴展性：Gflops 才是關鍵，而非參數量

![左：不同 Gflops 的 FID；右：DiT-XL/2 對比先前 U-Net 模型](imgs/bubbles.png)

本文最重要的實證結論是：DiT 的樣本品質由 Gflops 決定，而非參數量。作者掃過 4 個規模 × 3 個 patch 大小共 12 個模型，發現「加深加寬」與「縮小 patch（增加 token 數）」都能穩定降低 FID。特別是固定模型規模、只縮小 patch 時，總參數量幾乎不變（甚至略減），單純是 Gflops 上升，FID 卻顯著下降——這說明參數量無法唯一決定品質。

![Transformer Gflops 與 FID-50K 高度相關](imgs/gflops_fid.png)

把 12 個模型在 400K 步的 FID-50K 對 Gflops 作圖，可見很強的負相關：Gflops 相近的不同配置（例如 DiT-S/2 與 DiT-B/4）得到相近的 FID。作者據此主張「增加模型算力」是改善 DiT 的關鍵因素。他們也另外指出，增加「取樣時」算力（更多取樣步數）無法補償「模型算力」的不足——小模型即使取樣步數遠多於大模型，FID 仍追不上。

![固定 patch 或固定規模時，放大 DiT 都改善 FID](imgs/scaling.png)

在最終的 SOTA 比較上，DiT-XL/2 持續訓練到 7M 步後，加上 $s=1.5$ 的 classifier-free guidance，把先前由 LDM 保持的 3.60 FID 紀錄推進到 2.27，並在 Inception Score、Recall 等次要指標上也表現優異；在 512×512 上也把 ADM 的 3.85 改善到 3.04。下表為 256×256 的主要對照：

| Model | FID↓ | IS↑ | Precision↑ | Recall↑ |
|-|-|-|-|-|
| ADM-G, ADM-U | 3.94 | 215.84 | 0.83 | 0.53 |
| LDM-4-G (cfg=1.50) | 3.60 | 247.67 | 0.87 | 0.48 |
| StyleGAN-XL | 2.30 | 265.12 | 0.78 | 0.53 |
| DiT-XL/2-G (cfg=1.50) | 2.27 | 278.24 | 0.83 | 0.57 |

## 🧪 Critical Assessment

### 問題是否真實、是否重要

「U-Net 是否為擴散模型的必要條件」是一個貨真價實、可證偽的科學問題，而非人為包裝的假議題。在本文之前，整個社群預設 U-Net 不可或缺，DiT 用嚴謹的對照實驗把這個預設打破，並提供一組乾淨的縮放基線，對後續研究（Stable Diffusion 3、Sora、PixArt 等皆採用 DiT 系骨幹）確有實質影響。這一點上，本文的重要性經得起時間檢驗。

### 基線、消融與量測是否充分

實驗設計相當扎實：條件注入的四選一消融、12 個模型的規模 × patch 雙軸掃描、以及「模型算力 vs 取樣算力」的交叉比較，都直接支撐其主張；並刻意使用 ADM 的官方 TensorFlow FID 評測套件以確保可比性。但仍有值得保留之處。其一，FID 對實作細節極度敏感，而 DiT 是 JAX/TPU 實作、比較對象多為 PyTorch/GPU 實作，跨框架的絕對數字比較天然帶有噪聲；作者雖統一評測套件，仍無法完全消除此類差異。其二，adaLN-Zero 雖被宣稱「最佳」，但四種區塊的參數量並不相同（in-context 449M、cross-attention 598M、adaLN 600M、adaLN-Zero 675M），因此「adaLN-Zero 較好」與「adaLN-Zero 剛好參數也較多」在此消融中並未被完全解耦，這與本文主軸「參數量不重要、Gflops 才重要」之間存在一絲張力。

### 這是新方法還是既有元件的重組

平心而論，DiT 幾乎不含全新的數學元件：ViT 的 patchify、DDPM 的訓練目標、LDM 的潛空間、FiLM/adaLN 的條件注入、classifier-free guidance——全都是既有技術。本文的貢獻在於「證明這個組合可行且可擴展」，而非發明新機制。這是一種價值明確但屬於系統性、實證性的貢獻。需要留意的是，其「可擴展性」結論是在自己定義的 Gflops 座標軸與 ImageNet class-conditional 這個特定基準上成立的；把複雜度定義為 Gflops，本身就對「參數效率高、但 Gflops 高」的 Transformer 較有利，論證帶有一定程度的自我選定框架色彩。

### 宣稱的問題是否真的被解決、是否有現實意義

作者宣稱的「U-Net 並非必要」在其實驗範圍內確實被證立，且後續產業採用也提供了強力的外部驗證。但也應誠實指出本文未涵蓋的部分：全部結論僅來自 class-conditional ImageNet 的兩個解析度，並未觸及文生圖（text-to-image）這個擴散模型最主要的現實應用；作者自己也只把 text-to-image 列為 future work。此外，DiT-XL/2 在 TPU v3-256 上以約 5.7 iterations/second 訓練並跑到 7M 步，其可擴展性是以可觀的算力為代價換來的——「大模型更省算力」是在達到同一 FID 的相對意義下成立，並不意味絕對成本低廉。因此更精確的結論是：在有足夠算力的前提下，Transformer 是比 U-Net 更值得投資的擴散骨幹，而非「擴散模型變便宜了」。

## 🔗 Related notes

- [DDPM](../diffusion/)
- [ViT](../ViT/)
