# xLSTM — Research Note
> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | xLSTM: Extended Long Short-Term Memory |
| Venue | Neural Information Processing Systems (NeurIPS) |
| Year | 2024 |
| Authors | Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter |
| Official Code | https://github.com/NX-AI/xlstm |
| Venue Kind | paper |

> 本篇為基於 arXiv 全文（`2405.04517`，2024-05-07 首版）的研究筆記。NeurIPS 2024 的正式版由 Semantic Scholar 的 publication venue 欄位交叉確認；截至 2026-07-03，該紀錄回報約 632 次引用（Semantic Scholar，可能隨時間變動）。

## First Principles

### 從 constant error carousel 到三個致命限制

LSTM 的核心是 1990 年代提出的 constant error carousel（固定誤差旋轉木馬）與 gating（閘控）：cell state 以加法方式被更新，由 sigmoid 閘控制寫入與遺忘，藉此繞過 RNN 的梯度消失問題。作者主張 LSTM 有三個主要限制：(i) 無法修正已儲存的決策（storage decision），因為 sigmoid forget gate 難以在看到更相似向量時覆寫舊值；(ii) 儲存容量有限，資訊被壓縮進純量 cell state；(iii) 因為 memory mixing（hidden-hidden 連接）而無法平行化。xLSTM 的整篇論文就是要在保留 LSTM 精神的前提下逐一鬆綁這三點。

原始 LSTM 的 cell state 與 hidden state 更新可寫成下式，其中 $f_t, i_t, o_t$ 為 sigmoid 閘、$z_t$ 為 cell input、$\psi$ 為壓縮函數：

$$
c_t = f_t \, c_{t-1} + i_t \, z_t, \qquad h_t = o_t \, \psi(c_t)
$$

### sLSTM：指數閘控（exponential gating）與純量記憶

sLSTM 的第一項改動是把 input gate（以及可選的 forget gate）的激活函數從 sigmoid 換成 exponential（指數），讓閘值不再被壓在 $[0,1]$，因此模型能大幅放大新輸入、有效「覆寫」舊記憶——這正對應限制 (i)。指數會爆炸，所以作者額外引入一個 normalizer state $n_t$（累加 input gate 乘上所有後續 forget gate）把輸出正規化，並用一個 stabilizer state $m_t$（取對數域最大值）做數值穩定；論文證明這個穩定化不改變網路輸出與梯度。

sLSTM 的純量前向遞迴與正規化如下：

$$
c_t = f_t \, c_{t-1} + i_t \, z_t, \quad n_t = f_t \, n_{t-1} + i_t, \quad \tilde{h}_t = c_t / n_t, \quad h_t = o_t \, \tilde{h}_t
$$

sLSTM 也保留多個 memory cell，並可切成多個 head：在同一個 head 內部透過 recurrent 矩陣 $R$ 做 memory mixing，但 head 之間不混合。作者強調，把 head 與指數閘控結合起來，構成了一種新的 memory mixing 形式，而這正是 sLSTM 相對於 SSM 與 linear attention 的關鍵差異點——後兩者沒有 memory mixing，因此無法做 state tracking。

### mLSTM：矩陣記憶與 covariance 更新規則

mLSTM 針對限制 (ii)：把純量 cell state $c \in \mathbb{R}$ 升級為矩陣記憶 $C \in \mathbb{R}^{d \times d}$，於是「檢索」變成一次矩陣乘法。借用 Transformer 的術語，每個時間步存入一對 key $k_t$ 與 value $v_t$，日後由 query $q_{t+\tau}$ 取回,這其實是 Bidirectional Associative Memory 的設定。存入採用 covariance（外積）更新規則 $C_t = C_{t-1} + v_t k_t^\top$；把它套進 LSTM 框架後，forget gate 就對應 decay rate、input gate 對應 learning rate、output gate 縮放取回的向量。

矩陣記憶的更新與檢索（含 normalizer state $n_t$ 與下界為 1 的分母）可寫成：

$$
C_t = f_t \, C_{t-1} + i_t \, v_t k_t^\top, \quad n_t = f_t \, n_{t-1} + i_t \, k_t, \quad \tilde{h}_t = \frac{C_t \, q_t}{\max\{\,|n_t^\top q_t|,\; 1\,\}}
$$

因為 mLSTM 拿掉了 memory mixing（沒有 hidden-hidden 連接），這條遞迴可以被改寫成一個平行版本，訓練時像 attention 一樣沿序列平行計算；代價是每步都要處理 $d \times d$ 的矩陣，計算量大，但這些矩陣運算不含參數且能在 GPU 上平行，牆鐘時間的額外負擔有限。

### xLSTM 區塊與 xLSTM[a:b] 架構

兩種 cell 各自被包進不同的殘差區塊：sLSTM 用 post up-projection block（像 Transformer：先在原空間非線性摘要過去，再投影到高維、非線性、投回），mLSTM 用 pre up-projection block（像 SSM：先投到高維再摘要，因為高維空間裡矩陣記憶容量更大）；設計動機引用 Cover 定理——高維空間中非線性嵌入的樣式更容易被線性分開。整體架構就是把這些區塊以 pre-LayerNorm 殘差方式堆疊。論文用 xLSTM[$a$:$b$] 表示 mLSTM 與 sLSTM 區塊的比例：例如 xLSTM[7:1] 代表每八個區塊有七個 mLSTM、一個 sLSTM；For a common total block number of 48, this translates to 6 個 sLSTM 區塊與 42 個 mLSTM 區塊。

### 一次「LSTM → xLSTM」的具體推演

理解 xLSTM 各元件貢獻的最好方式，是看作者在 15B token SlimPajama 上把一個 vanilla LSTM 逐步 morph 成 xLSTM 的消融：從最原始的多層 LSTM 出發，依序加上 ResNet backbone、up-projection backbone、指數閘控、矩陣記憶，觀察 validation perplexity（越低越好）如何一路下降。

| 模型階段 | 指數閘控 | 矩陣記憶 | #Params (M) | SlimPajama (15B) ppl ↓ |
|-|-|-|-|-|
| Vanilla 多層 LSTM | ✗ | ✗ | 607.8 | 2417.86 |
| + ResNet backbone | ✗ | ✗ | 506.1 | 35.46 |
| + up-projection backbone | ✗ | ✗ | 505.9 | 26.01 |
| xLSTM[0:1]（加指數閘控） | ✓ | ✗ | 427.3 | 17.70 |
| xLSTM[7:1]（再加矩陣記憶） | ✓ | ✓ | 408.4 | 13.48 |

這條軌跡把故事講得很清楚：光是把一個約 6 億參數的裸 LSTM 放進殘差 backbone，perplexity 就從 2417.86 崩落到 35.46（裸 LSTM 這種規模幾乎訓不動）；up-projection 再降到 26.01；換上指數閘控（此時已是純 sLSTM 的 xLSTM[0:1]）跳到 17.70；最後補上 mLSTM 的矩陣記憶得到 13.48。作者據此把主要增益歸因於指數閘控與矩陣記憶兩者，而不是單一元件——這也是全文最有說服力的一張表。

### 主要語言模型結果

在同樣的 15B token 設定下，作者拿 xLSTM 對打各家 350M 級模型（皆對齊 GPT-3 350M 的維度）。下表節錄各類別代表的 validation perplexity：

| 模型（類別） | #Params (M) | SlimPajama (15B) ppl ↓ |
|-|-|-|
| Llama (Transformer) | 407 | 14.25 |
| Mamba (SSM) | 423 | 13.70 |
| RWKV-5 (RNN) | 456 | 14.25 |
| HGRN2 (RNN) | 411 | 14.32 |
| **xLSTM[1:0]** | 409 | **13.43** |
| xLSTM[7:1] | 408 | 13.48 |

在這張表裡 xLSTM outperforms all existing methods in validation perplexity，其中純 mLSTM 的 xLSTM[1:0] 以 13.43 拿下全場最佳，略勝 Mamba 的 13.70 與 Llama 的 14.25。接著作者把資料放大 20 倍、訓練 300B token，並在 125M–1.3B 四種規模上比較 xLSTM、Llama、Mamba、RWKV-4。以下節錄 1.3B 規模的驗證困惑度與下游任務：

| 模型 (1.3B) | #Params (M) | SlimPajama ppl ↓ | LAMBADA acc ↑ | HellaSwag acc ↑ | 平均 acc ↑ |
|-|-|-|-|-|-|
| RWKV-4 | 1515.2 | 9.83 | 49.78 | 56.20 | 54.78 |
| Llama | 1420.4 | 9.44 | 57.44 | 57.81 | 56.99 |
| Mamba | 1475.3 | 9.14 | 55.64 | 60.45 | 58.41 |
| **xLSTM[1:0]** | 1422.6 | **8.89** | 57.83 | 60.91 | 58.48 |

在 1.3B 上 xLSTM[1:0] 的驗證困惑度 8.89 仍最低。更細緻的證據來自 PALOMA 的 571 個文本領域：作者回報 xLSTM[1:0] 在 568 out of 571（99.5%）的領域上困惑度低於 Mamba、在 85.1% 的領域低於 Llama、在 99.8% 的領域低於 RWKV-4。此外，訓練在 context 2048、測到 16384 的長度外推實驗中 xLSTM 的困惑度維持穩定，而其 recurrent 特性讓生成時間隨序列線性增長、可用比 Llama 更大的 batch 而取得更高吞吐。

## 🧪 Critical Assessment

### 線性時間序列模型追趕 Transformer 是真實需求

論文提出的問題本身是真實的：在 Transformer 的二次複雜度成為長序列與推論成本瓶頸的當下，How far do we get in language modeling 這種「線性時間、常數記憶的序列模型能否追上 Transformer」的追問，與 SSM/RWKV 一整條研究線同源，並非人造需求。合理的判讀是，這是一個有實際工程價值的問題，且 xLSTM 提供了一個「把 LSTM 現代化」而非另起爐灶的角度，這一點有其獨立意義。

### 廣泛基線之下，單一語料與 1.3B 上限削弱外部效度

基線覆蓋面相當廣（Transformer、SSM、多代 RWKV、GLA、HGRN2、RetNet 等），LSTM→xLSTM 的逐步消融也做得漂亮，這是強項。值得懷疑的是外部效度：所有語言模型結果都建立在單一語料 SlimPajama 上（125M, 350M, 760M, 1.3B），最大只到 1.3B，且主要度量高度集中在 validation perplexity。以困惑度為主軸、規模停在 1.3B，讓「能與最先進 Transformer 抗衡」的宣稱在真正的 LLM 尺度上仍屬外推而非實證。

### mLSTM 與線性注意力同源，sLSTM 的 memory mixing 才是區辨點

需要誠實區分兩塊。mLSTM 的矩陣記憶與外積更新並非全新：論文自己就寫明 The covariance update rule is equivalent to Fast Weight Programmers，與 linear attention、Retention、RWKV-5/6 共享同一數學骨架，這部分較接近把既有機制重新編排並補上指數閘控與正規化。真正較難在其他線性模型中找到對應物的，是 sLSTM 帶 head 的 memory mixing 所宣稱的 state tracking 能力——這是 xLSTM 敘事中最具區辨性的貢獻，也是它與 SSM 拉開概念差距之處。

### 形式語言實驗圍繞 sLSTM 的強項設計

形式語言（formal language）實驗需要保留戒心：作者展示 Transformer 與 SSM cannot solve, e.g. regular grammars like the parity task，而具 memory mixing 的 sLSTM 可以。這個結論方向雖與既有理論一致，但基準本身正好挑在「需要 memory mixing」的任務族上，等於是圍繞自家 sLSTM 的強項來定義題目；它證明的是「有 memory mixing 比沒有好」，而未必等同於在主流下游任務上的普遍優勢。

### 1.3B 以下成立，但 kernel 未最佳化使大規模部署仍未證

在 1.3B 以下、以困惑度衡量的範圍內，「把 LSTM 擴展到能與 Transformer/SSM 抗衡」大致成立；但論文並未宣稱問題已完全解決。作者在 Limitations 明言 CUDA kernel 尚未最佳化、mLSTM 約比 FlashAttention 慢 4 倍，且坦承 we did neither fully optimize the architecture nor the hyperparameters，並預期 an extensive optimization process is needed for xLSTM to reach its full potential。因此就真實世界部署而言，xLSTM 目前更像是一個有力的存在性證明，能否在 7B 以上規模、以下游品質與實際吞吐同時勝出，仍屬未被證明。

## 🔗 Related notes

- [Attention Is All You Need](../AttentionIsAllYouNeed/) — xLSTM 全篇對標的 Transformer 基線與 self-attention 起點。
