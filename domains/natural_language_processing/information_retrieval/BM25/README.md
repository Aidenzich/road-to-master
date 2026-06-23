| Property | Data |
|-|-|
| Created | 2026-06-23 |
| Updated | 2026-06-23 |
| Author | @Aiden |
| Tags | #study #information-retrieval |

# BM25

## 核心定義

- **全名**：Best Matching 25，又常稱為 Okapi BM25。
- **核心概念**：BM25 是 TF-IDF 的 ranking 改良版，用來衡量一份文件（document）與一個查詢（query）之間的文字相關性。
- **解決痛點**：修正傳統 TF-IDF 中局部 TF 線性膨脹、長 document 天然佔優的問題。

BM25 仍然保留 TF-IDF 的核心直覺：一個 term 若在 query 和 document 中都出現，且它在整個 corpus 中相對稀有，這份 document 就更可能相關。差別在於 BM25 對 term frequency 和 document length 做了更合理的控制。

## 數學公式

對於 query $q$ 和 document $D$，BM25 分數通常寫成：

- $\color{orange}{k_1}$：控制 TF saturation。
- $\color{cyan}{b, |D|, \text{avgdl}}$：控制 document length normalization。

$$
\text{BM25}(D, q) =
\sum_{t \in q}
\text{IDF}(t)
\cdot
\frac{
    \text{freq}(t, D)( {\color{orange}{k_1}} + 1)
}
{
    \text{freq}(t, D) + {\color{orange}{k_1}} \left(1 - \color{cyan}{b} + {\color{cyan}{b} \cdot \frac{\color{cyan}{|D|}}{\color{cyan}{\text{avgdl}}}}\right)}
$$

其中：
- $\text{freq}(t, D)$：term $t$ 在 document $D$ 中出現的次數。
- $t$：query 中的某個詞項（term）。
- $\color{orange}{k_1}$：控制 TF saturation 的強度，常見範圍約 $1.2$ 到 $2.0$。
- $\color{cyan}{|D|}$：document $D$ 的長度，通常是 term 數量。
- $\color{cyan}{\text{avgdl}}$：corpus 中所有 document 的平均長度。
- $\color{cyan}{b}$：控制 document length normalization 的強度，常見預設值約 $0.75$。
- $\text{IDF}(t)$：term $t$ 的全域稀有度權重。

常見的 BM25 IDF 平滑版本為：

$$
\text{IDF}(t) =
\log \left(
\frac{N - \text{DF}(t) + 0.5}
{\text{DF}(t) + 0.5} + 1 \right)
$$

- $N$：corpus 中的總 document 數量。
- $\text{DF}(t)$：包含 term $t$ 的 document 數量。
- $+0.5$：平滑項，避免極端 DF 導致分數不穩。
- 外層 $+1$：避免 IDF 變成負值或過度懲罰高頻 term。

## IDF Smoothing 的意義

BM25 的 IDF 不是單純使用 $\log(N / \text{DF})$，而是使用一種平滑後的 odds-like 形式。它的目標不只是避免分母為 0，而是讓極端常見或極端稀有的 term 在 ranking 中更穩定。

### 和 TF-IDF smoothing 的差異

TF-IDF 常見的 smoothing 寫法是：

$$
\text{IDF}(t) = \log \left( \frac{N}{\text{DF}(t) + 1} \right)
$$

這裡的 $+1$ 主要是工程上的保護：避免 $\text{DF}(t)=0$ 時分母為 0。它解決的是「能不能算」的問題。

BM25 的 smoothing 則更偏向 ranking 穩定性：

$$
\text{IDF}(t) =
\log \left(
\frac{N - \text{DF}(t) + 0.5}
{\text{DF}(t) + 0.5} + 1 \right)
$$

這裡其實是在比較：

- $N - \text{DF}(t)$：不包含該 term 的 document 數量。
- $\text{DF}(t)$：包含該 term 的 document 數量。

也就是說，BM25 的 IDF 更像是在問：這個 term 比較像是「少數 document 才有的鑑別訊號」，還是「大多數 document 都有的背景噪音」？

### 為什麼加 $0.5$？

$+0.5$ 是對 document frequency 的平滑，避免在極端情況下讓 IDF 過度爆炸或過度不穩。

- 當 $\text{DF}(t)$ 很小時，term 很稀有，IDF 會高，但 $+0.5$ 會避免分數過度爆衝。
- 當 $\text{DF}(t)$ 很大時，term 很常見，IDF 會低，但 $+0.5$ 會避免分數在邊界處劇烈震盪。

直覺上，$+0.5$ 等於在統計上保留一點不確定性：不要因為 corpus 裡目前觀察到的 DF 太極端，就讓 ranking 分數做出過度自信的判斷。

### 為什麼外層加 $1$？

原始 odds-like IDF：

$$
\log \left(
\frac{N - \text{DF}(t) + 0.5}
{\text{DF}(t) + 0.5}
\right)
$$

在 term 出現在超過一半 document 時，分子會小於分母，IDF 可能變成負數。負 IDF 的意思是：這個 term 太常見，甚至應該扣分。

但在很多搜尋系統中，負分會讓 ranking 行為比較難解釋，也可能讓常見但仍有一點語意價值的 term 被過度懲罰。因此常見實作會在比例外加 $1$，讓 IDF 保持非負或至少更穩定。

簡單說：

| 平滑項 | 主要作用 |
| :--- | :--- |
| TF-IDF 的 $\text{DF}+1$ | 避免除以 0，讓公式可計算 |
| BM25 的 $+0.5$ | 平滑極端 DF，讓 ranking 更穩 |
| BM25 外層 $+1$ | 避免高頻 term 產生負 IDF 或過度扣分 |

## 從 TF-IDF 到 BM25

TF-IDF 的基本形式可以理解為：

$$
\text{TF-IDF}(t, D) = \text{TF}(t, D) \times \text{IDF}(t)
$$

BM25 可以視為把其中的 $\text{TF}(t, D)$ 換成一個更成熟的 scoring function：

$$
\text{BM25-TF}(t, D) =
\frac{\text{freq}(t, D)( {\color{orange}{k_1}} + 1)}
{\text{freq}(t, D) + {\color{orange}{k_1}} \left(1 - {\color{cyan}{b}} + {\color{cyan}{b}} \cdot \frac{
    {\color{cyan}{|D|}}}{{\color{cyan}{\text{avgdl}}}} \right)}
$$

也就是說：

$$
\text{BM25}(D, q)
= \sum_{t \in q} \text{IDF}(t) \times \text{BM25-TF}(t, D)
$$

## BM25 解決了什麼？

### 1. TF Saturation：詞頻飽和

傳統 TF-IDF 常把 term frequency 當成近似線性訊號：term 在 document 中出現越多次，分數就越高。問題是 term 的資訊貢獻不應該無限線性增加。

BM25 的 TF 部分會讓分數隨著 $\text{freq}(t, D)$ 增加而上升，但上升速度會逐漸變慢：

$$
\frac{\text{freq}(t, D)({\color{orange}{k_1}} + 1)}
{\text{freq}(t, D) + \color{orange}{k_1}}
$$

這個簡化式先忽略 document length normalization，只看 saturation。當 $\text{freq}(t, D)$ 從 1 增加到 2 時，分數會明顯上升；但當 $\text{freq}(t, D)$ 從 50 增加到 100 時，新增貢獻會非常有限。

這對 ranking 很重要：一份 document 不應該只因為刻意重複 query term 就無限加分。

### 2. Length Normalization：文件長度正規化

長 document 因為內容多，term 自然更容易出現。如果只看絕對 TF，長 document 會天然取得更高分，即使它並不比短 document 更聚焦。

BM25 用以下項目調整 document 長度：

$$
1 - \color{cyan}{b} + \color{cyan}{b} \cdot \frac{\color{cyan}{|D|}}{\color{cyan}{\text{avgdl}}}
$$

- 當 $|D| > \text{avgdl}$：document 比平均長，分母變大，term frequency 的加分被壓低。
- 當 $|D| < \text{avgdl}$：document 比平均短，分母變小，term 命中會被相對放大。
- 當 $b = 0$：完全不做長度正規化。
- 當 $b = 1$：完整根據 document 長度做正規化。

這讓 BM25 能處理「長 document 因為字數多而自然命中更多 query term」的偏差。

## 參數直覺

| 參數 | 控制對象 | 值變大時的效果 | 常見直覺 |
| :--- | :--- | :--- | :--- |
| $\color{orange}{k_1}$ | TF saturation | 詞頻可以帶來更多加分，飽和較慢 | 越大越接近傳統 TF |
| $\color{cyan}{b}$ | document length normalization | 長度懲罰更強 | 越大越重視 document 長短差異 |

### $k_1$

$k_1$ 控制 term frequency 的邊際效益遞減速度。

- $k_1$ 較小：term 重複幾次後很快飽和，比較不容易被 keyword stuffing 影響。
- $k_1$ 較大：term frequency 的影響變強，重複出現能帶來更多加分。

### $b$

$b$ 控制文件長度正規化的強度。

- $b = 0$：不管 document 長度，長 document 和短 document 只按 term frequency 比較。
- $b = 1$：完整依照 $|D| / \text{avgdl}$ 調整，長 document 會受到明顯懲罰。
- $b \approx 0.75$：常見預設值，在長度校正與保留 TF 訊號之間取得折衷。

## 例子：為什麼 BM25 比 TF-IDF 更穩？

假設 query 是「量子電腦」，有兩份 document：

| Document | 長度 | term 出現次數 | TF-IDF 直覺 | BM25 直覺 |
| :--- | :--- | :--- | :--- | :--- |
| A | 很長 | 20 次 | 分數很高 | 會被長度正規化壓低，且 TF 貢獻逐漸飽和 |
| B | 很短 | 5 次 | 可能輸給 A | 因為短且集中，可能被判斷為更聚焦 |

TF-IDF 容易把「term 出現很多次」直接當作強相關；BM25 則會問兩個更細的問題：

- 這個 term 的出現次數是否已經接近飽和？
- 這份 document 是否只是因為篇幅很長才命中較多次？

## WrapUP

BM25 是把 TF-IDF 的 ranking 直覺做得更像人類判斷：
- IDF 保留「全域稀有 term 更重要」的概念。
- TF saturation 避免 keyword stuffing 讓分數無限膨脹。
- Length normalization 避免長 document 單純因篇幅長而佔優。

因此在傳統 lexical search、搜尋引擎第一階段召回、RAG keyword retrieval 中，BM25 仍然是非常重要且實用的 baseline。
