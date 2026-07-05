# Scaling Laws for Neural Language Models — Research Note
> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Scaling Laws for Neural Language Models |
| Venue | arXiv preprint (arXiv:2001.08361) |
| Year | 2020 |
| Authors | Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei |
| Official Code | unknown |
| Venue Kind | paper |

這是一篇 OpenAI 與 Johns Hopkins University 合作、僅以 arXiv 預印本形式發表的實證研究，並未走過任何正式的同行評審流程，因此下方的場館等級（venue tier）在證據上只能標記為 unknown。本文基於 arXiv 全文（arXiv:2001.08361v1）撰寫；由於沒有後續的正式會議或期刊定稿版本，此預印本即為權威文本。

## First Principles

### 研究問題：把「規模」拆成三個可量測的軸

本文要回答的核心問題是：一個自回歸 Transformer 語言模型在交叉熵損失（cross-entropy loss）上的表現，究竟由哪些因素決定、以及如何隨這些因素改變。作者把「規模」拆解成三個彼此獨立、都可精確量測的軸：非嵌入參數量（number of non-embedding parameters）$N$、資料集的 token 數 $D$、以及訓練所用的計算量（compute）$C$。所有實驗都在 WebText2 上以 1024 個 token 的上下文最佳化自回歸 log-likelihood，並以此損失（單位為 nats）作為唯一的主要量測指標。

作者刻意把詞嵌入與位置嵌入的參數排除在 $N$ 之外，因為他們發現只有在扣除嵌入參數後，不同深度的模型才會收斂到同一條趨勢線；若把嵌入計入總參數量，趨勢會被層數污染而變得模糊。這個「用非嵌入參數量描述規模」的選擇，是後面所有乾淨冪律（power law）能成立的前提，而不是事後的美化。

### 三條基本冪律

當表現只被三軸中的其中一軸瓶頸住時，測試損失對該軸都呈現冪律關係。這是全文的骨幹結論，對應下面三個式子（數值為 WebText2 上的擬合值）：

$$ L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13} $$

$$ L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13} $$

$$ L(C_{\min}) = \left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050, \quad C_c^{\min} \approx 3.1 \times 10^{8} $$

這些關係橫跨了 $C_{\min}$ 的八個數量級、$N$ 的六個數量級、以及 $D$ 的兩個以上數量級，而且對模型形狀（深度、寬度、注意力頭數）幾乎不敏感。指數的絕對值很小這件事本身很有意義：$\alpha_N \approx 0.076$ 代表把參數量翻倍，損失只會乘上 $2^{-0.076} \approx 0.95$，也就是每翻一倍只換到約 5% 的損失下降——規模帶來的是穩定但邊際遞減的回報。作者特別強調 $N_c, D_c, C_c$ 的絕對數值依賴於詞彙表大小與 tokenization，因此沒有基礎性的物理意義，真正可攜的是指數。

### 從架構數清參數與計算量

要讓上面的 $N$ 與 $C$ 成為可計算的量，作者給出了 Transformer 的參數與計算量估算。在標準設定 $d_{\rm attn} = d_{\rm ff}/4 = d_{\rm model}$ 下，非嵌入參數量可近似為一個乾淨的閉式：

$$ N \approx 12\, n_{\rm layer}\, d_{\rm model}^2 $$

而每個訓練 token 的非嵌入計算量被估計為前向約 $2N$、加上反向的兩倍，合計約 $C \approx 6N$ 浮點運算。下面這張表把每個運算單元的參數與前向 FLOPs 拆開，是上述兩個近似式的來源：

| Operation | Parameters | FLOPs per Token |
|-|-|-|
| Embed | $(n_{\rm vocab}+n_{\rm ctx})d_{\rm model}$ | $4 d_{\rm model}$ |
| Attention: QKV | $n_{\rm layer} d_{\rm model} 3 d_{\rm attn}$ | $2 n_{\rm layer} d_{\rm model} 3 d_{\rm attn}$ |
| Feedforward | $n_{\rm layer} 2 d_{\rm model} d_{\rm ff}$ | $2 n_{\rm layer} 2 d_{\rm model} d_{\rm ff}$ |
| Total (Non-Embedding) | $N = 2 d_{\rm model} n_{\rm layer}(2 d_{\rm attn}+d_{\rm ff})$ | $C_{\rm forward} = 2N + 2 n_{\rm layer} n_{\rm ctx} d_{\rm attn}$ |

計算量之所以能近似成純粹由 $N$ 決定（而忽略與上下文長度相關的項），是因為作者主要研究 $d_{\rm model} \gg n_{\rm ctx}/12$ 的模型，此時與上下文相關的注意力計算只佔總計算量的一小部分。

### 一條式子同時描述模型與資料規模

真正把兩個軸接起來的，是同時刻畫 $N$ 與 $D$ 依賴、並且直接預測過擬合程度的聯合式：

$$ L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D} $$

這個形式不是隨便湊的，而是由三個原則約束出來：改變詞彙表只應對損失做整體縮放、固定一軸把另一軸推到無窮大時必須回到單軸律 $L(N)$ 或 $L(D)$、以及損失在 $D=\infty$ 附近應可用 $1/D$ 的整數冪級數展開。第三個原則（$1/D$ 展開）是三者中最具推測性的，作者自己也承認若沒有實證確認並不會太有把握，但它解釋了為何 $N$ 與 $D$ 在式中角色不對稱。對這個四參數式子的擬合結果如下表：

| Parameter | $\alpha_N$ | $\alpha_D$ | $N_c$ | $D_c$ |
|-|-|-|-|-|
| Value | $0.076$ | $0.103$ | $6.4 \times 10^{13}$ | $1.8 \times 10^{13}$ |

由 $\alpha_N/\alpha_D$ 可推得一條實用的擴充守則：為了在把模型放大時不落入過擬合，資料量只需以次線性的速度成長，$D \gtrsim (5 \times 10^3)\, N^{0.74}$。這也是「每把模型放大 8 倍、只需把資料放大約 5 倍」這句話的來源（因為 $8^{0.74} \approx 4.7$）。

### 訓練動態與計算最優分配

把訓練步數也納入後，損失可用模型規模與（批次調整後的）步數 $S_{\min}$ 來描述：

$$ L(N, S_{\min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}, \quad \alpha_S \approx 0.76,\ S_c \approx 2.1 \times 10^{3} $$

再結合臨界批次大小（critical batch size）$B_{\rm crit}(L) = B_*/L^{1/\alpha_B}$（$\alpha_B \approx 0.21$），作者在固定計算預算下對 $N$ 求損失極小，得到最優分配應如何隨計算量成長。最關鍵、也最反直覺的結論是：計算最優訓練應該幾乎把所有新增算力都投到「更大的模型」上，而序列訓練步數幾乎不增加。這由下表的指數具體量化——$N_{\rm opt}$ 隨 $C_{\min}^{0.73}$ 快速成長，而步數 $S_{\min}$ 只隨 $C_{\min}^{0.03}$ 成長，慢到甚至可能與零指數相容：

| Compute-Efficient Value | Power Law | Scale |
|-|-|-|
| $N_{\rm opt} = N_e \cdot C_{\min}^{p_N}$ | $p_N = 0.73$ | $N_e = 1.3 \times 10^{9}$ params |
| $B \ll B_{\rm crit} = B_e C_{\min}^{p_B}$ | $p_B = 0.24$ | $B_e = 2.0 \times 10^{6}$ tokens |
| $S_{\min} = S_e \cdot C_{\min}^{p_S}$ | $p_S = 0.03$ | $S_e = 5.4 \times 10^{3}$ steps |
| $D_{\rm opt} = D_e \cdot C_{\min}^{p_D}$ | $p_D = 0.27$ | $D_e = 2 \times 10^{10}$ tokens |

從 $L(N, S_{\min})$ 的解析求極小還可推出一個乾淨的訓練停止準則：計算最優的訓練應該在「比收斂損失高約 $\alpha_N/\alpha_S \approx 10\%$」的地方就停手，而不是訓練到收斂。這正式化了摘要裡「訓練非常大的模型、並在遠未收斂前就停止」這個實務指引。

### 一個具體的數值範例

以本文引用的 GPT-2 尺寸模型 $(n_{\rm layer}, d_{\rm model}) = (48, 1600)$ 為例走一遍。先用參數式估算非嵌入參數量：

$$ N \approx 12 \times 48 \times 1600^2 = 1.47 \times 10^{9} \ \text{(non-embedding params)} $$

代入單軸律 $L(N)$ 預測其在無限資料下的收斂損失：

$$ L(N) = \left(\frac{8.8 \times 10^{13}}{1.47 \times 10^{9}}\right)^{0.076} = (5.97 \times 10^{4})^{0.076} \approx 2.3 \ \text{nats/token} $$

接著用過擬合守則檢查資料是否足夠：$D \gtrsim 5 \times 10^3 \times (1.47 \times 10^9)^{0.74} \approx 3.0 \times 10^{10}$ tokens，也就是約 300 億 token。這正好解釋了為什麼 WebText2 的 22B token 對 $10^9$ 參數以下的模型幾乎不會過擬合，但對本文最大的模型會開始出現輕微過擬合——因為所需資料量剛好爬過了資料集規模。（以上代入與四則運算為本文作者的推導，非原論文逐字給出。）

下圖是本文的招牌圖：測試損失對計算量、資料量、參數量三者都呈現橫跨多個數量級的直線（對數-對數座標下），是「平滑冪律」這個核心主張最直接的視覺證據。

![測試損失對 compute、dataset size、parameters 皆為冪律的招牌圖](imgs/simple-power-laws.png)

最後，全文的擬合值可彙整成下表，作為所有預測的參數來源（數值皆依賴 tokenization）：

| Power Law | Scale |
|-|-|
| $\alpha_N = 0.076$ | $N_c = 8.8 \times 10^{13}$ params (non-embed) |
| $\alpha_D = 0.095$ | $D_c = 5.4 \times 10^{13}$ tokens |
| $\alpha_C = 0.057$ | $C_c = 1.6 \times 10^{7}$ PF-days |
| $\alpha_C^{\min} = 0.050$ | $C_c^{\min} = 3.1 \times 10^{8}$ PF-days |
| $\alpha_B = 0.21$ | $B_* = 2.1 \times 10^{8}$ tokens |
| $\alpha_S = 0.76$ | $S_c = 2.1 \times 10^{3}$ steps |

## 🧪 Critical Assessment

### 把算力分配化約成可外推冪律的實用價值

這篇論文問的問題本身是紮實而非造出來的：在 2020 年，「把模型與資料放大到什麼程度、算力該如何分配」是一個真金白銀的工程決策，而當時業界多半靠直覺與硬體限制在做。把這個決策化約成幾條可外推的冪律，具有清楚的實用價值，後續 GPT-3 的尺寸選擇與 Chinchilla 的重新檢視都直接建立在這個框架上，足見問題的真實性。這一點上我認為沒有灌水成分。

### 單一 WebText2 語料與純交叉熵指標的兩個缺口

作為一篇實證論文，它的內部消融相當充分：形狀無關性是靠固定 $N$、單獨掃描深度/寬度/頭數得到的；嵌入參數的取捨也有左右對照圖直接支持。但有幾個結構性侷限值得點名。第一，全部結論都建立在單一資料分布 WebText2 上，作者雖測了其他分布的遷移，卻沒有在不同語料上重新擬合指數，因此「指數與 tokenization 無關的可攜量」這個說法其實只是弱驗證，而非跨資料集的證明。第二，量測指標只有一種——上下文平均的交叉熵損失，完全沒有下游任務表現。損失平滑下降不代表任何具體能力平滑提升，作者自己在 Discussion 也承認「more is different」，這等於承認了核心量測與人們真正在意的能力之間存在未被橋接的鴻溝。

### 與 Hestness 相反的次線性資料守則，及 $1/D$ 展開的推測性

冪律縮放本身在 2020 年之前並非全新——Hestness、Rosenfeld 等人都已報告過模型/資料規模與表現的冪律。本文誠實地在 Related Work 引用了這些工作，並指出關鍵差異：前人（如 Hestness）發現資料量需超線性成長於模型規模，而本文得到的是次線性成長，兩者結論方向相反。這個反向結論加上「把計算量納為第三軸並推導最優分配」的閉環，構成了實質的新增貢獻，我不認為這只是換名詞的重新包裝。不過需要保留的是，$L(N,D)$ 那個聯合式的第三原則（$1/D$ 整數冪展開）帶有相當的推測性，作者也自承理論支持較弱，因此該式的「必然性」被包裝得比證據所能支撐的更強一些。

### 冪律必然失效的外推交點與「語言熵」詮釋的風險

最需要警惕的是，這套框架的「成功」幾乎是用它自己定義的量測（WebText2 交叉熵）來宣稱的——趨勢線漂亮，但那條線量的正是被最佳化的目標本身。真正的外推風險被作者自己揭露在「矛盾與猜想」一節：$L(C_{\min})$ 與 $L(D)$ 兩條外推線會在 $C^* \sim 10^4$ PF-days、$N^* \sim 10^{12}$ 附近相交而產生矛盾，意味著冪律必然在某處失效。作者把這個交點浪漫化為「自然語言熵的估計」，但也坦承交點位置對指數極度敏感，可上下浮動一個數量級。這是一個誠實但危險的猜想：把一個外推失效點重新詮釋成物理常數，本質上仍是未經證實的推斷。整體而言，論文的預測框架在其量測的內部區間內證據充分，但一旦跨出量測範圍或換到下游能力，其宣稱就從「已驗證」滑向「合理但未證實」，讀者不應把冪律當成無條件的定律來背。

## 🔗 Related notes

- [Attention Is All You Need](../AttentionIsAllYouNeed/)
- [BERT](../BERTSummary/)
