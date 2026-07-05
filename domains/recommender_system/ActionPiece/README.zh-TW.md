# ActionPiece — Research Note

> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation |
| Venue | ICML |
| Year | 2025 |
| Authors | Yupeng Hou, Jianmo Ni, Zhankui He, Noveen Sachdeva, Wang-Cheng Kang, Ed H. Chi, Julian McAuley, Derek Zhiyuan Cheng |
| Official Code | https://github.com/google-deepmind/action_piece |
| Venue Kind | paper |

作者來自 University of California, San Diego 與 Google DeepMind（第一作者 Yupeng Hou 於 Google DeepMind 擔任 student researcher 期間完成本工作）。論文 LaTeX 原始碼以 `\usepackage[accepted]{icml2025}` 標記為 ICML 2025 已接受論文。以下所有數值與引文均以 arXiv `2502.13581` 的 LaTeX 原始碼為準（camera-ready 版本可能略有差異）。

## First Principles

### 問題:為什麼「同一個 action」不該永遠對應同一組 token

生成式推薦(generative recommendation, GR)把使用者的互動序列切成離散 token,再讓一個自回歸模型(如 T5 encoder-decoder)逐一生成 token,最後把生成的 token 解析回推薦物品。它的好處是 token 共用一個「不隨物品池大小成長」的緊湊詞彙表,因此在記憶體與擴展性上優於傳統為每個物品維護一列 embedding 的作法。

論文指出既有 GR 方法的共同缺陷:**每個 action 都被獨立地 tokenize**,同一個 action(例如買了同一件商品)在任何序列裡都被指派到相同的固定 token,完全不看上下文。作者的類比很清楚:語言模型早期的 word-level tokenization 也是上下文無關的,而現代 LLM 幾乎都改用 BPE、Unigram 這類上下文感知(context-aware)的子詞切分,讓同一個詞根依鄰近上下文切成不同 token。ActionPiece 想做的,就是把這一步搬到 action 序列上,成為第一個上下文感知的 action 序列 tokenizer。

真正的技術難點在於:文字天生是一維字元序列,但一個 action 關聯的特徵(類別、品牌、價格……)是一個**無序集合(unordered set)**。所以整個演算法必須跑在「集合的序列(sequence of sets)」上,而不是一維序列上。

### 把 action 表示成特徵集合的序列

給定使用者歷史 $S=\{i_1,\dots,i_t\}$,把每個物品 $i_j$ 換成它的特徵集合 $\mathcal{A}_j$(每個物品 $m$ 個特徵,第 $k$ 個特徵記為 $f_{j,k}\in\mathcal{F}_k$),整段輸入就變成一個依時間排序的集合序列 $S'=\{\mathcal{A}_1,\dots,\mathcal{A}_t\}$。集合**內部**無序,但集合**之間**有時間序。Tokenizer 要輸出 token 序列 $C=\{c_1,\dots,c_l\}$,其中 token 數 $l$ 通常大於 action 數 $t$。用無序集合(而非 RQ-VAE 產生的有序 semantic ID)有兩個好處:不必為特徵指定順序、且能自然容納 category / brand / price 這類一般離散與數值特徵。

### 詞彙建構:對「集合序列」做加權版 BPE

ActionPiece 沿用 BPE 的 bottom-up 精神:初始詞彙 $\mathcal{V}_0$ 讓每個 token 代表「一個單一特徵的集合」,然後反覆執行 count → update,每輪把最常共現的一對 token 合併成新 token,直到詞彙量達到目標 $Q$。

關鍵差別在 **count 這一步要考慮集合結構**。集合序列裡有兩種共現:(1) 兩個 token 在同一個集合內、(2) 兩個 token 在相鄰的兩個集合裡。作者不像文字 BPE 那樣把每對 token 等權計數,而是用一個「隨機把集合攤平成一維序列後,兩 token 相鄰的期望機率」來定義權重。對同一集合內的兩個 token:

$$P(c_1, c_2) = \frac{|\mathcal{A}_i| - 1}{\tbinom{|\mathcal{A}_i|}{2}} = \frac{2}{|\mathcal{A}_i|}, \quad c_1, c_2 \in \mathcal{A}_i$$

對相鄰兩集合的 token:

$$P(c_1, c_3) = \frac{1}{|\mathcal{A}_i| \times |\mathcal{A}_{i+1}|}, \quad c_1 \in \mathcal{A}_i,\; c_3 \in \mathcal{A}_{i+1}$$

![ActionPiece 詞彙建構時的加權共現計數示意(左集合 4 個 token、右集合 3 個 token;集合內配對 ⟨○,○⟩、⟨□,□⟩ 與跨集合配對 ⟨○,□⟩ 使用不同權重)](imgs/weight.png)

**update 這一步的資料結構是全篇最工程的部分**。合併同一集合內的 token 很直接;但合併「跨相鄰集合」的 token 時,新 token 該放進哪個集合並不明確。作者用雙向鏈結串列維護每條序列,並引入「中間節點(intermediate node)」專門存放橫跨多個 action 的 token:兩個 action 節點的 token 合併時,在兩者之間插入一個中間節點承接新 token;action 節點與既有中間節點合併時,新 token 直接取代中間節點裡的舊 token。此規則保證任兩個 action 節點之間至多一個中間節點、且每個中間節點至多一個 token,計算共現權重時把中間節點當作大小為 1 的集合處理。

樸素地每輪重掃全語料的複雜度是 $O(QNLm^2)$;作者用倒排索引(把 token pair 映射到含它的序列)加上帶 lazy-update 的全域 heap,只增量更新受影響的部分,把複雜度降到 $O(\log Q \log H \cdot NLm^2)$,其中 $H=O(NLm)$ 為 heap 最大長度。

### Segmentation:用集合置換正則化(SPR)避免只用到少數 token

有了詞彙,還要把原始序列切(segment)成對應 token。若直接沿用建詞彙時的貪婪固定順序合併,作者觀察到會產生偏差——只有一部分 token 被頻繁使用。為此提出 **set permutation regularization (SPR)**:對每個集合隨機生成一個排列、當成一維序列,再把所有排列串接後用傳統 BPE segmentation 切分。不同排列會切出「語義相同但 token 序列不同」的多個版本,訓練時每個 epoch 重切一次當作資料增強,推論時對同一輸入切 $q$ 次、把 $q$ 份排序結果的分數平均做集成(inference-time ensembling)。

下表(論文 Table 3)把 BPE 與 ActionPiece 的差異並列,可看出 ActionPiece 本質上是「把 BPE 從一維 byte 序列搬到特徵集合序列」並補上跨集合合併與 SPR:

| Aspect | BPE | ActionPiece |
|-|-|-|
| Data Type | text sequences | action(無序特徵集合)序列 |
| Token | a byte sequence | a feature set |
| Merging Unit | 相鄰 byte pair | 集合內 或 相鄰集合間的特徵對 |
| Co-occurrence Weighting | raw frequency | 機率加權(上方兩式) |
| Segmentation | 貪婪固定順序合併 | set permutation regularization |
| Intermediate Structures | N/A | 跨 action 合併用的中間節點 |

### 一個帶真實數字的具體例子

以 Beauty 資料集為例,每個物品的特徵是用 OPQ 量化出的 4 個 code 再加 1 個防衝突的識別 code,共 $m=5$ 個特徵——正好對應論文 case study 中「每個物品由五個特徵構成」的設定。把這 5 個特徵當一個集合,建詞彙時:

- 集合**內部**任一對特徵每次共現貢獻的權重是 $P=\tfrac{2}{|\mathcal{A}|}=\tfrac{2}{5}=0.4$;
- 若前後兩個物品都是 5 個特徵,**跨集合**任一對特徵的權重是 $P=\tfrac{1}{5\times 5}=0.04$。

也就是說,同一物品內的特徵配對,權重是跨物品配對的 **10 倍**。這正是「加權計數」的設計意圖:先把「同一件商品內部反覆同現的特徵組合」壓成一個 token,再讓較稀有的跨商品組合去捕捉真正的上下文訊號。經過合併,一個 token 可能對應到:某物品的部分特徵、單一特徵、某物品的全部特徵、或橫跨多個物品的特徵——論文 case study 中 token `14844` 對應 T-shirt 的特徵 `747` 與 `923`,而 token `⟨19,895⟩` 則同時包含 socks 的特徵 `1100` 與 shorts 的特徵 `560`、`943`,示範了「同一 action 依鄰近上下文被切成不同 token」。

![ActionPiece tokenization 的 case study:同一段 action 序列在不同置換下切出語義相同但 token 不同的兩種結果,並標註四種 token 類型](imgs/case.png)

推薦效果上,以 Beauty 的 NDCG@10 為例:ActionPiece 為 0.0424,而該欄最強的 baseline(P5-CID,0.0400)的相對提升即 $(0.0424-0.0400)/0.0400 = 6.00\%$,與論文 Improv. 欄一致。整體而言 ActionPiece 在三個資料集的 NDCG@10 相對最佳 baseline 提升 6.00% 到 12.82%。模型端沿用 TIGER 式的 T5 encoder-decoder:4 層、6 個 head(每個 head 維度 64)、token embedding 維度 128、FFN 維度 1024,對 Sports/Beauty 約 4.46M 非嵌入參數;詞彙量固定 40k、推論集成 $q=5$、beam size 50。

## 🧪 Critical Assessment

### 問題是否真實,重要性有多高

「既有 action tokenizer 都上下文無關」這個觀察是準確且可驗證的:論文 Table 1 逐一點名 VQ-Rec、TIGER、HSTU、SPM-SID 等都在 Contextual 欄打叉,與 LLM 從 word-level 走向 subword 的歷史類比也相當貼切。問題本身成立。但要注意「重要性」與「效果幅度」是兩回事:三個資料集全部來自 Amazon Reviews(Sports/Beauty/CDs),物品數 12k–64k、序列平均長度 8–15,屬於學術基準的偏小規模;論文結論裡把方法外推到「audio modeling、sequential decision-making、time series」目前只是 future work 的宣稱,沒有任何實驗支撐,讀者不宜把它當成已證實的通用性。

### Baseline、消融、資料與指標是否充分

方法論的嚴謹度整體不錯:baseline 橫跨 ID-based、feature+ID、generative 三類共十個方法,跑五個隨機種子並回報標準差,消融同時檢驗了詞彙量、context-aware、加權計數、SPR 訓練/推論拆分。特別值得肯定的是為了預先反駁「提升只是詞彙變大」,作者刻意把 TIGER 的詞彙從 192 一路調到 66k,顯示更大詞彙的 TIGER(49k、66k)反而更差。

但有兩個公平性缺口值得指出。其一,除 ActionPiece 外的多數 baseline 由作者自行重現,唯獨 LMIndexer 直接引用原論文數字,且它在 CDs 全欄與若干 R@10 欄位是 `---`(未收斂/缺值),無法與其他方法對齊比較。其二,也是更關鍵的一點:**ActionPiece 的推論用了 $q=5$ 份切分做集成,等於 5 倍前向計算,而 baseline 都是單次推論**。論文用「置換在 CPU 上非同步、增廣版本可跨裝置平行」來論證延遲相當,但這掩蓋不了 FLOPs 是 5 倍的事實;把「單次推論的 baseline」與「5 次集成的本方法」直接並列,headline 的提升裡有多少來自更好的 tokenization、多少來自純粹的測試期集成,並沒有被完全拆乾淨。消融 (3.1)(只在推論用 SPR)掉到 Sports 0.0192、(3.3)(TIGER+SPR)也沒起色,確實佐證了「SPR 要搭配上下文感知詞彙才有用」,但這仍不等於證明在**相同推論預算**下 ActionPiece 勝過 baseline。

### 從 BPE 到集合序列:是真創新還是包裝

老實說,ActionPiece 的核心就是「把 BPE 搬到特徵集合序列上」,論文 Table 3 自己也是這樣並列的。真正原創的成分有三塊:跨集合合併所需的中間節點資料結構、依集合大小的機率加權計數、以及 SPR。其中 SPR 看來是提升的主要來源——消融顯示拿掉 SPR(退回樸素 segmentation)或只在單邊使用都會明顯掉分,而 SPR 帶來的 token 利用率從 56.89% 升到第一個 epoch 的 87.01%(附錄進一步稱到第 5 個 epoch 達 95.33%)。這是紮實的工程貢獻,但也意味著「上下文感知 tokenization」這個標題賣點,和「靠置換增強+集成把 token 用好」這個真正的效能引擎,在論文敘事裡是被綁在一起講的;若把 SPR 看成一種資料增強/集成技巧,它與「上下文感知」的因果歸屬其實部分交纏。

### 自訂基準與是否真正解決問題

作者主動引入了規模較大的 CDs(互動量約為 Sports 的 4 倍)來測擴展性,Sports/Beauty 則沿用社群標準基準,這降低了「挑對自己有利的資料集」的疑慮,是誠實的一面。但有一個容易被忽略的設定差異:ActionPiece 用 OPQ 量化出 4 code(+1 識別 code),TIGER/SPM-SID 用 RQ 的 3 code(+1 識別 code),不同方法的底層特徵表示並不完全一致,這種量化方式的差異可能對結果構成潛在干擾,論文雖聲明「為公平比較統一成 4 code」,但 RQ 與 OPQ 的本質差異仍在。此外「Improv.」欄所對比的「最強 baseline」並非逐格一致——例如 Sports R@5 欄,P5-CID(0.0287)其實高於被標為次佳(underline)的 SPM-SID(0.0280),而該欄 +12.86% 的提升是相對 0.0280 算出的;這不影響 ActionPiece 為當欄最佳的結論,但顯示「相對最強 baseline」的敘述在個別格子上並不精確,讀者引用提升幅度時應回表核對。整體而言,在其設定的離線 Amazon 基準上「上下文無關 tokenization 是次優的」這個假說得到了支持;但「在真實、大規模、線上場景是否成立」則完全沒有被觸及——沒有工業級資料、沒有線上 A/B、5 倍推論成本對線上部署的影響也未量化。

![NDCG@10(↑)與正規化序列長度 NSL(↓)隨詞彙量的變化,顯示效能/序列長度/記憶體的取捨](imgs/vocab_size.png)

## 🔗 Related notes

- [SASRec](../SASRec/) — 本文 ID-based baseline 之一,自注意力序列推薦的代表。
- [S3Rec](../S3Rec/) — 本文 feature-enhanced baseline,利用自監督預訓練關聯物品特徵與 ID。
- [BERT4Rec](../Bert4Rec/) — 本文 ID-based baseline,雙向 Transformer 序列推薦。
