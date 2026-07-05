# MMoE 與多目標推薦系統 (Multi-Task Learning)

> [English](./README.md) | **繁體中文**

> 針對影片推薦系統中，如何同時優化多個目標 (如 CTR、完播率、按讚、分享)，以及 MMoE (Multi-gate Mixture-of-Experts) 的核心架構解析。

---

## 1. 為什麼影片推薦需要多目標學習 (Multi-Task Learning, MTL)？

在影片推薦場景中，使用單一目標 (例如只預測點擊率 CTR) 容易導致「標題黨」問題，損害使用者長期體驗。為了全面評估推薦品質，系統通常需同時預測多個目標：
- **CTR (Click-Through Rate)**：使用者是否點擊影片。
- **CVR (Conversion Rate) / Watch Time**：點擊後是否完整觀看，或是觀看時長長短。
- **Engagement (互動)**：是否按讚、收藏、分享、留言。

這些任務之間可能存在**衝突** (例如：騙點擊的影片觀看時長很短；偏門冷知識影片觀看時長長但點閱率低)，這就是多目標學習中常見的**蹺蹺板現象 (Seesaw Phenomenon)**：優化了 A 任務，卻導致 B 任務效能下降。

---

## 2. 傳統 Shared-Bottom 架構的問題

在 MMoE 出現之前，業界常使用 **Shared-Bottom** 架構：
- 底層使用相同的幾層神經網路 (Shared Bottom) 提取共用特徵。
- 頂層為每個任務分出獨立的 Task Tower 進行預測。
- **缺點**：如果多個任務之間的相關性較低（甚至是互斥的），Shared-Bottom 會因為不同任務傳遞下來的梯度方向衝突 (Gradient Conflict)，導致底層特徵抽取器無法學到好的特徵，進而引發蹺蹺板現象。

---

## 3. MMoE (Multi-gate Mixture-of-Experts) 核心突破
![alt text](imgs/image.png)

> [!IMPORTANT] 【MMoE 的精髓】
> MMoE 透過引入「專家網路 (Experts)」與「門控網路 (Gates)」，取代了傳統的 Shared-Bottom，讓不同任務能「動態且選擇性地」組合底層特徵。

要深刻理解 MMoE，必須釐清以下三個核心組件的真正定位：

### 1. Task Tower (任務塔) —— 真正有 Loss function 的地方
- **定位**：**這才是整個模型真正產生 Loss (損失函數) 並發起回傳梯度的源頭。**
- 每個任務都有自己獨立的 Task Tower（例如一個算 CTR Loss，一個算 Watch Time Loss）。
- 整個 MMoE 架構的學習，是由各個 Task Tower 的 Loss 往下 Backpropagation (反向傳播) 來驅動的。

### 2. Experts (專家網路) —— 只是「特徵提供器」
- **定位**：Expert 本身**沒有**專屬的 Loss function！它們絕對不是在做最終決策。
- 它們的作用僅僅是將原始輸入 (Input) 轉換成多個高階的 Dense 特徵空間。
- 為什麼叫 Expert？因為在多個 Task Tower 的梯度共同拉扯 (優化) 下，不同的 Expert 會自動分化，隱式地學會捕捉不同面向的模式（例如 Expert A 可能對視覺特徵敏感，Expert B 對時間序列特徵敏感）。

### 3. Gate (門控網路) —— 「特徵權重分配器」
- **定位**：為「每個任務」配置一個專屬的 Gate。
- Gate 的輸入同樣是原始資料向量，輸出則是一個 Softmax 權重向量。
- **作用**：針對當前輸入的這筆資料 (User-Video Pair)，Gate 決定該任務要從哪幾個 Expert 那裡「拿多少比例的特徵」。
- 例如：預測「按讚」的 Gate 發現這筆資料需要注重 Expert C 的表徵，就會給 Expert C 較高的權重，將各 Expert 的輸出做加權總和後，送入「按讚」的 Task Tower 進行預測。

---

## 4. MMoE 如何解決蹺蹺板現象？

> [!TIP] 【追問與亮點】
> **Q: MMoE 為什麼不怕任務衝突？**
> **A**: 在 Shared-Bottom 中，所有任務的梯度都強迫在同一個網路中妥協；但在 MMoE 中，如果 CTR 任務和 Watch Time 任務所需特徵衝突，這兩個任務的 Gate 會自動把權重分配給**不同的 Experts**。
> 
> 也就是說，MMoE 實現了**「特徵層面的軟性隔離 (Soft Routing)」**。高度相關的任務可以共享相同的 Experts；互斥的任務則透過 Gate 各自挑選不同的 Experts，互不干擾。這樣既能利用 MTL 帶來的大數據泛化優勢，又能避免梯度衝突導致的效能下降。

---

## 5. 多任務結果融合 (Score Fusion) —— 如何把不同 Task Tower 的結果變成最終排序分數？

雖然 MMoE 幫我們針對各個任務（例如 CTR, CVR, 完播率, 互動率）都訓練出優秀的專屬神經網路結構 (Task Tower)，但最終推薦給使用者時，系統只能根據「一個排序列表」來展示。因此進入最後排序 (Ranking) 階段，需要把不同 Task Tower 輸出的預測值融合為單一排序分數 (Final Ranking Score)。

業界常見的融合方式有以下幾種：

### 1. 乘法融合 (Multiplicative / Log-Linear Fusion)：業界最主流
利用指數加權相乘，不同任務的重要性透過指數超參數來進行調控：
$$ \text{Final Score} = (\text{pCTR})^{\alpha} \times (\text{pCVR})^{\beta} \times (\text{pWatchTime})^{\gamma} \times (\text{pLike})^{\delta} $$

- **優點**：
  - 具有「一票否決權」的特性（只要其中一個關鍵指標接近 0，總分就會很低）。
  - 可以快速且直接地調控線上業務目標（例如最近需要衝按讚數，就直接調大 $\delta$）。

### 2. 加法融合 (Linear Combination)
給予不同預測目標對應的權重直接相加：
$$ \text{Final Score} = w_1 \times \text{pCTR} + w_2 \times \text{pCVR} + w_3 \times \text{pWatchTime} + \dots $$

- **缺點與注意事項**：
   各個任務預測機率的絕對值大小差異通常極大（例如點閱率可能 5%，但點讚率只有 0.1%），直接相加會導致大機率的目標強勢掩蓋小機率目標。因此若要用加法，通常需要先對各自分數進行正規化 (Normalization / Calibration) 後再相加。

### 3. 以期望價值直接相乘 (Expected Value)
在電商或某些嚴格定義收益的場景中，直接利用概率計算期望價值：
- 電商場景：$\text{eCPM} = \text{pCTR} \times \text{pCVR} \times \text{Price} \times 1000$
- 影片場景：將預測觀看時長作為價值乘積，例如 $\text{pCTR} \times \text{E(WatchTime)}$

### 4. 模型自動融合 (L2 Ranker)
與其依靠人工制定規則，乾脆把這些 Task Tower 的輸出（機率或 logits）當作「特徵」，再餵給一個輕量級的模型（例如小型神經網路或 Tree-based 模型 XGBoost 等）進行二次排序。

> [!TIP] 【進階追問】
> **Q: 在多目標 Score Fusion 的乘法公式中，權重 $\alpha, \beta, \gamma$ 是怎麼決定的？**
> **A**: 
> 1. **初期階段**：主要靠業務經驗（看重什麼就調高什麼），或是透過 Grid Search 掃描參數。
> 2. **上線階段**：大量依賴線上 **A/B Testing** 來觀察不同權重組合對北極星指標的影響。
> 3. **進階階段**：引入自動化參數搜尋機制，例如利用貝氏最佳化 (Bayesian Optimization) 或進化演算法自動不斷尋找最佳的融合權重，減少人工調參的成本。
