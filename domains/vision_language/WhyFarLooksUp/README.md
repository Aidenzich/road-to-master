# WhyFarLooksUp — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Why Far Looks Up: Probing Spatial Representation in Vision-Language Models |
| Venue | ECCV 2026 |
| Year | 2026 |
| Authors | Cheolhong Min, Jaeyun Jung, Daeun Lee, Hyeonseong Jeon, Yu Su, Jonathan Tremblay, Chan Hee Song, Jaesik Park |
| Official Code | https://github.com/cheolhong0916/contrastive-probing |
| Venue Kind | paper |

> 本文以 arXiv 預印本 `2605.30161v1`（2026-05-28）為全文依據撰寫。專案頁與程式庫將投稿定位為 ECCV 2026；正式 camera-ready 版本若有更動，數值與敘述可能與此處略有差異。所有數字均取自預印本 LaTeX 原始檔。

## First Principles

### 一句話問題：高分是「真的懂 3D」還是「猜對了統計捷徑」

視覺語言模型（vision-language models, VLMs）在空間推理基準上分數很高，但這些模型幾乎只看單張 2D RGB 影像，本身沒有深度感測。它們回答「椅子是不是比桌子離相機近？」時，究竟是重建了場景的 3D 結構，還是抓住了自然照片裡「剛好」與深度相關的表面線索？本文的核心貢獻，是把這個問題從「行為層（答對率）」推進到「表徵層（模型內部怎麼編碼空間）」，並指出一個貫穿多個模型家族的系統性缺陷：**垂直-距離糾纏**（vertical-distance entanglement）。

![模型常靠透視捷徑答空間題：越高處的物體被當成越遠。弱模型在 counter 例子上系統性失敗，強模型則在真實與合成情境下都保持一致。](imgs/teaser.png)

### 捷徑從何而來：透視投影把「高」和「遠」綁在一起

在日常照片裡，站在地面上的觀察者，看到共處同一地平面的物體時，越遠的物體會出現在影像中越高的位置（越靠近地平線）。這是古典的仰角深度線索（elevation cue）。問題在於：這個相關性只是「照片統計」，不是「幾何必然」。模型若把「above ≈ far、below ≈ close」內化成一條捷徑，就能在多數自然影像上答對，卻在違反此捷徑的情況下崩潰。

為了量化，作者把所有與深度有關的題目依真值是否符合此捷徑分成兩類。判準很直接：比較兩個被問物體在像素空間的垂直中心座標，若「較遠的物體」有較小的 $y$ 座標（在影像中較高），則此例為 **consistent**；反之為 **counter**。若模型沒有糾纏，兩類的準確率應該相當；一旦出現系統性差距，就是模型依賴垂直位置捷徑的證據。

一個關鍵的觀察是：真實基準本身就嚴重偏向 consistent。在 EmbSpatial-Bench 上 consistent 佔 80.9%、counter 只佔約 10.7%；CV-Bench-3D 上 consistent 佔 60.5%、counter 佔 10.8%。這種偏斜正好複製了真實照片的自然統計——所以「在真實基準上分數高」本身可能就是被資料分布放大的假象。

### 真實基準上的證據：所有模型、所有訓練規模都在 counter 上失分

作者在 EmbSpatial-Bench 與 CV-Bench-3D 的深度題上，分別回報 consistent 與 counter 兩個子集的準確率。跨越三個模型家族（Molmo-7B、NVILA-Lite-2B、Qwen2.5-VL-3B）、四個空間微調規模（80k / 400k / 800k / 2M），以及大規模參考模型 Qwen3-VL-235B，**沒有任何一個模型的 counter 準確率高於 consistent**。舉例：Qwen2.5-VL 微調到 2M 樣本後，在 EmbSpatial-Bench 的 consistent 子集達 60.9%，但 counter 只有 24.0%，落差高達 36.9 個百分點。這種普遍性說明糾纏不是某個架構、某個訓練配方或某個資料規模的偶然產物。

### SpatialTunnel：用合成場景把「垂直位置」和「深度」硬拆開

真實照片會同時混雜多種深度線索（垂直位置、視覺大小、遮擋），無法乾淨地隔離單一線索的貢獻。作者因此在 Blender 建了一個合成資料集 **SpatialTunnel**：一條單點透視的隧道走廊，牆面、天花板、地面對相機光軸對稱。關鍵設計是——因為靠近影像上緣與下緣的物體可以與相機等距，「越高就越遠」的啟發式在此**不再成立**。

![SpatialTunnel 把兩個物體固定在各自的深度，只掃動它們在隧道橫截面上的角位置 $\theta$，讓 2D 影像佈局改變、但深度排序不變。](imgs/spatialtunnel.png)

每個物體以深度 $z$ 與橫截面上的角位置 $\theta$ 參數化。固定 $z$、變動 $\theta$，物體就在影像中上下左右移動，但深度排序不變，於是能構造出「翻轉垂直排列、保持深度不變」的配對反事實樣本。作者把橫截面離散成 16 個角位置，對兩個物體形成 $16\times16$ 的笛卡兒網格，可畫成熱圖診斷。程式庫釋出的 `phase_variation` 設定即為此 $16\times16$ 網格，共 3,072 列。

評分方式沿用前人做法，取第一個生成 token 上 `Yes` 與 `No` 的 logit，定義局部機率：

$$p=\sigma\!\bigl(\ell_{\texttt{Yes}}-\ell_{\texttt{No}}\bigr).$$

單題的正確性分數在真值為 `Yes` 時取 $v=p$、真值為 `No` 時取 $v=1-p$。四項指標分別是平均正確性 $v$、consistent 子集分數 $v_\text{cons}$、counter 子集分數 $v_\text{ctr}$，以及量化糾纏的準確率差 $\Delta = v_\text{cons} - v_\text{ctr}$；無方向偏誤的模型應有 $\Delta \approx 0$。

在這個「去除了評估集偏斜」的平衡合成基準上，糾纏依然普遍存在——所有 base 與微調模型的 $\Delta$ 都是正值，證明糾纏是模型內生的，而非單純繼承自偏斜的評估資料。以下摘錄部分結果：

| Model | $v$ | $v_\text{cons}$ | $v_\text{ctr}$ | $\Delta$ |
|-|-|-|-|-|
| Qwen2.5-VL-3B (base) | 0.570 | 0.776 | 0.360 | +0.416 |
| Qwen2.5-VL-3B + 2M | 0.500 | 0.648 | 0.353 | +0.295 |
| NVILA-Lite-2B + 400k | 0.669 | 0.804 | 0.538 | +0.267 |
| NVILA-Lite-2B + 2M | 0.812 | 0.875 | 0.749 | +0.127 |
| RoboRefer-2B-SFT | 0.793 | 0.816 | 0.770 | +0.046 |
| Qwen3-VL-235B | 0.908 | 0.948 | 0.880 | +0.068 |

值得注意的是資料規模與糾纏的關係並非單調：NVILA 的 $\Delta$ 從 base 的 +0.033 在 400k 衝到 +0.267（峰值），再隨資料增加回落到 2M 的 +0.127；而 Qwen2.5-VL 反而在真實基準上越訓練差距越大。大規模訓練（RoboRefer 逾 20M 樣本、Qwen3-VL-235B 超大預訓練）才把 $\Delta$ 壓到 +0.046~+0.068，同時維持高準確率。

### 表徵層探測：delta 向量、軸一致性、與 VD-EI

行為層的差距只告訴我們「有沒有糾纏」，但不能解釋「為什麼」。作者的核心方法是**對比探測**（contrastive probing）：對同一張影像構造一對只交換被問物體順序的問題（例如「A 在 B 的左邊還是右邊？」對「B 在 A 的左邊還是右邊？」），真值會被翻轉成空間反向。在某個固定的中間層 $L^*$ 取最後一個 token 的隱狀態 $h_q \in \mathbb{R}^d$，對配對 $(q_1,q_2)$ 定義 **delta 向量**：

$$\delta = h_{q_2} - h_{q_1}.$$

直覺上，delta 向量抵銷了兩題共有的視覺內容，只留下「交換空間關係」造成的表徵位移。把大量影像的 delta 依空間類別（above / below / far / close / left / right）聚合後，就能檢視模型內部如何組織空間方向。

![對比探測：交換物體順序構造最小問題對，取中間層最後 token 隱狀態之差為 delta 向量，聚合後即可診斷空間軸的組織與糾纏。](imgs/contrastive_probing.png)

第一個指標是**軸一致性**（axis coherence）。對每條軸（水平、垂直、距離），把兩個對立類別的 delta 都翻正到同一標準方向（例如距離軸把 close 的 delta 取負，全部指向 far），再算符號校正後集合的平均兩兩餘弦相似度：

$$\mathrm{Coh}_{\mathrm{axis}} = \frac{2}{N(N-1)} \sum_{i < j} \cos(\tilde{\delta}^{(i)},\; \tilde{\delta}^{(j)}).$$

一致性高，代表模型用一條穩定、一致的方向來編碼該軸。第二個指標是 **VD-Entanglement Index（VD-EI）**，量化垂直軸與距離軸方向上的耦合。對 above / below / far / close 各算平均 delta $\mu_c$：

$$\mathrm{VD\text{-}EI} = \tfrac{1}{4} \bigl[ \cos(\mu_{\text{above}}, \mu_{\text{far}}) + \cos(\mu_{\text{below}}, \mu_{\text{close}}) - \cos(\mu_{\text{above}}, \mu_{\text{close}}) - \cos(\mu_{\text{below}}, \mu_{\text{far}}) \bigr].$$

前兩項量測「透視對齊」配對（above↔far、below↔close）的相似度，後兩項量測「透視相反」配對。VD-EI 為正，代表垂直與距離表徵按透視投影所預測的方式耦合在一起；為零則代表兩軸獨立。隱狀態取自 EmbSpatial-Bench 影像，各家族固定一個中間層（Molmo-7B 第 23 層、NVILA-Lite-2B 第 20 層、Qwen2.5-VL-3B 第 28 層、Qwen3-VL-235B 第 87 層）。

### 一個帶真實數字的走查：距離軸為什麼是最弱的一環

下表摘錄探測結果。跨所有模型與訓練規模，**距離一致性 $\mathrm{Coh}_{\mathrm{D}}$ 都是三軸中最低的**。微調能大幅拉高垂直一致性（Molmo $0.23\to0.57$、Qwen $0.29\to0.59$），但 $\mathrm{Coh}_{\mathrm{D}}$ 的成長幅度小得多。

| Model | $\mathrm{Coh}_{\mathrm{H}}$ | $\mathrm{Coh}_{\mathrm{V}}$ | $\mathrm{Coh}_{\mathrm{D}}$ | VD-EI |
|-|-|-|-|-|
| NVILA-2B (base) | 0.323 | 0.289 | 0.052 | 0.539 |
| NVILA-2B + 2M | 0.241 | 0.553 | 0.104 | 0.550 |
| RoboRefer-2B | 0.649 | 0.830 | 0.182 | 0.362 |
| Qwen2.5-3B (base) | 0.367 | 0.293 | 0.043 | 0.457 |
| Qwen2.5-3B + 2M | 0.485 | 0.586 | 0.052 | 0.472 |

以 Qwen2.5-VL 為例走一遍：它在 SpatialTunnel 上的 base 版本 $v_\text{cons}=0.776$、$v_\text{ctr}=0.360$，$\Delta=0.416$ 是全表最大——強烈依賴垂直位置捷徑。對應到探測，它的 $\mathrm{Coh}_{\mathrm{D}}$ 從 base 到 2M 幾乎不動（$0.043\to0.052$），counter 準確率反而下滑。相對地，與 NVILA 共用 base 架構的 RoboRefer 佔據一個獨特位置：家族中最高的 $\mathrm{Coh}_{\mathrm{D}}=0.182$ 與最低的 $\text{VD-EI}=0.362$，對應到 EmbSpatial-Bench 上 59.7% 的 counter 準確率，遠高於 NVILA(2M) 的 41.1%。這條「距離一致性成長 → counter 準確率提升」的關係在 NVILA（80k 起）與 Molmo（400k 起）家族內都成立；唯獨 Qwen 的 $\mathrm{Coh}_{\mathrm{D}}$ 停滯，continued scaling 也解不開糾纏。

![delta 向量的 PCA。Molmo/NVILA/Qwen 的 2M 版本在水平與垂直軸上已能分離，但距離 delta 仍糾纏不清；RoboRefer 與 Qwen3 則呈現三個乾淨分離、各自對齊一個主成分的叢集。](imgs/pca_delta.png)

作者進一步驗證 $\mathrm{Coh}_{\mathrm{D}}$ 的跨域效力：在 SpatialTunnel 上算出的 $\mathrm{Coh}_{\mathrm{D}}$，與 EmbSpatial-Bench、CV-Bench-3D 上的 counter 準確率分別呈 $\rho=0.759$ 與 $0.804$ 的相關（皆 $p<10^{-3}$）。這支持 $\mathrm{Coh}_{\mathrm{D}}$ 捕捉的是一種可重用的表徵性質，而非某個基準的計算偽影，因此可當作「訓練是否真的在改善空間表徵」的實用診斷訊號。

### 程式庫的一個實作細節

官方程式庫 `probing.py` 中，`compute_axis_coherence` 忠實實作了論文的一致性公式（符號校正後取平均兩兩餘弦）。但 `compute_vd_ei_per_layer` 實際計算的是「符號校正後、垂直群平均 delta 與距離群平均 delta 之間的餘弦」這個兩群簡化式，而非論文 Eq. (3) 的四項 above/below/far/close 版本；README 也把 VD-EI 描述為「由 6×6 矩陣離線計算的互補診斷」。兩者在直覺上一致，但不是同一條算式，讀者若要精確重現論文表格中的 VD-EI 數值，需注意用的是四項版本而非程式預設的兩群版本。

## 🧪 Critical Assessment

### 問題本身是否真實且重要

是。VLM 越來越多被部署到機器人、具身代理與多模態助理，這些場景真的需要從單張 RGB 推理 3D。而「基準高分未必等於真懂空間」是一個具體、可證偽的疑慮，作者也用 consistent/counter 切分把它變成可量測的現象，而非空泛批評。把分析從行為層推到表徵層、並提出可計算的診斷指標（$\mathrm{Coh}_{\mathrm{D}}$、VD-EI），是這篇論文最扎實的地方。

### 基線、消融、資料與度量是否充分

涵蓋面相當廣：三個異質家族、四個微調規模、外加 RoboRefer 與 Qwen3-VL-235B 兩個大規模參考點，跨 EmbSpatial-Bench、CV-Bench（2D/3D）、BLINK 五個切分。但幾個地方值得保留：（1）核心的「表徵結構 → 穩健推理」關係基本上是**相關性**，而非因果。全篇沒有一個直接干預 $\mathrm{Coh}_{\mathrm{D}}$（例如用表徵編輯強行拉高距離一致性）再觀察 counter 準確率變化的實驗，因此「結構化表徵『導致』更可靠推理」的因果措辭，證據強度弱於敘述語氣。（2）跨域相關 $\rho=0.759/0.804$ 看似漂亮，但底層資料點只是少數 checkpoint（每家族約 5 個），$n$ 很小，相關係數對單點很敏感。（3）$L^*$ 是各家族**事後**從「一致性與 VD-EI 明顯的層」挑出來的，屬於研究者自由度；作者在附錄以跨模型 $\mathrm{Coh}_{\mathrm{D}}$ 排名的 Spearman $\rho=0.928$ 論證穩健性，這點做得不錯，但主文若能把層選擇的敏感度攤開會更有說服力。

### 是新機制，還是既有技巧的重新包裝

delta 向量（交換配對取隱狀態差）本質上是概念方向探測（concept-direction probing）在空間關係上的應用，這類「用對比對消除共有成分」的手法在可解釋性文獻裡並不新。真正的新意在於：把它專門化到三條空間軸、定義出 $\mathrm{Coh}_{\mathrm{D}}$ 與 VD-EI 兩個可操作指標，並用它們解釋一個具體的行為缺陷。這是有價值的組合，但不宜被讀成全新的探測範式。

### 自建基準的循環風險，與真實世界相關性

SpatialTunnel 是作者自己設計、且用來驗證自己方法的合成基準，這裡有需要警覺的環節：它同時扮演「暴露糾纏的診斷集」與「驗證 $\mathrm{Coh}_{\mathrm{D}}$ 跨域效力的測試集」，方法與基準由同一批人針對同一現象設計，存在輕微的循環論證味道——基準是圍繞作者方法所要凸顯的弱點來定義的。此外 SpatialTunnel 的場景極度簡化（對稱隧道、兩個物體、隨機外觀），與真實部署場景的視覺複雜度差距很大，因此「在 SpatialTunnel 上 $\Delta$ 小」能否外推到真實機器人場景的穩健性，仍是開放問題。

### 這個問題真的被解決了嗎

沒有，作者也沒宣稱解決。論文是**診斷性**的：它精準刻畫並量化了糾纏，指出大規模資料或深度監督「相關於」較低的糾纏，但沒有提出可控的修正方法。尤其 RoboRefer 這個最強對照，同時在**訓練規模、深度監督、資料設計**三個維度都與其他模型不同，作者自己也明白表示只把它當作「說明性的參考點」，不將其優勢歸因於單一因素——這是誠實的處理，但也意味著「該怎麼訓練才能拆開糾纏」仍缺乏乾淨的可操作結論。另外，摘要中「糾纏在資料規模擴大時加劇」的說法，其實只在 Qwen 家族的真實基準上清楚成立；NVILA 與 Molmo 的 $\Delta$ 是先升後降，這個非單調性在標題級敘述裡被稍微淡化了。

## 🔗 Related notes
