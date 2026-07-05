# WhyFarLooksUp — 空間表徵探測研究筆記
> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Why Far Looks Up: Probing Spatial Representation in Vision-Language Models |
| Venue | ECCV 2026 |
| Year | 2026 |
| Authors | Cheolhong Min, Jaeyun Jung, Daeun Lee, Hyeonseong Jeon, Yu Su, Jonathan Tremblay, Chan Hee Song, Jaesik Park |
| Official Code | https://github.com/cheolhong0916/contrastive-probing |
| Venue Kind | paper |

> 本筆記依據 arXiv e-print（arXiv id `2605.30161`，抓取於 2026-07-03）的 LaTeX 原始檔撰寫；`Venue` 欄記錄論文自述的投稿場所 ECCV 2026，其同儕審查與錄取狀態未在原文獨立佐證，正式版數據可能與此預印本略有差異。

## First Principles

### 透視投影如何把「上方」偷換成「遠方」

單張 RGB 照片只是 3D 場景的一個 2D 投影，模型要回答「椅子是否比桌子離相機更近」這類問題，只能從間接視覺線索反推空間結構。問題在於：日常照片裡存在一條穩定的相關性——地面上的物體，離觀察者越遠，在影像中就出現得越高（趨近地平線）。這條「仰角線索」（elevation cue）讓模型有機會走一條捷徑：被問到深度時，它其實部分是在讀物體的垂直位置，而非真的在推理 3D 幾何。作者把這種傾向命名為 vertical-distance entanglement（垂直—距離糾纏），也就是模型內部把 *above* 約等於 *far*、*below* 約等於 *close*。

![透視捷徑示意：位置越高被當成越遠，counter 樣本因此系統性失敗](imgs/teaser.png)

這條捷徑之所以危險，是因為 behavioral benchmark 只量「答對沒有」，卻看不到「怎麼答對的」。兩個在基準上分數相近的模型，內部機制可能完全不同：一個把空間關係編碼成結構化、可分離的方向，另一個只是依賴自然影像裡的相關線索，一旦分布偏移就變脆。要區分這兩者，必須直接檢視空間資訊在模型內部是怎麼被表徵的。

### 用 consistent / counter 切分把捷徑變成可量測的差距

作者把所有跟深度有關的樣本，依「真實答案是否與仰角捷徑一致」切成兩組：farther 物體在影像中較高就是 consistent，farther 物體反而較低就是 counter。實作上比較兩個被詢問物體的垂直中心座標，較遠者 $y$ 值較小（位置較高）即為 consistent，否則為 counter。如果模型沒有糾纏，兩組準確率應該相近；一旦出現系統性差距，就是它在吃垂直位置捷徑的證據。

![consistent 與 counter 樣本：前者遠物較高、後者遠物較低](imgs/consistent_counter.png)

真實基準的分布本身就嚴重偏向 consistent。在 EmbSpatial-Bench 裡 consistent 佔 976 題（80.9%）、counter 只有 129 題（10.7%）；CV-Bench-3D 則是 consistent 363 題（60.5%）、counter 65 題（10.8%）；其餘為 ambiguous（$\Delta y$ 小於影像高度 5%）。這正好複製了真實照片的自然統計：大多數場景裡遠物確實比較高。於是所有模型在 counter 子集上一致地大幅退步——例如 Qwen2.5-VL-3B 微調 2M 樣本後，在 EmbSpatial-Bench 的 consistent 上有 60.9%，counter 卻只剩 24.0%，落差高達 36.9 個百分點；而且這個現象跨越 Molmo、NVILA、Qwen2.5-VL 三個家族、跨越模型大小與微調資料量都成立。

### SpatialTunnel:在合成走廊裡把 2D 高度和 3D 深度解耦

真實照片會把多個深度線索（垂直位置、視覺大小、遮擋）攪在一起，很難單獨隔離某一個。作者因此用 Blender 造了一個 tunnel（單點透視走廊）合成資料集：牆、天花板與地板對相機光軸對稱，物體可以貼在走廊內壁任意位置。關鍵在於，走廊頂端和底端的物體可以與相機等距，於是「越高越遠」的捷徑在幾何上直接失效。每個物體以深度 $z$ 與截面角度 $\theta$ 參數化，固定 $z$ 只掃 $\theta$，就能在不改變深度排序的前提下讓物體在影像裡上下左右移動，構成一組配對的 counterfactual。

![SpatialTunnel 固定兩物深度、掃描截面角度,使 2D 佈局獨立於深度排序而變化](imgs/spatialtunnel.png)

實作上把截面離散成 16 個角度，對兩個物體形成 $16\times16$ 的 $(\theta_1,\theta_2)$ 網格；每個配置渲染 12 個隨機化場景（形狀、顏色、大小、光照都隨機），得到 $16\times16\times12=3{,}072$ 張影像，配上 4 種問句模板即 $12{,}288$ 個問題。評分不看生成的字面答案，而是取第一個生成 token 上 `Yes` 與 `No` 的 logits，定義：

$$
p=\sigma\!\bigl(\ell_{\texttt{Yes}}-\ell_{\texttt{No}}\bigr)
$$

單題正確度 $v=p$（若標準答案是 `Yes`）或 $v=1-p$（若為 `No`）。再依 consistent / counter 切分報告四個指標：平均正確度 $v$、consistent 正確度 $v_\text{cons}$、counter 正確度 $v_\text{ctr}$，以及量化糾纏強度的準確率落差 $\Delta=v_\text{cons}-v_\text{ctr}$；沒有方向性偏差的模型應該有 $\Delta\approx0$。

因為 $16\times16$ 網格窮舉了兩物的所有截面角度組合，可以把每個格子的正確度畫成熱力圖，直接看偏差落在哪些配置上。下圖用 Molmo-7B 的三個微調規模（base → 400k → 2M）並排 consistent 與 counter 兩張熱力圖：consistent 側的紅色（高正確度）區域隨資料規模穩定擴張、到 2M 已大面積深紅；counter 側則相反，base 時整片偏淡（接近亂猜），400k 掉到最深的藍（糾纏最嚴重、落差最大），到 2M 雖回升成大片紅色，但紅得比 consistent 側淺、且夾著一條偏白/藍的帶——也就是原文說的「部分回升，但 counter 依舊實質更難」。這張圖把「加資料主要在補 consistent、counter 只是被動跟上」的機制攤在同一個色階上。

![Molmo-7B 在 SpatialTunnel 的正確度熱力圖：consistent 側紅區隨 base→400k→2M 擴張，counter 側 400k 最深藍、2M 部分回升但仍淺於 consistent](imgs/heatmap_compare_molmo_consistent.png)

![同上的 counter 子集：base 近亂猜、400k 落至最深藍（最大落差）、2M 回升為大片紅但仍夾雜偏白/藍帶](imgs/heatmap_compare_molmo_counter.png)

### 走一遍 Qwen2.5-VL-3B 的 SpatialTunnel 聚合結果(worked example)

以基礎版 Qwen2.5-VL-3B 在完整 SpatialTunnel 評估上的聚合結果為例，具體看數字怎麼落地。走廊裡每張圖放兩個固定深度的物體，問「obj$_1$ 是否比 obj$_2$ 離相機更遠」；模型讀入 RGB 影像與問句、跑前向，在第一個生成位置讀出 $\ell_{\texttt{Yes}}$ 與 $\ell_{\texttt{No}}$，代入 $p=\sigma(\ell_{\texttt{Yes}}-\ell_{\texttt{No}})$ 得到單題信心。把 $16\times16$ 網格、12 個隨機場景與 4 種問句模板上所有 consistent 樣本的 $v$ 平均，得到 $v_\text{cons}=0.776$；所有 counter 樣本平均則只有 $v_\text{ctr}=0.360$，於是 $\Delta=+0.416$——這是全表最大的糾纏落差（這些是整個資料集的聚合統計，不是單一格子或單次前向的值），顯示這個基礎模型幾乎是靠「高即遠」在作答，而非真的比較深度。作為對照，同屬 NVILA 基座、但吃了 20M 以上含深度監督資料的 RoboRefer-2B-SFT，在同一測試上 $v=0.793$、$\Delta$ 只有 $+0.046$；資料規模化到極致的 Qwen3-VL-235B 則 $v=0.908$、$\Delta=+0.068$。同一個「二選一深度題」，糾纏可以從 0.416 一路壓到 0.05 附近，差別不在題目而在表徵。

### contrastive probing:用 delta vector 直接讀出空間軸

要看內部表徵，作者設計了 contrastive probing。給定一張圖，構造一對只差在物體順序的問句，例如把「A 在 B 的左邊還是右邊」換成「B 在 A 的左邊還是右邊」，於是標準答案剛好被反轉（left 變 right）。對每個問句取固定中間層 $L^*$ 的最後一個 token 隱狀態 $h_q\in\mathbb{R}^d$，定義 delta vector 為交換前後的位移 $\delta=h_{q_2}-h_{q_1}$；跨大量影像重複，就得到每個空間類別（above、below、far、close、left、right）的一組 delta 向量，把共通的視覺成分抵消掉，只剩下空間方向的潛在編碼。

![contrastive probing:交換物體順序取隱狀態差,得到 delta 向量](imgs/contrastive_probing.png)

在 delta 向量上定義兩個指標。第一個是 axis coherence（軸一致性）：對每條軸（水平、垂直、距離），把兩個相反類別的 delta 集中起來，並對相反那一側取負號使全部朝同一正向（$\tilde{\delta}$），再算兩兩餘弦相似度的平均：

$$
\mathrm{Coh}_{\mathrm{axis}} = \frac{2}{N(N-1)} \sum_{i < j} \cos\!\bigl(\tilde{\delta}^{(i)},\; \tilde{\delta}^{(j)}\bigr)
$$

高一致性代表模型把該軸編碼成一個穩定、方向一致的向量。第二個是 VD-Entanglement Index（VD-EI），對 above、below、far、close 各算平均 delta $\mu_c$，量測垂直與距離兩軸的方向重疊：

$$
\mathrm{VD\text{-}EI} = \tfrac{1}{4}\bigl[\cos(\mu_{\text{above}},\mu_{\text{far}}) + \cos(\mu_{\text{below}},\mu_{\text{close}}) - \cos(\mu_{\text{above}},\mu_{\text{close}}) - \cos(\mu_{\text{below}},\mu_{\text{far}})\bigr]
$$

前兩項是透視一致的配對（above↔far、below↔close），後兩項是透視相反的配對；VD-EI 為正，代表垂直與距離表徵正如透視投影預測地耦合在一起，為零則代表兩軸獨立。作者在 EmbSpatial-Bench 影像上、於各家族固定的一層抽取隱狀態；分析層落在網路約 71% 到 93% 深度處（Molmo $L^*{=}23$／共 32 層 ≈72%、NVILA $L^*{=}20$／共 28 層 ≈71%、Qwen2.5-VL $L^*{=}28$／共 36 層 ≈78%、Qwen3-VL-235B $L^*{=}87$／共 94 層 ≈93%）。前三個中小模型落在約 71–78% 的中後段，符合「空間表徵在中段層成形、末段層轉為輸出專用」的既有觀察；Qwen3-VL-235B 則是例外——原文明說它三軸的連貫表徵「很晚才形成」，因此選層被推到 93% 深度，作者推測與其 94 層 MoE 架構延後穩定空間表徵的形成有關。核心流程可寫成：

```python
# 每張圖建一組最小對比對:交換兩物體順序,標準答案被反轉
q1 = "Is A closer to / farther from the camera than B?"
q2 = "Is B closer to / farther from the camera than A?"
h1 = hidden_state(model, image, q1, layer=L_star)[-1]   # 最後一個 token
h2 = hidden_state(model, image, q2, layer=L_star)[-1]
delta = h2 - h1                                          # delta vector δ
# 對每條軸:相反類別取負號對齊,再算兩兩餘弦相似度平均 -> Coh_axis
# above/below/far/close 的平均 delta 之間算餘弦 -> VD-EI
```

### 距離軸最弱,而它的成長預測 counter 穩健度

把三條軸的一致性攤開來看，結論很乾脆：距離軸一致性 $\mathrm{Coh}_\mathrm{D}$ 在每一個模型、每一個訓練規模上都是三軸中最低的。微調會把垂直一致性拉得很高（Molmo 從 0.23 到 0.57、Qwen 從 0.29 到 0.59），但 $\mathrm{Coh}_\mathrm{D}$ 的成長幅度小得多。下表節錄幾個代表性列（完整表見原文 Table 5）：

| Model | $\mathrm{Coh}_\mathrm{H}$ | $\mathrm{Coh}_\mathrm{V}$ | $\mathrm{Coh}_\mathrm{D}$ | VD-EI |
|-|-|-|-|-|
| Molmo-7B (base) | 0.143 | 0.228 | 0.075 | 0.279 |
| Molmo-7B (+2M) | 0.239 | 0.574 | 0.112 | 0.474 |
| NVILA-2B (base) | 0.323 | 0.289 | 0.052 | 0.539 |
| NVILA-2B (+2M) | 0.241 | 0.553 | 0.104 | 0.550 |
| RoboRefer-2B | 0.649 | 0.830 | 0.182 | 0.362 |
| Qwen2.5-3B (base) | 0.367 | 0.293 | 0.043 | 0.457 |
| Qwen2.5-3B (+2M) | 0.485 | 0.586 | 0.052 | 0.472 |

真正有意思的是 $\mathrm{Coh}_\mathrm{D}$ 與行為穩健度的連動。當距離一致性隨資料規模成長，counter 準確率會跟著上升：NVILA 的 $\mathrm{Coh}_\mathrm{D}$ 從 0.052 翻倍到 0.104，EmbSpatial-Bench counter 準確率從 27.1% 升到 41.1%；反之 Qwen2.5-VL 的 $\mathrm{Coh}_\mathrm{D}$ 幾乎不動（0.043→0.052），counter 準確率反而從 32.6% 掉到 24.0%，落差越拉越大。換句話說，距離表徵沒長出來，繼續加資料並不會解決糾纏。

![Counter 準確率對距離一致性:Molmo、NVILA 沿右上軌跡上升,Qwen 卡在低 CohD,RoboRefer 位於右上角](imgs/cohd_vs_counter_acc.png)

換一個角度、把 $\mathrm{Coh}_\mathrm{D}$（縱軸）對 VD-EI（橫軸）攤開，同一個 NVILA 家族的內部幾何就更清楚：一般微調變體（80k→2M 的橘點，帶箭頭標出規模化軌跡）全擠在圖的右下角——高 VD-EI（約 0.56–0.64）、低 $\mathrm{Coh}_\mathrm{D}$（約 0.05–0.09），也就是垂直與距離仍高度糾纏、距離軸還沒長好；RoboRefer（深紅點）則獨自落在左上角，是全家族唯一同時做到低糾纏與高距離一致性的點。要注意這張散點圖是各模型在圖用選層上算出的，數值與正文 Table 4 逐列報告的選層值略有出入（例如 RoboRefer 圖上約 $\mathrm{Coh}_\mathrm{D}\approx0.17$／VD-EI$\approx0.38$，表列為 0.182／0.362；NVILA-2M 圖上約 0.09／0.56，表列為 0.104／0.550），但兩者呈現的相對結構一致。

![CohD 對 VD-EI 散點:NVILA 一般微調變體聚在右下角(高 VD-EI、低 CohD),RoboRefer 深紅點獨佔左上角(低 VD-EI、高 CohD)](imgs/cohd_vdei.png)

而且 $\mathrm{Coh}_\mathrm{D}$ 不只是某個基準的內部產物。在 SpatialTunnel 上計算的 $\mathrm{Coh}_\mathrm{D}$，與另外兩個基準的 counter 準確率跨域相關：對 EmbSpatial-Bench 是 $\rho=0.759$、對 CV-Bench-3D 是 $\rho=0.804$（皆 $p<10^{-3}$）。這種跨域一致性支持「距離一致性捕捉到一種可重用的表徵訊號，而非某個資料集的計算假象」。

進一步把 $\mathrm{Coh}_\mathrm{D}$ 分別在合成走廊（SpatialTunnel）與真實影像（EmbSpatial-Bench）上重算並排比較，可以看到雖然兩域的絕對值不同（合成走廊普遍偏高），但每個家族內部的相對排序大致保留——下圖紅框標出的 NVILA 家族，其 RoboRefer $>$ 2M $>$ 400k$\approx$800k $>$ 80k $>$ base 的次序在合成與真實兩域完全一致。這佐證了「$\mathrm{Coh}_\mathrm{D}$ 的絕對大小受環境影響，但在相同資料條件下提供可靠的相對比較」。

![合成走廊(灰條)與真實 EmbSpatial-Bench(紅點)上的 CohD 並排:絕對值不同但家族內排序保留,紅框標出 NVILA 家族在兩域排序一致](imgs/cross_domain_distance_coherence.png)

### 什麼樣的表徵才算「強」

把同基座的 NVILA 系列與 RoboRefer 放在一起做 PCA，可以直接看到差異。基礎 NVILA 的距離 delta 向量塌縮在原點附近，根本沒形成一條可辨認的軸；微調到 2M 開始出現方向擴散，但垂直與距離的群集仍然重疊；RoboRefer 則是三條軸各自乾淨地分成獨立群集、對齊到不同主成分。量化上，NVILA 整條規模化軌跡的 $\mathrm{Coh}_\mathrm{D}$ 只有邊際成長、VD-EI 一直高掛在 0.54–0.64；RoboRefer 佔據一塊獨特區域——家族內最高的 $\mathrm{Coh}_\mathrm{D}$（0.182）與最低的 VD-EI（0.362），對應到 59.7% 的 EmbSpatial-Bench counter 準確率，遠高於 NVILA-2M 的 41.1%。

![各模型 delta 向量的 PCA:Molmo/NVILA/Qwen 的 2M 版距離軸仍糾纏,RoboRefer 與 Qwen3 三軸清楚分離](imgs/pca_delta.png)

也因此，行為準確率單看一個數字並不可靠：NVILA(2M) 在 CV-3D Depth 有 93.8%，到 BLINK Spatial Relation 卻掉到 62.9%；Qwen(2M) 在 BLINK Spatial Relation 有 78.3%，到 CV-3D Distance 又只剩 52.2%。相對地，表徵結構最乾淨的 RoboRefer 與 Qwen3-VL-235B 在各基準上一致地高。作者據此主張，高 $\mathrm{Coh}_\mathrm{D}$（搭配低 VD-EI 作為互補訊號）可以當成「這次訓練有沒有真的改善空間表徵」的實用診斷。不過作者也明說，RoboRefer 同時改了訓練規模與監督方式，因此只當作示意性的參照，而非把好處歸因於單一因素。

## 🧪 Critical Assessment

### 這個捷徑是真實缺陷,還是被基準分布放大的假象

論文最有價值的一步，其實是先看穿自己的證據可能有問題。真實基準本身嚴重偏 consistent（EmbSpatial 80.9%、CV-Bench-3D 60.5%），所以「counter 掉分」有兩種解釋：模型內化了偏差，或只是評測集偏斜。作者用 SpatialTunnel 這個平衡合成集把後者切掉，在幾何上對稱、consistent 與 counter 均衡時仍看到正的 $\Delta$，這讓「model-intrinsic」的主張站得住腳，方法論上是誠實且到位的。可補強的是：counter 子集在真實基準上樣本量很小（EmbSpatial 只有 129 題、CV-Bench-3D 只有 65 題），單一模型 counter 準確率的信賴區間會相當寬。以 Qwen2.5-VL-3B(+2M) 為例做個粗估：EmbSpatial counter 24.0%（$n=129$）的常態近似 95% 信賴區間約為 $[16.6\%, 31.4\%]$，CV-Bench-3D counter 53.8%（$n=65$）約為 $[41.7\%, 65.9\%]$（區間為本筆記依二項常態近似自行推算，非原文報告值）——區間寬達 15–24 個百分點。論文把不少 3–5 個百分點的差異拿來敘事，這類小樣本比較的統計穩定度值得保留。

### 把 RoboRefer 當「強表徵」對照是否公平

整篇的正向錨點幾乎都壓在 RoboRefer 與 Qwen3-VL-235B 身上，但兩者都不是乾淨的受控變因。RoboRefer 相對 NVILA 系列同時動了資料規模（20M+）與監督型態（含 RGB-D 深度監督），Qwen3-VL-235B 更是換了一個數量級的預訓練規模。作者自己也承認 RoboRefer「只當示意參照」，這是恰當的謹慎；但論文的核心敘事（結構化表徵帶來穩健度）在很大程度上依賴這兩個混淆了多個因素的點，真正受控的家族內證據其實只有「$\mathrm{Coh}_\mathrm{D}$ 隨規模成長 ↔ counter 上升」這條，且它在 Qwen 家族上是反例。把單一深度監督模型當成「結構化表徵」的存在性證明可以，但要支撐「這是達成穩健度的路徑」則證據偏薄。

### CohD 與穩健度:相關性被當成因果的風險

$\mathrm{Coh}_\mathrm{D}$ 與 counter 準確率的關係，通篇是相關性證據（跨域 $\rho=0.759/0.804$、家族內同向軌跡），而敘事語氣時常滑向因果（「形成連貫的距離表徵使模型更穩健」）。這裡有循環的疑慮：$\mathrm{Coh}_\mathrm{D}$ 與 counter 準確率都在 EmbSpatial-Bench 的同一批影像上計算，兩者共享輸入分布，跨域驗證雖然緩解了一部分，但相關並不排除「兩者同為某個更根本能力的共變量」。論文沒有做介入式實驗（例如直接沿距離軸編輯表徵、或在訓練中顯式提升 $\mathrm{Coh}_\mathrm{D}$ 再看 counter 是否改善），因此「診斷訊號」的定位是誠實的，但把它讀成「訓練目標」則超出了現有證據。

### 合成走廊的外部效度與 near-chance 混淆

SpatialTunnel 用對稱走廊換來了乾淨的控制，代價是視覺分布與真實照片差很遠——單點透視、隨機貼在內壁的物體、隨機外觀，這種域落差本身可能壓低所有模型的絕對表現，也可能引入合成專屬的線索。不過作者用同一套走廊多做了一個對照來緩解「只抓到高度捷徑」的疑慮：固定 $s_1+s_2=0.4$、只掃物體外觀大小，讓遠物由小變大（size-conflicting）。下圖三個模型的曲線顯示，Molmo 與 NVILA 在遠物變大時正確率明顯下滑（微調版尤其陡，落差可達 0.2 以上），代表它們同樣吃「大即近」的視覺大小捷徑；而 Qwen 全程貼在 0.5（亂猜）附近幾乎不動——但這不是穩健，而是它在這個設定下根本沒有深度判別力，正好呼應下面談的 near-chance 混淆。也就是說，垂直位置只是眾多會與深度相關的捷徑之一，模型的「深度判斷」其實是多個表面線索的疊加。

![物體大小消融:橫軸為遠物 obj1 大小(上軸為近物 obj2),Molmo/NVILA 隨遠物變大正確率下滑,Qwen 全程貼近 0.5 亂猜](imgs/correctness_vs_size.png)

另一個要小心的是 near-chance 混淆：基礎 NVILA 的 $\Delta$ 只有 +0.033 看似「無偏」，但它整體 $v<0.5$、其實是接近亂猜，小落差反映的是無能力而非無偏差。論文有點到這一點（NVILA base 近乎隨機），值得肯定；但表格中不少 $v$ 落在 0.5 附近的列，若只看 $\Delta$ 容易誤讀，理想上應搭配對隨機基準的顯著性檢定一起看。

### 指標、消融與可重複性的覆蓋是否足夠

分析層 $L^*$ 是逐家族手選的單一層，論文以附錄的一致性平原（Spearman $\rho=0.928$）論證選層穩健，這是必要的辯護；但 VD-EI 在高層會震盪、Qwen3-235B 沒有明顯平原（程式碼註解自承），提醒讀者這些絕對數值對選層仍有一定敏感度。整體而言，問題定義清楚、指標可操作、且附了可執行的探測程式碼（`probing.py` 實作 delta、coherence 與 6×6 相似度矩陣），可重複性相對紮實。真實世界關聯上，垂直—距離捷徑對部署在機器人／具身代理的 VLM 確實是實際風險（視角一變捷徑就失效），這個問題是真的；但論文停在「診斷」，並未證明所提指標能反過來驅動出更穩健的模型，離「解決」還有明確距離——這點論文表述得算克制，沒有過度宣稱。

## 🔗 Related notes

<!-- 目前 vision_language 網域下沒有與本主題直接相關且可解析的既有筆記,保留標題留空。 -->
