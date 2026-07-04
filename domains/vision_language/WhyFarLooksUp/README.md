# Why Far Looks Up — Research Note

## 📇 Academic Context

| Field | Value |
|-------|-------|
| Title | Why Far Looks Up: Probing Spatial Representation in Vision-Language Models |
| Venue | ECCV |
| Venue Tier | unknown |
| Year | 2026 |
| Citation Count | unavailable |
| Authors | Cheolhong Min, Jaeyun Jung, Daeun Lee, Hyeonseong Jeon, Yu Su, Jonathan Tremblay, Chan Hee Song, Jaesik Park |
| Official Code | https://github.com/cheolhong0916/contrastive-probing |
| Venue Kind | paper |

## 第一原理與空間表徵探測

本篇論文研究視覺語言模型（VLMs）在推論 3D 空間關係時，是否真正理解 3D 幾何結構，還是僅僅依賴於 2D 影像的表面捷徑。具體而言，作者指出了一種稱為「垂直-距離糾纏」（vertical-distance entanglement）的現象：由於在現實世界的透視投影（perspective projection）中，位於地面上較遠的物體在 2D 影像上通常顯得較高（靠近地平線），VLMs 往往會將「在影像上方」與「距離較遠」錯誤地綁定在一起。

### 捷徑與反常理樣本的表現落差

在現有的空間推理基準測試（例如 EmbSpatial-Bench 和 CV-Bench-3D）中，樣本分佈嚴重偏向「一致性（consistent）」的例子，即較遠的物體確實在影像中較高（比例分別高達 80.9% 和 60.5%）。然而，當作者將測試集拆分為一致性樣本與「反常理（counter）」樣本（較遠的物體在影像中較低）時，所有接受測試的模型都呈現了巨大的準確率落差。

這證明了高基準測試分數可能只是來自於學習了透視捷徑，而非建立穩健的 3D 深度表徵。

### SpatialTunnel 基準測試

為了解除深度與 2D 垂直位置的耦合，作者利用 Blender 構建了名為 `SpatialTunnel` 的合成場景。透過固定物體的 3D 深度，僅改變其在隧道截面上的角位置，可以隨意調整物體在 2D 影像平面的垂直與水平位置，而不會改變它們的真實相對深度。實驗結果顯示，在這樣沒有透視偏差的測試中，多數模型依然表現出強烈的垂直-距離糾纏，這證實了該捷徑是模型內在的偏見，而非僅是評估資料集的巧合。

### 對比探測 (Contrastive Probing)

為了探究模型內部的空間特徵，作者引入了對比探測技術。給定一組空間問答，將問題中的目標物體順序對調以反轉真實關係（例如將「左」換成「右」），並提取模型中間層的隱藏狀態，計算它們的差異向量 $\delta = h_{q_2} - h_{q_1}$。

透過分析這些 $\delta$ 向量，作者定義了軸向連貫性（Axis Coherence）。例如距離軸的連貫性 $\mathrm{Coh}_{\mathrm{D}}$ 計算如下：

$$
\mathrm{Coh}_{\mathrm{axis}} = \frac{2}{N(N-1)} \sum_{i < j} \cos(\tilde{\delta}^{(i)},\; \tilde{\delta}^{(j)}).
$$

研究發現，透過 PCA 降維可見，一般微調模型（如 Molmo 2M、NVILA 2M 與 Qwen 2M）在水平與垂直軸上呈現分離，但距離特徵的群集依然難以區分；只有在使用深度監督或極大規模資料預訓練的模型（如 RoboRefer 和 Qwen3）中，距離特徵才會形成獨立的群集。

### 實際數據演示：捷徑效應與內部表徵的量化落差

為具體說明這項探測技術如何連結內部表徵與外部表現，以 NVILA-Lite-2B 模型家族與其變形 RoboRefer 為例：一般經過 200 萬筆樣本微調的 NVILA (2M) 模型，在面對 EmbSpatial-Bench 基準測試的反常理樣本（即違反透視捷徑的樣本）時，準確率僅有 41.1%。

相反地，RoboRefer 模型在其家族中佔據了一個獨特的空間特徵區域：它展現了最高的距離軸連貫性（$\mathrm{Coh}_{\mathrm{D}}$ 為 0.182）以及最低的垂直-距離糾纏指數（VD-EI 為 0.362）。反映在實際的空間問答上，這使得 RoboRefer 在 EmbSpatial-Bench 的反常理樣本上準確率達到了 59.7%，顯著優於 NVILA (2M)。這個具體的比較展示了：唯有內部表徵中的 $\mathrm{Coh}_{\mathrm{D}}$ 提高且糾纏降低，模型才能真正抵抗視覺透視偏差。

![Coherence vs Entanglement Index](imgs/cohd_vdei.png)

## 🧪 Critical Assessment

### 垂直-距離糾纏在具身 AI 的致命影響
探究 VLM 如何理解 3D 空間這項議題極具真實性。隨著具身 AI（Embodied AI）與機器人技術的發展，模型若無法正確區分 2D 垂直位置與 3D 深度，將在與物理環境互動時發生災難性失敗。該研究明確指出了目前排行榜高分模型可能存在的「捷徑依賴」陷阱。

### 透過 SpatialTunnel 分離混淆變數的有效性
論文不僅在現有資料集上區分了一致性與反常理樣本，更提出了 `SpatialTunnel` 這個能控制混淆變數的合成資料集。透過將 3D 深度與 2D 座標完全脫鉤，這種評估設計在度量垂直-距離糾纏上是非常充分且精準的。然而，合成隧道的視覺多樣性有限，未必能完全反映真實場景中遮擋、相對大小等其他深度線索的複雜交互作用。

### 偏向問題診斷而缺乏通用的架構解法
雖然該研究成功地「診斷」了垂直-距離糾纏，並證明了僅看 Benchmark 總分是不可靠的，但它並沒有提出一個通用的架構或訓練損失函數來主動「解決」這個糾纏問題。因此，這篇論文的貢獻更偏向於分析與評估方法，為未來的 VLM 架構設計提供了診斷標準。

## 🔗 Related notes
