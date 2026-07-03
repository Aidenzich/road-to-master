# Segment Anything — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Segment Anything |
| Venue | ICCV |
| Year | 2023 |
| Authors | Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick |
| Official Code | https://github.com/facebookresearch/segment-anything |
| Venue Kind | paper |

## 可提示的分割介面

Segment Anything 把影像分割重新整理成提示分割 (promptable segmentation)：輸入可以是前景點、背景點、框、粗 mask 或文字，輸出不必唯一，但必須是對該提示合理的物件 mask。這個定義的核心不是發明新的 pixel loss，而是把分割模型做成可被其他系統呼叫的介面；當一個點同時可能指向衣服、人體或局部零件時，SAM 允許多個有效答案，而不是強迫模型平均成一個模糊輪廓。

![SAM model diagram](imgs/model_diagram.png)

SAM 的架構拆成三段：重的 image encoder 先把影像編成可重用的 embedding，prompt encoder 把點、框、mask 或文字變成提示 embedding，輕量 mask decoder 再把兩者融合成 mask。論文與公開程式碼一致地使用 1024×1024 輸入、ViT patch size 16、64×64 image embedding、256 維 prompt embedding，並讓 decoder 預設輸出 3 個候選 mask 加上估計 IoU，讓單點提示可以保留「整體、部分、子部分」這類歧義。

$$
\text{image }1024\times1024 \rightarrow E\in\mathbb{R}^{64\times64\times256},\quad
(E,\ \text{prompt tokens}) \xrightarrow{\text{two-way decoder}} \{\hat{M}_1,\hat{M}_2,\hat{M}_3,\widehat{IoU}\}
$$

一個具體前向流程是：先把一張影像縮放並 padding 到 1024×1024，ViT-H/16 產生 64×64 的影像格點，每格 256 維；若使用者給一個前景點，prompt encoder 以位置編碼加上前景點類型 embedding 形成 sparse token；two-way decoder 做 2 層 token self-attention、token-to-image cross-attention、MLP、image-to-token cross-attention，最後上採樣 4× 到相對輸入 1/4 的 mask grid。若提示是單點，decoder 回傳 3 個 mask 候選，應用端通常選 estimated IoU 最高者；若提示已有多個點或框，歧義降低，模型可回傳單一 mask。論文聲稱在已預先計算 image embedding 時，prompt encoder 與 mask decoder 可在瀏覽器 CPU 約 50ms 內完成一次互動。

| 實驗切面 | 論文中的設定 | 代表數字 |
|-|-|-|
| 自動資料生成 | 32×32 foreground point grid，每點多個候選，經 predicted IoU、stability 與 NMS 過濾 | 11M images / 1.1B masks |
| 單點泛化 | 23 個多樣分割資料集，中心點提示，主要和 RITM 比 | SAM 在 16/23 datasets 較高，oracle 選 3 個候選中最佳者時全部較高 |
| Edge detection | BSDS500，16×16 點格產生 768 predicted masks，再轉成 edge map | ODS .768、OIS .786、AP .794、R50 .928 |
| Object proposals | LVIS v1，mask AR@1000 | SAM all 59.3；ViTDet-H all 63.0；SAM rare 65.8 高於 ViTDet-H rare 58.3 |
| Instance segmentation | 以 ViTDet boxes 提示 SAM | COCO AP 46.5 vs ViTDet-H 51.0；LVIS AP 44.7 vs 46.6 |

資料引擎是這篇工作的真正放大器。作者先用模型輔助人工標註，再用半自動流程讓模型預填高信心物件、人工補未標註物件，最後在 11M 張授權且處理隱私的影像上全自動產生 1.1B masks。作者抽樣 500 張影像、約 50k masks 讓專業標註者修正，回報 94% 自動 mask 與修正版 IoU 大於 90%，因此最終 SA-1B 只釋出自動生成的 masks；這是強主張，因為資料品質判斷主要來自作者自己的抽樣與標註流程。

評估設計刻意避開只看單一 IoU 的陷阱。單點提示本來就可能對應多個有效物件，所以論文同時報告最信心 mask、三個候選中的 oracle mask，以及人類對 mask 品質的 1 到 10 分評分。這讓結果更符合提示分割的使用情境，但也讓比較變得不完全對稱：RITM、SimpleClick、FocalClick 是單一互動分割器，而 SAM 的三輸出設計在歧義案例有額外自由度。

公開程式碼和論文描述大致對齊：`build_sam.py` 中 `prompt_embed_dim = 256`、`image_size = 1024`、`vit_patch_size = 16`，ViT-H/L/B 三種 encoder 以 registry 暴露；`mask_decoder.py` 的 `num_multimask_outputs = 3`、`num_mask_tokens = num_multimask_outputs + 1` 與 `iou_prediction_head` 也對應論文的多候選 mask 與品質估計頭。這裡只做靜態檢查，沒有執行 cloned repository 的任何程式、測試或安裝指令。

## 🧪 Critical Assessment

### 問題是否真的值得重新定義

把分割做成可提示介面是實用問題，因為很多下游系統已經能產生點、框、文字或 gaze 之類的弱定位訊號，缺的是穩定把訊號轉成 mask 的模組。SAM 的價值在於提供可組合的 mask primitive，而不是取代所有語意、實例或互動分割方法；論文自己的限制段落也承認，細結構、斷裂小區塊、非常高 IoU 的互動場景、text-to-mask 穩健性，以及 semantic/panoptic segmentation 的提示設計仍未完全解決。

### 評估是否足以支撐大範圍泛化

23 個資料集涵蓋 microscopy、X-ray、underwater、driving、painting、egocentric 等影像型態，比只在 COCO/LVIS 上報告更有說服力；edge detection、object proposals、instance segmentation 也測到不同層級的視覺任務。不過，很多 headline 結果仍依賴作者設計的 prompt engineering 與評估協定。例如 object proposal 用 64×64 point grid 與 NMS threshold 0.9 調到約 900 masks/image；edge detection 則把 masks 經 Sobel 與 edge NMS 轉成 edge map。這些流程證明 SAM 可被工程化成多種工具，但不等於模型本身直接學會那些任務。

消融實驗有助於說明規模與資料引擎不是裝飾：累積三個資料階段會提升 23-dataset single-point mIoU，只用自動 masks 比使用全部資料只低約 0.5 mIoU，1M images 已接近 11M images，而 0.1M images 明顯掉分；ViT-H 也明顯優於 ViT-B，但相對 ViT-L 的增益已變小。限制是這些消融仍集中在同一套 single center point protocol，且圖中呈現的是趨勢而非完整 per-dataset 誤差分析，所以它支持「資料規模、模型容量、資料階段都有貢獻」，但不足以單獨證明每個下游任務都會同樣受益。

### 新意主要在規模、介面與閉環資料

方法新意不是單一網路模組，而是任務定義、ambiguity-aware decoder、互動速度約束，以及資料引擎的組合。ViT image encoder、Transformer decoder、focal/dice loss、IoU head、NMS 都不是孤立的新發明；真正難複製的是 11M images / 1.1B masks 的資料閉環、專業標註流程與運算資源。這也意味著，若讀者只有一般規模資料，最可遷移的是「把分割模型設計成 promptable component」這個系統觀，而不是完整重建 SA-1B。

### 現實部署還需要邊界條件

SAM 很適合作為互動標註、資料預標、偵測器後處理或開放世界候選 mask 產生器，但直接部署在高風險場景仍要小心。作者的 RAI 分析顯示人像分割在部分屬性上差異不大，但 clothing segmentation 在 perceived gender presentation 上有一點提示下的差距；資料地理分布也不是均勻世界樣本。再加上 SA-1B 的 captions 與 inferred locations 不公開，外部研究者難以完整重做偏差分析。因此，這篇論文解決的是「通用 mask 介面可否大規模成立」的大部分工程問題，而不是保證所有領域、所有族群、所有語意任務都已可安全使用。

## 🔗 Related notes
