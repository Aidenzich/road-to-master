# Qwen Image Edit 2511 與 Codex 內建生圖：官方男女參考圖合成對照

實驗日期：2026-09-13。單次、同素材與同指令的功能對照；非等參數效能基準。

## 來源與輸入

- [官方模型頁](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [官方第 5 張展示圖](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/%E5%B9%BB%E7%81%AF%E7%89%875.JPG)
- `official-slide-5.jpg` 是下載的官方拼圖；兩張參考圖僅做像素裁切並存為 PNG，沒有重新生成、修臉或放大。
- 原始拼圖為 1280×720。裁切座標採左上原點、右下不包含：男性 `(62,192,380,700)`；女性 `(380,192,710,700)`。已目視確認不包含右側官方合照或文字。

| Image 1：男性，318×508 | Image 2：女性，330×508 |
|---|---|
| ![男性](assets/qwen-official-couple-20260913/reference-1-man.png) | ![女性](assets/qwen-official-couple-20260913/reference-2-woman.png) |

官方展示（右侧是官方結果，**不是本次輸出**）：

![官方展示圖](assets/qwen-official-couple-20260913/official-slide-5.jpg)

## 重測條件

使用官方展示圖上的指令，不自行補寫外貌：

```text
两个人，一起做一个“嘘”的手势。
```

| 項目 | 設定 |
|---|---|
| 模型 | 現有 Qwen Image Edit 2511 FP8 mixed |
| 編碼器 | Qwen 2.5 VL 7B FP8 scaled，device=default |
| 採樣 | Euler/simple，40 steps，CFG 4，denoise 1 |
| Shift／CFGNorm | 3.1／1 |
| 尺寸 | 640×1024，接近男性參考圖比例；既有 adapter 的第一圖縮放流程保留 |
| Seed | 20260913，本次指定，不是已知官方 seed |
| 候選／外部 LoRA | batch 1／無 |
| 服務 | 5090 現有 ComfyUI，等待其他工作完成，不中斷其他任務 |

40 steps／CFG 4 來自官方模型頁的通用範例；該展示案例本身沒有公開完整 seed／參數。輸入也是展示 JPEG 的裁切，而非官方未壓縮原件。因此這是同素材重測，不是精確復現，也不是與官方工作流的配對 A/B。

## 結果與紀錄

Qwen 執行成功，輸出 640×1024。ComfyUI provider 執行區間 173.473 秒；sampler 節點觀察區間 155.957 秒（可能包含模型載入，不稱純採樣）；runner 至結果保存 202.218 秒，不含前置等待既有 H3 工作結束的時間。

本機證據目錄 `isuper/sample/qwen-official-couple-20260913/` 的 `verify.py` 使用隔離資料庫 schema，保存 request/events／本機成品。清理憑證確認本次兩個遠端 input、一個 output 已移除，本機原件保留，隔離 schema 已刪除；未清除其他任務檔案。

### Qwen 與 Codex 並排

| Qwen Image Edit 2511 FP8 mixed | Codex 內建生圖 |
|---|---|
| ![Qwen 結果](assets/qwen-official-couple-20260913/qwen-output.png) | ![Codex 結果](assets/qwen-official-couple-20260913/codex-output.png) |
| 640×1024，40 steps／CFG 4；runner 202.218 秒 | 1007×1562；工具呼叫約 22.3 秒 |

Qwen 目視：兩個主體完整出現，兩人都以食指靠近嘴唇；黑色短髮男性與雙辮女性、白色服裝能對應參考。女性在左、男性在右，背景偏向男性原圖的水面；原 prompt 沒有指定左右與背景，因此這些變化不算違反明示要求。臉部質感較平滑，角度與五官比例有變化，不能稱精確身份保留。

兩邊本次都達成雙人與手勢。Codex 圖目視有較多頭髮／衣物細節，但解析度不同，沒有盲評或身份分數，不能以單張建立模型排名。這次雙人結果也不能證明三參考圖問題已解決，或證明上一輪失敗就是 12 steps／裁切造成；素材、指令、參考數量、尺寸與步數均有變動。

## Codex 內建生圖對照

Codex 使用 imagegen 內建工具測試；未改用 CLI/API，也未把官方右側成品當作參考。輸入順序與 Qwen 相同，完整工具 prompt 同為：`两个人，一起做一个“嘘”的手势。` 沒有追加外貌、構圖或身份提示。

![Codex 內建生圖結果](assets/qwen-official-couple-20260913/codex-output.png)

目視：兩個人都有出現，兩人食指靠近嘴唇，符合「噓」手勢；深色短髮／雙辮子與白色服裝可對應參考。背景偏向女性原圖的室內場景，人物角度及構圖改變。這不是量化身份分數，也不代表精確保留所有面部特徵。

工具呼叫約 22.3 秒，包含工具端處理，不能當成純 GPU 推理耗時。工具沒有提供可核實的底層模型版本、seed、steps 或 CFG；實際輸出 1007×1562，未與 Qwen 鎖定。因此只能比較此介面的單次任務效果，不是等算力或等參數基準。原圖直接保留，沒有後製。

## 其他候選模型（未在本組測試）

- 雲端：FLUX.2 [max]/[pro] 有官方多人物、多參考圖合成展示；Gemini 3 Pro Image（Nano Banana Pro）有角色參考支援。這是功能證據，不是與本組 Qwen 的配對勝率。
- 本機開放權重：FLUX.2 [klein] 9B 可作下一個獨立模型對照，官方列多參考圖支援；9B 是非商業授權，不應統稱 Apache 開源或保證一定勝過 Qwen。
- [BFL 模型比較](https://docs.bfl.ai/flux_2/flux2_overview)、[Google 圖片生成文件](https://ai.google.dev/gemini-api/docs/image-generation)。

