# H3 Ref2VA 最短 5 幀雙人物實驗

## 目的與邊界

測試 H3 能否用兩張人物參考圖產生極短影片，再擷取圖片。不是 H3 原生單張生圖，也不是完整動作驗證。5 幀低於常規訓練時長；可接受參數不等於保證生成品質。

沿用 [Qwen／Codex 比較](../../qwen-image-edit-2511-official-couple-codex-comparison.md)的官方男女獨立裁切圖，未將官方合成結果當輸入。兩圖順序與角色定義固定：男性 → Picture 1 → Subject 1，女性 → Picture 2 → Subject 2。

| 項目 | 設定 |
| --- | --- |
| 工作流 | MiniMax H3 Ref2VA，standard，無 Turbo LoRA |
| 尺寸 | 640 × 1024 |
| 幀數／播放率 | 明確指定 length=5，24 fps，約 0.2083 秒 |
| 採樣 | 20 steps，Euler／linear_quadratic |
| Seed | 20260913 |
| 參考圖 | 2 張，ref_image_size=match |
| Batch | 1 |
| Prompt | [完整實際提示詞](prompt.txt) |
| 完整實際工作流與事件 | 執行後見 result.json |

## 導演與參考圖規劃

單一固定眼平中近景，兩人面向鏡頭；男左女右，臉部及做手勢的手保持可見。極短片段從首幀即保持「噓」姿勢，不安排伸手到嘴邊的過渡動作。參考圖保留各自臉型、頭髮及衣服，不沿用原背景或姿勢。無對白、無字幕、無配樂，只有室內環境聲。固定鏡頭有利於檢查人物和手部，不需要運鏡。

採用 director skill 的六欄 Ref2VA 格式及 Subject／Picture 明確對應；結構 lint 以靜默片段（0 句對白）通過。H3 提示詞為模型專用展開，**與 Qwen／Codex 的一句中文提示詞不完全相同**，因此不是嚴格控制全部變因的模型排名。

## 執行隔離

`verify.py` 只在獨立 Python 程序中指定 5 幀與小數秒，避免現有產品整數秒轉換；不修改 Veritas 產品程式或配置。等待共享 GPU 空閒才提交，不打斷其他任務。使用自有 PostgreSQL schema、素材、本機 cleanup journal；成功保存並驗證後，只清理本次遠端 input/output，再移除自有 schema。既有的 GPU 記憶體策略交由 vision-flow 控制。

## 結果

**ComfyUI 推論成功，實際產出 5 幀；產品影音長度檢查未通過。** 已從保留的原始影片擷取全部 5 張圖片，沒有重跑或補幀。

![H3 首幀](frame-01.png)

[第 2 幀](frame-02.png) · [第 3 幀](frame-03.png) · [第 4 幀](frame-04.png) · [第 5 幀](frame-05.png) · [原始影片](h3-output.mp4)

| 檢查 | 實測 |
| --- | --- |
| Provider 狀態 | success，無 cached nodes |
| 實際影像 | 640 × 1024，24 fps，5 幀，0.208333 秒 |
| 實際音軌 | 0.200000 秒 |
| Provider 執行時間 | 178.075 秒，依本次 prompt 的 history 起訖時間 |
| Runner 總時間 | 201.783 秒，含提交、等待、收集與失敗處理，不含結尾清理 |
| 執行開始 → 進入採樣節點 | 約 125.057 秒，含編碼等前處理 |
| 採樣節點開始 → 第 20 步 | 約 50.560 秒，可能含模型載入 |
| 第 1 步回報 → 第 20 步回報 | 約 17.483 秒；不包含第一步完成前的成本 |
| GPU／Runtime | RTX 5090，31.36 GiB，ComfyUI 0.33.0，PyTorch 2.7.1+cu128 |
| 裝置策略 | vision-flow H3 low-VRAM、reserve 6 GiB；CLIPLoader=device=default，日誌實際 encoder 在 CPU |

視覺檢查全部 5 幀：兩人均出現並做「噓」手勢；男性白色無袖上衣、女性白色短袖與雙辮子均保留。顔部仍有柔化與形狀差異，不能稱為完全一致。各幀構圖穩定，但有亮度／細節變化。這次支持「H3 最短片段可用於擷取雙人物圖片」的可行性，尚不足以證明人物一致性全面勝過 Qwen 或其他模型，也沒有測試三人以上。

過去同組 Qwen 測試為 173.473 秒 provider、202.218 秒 runner；本次 H3 約 178.075 秒 provider，因此此配置**沒有顯示明顯速度優勢**。兩者提示詞結構、模型、採樣步數及輸出種類不同，且不是交錯重複量測，不能解讀為公平效能排名。5 幀仍需付出編碼、參考圖處理及載入成本。

### 為什麼 result.json 顯示 failed？

產品 `_trim_video` 同時要求影像與音軌皆足夠指定時長。此次音軌比 5/24 秒短約 8.333 毫秒，觸發 `result_corrupt`（SUP-8FEB1B77），不是推論失敗或影片無法解碼。未放寬產品檢查；圖片直接從本機 journal 原始 MP4 擷取。若正式加入「H3 生成圖片」功能，應設計獨立靜態圖收集流程，而不是將一般影片的影音校驗全面關閉。

### 收據與清理

- Provider ID：`81cdae87-45d8-494a-bd47-230e1b913e81`。
- [完整工作流／階段事件](result.json)、[提交設定](run.json)、[清理憑證](cleanup.json)。
- 原始 MP4 SHA-256：`5f62b0cc6e61966610f4989bf5a229d077c0832aefaaf5b98b7e52235472df91`。
- 2 個遠端輸入與 1 個遠端輸出已精確刪除；本機參考图、原始影片、5 幀及 journal 保留。獨立測試 schema 已移除，無修改產品資料與配置。
