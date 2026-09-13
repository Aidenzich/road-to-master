# Qwen Image Edit 2511：RTX 5090 異質 Batch 1／2／4 實驗

實驗日期：2026-09-13。性質：單輪工程試跑與效能診斷，不是模型排行榜或正式上線驗收。

## 摘要

不同提示詞、不同參考圖與不同 seed 可以組成同一個 Qwen 圖像編輯採樣批次；
本次 batch 1、2、4 都成功產圖，輸出已下載、解碼、雜湊驗證並清理遠端暫存。
但**可批次執行不等於有明顯加速**：

- 兩次暖機單張共 306.281 秒，batch 2 為 303.429 秒，全程只縮短 0.93%。
- 採樣本身由 106.610 秒變成 103.365 秒，只縮短 3.04%；單輪差距可能受波動影響。
- Batch 4 花 600.680 秒，平均每張 150.170 秒；未做四組完全配對的單張基準，不能宣稱其加速比。
- 約三分之二時間用在串行 CPU 編碼；此外，實驗計時 ID 不當進入快取鍵，造成不必要的重新編碼。
- 上述問題能解釋端到端效益受限，**不能完整解釋採樣為何也幾乎沒有加速**；底層 profiling 尚未完成。

本篇集中保存本輪實驗設計、測量、品質觀察及歸因修正，不把尚未執行的改善當成結論。

## 1. Batch 指的是什麼

本實驗的 heterogeneous batch（異質批次）是每張各自帶入 prompt、ordered references 與 seed，
將編碼後的條件／latent 沿 batch 維度合併，呼叫一次 sampler，再依索引分離輸出。

- Batch 2：兩組不同輸入，輸出兩張；不是兩張參考圖合成一張。
- Batch 4：四組不同輸入，輸出四張；不是同一個 prompt 重複四次。
- 不是同時送出四個 HTTP request，也不是用 RepeatLatentBatch 冒充異質批次。
- API／sampler 的 batch 大小與底層每次模型 forward 的有效 batch 大小，是不同的驗證層級。

目前自訂節點支援 1–4 筆、每筆 1–3 張參考圖，但這輪**每筆只用一張參考圖**。
各筆必須有相同尺寸、steps、CFG、參考圖張數，以及相容的 reference latent shape；不相容就拒絕，
不偷偷裁掉參考圖或退回逐張採樣。不同 CFG 的原始研究條件不能硬湊成同一批。

## 2. 環境與固定參數

| 項目 | 本輪設定／證據 |
|---|---|
| GPU | 一張 NVIDIA GeForce RTX 5090，32,607 MiB 可見 VRAM |
| 主模型 | Qwen Image Edit 2511 FP8 mixed |
| 文字／影像編碼器 | Qwen 2.5 VL 7B FP8 scaled，明確設 `device=cpu` |
| VAE | Qwen image VAE |
| 生成尺寸 | 512 × 512 |
| Steps／CFG | 40／4 |
| Sampler／scheduler | Euler／simple |
| Denoise | 1.0 |
| Model sampling shift／CFGNorm | 沿用原 Qwen builder：3.1／1 |
| 額外 LoRA | 無 |
| PyTorch／runtime CUDA | 2.7.1+cu128／12.8 |
| NVIDIA driver | 595.58.03；nvidia-smi 顯示 CUDA 13.2，與 PyTorch runtime 版本分開記錄 |
| 實際部署 | 既有 ComfyUI Docker 容器；本機經 SSH tunnel 接 API |
| 容器 CPU／RAM | CPU quota 無上限、可见 CPU 0–15、查詢時 nr_throttled=0；無容器 RAM 上限 |
| 候選程式版本 | vision-flow `be705fe`，基於 Qwen prerequisite `1549d8a` |

Vision-flow 同時有 `make pod-deploy`、K3s 與 HAMi／Volcano 部署路徑，
**但本輪沒有走 Pod 路徑**。不能把 Docker 的限制檢查當成 Pod 配額驗證，
也不能把本輪數字當成正式 Pod 環境的 benchmark。

實驗自訂節點只安裝在既有 container writable layer；可隨容器 restart 保留，
container recreate 不保證保留。這不是完成正式 image／IaC 發布的證據。

## 3. 輸入與測試順序

| 識別 | 內容與修改目標 | Seed | 使用範圍 |
|---|---|---:|---|
| A | 2D 動畫人物，向畫面右方轉約 45 度 | 2026091360 | 單張、batch 2、batch 4 |
| B | 電影風格人物，向畫面左方轉成側臉 | 2026091362 | 單張、batch 2、batch 4 |
| C | 3D 動畫人物，向右轉約 45 度 | 2026091361 | batch 4 |
| D | 古裝人物，向右轉約 45 度 | 2026091363 | batch 4 |

每個身份跨批次保持原 prompt、參考圖與 seed。輸入圖的 SHA256、原始提示詞與輸出映射保存在本機證據中。
噪聲先以各筆 seed 生成 singleton，再 concatenate，避免批次位置改變初始噪聲。
這不保證不同 batch 的最終像素完全相同。

執行順序：

1. 單張 A 冷啟動：ComfyUI restart 後第一張，包含模型載入成本。
2. 單張 B 暖機：不重啟服務。
3. Batch 2：A＋B。
4. 再生成單張 A 暖機對照，避免使用冷啟動 A 做不公平比較。
5. 上述輸出下載與清理完成後，batch 4 自動接著執行 A＋B＋C＋D。

「暖機」只指服務已跑過任務，不代表所有模型保證常驻 VRAM，也不代表編碼快取命中。
本輪未做多輪隨機化、誤差區間，也沒有 C、D 的配對單張測速。
Batch 3 僅是原計畫的失敗備案；batch 4 成功，因此沒有跑 batch 3。

## 4. 時間結果

單位均為秒；平均值按該批輸出張數計算。

| 測試 | 張數 | Provider 全程 | 編碼 | 採樣 | 平均全程／張 | 平均採樣／張 |
|---|---:|---:|---:|---:|---:|---:|
| 單張 A 冷啟動 | 1 | 161.745 | 101.401 | 57.464 | 161.745 | 57.464 |
| 單張 B 暖機 | 1 | 153.335 | 99.453 | 53.529 | 153.335 | 53.529 |
| Batch 2：A＋B | 2 | 303.429 | 199.480 | 103.365 | 151.715 | 51.682 |
| 單張 A 暖機 | 1 | 152.946 | 99.637 | 53.081 | 152.946 | 53.081 |
| 暖機 A＋B 合計 | 2 | 306.281 | 199.090 | 106.610 | 153.141 | 53.305 |
| Batch 4：A＋B＋C＋D | 4 | 600.680 | 399.046 | 200.226 | 150.170 | 50.057 |

Batch 2 配對比較：

```text
全程時間減少 = (306.281 - 303.429) / 306.281 = 0.93%
採樣時間減少 = (106.610 - 103.365) / 106.610 ≈ 3.04%
```

不能把時間減少百分比與 throughput 增加百分比混用；本篇以上均指時間減少。
Batch 4 平均每張與 batch 2 接近，但案例組成不同，只能作描述性比較。

### 計時邊界

- Provider 全程：ComfyUI `execution_start` 到 `execution_success`，不含本機下載、遠端清理與前置排隊。
- 編碼：native reference VAE／文字與影像編碼路徑的觀測總和，含 CPU encoder 工作；不是純文字 tokenization。
- 採樣：`comfy.sample.sample` 呼叫前後同步 CUDA 計時，可能包含必要模型載入，不是純 kernel 時間。
- Pack/noise 另有紀錄：batch 2 約 0.209 秒、batch 4 約 0.391 秒，不是主要瓶頸。
- 完整 queue trace、provider history 與 node timing receipt 分開保留；cached UI receipt 不計成新計算。

## 5. GPU／VRAM 觀察

| 觀察時點 | GPU 使用率 | 顯存／其他數據 | 能說明什麼 |
|---|---:|---|---|
| Batch 4 CPU 編碼中 | 0% | nvidia-smi 約 21,956 MiB | 此刻 GPU 在等編碼，不代表採樣閒置 |
| Batch 4 GPU 採樣中 | 100% | 24,708 MiB，474／475 W，63°C | 當下 GPU 忙且接近設定功耗上限 |
| Batch 2 採樣結束 | 未連續採集 | PyTorch reserved 22,048 MiB；device free 9,356 MiB | 結束快照，不是每 job peak |
| Batch 4 採樣結束 | 未連續採集 | PyTorch reserved 24,000 MiB；device free 7,404 MiB | 四筆在本設定下成功完成，無 OOM |

PyTorch allocated、reserved、裝置整體使用量是不同口徑。
receipt 的 `process_lifetime_peak_allocated_bytes` 是程序存續期間峰值，不可標成單筆任務峰值。
GPU utilization 100% 也不等同於已證明 Tensor Core FLOPS 飽和；沒有 profiler 不能據此完成瓶頸歸因。

## 6. 為什麼 batch 沒有明顯加速

### 已確認：編碼仍然串行

每張各跑正／負兩次 native encode。Batch 2 四次、batch 4 八次，每次約 49–51 秒。
所以編碼由約 100 秒／張線性增長到約 200／400 秒。
這版只將採樣批次化，並未批次化整條生成流程。

### 已確認：計時識別污染快取

`build_batch` 對含 output prefix 的完整 graph 產生 batch_id，
再把 batch_id 放進 `VisionQwenTimedEncode` 的輸入。
換批次或輸出名稱就會讓內容相同的編碼無法重用快取。
這是實作的效能缺口，不應成為正式版本的默認行為。

本輪單張與 batch 都受到這個設計影響，所以表中可以比較這版 adapter 的未命中編碼成本，
但不能代表原生工作流跨任務快取命中時的效能。修快取也不等於已解釋採樣只有約 3% 的收益。

### 已確認的採樣證據，以及還缺什麼

採樣輸入 latent shape 分別為 `[1,16,1,64,64]`、`[2,16,1,64,64]`、`[4,16,1,64,64]`。
程式將不同條件沿 batch 軸合併、以整數 binary attention mask 補齊文字長度，
reference latents 按 slot 合併，且只呼叫一次 sampler。

這支持「不是外層逐張呼叫 sampler」，**不保證底層執行方式已最佳化**。
現場 ComfyUI sampler 原始碼會依條件相容性與可用記憶體決定是否合併 CFG 条件；
尚未追蹤本次實際 forward 的 shape、次數與拆分狀況。

可能解釋包括單張已接近有效算力上限、CFG 條件拆分、attention／矩陣 kernel 效率或記憶體搬運。
它們目前都只是待驗證假設，不是已確認的根因；不能用一般「batch 不一定快」的說法替代 profiling。

## 7. 圖片品質與 prompt 干擾修正

已目視檢查 A、B 的單張與 batch 2，以及 batch 4 四張成品：不同角色／畫風沒有明顯互換。
A 保留銀髮、紅眼、貓耳、黑色星形髮飾與深色服裝；A、B 的單張與 batch 輸出看起來相近。
Batch 4 另有 3D 卡通人物與古裝人物，輸出檔與 item ID 一一對應；這不是完整身份保留評分。
PIL RGB 比較確認 A、B 單張與 batch 2 不是逐像素完全相同。

**眼鏡案例必須排除 prompt 干擾後才可歸因，這輪不重測。**
原 B 參考圖沒有眼鏡，但單張／batch 輸出新增眼鏡；測試模板卻含有：

```text
Preserve ... all accessories including any glasses ...
```

雖然字面是「保留原有眼鏡」，不是命令新增，但原圖沒有的物件本就不該被模板額外提及。
這是 prompt 撰寫缺陷，可能誘導結果。撤回將新增眼鏡直接當成模型一致性不佳的結論，
也不能把它算成 batch 特有問題；沒有去除措辭的對照，無法證明因果。
依使用者決定只修正報告、不重測，原始 prompt、seed 與輸出均保留。

未來此類測試應只要求改變視角並保留人物外觀、服裝與原有配件，避免列出不存在的物件。
此建議不是本輪真正送出的 prompt，不能覆寫歷史輸入。

## 8. 證據、清理與未完成範圍

原始資料保留在研究工作區 `isuper/sample/qwen-batch-benchmark-20260913/`，
不複製機器絕對路徑、連線憑證或完整 runtime dump 到本 repo。
本篇為單一閱讀入口；成品及完整 JSON 仍在本機 sample，不另建立第二篇 batch 筆記。

| 證據 | 相對於上述實驗資料夾的檔案 |
|---|---|
| A／B 原始 prompt、seed、reference SHA | `inputs.json` |
| 四筆原始輸入 | `batch-four/inputs.json` |
| 完整 ComfyUI graph 與輸出映射 | 各測試 `*.plan.json`；其中 `request.prompt` 才是實際 graph |
| Provider 原始事件 | 各測試 `*.history.json`、`*.trace.json` |
| 階段耗時／GPU／參數 | 各測試 `*.receipt.json` |
| Batch 4 最終結果 | `batch-four/05-batch-four-warm.receipt.json` |
| 圖檔 SHA 與 cleanup receipts | `status.json`、`batch-four/status.json` |
| 程式 | `run.py`、`run_four.py` |

單張／batch 2 共保存五張 PNG，batch 4 保存四張 PNG。
兩輪相應遠端九張輸出、六個上傳輸入與五筆 provider history 已清除；task 輸出目录不存在。
同一參考圖不同輪次重新上傳按不同檔計數。既有服務、公開模型与 runtime 自訂節點不是此次成品暫存，仍保留。
原有 SSH tunnel 未移除。未知 job 狀態曾短暫出現後恢復成功，腳本沿用原 job identity 查詢，沒有盲目重送。

原先五組人物／多角度／CFG 實驗另外完成 18／35 張，剩下 17 張尚未由本腳本自動接續。
這輪 batch 對照不自動算入那 17 張，也不能宣稱原研究全部結束。

## 9. 下一步：建議而非已完成

1. 將 tracing identity 與內容快取分離；比較編碼 cache miss／hit，保留正確的計時歸屬。
2. 實測每個 step 的模型 forward 次數、有效 batch shape、CFG 分組及 kernel／搬運時間。
3. 再評估 encoder 放置、共用 reference VAE 計算或編碼批次化；保持 prompt／seed／輸出配對不變。
4. 在正式 Pod 路徑重做有界配對測試，記錄 CPU、GPU 配額與同卡競爭，不混用環境數據。
5. 多輪重複測量後再決定預設 batch；當前資料不足以推薦以 batch 4 作為加速策略。

結論：本輪驗證了異質 batch 的基本產圖能力，但沒有證明有實用的吞吐量收益。
已知編碼與快取實作問題應與仍未定位的採樣瓶頸分開處理，不能互相替代解釋。
