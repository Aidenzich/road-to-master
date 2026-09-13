# 六小時多角色參考圖實驗：Codex／Qwen Edit／H3

資料更新：2026-09-13T15:50:34.665266+00:00。**持續實驗中，非最終結論。**

## 問題與方法

比較同一組參考角色在 1–5 人、不同互動動作下的外觀保留、數量、對應與手部表現。六小時窗口：台灣時間 2026-09-13 23:41:32 至 2026-09-14 05:41:32；到期後不新增提交，已提交任務完成、檢查與清理另計。

兩類角色 × 五種人數 × 三類動作 × 兩次重複，預先登記 60 個場景；完整窮舉所有角色子集／排列將遠超時窗，因此先做分層覆盖，再依剩餘時間增加單變因對照。未執行、流程不支援、執行失敗與未人工檢查皆分開列出。

三類動作：各自揮手；共同看同一本綠色書（單人為自己看書）；兩兩擊掌（單人為雙手持藍色杯）。比較靜態完成姿勢，並非完整動作序列。H3 要求首幀即持有該姿勢並保持 5 幀；固定眼平構圖。

| 模型 | 實驗配置 | 限制 |
|---|---|---|
| Codex 內建生圖 | 同一語意 prompt + 3:2 構圖要求 | 工具未提供可核實 seed、steps、CFG、GPU 與底層模型版本；實際尺寸逐張記錄 |
| Qwen Image Edit 2511 FP8 mixed | 768×512、40 steps、CFG4、Euler/simple、denoise1、shift3.1、CFGNorm1、batch1 | 現有 adapter／TextEncodeQwenImageEditPlus 接受至多3張獨立參考圖；4–5張原生條件不支援，不代表模型架構永遠不可能 |
| H3 Ref2VA INT8 ConvRot | 768×512、20 steps、Euler/linear_quadratic、5幀、24fps、batch1、無Turbo LoRA | 專用六欄 prompt；5幀低於常規片長，視為取圖實驗。保留全部5幀及MP4，主比較固定第1幀，不挑最佳幀 |

本地模型使用 RTX 5090，共享 ComfyUI 的 vision-flow 裝置策略；H3 low-VRAM／reserve6GiB，Qwen 使用既有 image 策略。工作流快照含實際編碼器裝置選項與硬體資訊。不重啟服務、不全域中斷 GPU。六個場景為一小模型批段並交替模型順序，以降低頻繁換模；時間漂移與cache差異仍是限制。

## 來源與素材限制

- [芙莉蓮官方角色頁](https://frieren-anime.jp/character/chara_group1/1-1/)與[菲倫](https://frieren-anime.jp/character/chara_group1/1-5/)；[2026官方消息](https://frieren-anime.jp/news/)。
- [SPY×FAMILY 官方頁](https://spy-family.net/tvseries/)：Loid、Yor、Anya。
- [Netflix 官方 Wednesday Season 2 角色頁](https://www.netflix.com/tudum/articles/wednesday-season-2-character-cast-guide)：Wednesday、Enid、Bianca、Tyler、Dort。
- [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)列40 steps／CFG4範例；本地量化版本不等於官方BF16配置。

原始下載URL、尺寸與SHA-256見 [reference-receipts.json](assets/multichar-reference-20260913/reference-receipts.json)。官方圖版權屬各權利人，僅作本報告的具名比較與來源證據；不是本專案原創或可再授權素材。生成結果為虛構場景。

動漫素材含透明背景，實際RGB／alpha處理由各adapter不同流程處理，並非完全相同的編碼輸入；真人素材是含大型字母、品牌、特殊效果的官方海報，不是乾淨肖像。原圖姿勢與道具均明確不要求保留，且未用任一比較模型先重畫參考圖。這些素材品質／前處理差異須列為解讀限制；不能將結果只歸因於模型本身。Tyler參考服裝限制手部活動，是額外的姿勢轉移難例。

## 進度與分母

| 模型 | 預登記格數 | 已提交 | 推論／取圖成功 | 執行失敗 | 不支援 | 已目視評估 |
|---|---:|---:|---:|---:|---:|---:|
| Codex built-in | 60 | 1 | 1 | 0 | 0 | 1 |
| Qwen Edit 2511 | 60 | 1 | 1 | 0 | 24 | 1 |
| H3 Ref2VA 5 frames | 60 | 0 | 0 | 0 | 0 | 0 |

評分：0明確失敗、1部分符合或不確定、2明確符合；null未審查／不適用。人工目視評分不是生物辨識身份驗證，也不是盲測或多評審共識。尚未有足夠重複樣本前，不宣稱統計顯著或模型優劣排名。

[完整CSV](assets/multichar-reference-20260913/results.csv) · [JSON](assets/multichar-reference-20260913/results.json) · [預登記計畫](assets/multichar-reference-20260913/plan.json)

## 個別結果

| 場景 | Codex | Qwen | H3 |
|---|---|---|---|
| anime-01-wave-r1 | pending | pending | pending |
| live-01-wave-r1 | pending | pending | pending |
| anime-02-wave-r1 | [succeeded](assets/multichar-reference-20260913/runs/codex/anime-02-wave-r1/result.json) | [succeeded](assets/multichar-reference-20260913/runs/qwen/anime-02-wave-r1/result.json) | pending |
| live-02-wave-r1 | pending | pending | pending |
| anime-03-wave-r1 | pending | pending | pending |
| live-03-wave-r1 | pending | pending | pending |
| anime-04-wave-r1 | pending | unsupported | pending |
| live-04-wave-r1 | pending | unsupported | pending |
| anime-05-wave-r1 | pending | unsupported | pending |
| live-05-wave-r1 | pending | unsupported | pending |
| anime-01-book-r1 | pending | pending | pending |
| live-01-book-r1 | pending | pending | pending |
| anime-02-book-r1 | pending | pending | pending |
| live-02-book-r1 | pending | pending | pending |
| anime-03-book-r1 | pending | pending | pending |
| live-03-book-r1 | pending | pending | pending |
| anime-04-book-r1 | pending | unsupported | pending |
| live-04-book-r1 | pending | unsupported | pending |
| anime-05-book-r1 | pending | unsupported | pending |
| live-05-book-r1 | pending | unsupported | pending |
| anime-01-contact-r1 | pending | pending | pending |
| live-01-contact-r1 | pending | pending | pending |
| anime-02-contact-r1 | pending | pending | pending |
| live-02-contact-r1 | pending | pending | pending |
| anime-03-contact-r1 | pending | pending | pending |
| live-03-contact-r1 | pending | pending | pending |
| anime-04-contact-r1 | pending | unsupported | pending |
| live-04-contact-r1 | pending | unsupported | pending |
| anime-05-contact-r1 | pending | unsupported | pending |
| live-05-contact-r1 | pending | unsupported | pending |
| anime-01-wave-r2 | pending | pending | pending |
| live-01-wave-r2 | pending | pending | pending |
| anime-02-wave-r2 | pending | pending | pending |
| live-02-wave-r2 | pending | pending | pending |
| anime-03-wave-r2 | pending | pending | pending |
| live-03-wave-r2 | pending | pending | pending |
| anime-04-wave-r2 | pending | unsupported | pending |
| live-04-wave-r2 | pending | unsupported | pending |
| anime-05-wave-r2 | pending | unsupported | pending |
| live-05-wave-r2 | pending | unsupported | pending |
| anime-01-book-r2 | pending | pending | pending |
| live-01-book-r2 | pending | pending | pending |
| anime-02-book-r2 | pending | pending | pending |
| live-02-book-r2 | pending | pending | pending |
| anime-03-book-r2 | pending | pending | pending |
| live-03-book-r2 | pending | pending | pending |
| anime-04-book-r2 | pending | unsupported | pending |
| live-04-book-r2 | pending | unsupported | pending |
| anime-05-book-r2 | pending | unsupported | pending |
| live-05-book-r2 | pending | unsupported | pending |
| anime-01-contact-r2 | pending | pending | pending |
| live-01-contact-r2 | pending | pending | pending |
| anime-02-contact-r2 | pending | pending | pending |
| live-02-contact-r2 | pending | pending | pending |
| anime-03-contact-r2 | pending | pending | pending |
| live-03-contact-r2 | pending | pending | pending |
| anime-04-contact-r2 | pending | unsupported | pending |
| live-04-contact-r2 | pending | unsupported | pending |
| anime-05-contact-r2 | pending | unsupported | pending |
| live-05-contact-r2 | pending | unsupported | pending |

## 圖片對照（包含失败成像，不做優勝挑選）

### anime-02-wave-r1

[共同要求](assets/multichar-reference-20260913/cases/anime-02-wave-r1/prompt.txt) · [H3實際prompt](assets/multichar-reference-20260913/cases/anime-02-wave-r1/h3-prompt.txt)

| Codex | Qwen | H3首幀 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-02-wave-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/anime-02-wave-r1/output.png) | 未產出／待執行 |

## 重現與失敗歸類

場景與角色對應可由 [prepare_cases.py](assets/multichar-reference-20260913/prepare_cases.py) 重建；[gpu_runner.py](assets/multichar-reference-20260913/gpu_runner.py) 使用現有 Veritas adapter、PostgreSQL 的自有 schema 與本機清理 journal，需自行提供本地服務配置（此PR不含env或密鑰）。腳本含作者環境路徑，移植時須調整，不能當作通用一鍵執行套件。Codex 使用內建 image_gen 逐張呼叫，實際prompt与來源順序保存在各run.json，不宣稱可由seed重現。

H3 5幀取圖保存原始MP4；若音軌0.20秒短於5/24秒，現有一般影片collector會拒絕影音等長檢查。此時分別記錄provider成功與catalog失敗，從已驗證的本機journal影片取圖；不重試、不補幀、不放寬產品校驗。只有實際解出5幀才記為取圖成功。所有已完成實验的遠端輸入／輸出需有hash比對與清理收據，不能用刪整個資料夾代替。

