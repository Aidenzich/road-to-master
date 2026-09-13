# 多角色參考圖生成比較：Codex／Qwen Edit／H3

## 摘要

本研究比較2–5名動漫／影集角色在揮手、共同閱讀與兩兩擊掌場景中的參考圖遵循。基準資料包含101張生成結果，另有13張對照結果。角色對應、外觀細節與互動動作分開評估，不以單一總分排列模型。

- 在相同的8組三人場景，Codex與H3的角色對應均為8/8明確符合，Qwen為2/8；全部動作要求分別為3/8、2/8、0/8。
- 可辨認角色不等於保留所有細節。服裝配件、指定左右手、書頁接觸位置與旁觀者狀態，是主要差異。
- H3在本次多人條件的角色對應較穩定，但4–5張參考圖的總耗時中位數約9–10分鐘。此為5幀取圖，不代表長影片的角色或時間一致性。
- Qwen完整第一參考圖的單一對照改善可見細節；尚不足以推論普遍效果。

## 實驗設計

兩類角色 × 四種人數（2–5人）× 三類動作 × 兩次生成，共48個場景、144個模型條件。各場景使用相同來源與順序的角色參考圖；H3使用專用Ref2VA描述。

| 模型 | 設定 |
|---|---|
| Codex | 內建image_gen；要求3:2，實際1536×1024；底層版本、seed、steps、CFG未知 |
| Qwen Edit 2511 | FP8 mixed、768×512、40steps、CFG4、Euler/simple、denoise1、shift3.1、CFGNorm1、batch1 |
| H3 Ref2VA | INT8 ConvRot、768×512、20steps、Euler/linear_quadratic、5幀／24fps、batch1、無Turbo；固定比較第1幀 |

本地模型使用RTX5090；H3採low-VRAM／reserve6GiB策略。Qwen現有adapter最多接受3張獨立參考圖，因此4–5人條件不執行原生比較。未知或未執行條件不填零分。

動作：各自單手揮手；第一人雙手持綠色書、第二人指指定書頁、其餘人觀看；依序兩兩擊掌、奇數人數的最後一人旁觀。

| 模型 | 條件數 | 已生成並評估 | 不支援 | 未執行 |
|---|---:|---:|---:|---:|
| Codex | 48 | 48 | 0 | 0 |
| Qwen Edit 2511 | 48 | 16 | 24 | 8 |
| H3 Ref2VA 取幀 | 48 | 37 | 0 | 11 |

[研究配置](assets/multichar-reference-20260913/study-design.json) · [逐筆評分](assets/multichar-reference-20260913/results.csv)

## 評分與限制

0＝明確不符，1＝部分符合或不確定，2＝明確符合。外觀指整組角色可見特徵及服裝細節；角色對應指參考角色與位置／分配相符。手部解剖合理不代表執行了正確動作。評分為單一助理目視，非盲測、非生物辨識，兩次生成不足以估計統計顯著性。

各模型尺寸、量化、參考圖前處理與可控制seed不同，結果不能視為等算力模型排名。動漫參考圖有透明背景，影集參考圖包含海報文字與特殊效果；未見表面也沒有多視角真值。Qwen第一張參考圖的中心裁切是解讀限制。8筆Codex多角色揮手請求缺少提交時參考圖雜湊，來源追溯完整性有限。

## 同場景比較

三模型均生成並完成評估的交集為16個場景。以下只比較此交集，仍有完整案例選擇偏差。

| 人數 | 模型 | 人數符合 | 外觀完整 | 角色對應 | 全部動作 | 手部合理 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Codex | 8/8 | 6/8 | 8/8 | 4/8 | 8/8 |
| 2 | Qwen Edit 2511 | 8/8 | 0/8 | 6/8 | 2/8 | 7/8 |
| 2 | H3 Ref2VA 取幀 | 8/8 | 0/8 | 8/8 | 1/8 | 5/8 |
| 3 | Codex | 8/8 | 0/8 | 8/8 | 3/8 | 8/8 |
| 3 | Qwen Edit 2511 | 6/8 | 0/8 | 2/8 | 0/8 | 3/8 |
| 3 | H3 Ref2VA 取幀 | 8/8 | 0/8 | 8/8 | 2/8 | 5/8 |

## 各人數的完整可用資料

| 人數 | 模型 | 人數符合 | 外觀完整 | 角色對應 | 全部動作 | 手部合理 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Codex | 12/12 | 9/12 | 12/12 | 4/12 | 12/12 |
| 2 | Qwen Edit 2511 | 8/8 | 0/8 | 6/8 | 2/8 | 7/8 |
| 2 | H3 Ref2VA 取幀 | 10/10 | 0/10 | 10/10 | 1/10 | 7/10 |
| 3 | Codex | 12/12 | 0/12 | 12/12 | 3/12 | 12/12 |
| 3 | Qwen Edit 2511 | 6/8 | 0/8 | 2/8 | 0/8 | 3/8 |
| 3 | H3 Ref2VA 取幀 | 10/10 | 0/10 | 10/10 | 2/10 | 7/10 |
| 4 | Codex | 12/12 | 0/12 | 12/12 | 4/12 | 12/12 |
| 4 | Qwen Edit 2511 | 未評估 | 未評估 | 未評估 | 未評估 | 未評估 |
| 4 | H3 Ref2VA 取幀 | 9/9 | 0/9 | 9/9 | 0/9 | 6/9 |
| 5 | Codex | 12/12 | 0/12 | 12/12 | 3/12 | 12/12 |
| 5 | Qwen Edit 2511 | 未評估 | 未評估 | 未評估 | 未評估 | 未評估 |
| 5 | H3 Ref2VA 取幀 | 8/8 | 0/8 | 8/8 | 2/8 | 4/8 |

[完整分組數據](assets/multichar-reference-20260913/quality-summary.json) · [兩次生成的配對數據](assets/multichar-reference-20260913/repeat-summary.json)

## 耗時

以下為成功基準組總耗時中位數；Codex僅有內建工具牆鐘時間，本地模型包含排隊與保存，不可直接用來比較算力效率。VRAM是提交前快照而非峰值。

| 參考圖數 | Codex工具秒數 | Qwen總秒數 | H3總秒數 |
|---:|---:|---:|---:|
| 2 | 38.0（n=12） | 129.4（n=8） | 164.4（n=10） |
| 3 | 38.6（n=12） | 194.6（n=8） | 416.7（n=10） |
| 4 | 41.5（n=12） | 不適用 | 555.9（n=9） |
| 5 | 42.9（n=12） | 不適用 | 595.7（n=8） |

[逐筆階段耗時](assets/multichar-reference-20260913/timings.csv)

## 圖片對照

### anime-02-wave-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-02-wave-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-02-wave-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-02-wave-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/anime-02-wave-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/anime-02-wave-r1/output.png) |

### live-02-wave-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-02-wave-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-02-wave-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-02-wave-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/live-02-wave-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/live-02-wave-r1/output.png) |

### anime-03-wave-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-03-wave-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-03-wave-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-03-wave-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/anime-03-wave-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/anime-03-wave-r1/output.png) |

### live-03-wave-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-03-wave-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-03-wave-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-03-wave-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/live-03-wave-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/live-03-wave-r1/output.png) |

### anime-04-wave-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-04-wave-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-04-wave-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-04-wave-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-04-wave-r1/output.png) |

### live-04-wave-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-04-wave-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-04-wave-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-04-wave-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/live-04-wave-r1/output.png) |

### anime-05-wave-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-05-wave-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-05-wave-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-05-wave-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-05-wave-r1/output.png) |

### live-05-wave-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-05-wave-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-05-wave-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-05-wave-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/live-05-wave-r1/output.png) |

### anime-02-book-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-02-book-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-02-book-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-02-book-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/anime-02-book-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/anime-02-book-r1/output.png) |

### live-02-book-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-02-book-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-02-book-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-02-book-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/live-02-book-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/live-02-book-r1/output.png) |

### anime-03-book-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-03-book-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-03-book-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-03-book-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/anime-03-book-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/anime-03-book-r1/output.png) |

### live-03-book-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-03-book-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-03-book-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-03-book-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/live-03-book-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/live-03-book-r1/output.png) |

### anime-04-book-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-04-book-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-04-book-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-04-book-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-04-book-r1/output.png) |

### live-04-book-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-04-book-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-04-book-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-04-book-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/live-04-book-r1/output.png) |

### anime-05-book-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-05-book-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-05-book-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-05-book-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-05-book-r1/output.png) |

### live-05-book-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-05-book-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-05-book-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-05-book-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/live-05-book-r1/output.png) |

### anime-02-contact-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-02-contact-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-02-contact-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-02-contact-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/anime-02-contact-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/anime-02-contact-r1/output.png) |

### live-02-contact-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-02-contact-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-02-contact-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-02-contact-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/live-02-contact-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/live-02-contact-r1/output.png) |

### anime-03-contact-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-03-contact-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-03-contact-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-03-contact-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/anime-03-contact-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/anime-03-contact-r1/output.png) |

### live-03-contact-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-03-contact-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-03-contact-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-03-contact-r1/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/live-03-contact-r1/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/live-03-contact-r1/output.png) |

### anime-04-contact-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-04-contact-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-04-contact-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-04-contact-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-04-contact-r1/output.png) |

### live-04-contact-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-04-contact-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-04-contact-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-04-contact-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/live-04-contact-r1/output.png) |

### anime-05-contact-r1

[通用提示詞](assets/multichar-reference-20260913/cases/anime-05-contact-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-05-contact-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-05-contact-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-05-contact-r1/output.png) |

### live-05-contact-r1

[通用提示詞](assets/multichar-reference-20260913/cases/live-05-contact-r1/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-05-contact-r1/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-05-contact-r1/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/live-05-contact-r1/output.png) |

### anime-02-wave-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-02-wave-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-02-wave-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-02-wave-r2/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/anime-02-wave-r2/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/anime-02-wave-r2/output.png) |

### live-02-wave-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-02-wave-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-02-wave-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-02-wave-r2/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/live-02-wave-r2/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/live-02-wave-r2/output.png) |

### anime-03-wave-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-03-wave-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-03-wave-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-03-wave-r2/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/anime-03-wave-r2/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/anime-03-wave-r2/output.png) |

### live-03-wave-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-03-wave-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-03-wave-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-03-wave-r2/output.png) | ![qwen](assets/multichar-reference-20260913/runs/qwen/live-03-wave-r2/output.png) | ![h3](assets/multichar-reference-20260913/runs/h3/live-03-wave-r2/output.png) |

### anime-04-wave-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-04-wave-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-04-wave-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-04-wave-r2/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-04-wave-r2/output.png) |

### live-04-wave-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-04-wave-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-04-wave-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-04-wave-r2/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/live-04-wave-r2/output.png) |

### anime-05-wave-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-05-wave-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-05-wave-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-05-wave-r2/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-05-wave-r2/output.png) |

### live-05-wave-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-05-wave-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-05-wave-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-05-wave-r2/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/live-05-wave-r2/output.png) |

### anime-02-book-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-02-book-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-02-book-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-02-book-r2/output.png) | 未執行 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-02-book-r2/output.png) |

### live-02-book-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-02-book-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-02-book-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-02-book-r2/output.png) | 未執行 | ![h3](assets/multichar-reference-20260913/runs/h3/live-02-book-r2/output.png) |

### anime-03-book-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-03-book-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-03-book-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-03-book-r2/output.png) | 未執行 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-03-book-r2/output.png) |

### live-03-book-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-03-book-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-03-book-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-03-book-r2/output.png) | 未執行 | ![h3](assets/multichar-reference-20260913/runs/h3/live-03-book-r2/output.png) |

### anime-04-book-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-04-book-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-04-book-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-04-book-r2/output.png) | 不支援 | ![h3](assets/multichar-reference-20260913/runs/h3/anime-04-book-r2/output.png) |

### live-04-book-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-04-book-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-04-book-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-04-book-r2/output.png) | 不支援 | 未執行 |

### anime-05-book-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-05-book-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-05-book-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-05-book-r2/output.png) | 不支援 | 未執行 |

### live-05-book-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-05-book-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-05-book-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-05-book-r2/output.png) | 不支援 | 未執行 |

### anime-02-contact-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-02-contact-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-02-contact-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-02-contact-r2/output.png) | 未執行 | 未執行 |

### live-02-contact-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-02-contact-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-02-contact-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-02-contact-r2/output.png) | 未執行 | 未執行 |

### anime-03-contact-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-03-contact-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-03-contact-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-03-contact-r2/output.png) | 未執行 | 未執行 |

### live-03-contact-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-03-contact-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-03-contact-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-03-contact-r2/output.png) | 未執行 | 未執行 |

### anime-04-contact-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-04-contact-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-04-contact-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-04-contact-r2/output.png) | 不支援 | 未執行 |

### live-04-contact-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-04-contact-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-04-contact-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-04-contact-r2/output.png) | 不支援 | 未執行 |

### anime-05-contact-r2

[通用提示詞](assets/multichar-reference-20260913/cases/anime-05-contact-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/anime-05-contact-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/anime-05-contact-r2/output.png) | 不支援 | 未執行 |

### live-05-contact-r2

[通用提示詞](assets/multichar-reference-20260913/cases/live-05-contact-r2/prompt.txt) · [H3提示詞](assets/multichar-reference-20260913/cases/live-05-contact-r2/h3-prompt.txt)

| Codex | Qwen | H3 |
|---|---|---|
| ![codex](assets/multichar-reference-20260913/runs/codex/live-05-contact-r2/output.png) | 不支援 | 未執行 |

## 對照：完整第一張參考圖

保持提示詞、seed、參考圖順序與採樣參數不變，只把Qwen正／負文字編碼的第一參考圖接回完整載入圖；輸出latent尺寸保持不變。此案例人物細節改善，樣本數1，不推論普遍效果。

| 原流程 | 完整第一參考圖 |
|---|---|
| ![baseline](assets/multichar-reference-20260913/runs/qwen/anime-02-wave-r1/output.png) | ![fullref](assets/multichar-reference-20260913/runs/qwen-fullref/anime-02-wave-r1/output.png) |

## 對照：空間排列反轉

12組Codex對照只反轉提示詞中的左右排列，參考圖上傳順序與角色動作分工不變。12組人數及角色對應均明確符合；動作細節仍部分符合。Codex seed不可控制，因此不是固定噪聲因果試驗。

### anime-02-book-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/anime-02-book-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/anime-02-book-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/anime-02-book-r1-reverse-position/output.png) |

### live-02-book-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/live-02-book-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/live-02-book-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/live-02-book-r1-reverse-position/output.png) |

### anime-03-book-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/anime-03-book-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/anime-03-book-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/anime-03-book-r1-reverse-position/output.png) |

### live-03-book-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/live-03-book-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/live-03-book-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/live-03-book-r1-reverse-position/output.png) |

### anime-05-book-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/anime-05-book-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/anime-05-book-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/anime-05-book-r1-reverse-position/output.png) |

### live-05-book-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/live-05-book-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/live-05-book-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/live-05-book-r1-reverse-position/output.png) |

### anime-02-contact-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/anime-02-contact-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/anime-02-contact-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/anime-02-contact-r1-reverse-position/output.png) |

### live-02-contact-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/live-02-contact-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/live-02-contact-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/live-02-contact-r1-reverse-position/output.png) |

### anime-03-contact-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/anime-03-contact-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/anime-03-contact-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/anime-03-contact-r1-reverse-position/output.png) |

### live-03-contact-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/live-03-contact-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/live-03-contact-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/live-03-contact-r1-reverse-position/output.png) |

### anime-05-contact-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/anime-05-contact-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/anime-05-contact-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/anime-05-contact-r1-reverse-position/output.png) |

### live-05-contact-r1-reverse-position

[對照提示詞](assets/multichar-reference-20260913/cases/live-05-contact-r1-reverse-position/prompt.txt)

| 原排列 | 反轉排列 |
|---|---|
| ![base](assets/multichar-reference-20260913/runs/codex/live-05-contact-r1/output.png) | ![reverse](assets/multichar-reference-20260913/runs/codex/live-05-contact-r1-reverse-position/output.png) |

## 參考來源

- [葬送的芙莉蓮官方角色頁](https://frieren-anime.jp/character/chara_group1/1-1/)
- [SPY×FAMILY官方](https://spy-family.net/tvseries/)
- [Netflix Wednesday角色介紹](https://www.netflix.com/tudum/articles/wednesday-season-2-character-cast-guide)
- [Qwen Image Edit 2511模型卡](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [來源檔案與雜湊](assets/multichar-reference-20260913/reference-receipts.json)

參考圖版權屬各權利人；生成圖片為虛構場景。本報告收錄固定比較圖片、提示詞及研究數據；原始影片與執行期證據另存本機，不屬於Git發布內容。
