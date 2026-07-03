# PromptLanguageCodingAccuracy — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | How much does prompt language (English vs Chinese) change LLM coding accuracy, by task type and model generation? |
| Venue | unknown |
| Year | 2024–2026 |
| Authors | unknown |
| Official Code | unknown |
| Venue Kind | survey |

這是一篇跨論文的整合式研究筆記（survey note），回答一個實務問題：**用英文還是中文寫 coding agent 的 prompt，對產出正確率的影響有多大？答案會因為「function 級 vs repo 級任務」與「模型世代」而不同嗎？** 我們不逐篇摘要，而是把四篇主要來源放在同一組比較維度上互相對照。

## 📚 Sources

下表列出本筆記引用的主要來源與取用狀態。所有數字都已在對應論文的 `source/*.tex` 原文中重新定位並記錄於 `ledger.json`；凡本筆記保留的數值，皆以論文原文為準，與 issue 種子稿衝突之處一律以論文為準並在文中標註更正。

| # | Title | Venue | Year | arXiv | Access |
|-|-|-|-|-|-|
| 1 | HumanEval-XL: A Multilingual Code Generation Benchmark | LREC-COLING | 2024 | 2402.16694 | fetched |
| 2 | Exploring Multi-Lingual Bias of Large Code Models in Code Generation | preprint (unknown) | 2024 | 2404.19368 | fetched |
| 3 | From Effectiveness to Efficiency: Uncovering Linguistic Bias in LLM-based Code Generation | preprint (unknown) | 2024 | 2406.00602 | fetched |
| 4 | Mythbuster: Chinese Language Is Not More Efficient Than English in Vibe Coding | preprint (unknown) | 2026 | 2604.14210 | preprint-fetched |

來源 2、3、4 目前僅有 arXiv 預印本（尚未見同儕審查的正式出處），引用時視為預印本，正式版數值可能有差異；其中來源 4 標示作者單位為 Scam.ai，並自陳為「preliminary study」。原任務附帶的次要脈絡來源 `openai.com/index/introducing-gpt-5-5/`（GPT-5.5 公告頁）在擷取時回傳 HTTP 403，無法取用，故排除，也因此本筆記不引用任何來自該頁的 aggregate coding 分數；它原本的用途僅是佐證「目前沒有官方的中英 prompt head-to-head」。

## 核心結論（先講重點）

把四篇證據疊起來看，中文 prompt 相對英文的準確率損失，**方向與大小都是「小而不穩定」，而且高度取決於任務層級與模型**，並不存在「英文一定大贏」的通則：

- 在 **function 級**（單題、函式簽名已給定）的任務上，差距通常落在 0–8 個百分點之間，方向甚至會反轉：在低溫取樣的平均值上中文還略高於英文。
- 在 **repo 級**（SWE-bench Lite 這類需要讀 issue、定位檔案、產生 patch）的任務上，目前公開資料裡英文較常勝，差距約 4.5–9.9 個百分點，但樣本很小、且屬預印本層級的初步證據。
- 對 **現世代旗艦模型**（GPT-5.5 / Claude Opus 等）而言，並沒有任何公開的「同一批任務、英文 vs 中文 prompt」對照實驗，因此任何「現在還是要用英文」的說法都是外推，而非量測。

## Function 級任務：差距小，方向會反轉

HumanEval-XL 用 80 題平行題目、跨 23 種自然語言與 12 種程式語言，量測 pass@1。它最適合檢驗 issue 種子稿裡「GPT-4 在 Go 上中文反而較高（中文 67.50% vs 英文 63.75%）」這個說法。實際查對論文附錄的 Go 表格後，這個說法**不成立**：GPT-4 在 Go 上英文與中文都是 47.50 pass@1（完全相同），GPT-3.5 則兩者都是 2.50；種子稿引用的 63.75 / 67.50 與 7.50 / 6.25 並不在該表中。

X-HumanEval-X（來源 2）把觀察拉到一般 code generation。作者在九個 code LLM（StarCoder、CodeLlama、DeepSeek-Coder 各三種尺寸）上報告：把指令從英文換成中文，pass@1「至少下降 13%」。要小心這裡的 13% 是**相對降幅**，且是相對於較低的中文基準計算的偏差量 (EN−ZH)/ZH，不是絕對百分點。以 base models 的平均值看：Python 英文 37.32 → 中文 31.84（相對 Δ17.25%、絕對約 5.5 pp）、C++ 34.88 → 30.69（Δ13.65%、約 4.2 pp）、Java 33.47 → 25.74（Δ30.03%、約 7.7 pp）。換算成絕對百分點，差距約 4.2–7.7 pp。

第三個角度來自 linguistic-bias（來源 3）：52 題中英平行 Python 題、十個模型（八個開源家族加 GPT-3.5-Turbo、GPT-4）。它有個關鍵細節推翻了「英文一定較好」的直覺：**平均正確率在低溫（t=0.2）時中文 0.58 反而略高於英文 0.56，只有在高溫（t=0.8）才變成英文 0.61 高於中文 0.59**。GPT-4 單獨看則兩溫度都偏英文（t=0.2 為 0.65 vs 0.63、t=0.8 為 0.71 vs 0.65）。作者同時報告，平均約 12% 的題目會出現「一語言對、另一語言錯」的不一致，另有 39% 出現效率（複雜度）差異。

| 來源／設定 | 指標 | 英文 | 中文 | 差距 |
|-|-|-|-|-|
| HumanEval-XL · Go · GPT-4 | pass@1 | 47.50 | 47.50 | 0.0 pp（相同） |
| HumanEval-XL · Go · GPT-3.5 | pass@1 | 2.50 | 2.50 | 0.0 pp（相同） |
| X-HumanEval-X · base 平均 · Python | pass@1 | 37.32 | 31.84 | 5.5 pp |
| X-HumanEval-X · base 平均 · Java | pass@1 | 33.47 | 25.74 | 7.7 pp |
| linguistic-bias · 全體平均 · t=0.2 | correctness | 0.56 | 0.58 | −2 pp（中文較高） |
| linguistic-bias · GPT-4 · t=0.8 | correctness | 0.71 | 0.65 | 6 pp |

## 一個走到底的實例：Go 的「中文較強」其實是平手

以 issue 種子稿最搶眼的宣稱為例，端到端走一次。種子稿說：GPT-4 在 HumanEval-XL 的 Go 子集上，中文 67.50% 高於英文 63.75%，因此「Go 任務沒有足夠證據說英文優於中文，甚至中文較好」。查 `humaneval-xl` 快取的附錄 Go 表（`\label{tab:appendix_go}`）：GPT-4 的 English 列與 Chinese 列在 Go 欄位都是 47.50，GPT-3.5 都是 2.50。也就是說，正確的讀法不是「中文高 3.75 pp」，而是「在這張表上兩者完全相同」；而且 Go 對所有模型都是低分程式語言（GPT-4 跨 23 語言只在 38.75–50.00 之間），這個「相同」更可能反映的是題目本身難、天花板低，而非中文有優勢。結論方向雖仍成立（Go 沒有證據顯示英文大勝），但支撐它的具體數字被更正為一個平手，而不是中文領先——這正是為什麼不能把種子稿的數字直接搬進筆記。

## Repo 級任務：英文目前較穩，但只是初步證據

Mythbuster（來源 4）是唯一觸及 repo 級真實工程任務的來源。它在 SWE-bench Lite 抽 50 題，比較三個模型的英文與中文 prompt 解題率（resolution rate）：MiniMax-2.7 英文 66.0% vs 中文 61.5%（差 4.5 pp）、GPT-5.4-mini 36.0% vs 26.1%（差 9.9 pp）、GLM-5 64.6% vs 55.1%（差 9.5 pp）。三個模型都是英文較高，量級約 4.5–9.9 pp，和 function 級的差距落在同一個數量級。

但這份證據要打折看：只有 50 題、作者自陳未做顯著性檢定、且中文組可評估題數更少（MiniMax-2.7 中文只有 39 題可評，比英文的 50 題少 22%，因為較長的中文 prompt 更容易觸發 token 上限而產生空 patch），作者明說這可能讓中文解題率「被高估」。同時它測的是 MiniMax-2.7 / GPT-5.4-mini / GLM-5，而非 GPT-5.5 或 Claude Opus 這類現世代旗艦。作者自己的收斂結論其實是：**模型之間的差距（最好與最差差約 30 pp）遠大於語言差距，選模型比選語言重要得多。**

| 模型 · SWE-bench Lite | 英文解題率 | 中文解題率 | 差距 | 註 |
|-|-|-|-|-|
| MiniMax-2.7 | 66.0% | 61.5% | 4.5 pp | 中文僅 39/50 題可評 |
| GPT-5.4-mini | 36.0% | 26.1% | 9.9 pp | 無 reasoning 模式 |
| GLM-5 | 64.6% | 55.1% | 9.5 pp | 中文 token 反而較省 (0.98×) |

## 為什麼會有差異：token 化，而不是「中文比較省」

Mythbuster 的第二個貢獻是拆穿「中文比較省 token、所以更划算」的迷思。它指出 token 成本是**模型（tokenizer）決定的、方向不一**：MiniMax-2.7 中文要 1.28× 的 token，但 GLM-5 中文反而只要 0.98×。用五個 tokenizer 對 23 段 SWE-bench 描述量測，GLM 的 tokenizer 對中文是 0.923 的 ZH/EN 比（中文更省），而 GPT/Llama 的 cl100k_base 對中文多用 15% token。真正重要的是：即使某語言每次嘗試 token 較少，只要解題率較低就得重試，整體「每題成功成本」反而更高——這解釋了為什麼「只看輸入壓縮率」會系統性低估中文的實際成本。這條 tokenizer 機制也和 X-HumanEval-X「把中文指令先翻成英文可縮小偏差」的發現相容：偏差主要來自模型對非英文輸入的理解與編碼，而非中文本身「較難寫程式」。

## 實務建議（本筆記的推論，非任何單一來源的結論）

回到讀者真正的問題「我該用英文寫 coding agent 的 prompt 嗎？」：現有證據支持的答案是「**影響不大、且看情境**」。若你用的是英文偏重的模型、做的是 repo 級任務、又在意穩定度，英文是較安全的預設；若是 function 級小題、或用 CJK 詞表對齊的模型（如 GLM 系），中文的損失可能趨近於零甚至反向。沒有任何一篇來源直接驗證「背景用中文、但把 API／型別／函式簽名／測試條件保留英文」這條混合寫法，因此它只能當成本筆記依 tokenizer 與理解偏差機制所做的合理推論：把最容易被 token 化與翻譯扭曲的技術符號維持英文原樣，通常是低風險的做法，但目前缺乏直接實驗支撐。

## 🧪 Critical Assessment

### 各來源實驗設定的可比性
四篇來源幾乎無法直接相加。它們的指標不同（HumanEval-XL 與 X-HumanEval-X 用 pass@1、linguistic-bias 用 correctness rate、Mythbuster 用 resolution rate）、任務層級不同（function vs repo）、模型世代橫跨 GPT-3.5 到 2026 年的 MiniMax-2.7/GLM-5，語言集合也不同。因此本筆記給的「0–10 pp」是把不同量測放在同一數量級上的粗略歸納，屬於跨論文推論，不是任何單篇的量測值；任何把這些數字當成同一把尺的比較都應謹慎。

### Benchmark 真實性與是否射向自訂靶
三篇 function 級來源都建立在 HumanEval 家族的翻譯版本上，題目短、天花板受限（Go 這種低分語言尤其明顯），能否外推到真實工程並不清楚。X-HumanEval-X 與 linguistic-bias 都是作者自建的中英平行資料集，翻譯品質本身就是混淆變項——所謂「中文較差」有一部分可能是翻譯雜訊而非模型偏差。Mythbuster 雖然用了較真實的 SWE-bench Lite，卻只抽 50 題且中英題數不對齊，等於在一個窄且不平衡的子集上量測。

### 樣本與統計充分性
Mythbuster 自陳未做顯著性檢定，50 題規模下 4.5 pp 的差距很可能落在雜訊內；HumanEval-XL 每個 cell 只有 80 題，pass@1 的最小可分辨粒度是 1.25 pp。這些都讓「英文高 N pp」的宣稱在統計上偏脆弱。

### 對讀者實務問題的外部效度，以及仍未回答的部分
最關鍵的缺口是**世代錯配**：讀者關心的多半是當下在用的旗艦模型，但證據集中在較舊或較小、且英文偏重的模型上。沒有任何來源提供現世代模型「同一批任務、英中對照」的 head-to-head，Mythbuster 也明白警告不要從三個模型過度外推。因此本筆記對「現在仍應偏好英文」只能給到「合理但未證實」的信心度；隨模型 tokenizer 與多語訓練改善，這個結論很可能持續縮小甚至翻轉。

## 🔗 Related notes

<!-- 目前沒有可安全解析的相關筆記連結。 -->
