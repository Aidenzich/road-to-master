# Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo — Research Note
> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Teaching LLMs a Low-Resource Language: Enhancing Code Completion in Pharo |
| Venue | unknown |
| Year | 2026 |
| Authors | Kilian Kier, Alessandro Giagnorio, Omar AbedelKader, Oleksandr Zaitsev, Robert Peharz, Romain Robbes, Gabriele Bavota, Stéphane Ducasse |
| Official Code | https://doi.org/10.5281/zenodo.18833238 |
| Venue Kind | paper |

> 本文基於 arXiv preprint `2607.04939`（IEEEtran conference 格式，尚未見同儕審查發表資訊，故 Venue 標為 `unknown`）。正式發表版本可能與此有差異。

## First Principles

### 問題：當一個語言只有 ~2k 個 GitHub repo

主流 code LLM 的訓練假設是「網路上有海量該語言的程式碼」。Python 在 GitHub 上有約 26M 個公開 repo（作者以 2026 年 2 月 18 日的 GitHub 進階搜尋為準）。本文的研究對象 **Pharo**（一種 Smalltalk 方言）只有約 2k 個公開 repo——比 Python 少了整整四個數量級，也比過去文獻常稱為「low-resource」的 Lua（620k）、Julia（85k）、Racket（23k）還要再少一個數量級以上。這種極端資料稀缺直接反映在工具落差上：Pharo 的 IDE 至今只有 single-token 補全，遠不及主流 IDE 的多 token、脈絡感知補全。

但作者強調，Pharo 難的不只是「資料少」，而是三個因素疊加：

1. **Tonel 檔案格式混合了程式碼與 metadata。** Pharo 開發者平時不直接操作檔案，程式碼存進 git 時才序列化成 Tonel 格式。一個 Tonel 檔案把「與 Smalltalk 方言無關的 class 定義」和「按字母排序的 methods」寫在一起，且 class 定義用的語法與 IDE／文件裡的不同。模型若直接吃這種格式，容易學到把「打包用的 metadata」誤認為可執行語法的錯誤模式。

```smalltalk
"
Class comment
"
Class {
    #name : 'ClassName',
    #superclass : 'SuperClassName',
    #instVars : [ 'var1', 'var2', ... ],
    #classVars : [ 'default', 'current', ... ],
    #category : 'CategoryName',
    #package : 'PackageName',
    #tag: 'Tag'
}

{ #category : 'MethodCategory' }
ClassName >> methodSelector [
    " Method comment"
    MethodBody
]
```
*Tonel 檔案的一般結構（重繪自論文 Figure 1）：上半的 `Class { ... }` 是宣告 metadata，下半才是實際 method。*

2. **Smalltalk 語法本身就和主流語言差很遠。** 控制流程如 `if`、`while` 不是語法關鍵字，而是普通的 method（訊息）；method 呼叫是 keyword message，參數與方法名交錯，例如 `at: aSymbol ifAbsentPut: aBlock`；語句以句點 `.` 分隔而非分號。這些特性讓從高資源語言遷移學習變得困難。

### 端到端特化流程

作者的核心主張是：**在極端稀缺的資料下，「特化小模型」比「直接用巨大通用模型」更務實**，因為目標是能在開發者機器上跑、滿足即時延遲的 in-IDE 補全，而非離線的 code generation。流程分三段。

**(1) 資料策展。** 從帶有 `pharo` topic、且採 MIT 授權的 GitHub repo 出發（748 個），再用「能否匯入 Pharo 10–14 至少一個環境」與「是否為 Tonel 格式」過濾，降到 415 個 repo。為降低資料污染，以 2024 年 6 月 1 日為切點：之前建立的 repo 拿來訓練，之後的留作 repo-level 評測。作者另外自建了兩個工具——Pygments 的 Pharo lexer 與 tree-sitter 的 Pharo grammar——用來 tokenize 與 parse 出 AST，共抽出 **387,159 個 Pharo method**。

**(2) Continued pre-training（教語法）。** 用 causal language modeling。對 25% 的 method 做完整的左到右預測；對 75% 採 fill-in-the-middle（FIM）：挖掉一段連續 span 讓模型依前後文補回。挖空採 **AST-aware** 策略——隨機挑一個含 3–10 個 token 的 AST 節點來遮罩（太短沒意義、太長太難）。序列化時依模型各自的 FIM 樣板：Qwen 用 prefix–suffix–middle（PSM），Mellum 用 suffix–prefix–middle（SPM）。

**(3) Fine-tuning（對齊真實補全情境）。** 改用 **Random-AST** 遮罩：從 method body 隨機挑一個 token（或 token 片段）當起點，一路遮到所在 statement 節點結束。這模擬「開發者在任意游標位置請求補全」，而非只在漂亮的 AST 邊界。為避免忘掉 pre-training 學到的結構知識，混入 20% 的 AST-aware 樣本做 rehearsal，最終 fine-tuning 資料集為 **324,725** 筆。訓練全程用 LoRA（alpha=32, r=16, dropout=0.05），序列長 2,048，學習率 5×10⁻⁵，AdamW，通常 3 個 epoch 收斂。

被特化的是兩個「小」開源模型家族：Qwen2.5 Coder Base（0.5B / 1.5B / 3B / 7B）與 Mellum-base（4B）。對照組除了它們自己的 base 版，還有兩個「巨型」通用模型：Qwen3 Coder 480B A35B Instruct，以及 Claude Sonnet 4.5。

### 兩類 benchmark

作者自建了評測套件，分兩層：

- **Method-level（考語法，可執行測試）：** 把 HumanEval+ 的 164 題用 GPT-4o 初譯成 Pharo、再由人工校對，另收 Exercism Pharo track 的 47 題，合計 211 題。每題保留 canonical solution 與 test；隨機遮罩 canonical solution 的一段，讓模型補回後**跑測試**判對錯，用 `pass@1`（temperature=0.2，每題重複 20 次）。每題再分 AST-aware 與 Random-AST（r-AST）兩種遮罩。
- **Repo-level（考真實情境，相似度）：** 從 22 個測試 repo 的 488 個開發者 commit 挖出改動，取開發者新增的 AST 節點遮罩一段（每個受影響 method 最多遮 3 次），共 2,185 個任務。因無測試可跑，改用 **ChrF** 與 **CrystalBLEU** 衡量與真實程式碼的相似度。

各 benchmark 的 FIM 任務數：

| 層級 | Benchmark | # FIM 任務 |
|-|-|-:|
| Method-level | HumanEval+ AST-aware | 2,274 |
| Method-level | HumanEval+ r-AST | 990 |
| Method-level | Exercism AST-aware | 1,272 |
| Method-level | Exercism r-AST | 551 |
| Repo-level | 各 context 策略（No/Class/Package/Impacted）各 | 2,185 |

### 一個具體的前向例子

以 **Qwen2.5 Coder 3B** 在 HumanEval+ AST-aware 上為例：base checkpoint 的 `pass@1` 為 71.48%，經 pre-training + fine-tuning（SFT）後升到 **83.73%（+12.25）**，Odds Ratio = 3.81，代表產生正確補全的勝算約為原本的 3 倍。

一個更細的失敗轉正例子是 `TripleSumToZero`（HumanEval+ AST-aware，id `humanevalplus-40-10`）：任務本身很簡單（判斷集合中是否有三個元素相加為零），難點在於要還原正確的括號與 Pharo 的訊息優先權。被遮的片段需要補出 `(aCollection at: i)`。特化後的 Qwen2.5 Coder 7B-SFT 20 次全對；而所有 base 模型都會破壞括號結構（如寫成 `aCollection at: i)`）或改變求值順序，造成語法錯誤。這呼應了作者對失敗原因的統計：base 模型的失敗有 65.6% 是語法錯誤，其次才是非預期例外（17.9%）與斷言失敗（16.5%）；兩段式訓練平均把語法錯誤降低了 33%。

### 主要結果

**Method-level（`pass@1`，節錄）：**

| Model | HE+ AST | HE+ r-AST | Exercism AST | Exercism r-AST |
|-|-:|-:|-:|-:|
| Qwen2.5 Coder 7B (base) | 71.12 | 45.24 | 68.58 | 36.12 |
| **Qwen2.5 Coder 7B - SFT** | **89.04** | **52.76** | **85.84** | **42.95** |
| Qwen2.5 Coder 3B - SFT | 83.73 | 51.83 | 78.11 | 41.81 |
| Qwen3 Coder 480B Instruct | 91.95 | 45.13 | 90.35 | 36.92 |
| Claude 4.5 Sonnet | 95.07 | 51.53 | 91.75 | 41.11 |

值得注意的是分裂的圖像：在較「乾淨」的 AST-aware 上，巨型模型（Claude 95.07、Qwen3 480B 91.95）仍勝過特化 7B（89.04）；但在較貼近真實的 r-AST 上情勢反轉——特化的 3B / 7B 不只**超過** Qwen3 Coder 480B（後者參數約多 60–320 倍，r-AST 僅 45.13 / 36.92），連 Claude 4.5 Sonnet（51.53 / 41.11）也被壓過：Qwen2.5 Coder 7B-SFT 在 HumanEval+ r-AST 拿 52.76、Exercism r-AST 拿 42.95，3B-SFT 為 51.83 / 41.81，兩者在兩個 r-AST 欄位都高於 Claude。

**Repo-level（Qwen2.5 Coder 7B-SFT，不同 context）：** 從「無 context」到「提供同一 commit 內其他被改的 method（impacted methods）」，ChrF 從 60.05% 升到 75.96%（+15.91），CrystalBLEU 從 35.96% 升到 58.99%（+23.03）。作者還設了一個 control：隨機挑同樣數量的 method 當 context（random methods）——它在 48 個 model-metric 比較中有 37 個優於只給 class/package 簽章（並非全面勝出），但始終不如 impacted methods。結論是**脈絡的相關性比脈絡的量更重要**。Claude 4.5 Sonnet 仍是全場最佳（impacted methods 下 ChrF 83.02%、CrystalBLEU 70.52%），但作者指出它在無 context 時就有偏高分數，不排除訓練資料污染。

**延遲與量化：** 7B 模型用 llama.cpp 的 Q4_K_M 量化，記憶體從 14.19 GiB 降到 4.36 GiB（約 −70%），method-level `pass@1` 平均只掉 0.61%。延遲上，量化 7B 在 M3/M4 Max CPU 約 1.33 秒、在消費級 GPU RX 7800XT 約 0.53 秒；未量化的 3B 更快（0.62–0.73 秒）。7B 的 1.3 秒略高於互動補全常見的「次秒」門檻，但已接近可用。

## 🧪 Critical Assessment

### 問題是真實的，但「贏過 60× 大模型」是被挑過的框架

Pharo 只有 single-token 補全、社群確實缺工具，這個痛點是真的。特化能大幅拉高 base 模型也是紮實的（多數格子有統計顯著的綠色底線標記）。但論文摘要與 intro 反覆強調的「特化小模型勝過大 60 倍以上的 code LLM」需要放回脈絡看：這個「勝出」**只發生在 r-AST 與 repo-level**，而在 AST-aware 上巨型模型仍領先；而且被比下去的 Qwen3 Coder 480B 是 **Instruct** 模型，未必擅長論文所用的特定 FIM 補全格式，這讓「小勝大」有一部分是「專用格式 vs 通用指令模型」的不對稱，而非單純的能力差距。Claude 4.5 Sonnet 只在 AST-aware 欄位（HumanEval+ 95.07、Exercism 91.75）穩居第一；在較貼近真實的 r-AST 欄位，它（51.53 / 41.11）反而被特化 7B（52.76 / 42.95）與 3B（51.83 / 41.81）超過。因此更準確的敘述應是「特化能讓小模型在特定補全格式上逼近甚至局部超越通用大模型」，而非籠統的「小勝大」，也不是籠統的「大模型全勝」。

### 自建 benchmark 與相似度指標的雙重隱憂

Method-level 用可執行測試判對錯，這點值得肯定，避免了用自訂相似度指標自畫靶心、自圓其說的評測。但 HumanEval+ 是先由 GPT-4o 自動翻譯再人工校對而來的——翻譯品質、以及「把 generation benchmark 硬轉成 completion」引入的偏差，都可能讓分數不完全等同於原生 Pharo 能力。更關鍵的是 **repo-level 完全沒有可執行測試**，只用 ChrF／CrystalBLEU 衡量與開發者原始碼的字面相似度。作者自己也承認：相似度高不代表語意正確。相似度指標傾向獎勵「表面接近」的補全，一個語意錯但用字相近的補全可能拿到虛高分數；反之語意對但寫法不同的補全會被低估。因此 repo-level 的絕對數字（如 75.96% ChrF）不宜被讀成「補全正確率」。

### 對照設計的縫隙

有兩個對照被省略。其一，作者只評 continued-pre-trained 及其 fine-tuned 版，**沒有 fine-tuning-only 的對照**——他們以「先前研究指出 FIM 能力主要在 pre-training 習得」為由跳過，但這正是本文自己的流程主張，缺這個對照就難以量化 pre-training 那一段到底貢獻多少。其二，資料污染只對自訓小模型做了 8-gram 去重與時間切點，對 Claude／Qwen3 這兩個 baseline「無法保證」；Claude 在無 context 時分數就偏高，作者自陳可能認得測試 repo，那麼把 Claude 當「天花板」來對照就有被污染的風險。

### 新穎性是工程整合，而非新方法

就方法論而言，continued pre-training + LoRA fine-tuning + FIM 遮罩、以及「小特化模型可勝通用大模型」的結論，在 low-resource code 文獻（MultiPL-T、Giagnorio 等、MonoCoder/MPIrigen 等）中都已出現。本文真正的貢獻偏向**工程整合與領域落地**：Tonel-aware 的資料策展、Pharo 的 Pygments lexer 與 tree-sitter grammar、可執行的 Pharo 評測工具、以及把整套流程收斂到能在筆電即時跑的量化模型。作者也誠實地把「工程投入遠大於訓練本身」寫進 discussion，這是本文的定位——一份 low-resource 語言落地的 case study，而非一個新演算法。

### 距離「解決」還差一個真實使用者研究

即便數字漂亮，論文尚未證明它在真實 IDE 裡有用。所有評測都是「遮罩—補回」的模擬，沒有開發者實際接受／修改／拒絕建議的線上研究；延遲評測也只在少數高規機器上做，未涵蓋低階硬體。作者把「整合進 Pharo IDE 並做真人評估」列為 future work，並說正在開發外掛。就目前證據，合理的結論是：**在離線模擬下，特化把小模型推到了接近可即時部署的補全品質**——這是有力的可行性證明，但「是否真能提升 Pharo 開發者生產力」仍未被驗證。

## 一分鐘版

- **極端低資源** = 目標語言在 GitHub 上的程式碼比主流語言少四個數量級。例子：Pharo 只有約 2k 個公開 repo，Python 有約 26M 個。
- **特化流程** = 用三段式訓練把一個小開源模型教會冷門語言，而不是直接用巨型通用模型。例子：對 387,159 個 Pharo method 先做 AST-aware 的 FIM 預訓練、再用 Random-AST 遮罩 fine-tuning，全程 LoRA。
- **特化的效果** = 兩段式訓練能把小模型的補全正確率大幅拉高。例子：Qwen2.5 Coder 3B 在 HumanEval+ AST-aware 的 pass@1 從 71.48% 升到 83.73%，Odds Ratio 3.81（勝算約 3 倍）。
- **「小勝大」要看是哪個指標與設定** = 沒有一個模型全指標稱王：特化小模型只在部分欄位勝出，通用大模型在另一些欄位領先。例子：兩個 method-level r-AST pass@1 欄位上，特化 3B / 7B 都超過 Qwen3 Coder 480B 與 Claude 4.5 Sonnet；但在 AST-aware 欄位 Claude（95.07）與 Qwen3 480B（91.95）勝過特化 7B（89.04），而 repo-level 提供 impacted methods 的那幾列則由 Claude 領先。
- **repo-level 數字不等於正確率** = 沒有可執行測試，只用字面相似度衡量，語意錯但用字相近的補全也可能拿高分。例子：repo-level 的 75.96% ChrF 不宜被讀成「補全正確率」。
- **可行性而非已解決** = 特化證明小模型在離線模擬下逼近可即時部署的品質，但尚未有真實 IDE 使用者研究。例子：量化 7B 在 M3/M4 Max CPU 約 1.33 秒，略高於互動補全常見的「次秒」門檻。

## 🔗 Related notes

- [LoRA](../Lora/) — 本文 pre-training 與 fine-tuning 兩階段都以 LoRA 進行參數高效微調。
- [Fine-tuning vs In-context Learning vs RAG](../FineTuning-vs-ICL-vs-RAG/) — 本文引用的 Giagnorio 等研究即比較 low-resource code 上 fine-tuning 與 in-context learning 的取捨。
