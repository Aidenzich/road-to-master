# LetMeSpeakFreely — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models |
| Venue | unknown |
| Year | 2024 |
| Authors | Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, Chieh-Yen Lin, Hung-yi Lee, Yun-Nung Chen |
| Official Code | https://github.com/appier-research/structure-gen |
| Venue Kind | paper |

## 問題設定：結構化生成與格式限制

在真實的工業應用中，我們幾乎不會直接把 LLM 的自由文字回覆丟給下游系統，而是要求它輸出 JSON、XML 或 YAML 這類標準化格式，方便程式解析（parsing）出「答案」欄位。這種要求模型以固定 schema 產生輸出的做法稱為 structured generation，而施加此限制的手段則是 format restriction。這篇論文問的問題非常務實：當我們為了好解析而綁住模型的輸出空間時，是否也連帶損害了模型本身的推理與知識理解能力？

過去的 IFEval、INFOBENCH、FOFO 等 benchmark 都在評估 LLM「能不能遵守格式」，卻沒有回答「格式指令是否會傷害被產生內容的品質」。作者主張，這是首次系統性地研究 format-restricting instructions 與生成內容品質之間關係的工作（this is the first systematic investigation），而答案出乎意料：在推理任務上，格式限制帶來顯著且可觀的效能衰退。

## 三種格式限制方法與其嚴格程度光譜

論文把工業界常見的作法抽象成三種嚴格程度遞減的方法，這個光譜是全文比較的骨架：

| 方法 | 機制 | 嚴格程度 |
|-|-|-|
| Constrained Decoding (JSON-mode) | 在解碼時強制 token 空間，保證輸出合法 JSON | 最嚴格 |
| Format-Restricting Instructions (FRI) | 用提示詞要求依指定 schema 輸出，不強制 token 空間 | 中等 |
| NL-to-Format | 先用自然語言作答，再把答案轉成目標格式 | 最寬鬆 |

Constrained decoding 透過在生成過程中強制一個預先定義的 token 空間來限制 LLM 的輸出（enforcing predefined token space during the generation process），OpenAI 與 Gemini API 的 JSON mode 就是這個技術最普及的實例。FRI 則只是用指令引導模型輸出 JSON、XML、YAML 等標準格式（generate responses in standardized formats such as JSON, XML, and YAML），比 constrained decoding 寬鬆，因為它不鎖死 token 空間。NL-to-Format 是兩階段流程：先要模型用自然語言回答問題，再指示它把回答轉換成目標格式 schema（answer the question in natural language, and then instructs it to convert its response into the target format schema），把「內容生成」與「格式遵循」解耦，是三者中最寬鬆的一種。

## 核心發現：推理退化、分類任務可能反受益

論文最反直覺的結果是：在 GSM8K、Last Letter Concatenation、Shuffled Objects 這類推理任務上，越寬鬆的提示通常給出越好的結果——JSON-mode 大多最差，其次是 FRI，再來是 NL-to-Format，而純自然語言（NL）最好。也就是說，越嚴格的格式限制，在推理任務上帶來越大的效能退化。

![不同格式限制強度下的推理任務表現](imgs/reasoning_restriction.png)

分類任務卻呈現相反的趨勢。在 DDXPlus 上，Gemini 1.5 Flash 開啟 JSON-mode 後效能明顯提升（demonstrates a significant performance boost when JSON-mode is enabled）；跨其他分類資料集，JSON-mode 表現具競爭力，某些情況甚至超越其他三種方法（surpasses the other three methodologies）。作者的解讀是：JSON-mode 藉由限制答案空間、減少選錯答案的機會，反而幫助了本來就不需要長推理的分類任務。這使得結論帶有任務相依性：嚴格格式會傷害重推理任務，卻可能提升需要固定輸出集合的分類任務。

## 一次具體的前向推理：GSM8K 的 Eliza 工資題

論文封面圖用一題 GSM8K 工資題把機制講得很清楚。題目是 Eliza 前 40 小時時薪 $10、超時 5 小時時薪為 1.2 倍。用標準的「逐步推理，再給答案」文字格式時，GPT-3.5-turbo 老老實實算出 40×$10 = $400、5×$12 = $60，總和 $460，答對。

![標準提示答對、格式限制答錯的對照](imgs/cover_showcase.png)

一旦把同一題改成必須先產出 JSON 物件，模型的行為就變了：

```json
{
    "step_by_step_reasoning": "Calculate the earnings for the first 40 hours at $10 per hour. Then calculate the earnings for the additional 5 hours at $10 * 1.2 per hour. Add both amounts to find the total earnings for the week.",
    "answer": 490
}
```

在 JSON 的 `step_by_step_reasoning` 欄位裡，模型只寫了「計畫」而沒有真的把算式一步步展開，`answer` 直接跳成 490（錯誤）。更關鍵的證據來自 Last Letter 任務：作者檢查後發現，100% 的 GPT-3.5-turbo JSON-mode 回應都把 `answer` 鍵放在 `reason` 鍵之前（100\% of GPT 3.5 Turbo JSON-mode responses placed the "answer" key before the "reason" key），於是模型變成 zero-shot 直接作答，而非 zero-shot chain-of-thought，推理鏈被格式的鍵順序給截斷了。這說明退化的根源之一是「先寫答案還是先寫推理」的鍵排序，而不是格式本身神秘地讓模型變笨。

## 用數字量化退化：加上 schema 限制的代價

為了把退化量化，論文比較「只要求輸出某語言」與「額外附上 schema 約束」兩種設定在 GSM8K 上、對 9 種提示擾動取平均的分數（括號內為標準差）。以下摘錄自 Table「loose vs strict」的關鍵列：

| Model | 格式 | 無 schema | 加 schema 約束 |
|-|-|-|-|
| claude-3-haiku | JSON | 86.99 | 23.44 |
| gpt-3.5-turbo | JSON | 74.70 | 49.25 |
| gpt-3.5-turbo | XML | 60.45 | 45.06 |
| LLaMA-3-8B | YAML | 69.41 | 46.08 |

我們可以用一個簡單的量來描述退化（此符號為本文自訂）：

$$
\Delta_{\text{fmt}} = \text{Acc}_{\text{NL}} - \text{Acc}_{\text{fmt}}
$$

最戲劇性的一列是 claude-3-haiku 的 JSON：平均分數從 86.99 崩到 23.44，標準差同時從 0.2 暴增到 22.9，顯示加上 schema 不只拉低平均、還讓模型對提示措辭變得極度敏感（adding schema not only increase the sensitivity to prompt but also degrade in average performance）。這支持了一個實務建議：處理重推理任務時，與其硬塞死板 schema，不如放寬格式限制、保留模型原有的推理空間。

## 退化不是解析錯誤造成的

一個自然的懷疑是：文字與結構化格式的差距，會不會只是因為結構化輸出更難被 parser 正確擷取答案？論文的分析否定了這個假設。Gemini 1.5 Flash 與 GPT-3.5-turbo 在三種格式下的解析失敗率幾乎是零，卻仍看到分數下降；而在 LLaMA-3-8B 上，Last Letter 任務 JSON 格式的解析錯誤率只有 0.148%，卻存在高達 38.15% 的效能落差（the parsing error rate for the Last Letter task in JSON format is only 0.148\%, yet there exists a substantial 38.15\% performance gap）。這代表差距來自格式限制對模型推理與生成過程的干擾，而非解析環節的失誤。

## 緩解方法：讓模型先自由說話

論文提出的解法圍繞同一個原則——把內容生成與格式遵循分開。NL-to-Format 幾乎能維持與純自然語言相同的分數，因為兩者的答案都源自同一段自然語言回覆。針對真正出現的解析錯誤（Claude-3-Haiku 與 LLaMA-3-8B 最嚴重），只要用第二次提示把壞掉的輸出重新格式化，就能在 JSON 與 YAML 上把分數救回來。另外，較新的 gpt-4o-mini 提供以 context-free grammar 實作、保證 100% 合乎 schema 的 JSON-Schema（Structured Output）API；在三個推理資料集裡有兩個，純自然語言仍略優於 JSON-Schema（In 2 out of 3 reasoning datasets, NL still performs slightly better than JSON-Schema），但差距已比舊的 JSON-mode 小很多。成本面上，YAML 對 LLaMA-3-8B、Gemini-1.5-Flash 與 GPT-3.5-Turbo 是最省錢的格式（YAML is the most cost-effective format），這也讓「寬鬆格式」在效能與成本上同時佔優。

## 🧪 Critical Assessment

### 問題真實性與重要性

這個問題是真實而非人造的。任何把 LLM 接進生產系統的人都會遇到「要好解析就得綁格式、綁了格式又怕傷品質」的兩難，而在本研究之前，社群的 benchmark 只量測格式遵循度、沒人量測格式對內容品質的副作用。論文把這個工程痛點轉成可量測的實驗，方向紮實。

### 基線、消融、資料集與指標的充分性

實驗覆蓋三家閉源 API 與兩個開源模型、六個資料集、JSON/XML/YAML 三種格式、以及九種提示擾動，涵蓋面在同類研究中算充分。但有兩個設計選擇會影響可信度。其一，所謂的「完美解析器」其實是用 claude-3-haiku 當 LLM parser，理由是它與 gpt-4-turbo 的一致性（kappa）最高、達 0.86——這是「最一致」而非「無誤差」，把 parser 誤差完全排除的說法略嫌樂觀，parser 本身仍可能對不同格式有系統性偏好。其二，schema 被刻意限制成只有 reasoning 與 answer 兩個欄位，作者也承認這與真實世界複雜巢狀 schema 的落差是未解的外部效度問題。

### 是新發現還是既有現象的重新包裝

真正的新意在於「量測格式限制對內容品質的因果影響」這個角度，而非提出新模型或新技術；緩解手段（兩階段生成、錯誤重寫）也都是既有工程技巧的整理。需要保留的是：部分關鍵證據其實指向「實作瑕疵」而非「格式限制的本質傷害」。JSON-mode 在 Last Letter 崩潰，根因是模型把 answer 鍵排到 reason 之前，這是鍵排序與 chain-of-thought 被截斷的問題，只要調整 schema 欄位順序或改用會保留推理欄位在前的設定就能大幅緩解；把它算成「constrained decoding 本質有害」有以偏概全之虞。此外，評測任務的選擇也可能放大了效果——Task 280 本身就是已知對提示格式極度敏感、變異可達 56% 的任務，等於在對本方法最有利的地形上比較。

### 這個問題真的被解決了嗎，以及真實世界關聯

論文很誠實地把貢獻收斂在「揭示現象並提出可用的緩解」，而非宣稱徹底解決。有幾點削弱了結論的普遍性：受測模型都是較小、較便宜的世代（gpt-3.5-turbo、claude-3-haiku、gemini-1.5-flash、8–9B 開源模型），作者因成本無法納入 LLaMA-70B 或 GPT-4o 等更強模型；而當他們試用較新的 gpt-4o-mini JSON-Schema 時，退化已明顯縮小。這讓人合理推測：所觀察到的傷害有相當一部分與早期模型和早期 JSON-mode 實作綁定，隨著原生 constrained decoding 成熟與模型變強，效應可能持續收斂。對實務工作者，最穩健的可操作結論不是「不要用結構化輸出」，而是「重推理任務優先用 NL-to-Format 或放寬 schema，並確保推理欄位排在答案之前」。

## 🔗 Related notes

- [Instruction Tuning with GPT-4](../InstructioinTuningWithGPT4/)
