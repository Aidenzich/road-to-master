# Coding Agent 省 Token 工具與論文調查 — 手段、代價、使用者實證

> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | 號稱為 Claude Code / Codex 等 coding agent 節省 20\~40% token 的專案與論文:手段、代價、使用者實證 |
| Venue Kind | survey(跨 11 個代表性專案 + 30 餘個周邊工具/論文的市場稽核) |
| Year | 2026 |
| 調查日期 | 2026-07-16(所有網路來源抓取日期相同,除另註) |
| 方法 | read-only 稽核:引文只取自實際開啟讀過的頁面(官方 blog / arXiv 全文 / GitHub API / HN Algolia / Reddit 存檔 API / 靜態 clone);每個承重宣稱至少跑一次反證搜尋;未執行任何下載程式碼 |

> 這則筆記不是單篇論文摘要,而是一則市場稽核 survey。稽核對象是「為 coding agent 省 20\~40% token」這類宣稱本身。**所有廠商數字一律先當成未驗證的 seed**,凡與一手來源衝突,一手來源勝出;每個宣稱都追問「量的是什麼、什麼工作負載、baseline 是誰」。

---

## 一句話結論

> **「省 20\~40%」這個帶寬本身可信,但它是各家 60\~95% 行銷數字被獨立量測打折後的殘值,而不是任何一家的原始宣稱。** 手段的風險差異極大:輸出側精簡與延遲載入幾乎無損;token 級剪枝(LLMLingua 型)用在 coding 是災難;檢索取代整檔餵入方向對,但實作品質常吃掉理論收益。

一個對照最能說明問題:唯一一份**中立、同任務、同機器**的橫測(ComputingForGeeks,baseline 284,473 tokens),把一票自稱省 60\~95% 的工具實測成 **0\~43%**:

| 工具 | 自稱 | 中立橫測實測 |
|-|-|-|
| token-savior | −80% | **43%** |
| caveman | 65% | **38% / 37%** |
| ooples token-optimizer-mcp | 95%+ | **23%** |
| RTK | 60\~90% | **0%**(輸出乾淨時完全沒省) |
| code-review-graph | — | **5%**(小 repo 反而 0.7x 變貴) |

使用者印象中的「20\~40%」跟獨立量測對得上,跟廠商標題(60\~95%)對不上。

---

## 三個調查問題

這份稽核要回答的三個問題:

- **Q1 — 手段**:這些專案各自透過什麼機制減少 token?每個機制都要有**具體例子**(repo 內實際的檔案/函式,或 paper 內實際的演算法/表格),而不只是行銷語言的轉述。
- **Q2 — 代價**:除了 token 減少,這些手段是否會影響 LLM 的**能力與正確性**?各專案自己提供了什麼評測證據?有沒有獨立第三方評測?證據的強度與缺口在哪?
- **Q3 — 使用者實證**:使用者的心得分享、GitHub issues、社群討論回報的問題/bug,是否指向 Q2 的懷疑(漏掉關鍵 context、改錯檔、答非所問、agent 迷路)?

### 速查結論

| 問題 | 結論 |
|-|-|
| **Q1 手段** | 五類機制:①token 級 prompt 壓縮 ②語意/符號檢索取代整檔讀取 ③context 中介層/長期記憶 ④輸出側 diff-edit ⑤模型路由/快取 proxy。詳見〈手段分類〉 |
| **Q2 代價** | 差異巨大且可分層(見下方風險表)。最關鍵發現:token 級 perplexity(困惑度,衡量模型對一段文字的「意外」程度)剪枝會破壞 coding 的識別符/數值/語法(壓縮後 AST(抽象語法樹,Abstract Syntax Tree)正確率僅 **0.29%**);而**行級 task-aware 剪枝**(SWE-Pruner)在全量 SWE-bench Verified 上 −23\~38% token 且成功率**反升** 1.2\~1.4 個百分點(pts,percentage points) |
| **Q3 使用者實證** | 大量印證 Q2 的懷疑:Serena「用了反而更快撞 context 上限」+ 靜默改壞碼;claude-context 索引失同步「從來搜不到」;mem0 存出互相矛盾的記憶;Claude Code 自己的 /compact 是最大宗「壓縮後失憶」抱怨來源 |

---

## 核心:按風險分層(Q2:能力與正確性的代價)

把所有手段依「省 token 是否傷能力」排序,是這份調查最有用的產出,也是 Q2 的總答:

| 風險 | 手段 | 為什麼 |
|-|-|-|
| 🟢 **最低** | diff / targeted edits(輸出側)、噪音工具輸出壓縮(RTK 型)、tool-definition 延遲載入、prompt caching | 機制上近無損;Anthropic 自家 token-efficient tool use(平均 14%)是唯一「大規模部署且查無退化回報」的案例 |
| 🟡 **中** | 語意/符號檢索取代整檔餵入(Serena、claude-context);**行級 task-aware context 剪枝(SWE-Pruner)** | 方向有獨立正面證據(Cursor A/B(A/B test,對照實驗):+12.5% 準確率),但實作層(索引失同步、MCP(Model Context Protocol,模型上下文協定)固定開銷、符號編輯損毀)常吃掉收益;SWE-Pruner 證據面全場最強,但單一團隊、零重現、僅 Python |
| 🔴 **高** | **token 級 / perplexity 剪枝(LLMLingua 型)用於 coding**;分類器模型路由用於 agentic coding | 多個獨立來源一致顯示破壞識別符/數值/AST;路由在 OOD(out-of-distribution,分布外,即測試情境與訓練資料不同)coding 崩壞 + 協定不相容導致 tool use 整段報廢 |
| ⚫ **無法評估** | 零方法論的行為注入框架(SuperClaude 型)、零方法論的「95%+」MCP | 無 benchmark、無方法論,甚至有反向實測(框架自身先吃掉 4 萬多 token context) |

> 注意:🔴 的判定**只限 token 級剪枝**,不延伸到行級 task-aware 剪枝 —— 兩者機制不同族,見〈手段分類 ①〉與〈深入:SWE-Pruner〉。

### WRAPUP:逐工具總表

一眼看完的主表。**機制**:A=token 級壓縮、B=檢索取代整檔、C=context 中介/記憶、D=輸出側 diff、E=路由/快取。**證據強度**(與能力風險同一套顏色,🟢=好、🔴=差):🟢 有公開 benchmark 可(原則上)重現;🟡 只有廠商自測或樣本過小;🔴 無方法論或明文拒絕評測(含只用 LLM 自評)。**能力風險**沿用上方風險分層:🟢 最低、🟡 中、🔴 高、⚫ 無法評估(零方法論、連傷不傷能力都判斷不了)。

| 專案 | 機制 | 宣稱節省(量測口徑) | 證據強度 | 能力風險 | 使用者負面訊號 |
|-|-|-|-|-|-|
| SWE-Pruner | A(行級) | 23\~54% token(agent 總 token,成功率反升) | 🟢 單一團隊、零重現 | 🟡 | 未找到(但部署量小) |
| Aider(repo-map + diff) | B+D | 不宣稱 %(repo-map 是開銷) | 🟢 全場最誠實 | 🟢 diff(強模型)/🟡 repo-map | 有:#752 預算超標 16 倍、SEARCH-REPLACE 失敗群 |
| claude-context | B | \~40%(30 題 localization) | 🟡 n=30、弱模型 | 🟡 | 有:索引失同步 #145/#226/#232 |
| LLMLingua 家族 | A(token 級) | up to 20x(input tokens,CoT 任務) | 🟡 官方窄 / coding 被獨立證據打穿 | 🔴(coding) | 有:#89 錯答率 +18pts、#136 崩到 0.02 |
| mem0 | C | >90%(vs 重播全史) | 🟡 高度爭議 | 🟡 | 有:#5867 矛盾記憶 |
| Anthropic 官方功能 | C/D/E | 84% / 14\~70% / 85% / 98.7%(internal) | 🟡 internal-only,機制透明 | 🟢 | 少;唯 /compact 有大量「失憶」抱怨 |
| RouteLLM | E | 85% 省費 + 保 95%(無 coding) | 🟡 in-domain / 崩 OOD | 🔴(agentic) | 學術反證強、repo 無明顯抱怨 |
| Serena | B | 無數字(社群傳 70%,查無據) | 🔴 明文拒測、LLM 自評 | 🟡 | 有:Reddit「反而更耗」、#1529 靜默改壞碼 |
| SuperClaude | C | 70%→30-50%(零方法論) | 🔴 | ⚫ | 有:#286 框架自吃 43.8k context |
| claude-code-router | E | 自身無 % 宣稱 | 🔴 無評測 | 🔴 協定不相容 | 有:#1378 tool use 全滅 |
| token-savior | B/C/D | −80% 自稱 → 中立 43% | 🔴 自製 tsbench | 🟡 未嚴謹評測能力 | — |
| caveman | D(強制電報體輸出) | 65% 自稱 → 中立 38% | 🔴 自測 | 🟢 中立測「答案相同」;但 terse 工作負載可能反增 | — |
| ooples token-optimizer-mcp | B/D | 95%+ 自稱 → 中立 23% | 🔴 零方法論 | 🟡 未評測 | — |
| RTK | D(壓噪音輸出) | 60\~90% 自稱 → 中立 0% | 🔴 自測 | 🟢 只壓噪音、風險低 | — |
| code-review-graph | B(圖檢索) | 中立 5%(小 repo 反而 0.7x 變貴) | 🔴 | 🟡 未評測 | — |
| 其他(claude-mem、Headroom 等) | C/D | 10x、\~50%(自稱) | 🔴 自測 | 🟡 | 有:claude-mem #618 token 膨脹 |

---

## 手段分類(Q1:機制 + 具體例子)

### ① Token 級 prompt 壓縮 — LLMLingua 家族

- **代表**:LLMLingua / LongLLMLingua / LLMLingua-2(Microsoft,arXiv 2310.05736 / 2310.06839 / 2403.12968)
- **機制**:用小型語言模型算每個 token 的 perplexity,把「低資訊」token 直接刪掉。
- **具體例子**:官方 README 把一段 2,365-token 的 GSM8K CoT(chain-of-thought,思維鏈,要求模型逐步推理的 prompt)prompt 壓到 211 tokens(11.2x)。但官方自己的範例輸出就已損毀:*"He reanged five of boxes into packages of sixlters each..."* —— 行銷頁上的示例已展示了機制代價。
- **官方甚至內建「修復」**:LongLLMLingua 有一個 subsequence recovery 步驟,把被壓壞的實體(如 "209" 修回 "2009")事後補回 —— 等於官方承認壓縮會破壞字面。
- **宣稱口徑**:「up to 20x compression with minimal performance loss」量的是 **input prompt tokens**,工作負載是 GSM8K/BBH 的 CoT + GPT-3.5-0613。

### ② 語意 / 符號檢索取代整檔讀取

用精準檢索取代「把整個檔案塞進 context」。

- **Serena MCP**(26.5k★):以 LSP(Language Server Protocol,語言伺服器協定,IDE 用來取得符號/定義/引用的標準介面)做符號級檢索,工具用符號名路徑(`MyClass/my_method`)取代整檔讀取。
  - 程式碼:`FindSymbolTool`(`src/serena/tools/symbol_tools.py:132`,參數 `include_body` 註明 "Use judiciously")、`GetSymbolsOverviewTool`(`:36`)、`ReplaceSymbolBodyTool`(`:571`)。
  - **官方自家 eval 的真實例子**:`get_symbols_overview` 一次呼叫回 \~2.5KB JSON vs Grep \~3KB;官方原話節省幅度是 *"saves \~1 call and some context window tokens per navigation"* —— 遠小於社群流傳的「省 70%」(該 70% 查無出處)。
- **claude-context**(Zilliz,12.1k★):AST-aware 切塊 → embedding → Milvus 向量庫 → 混合檢索。
  - **唯一「原則上可重現」的 vendor 量測**(repo `evaluation/README.md`):30 題 SWE-bench Verified 檢索子任務、GPT-4o-mini,tokens 73,373 → 44,449(**−39.4%**)、F1(精確率與召回率的調和平均,越高代表找對檔案的能力越好)0.40 vs 0.40 持平。這就是行銷句「Cut Token Waste by 40%」的全部實驗基礎。
  - 弱點:n=30、限定 2-file 改動、用最受檢索恩惠的弱模型、只測 file localization(F1=0.40 表示兩組都過半機率找錯檔)。
- **Aider repo-map**(全場定位最誠實):用 tree-sitter 抽符號 + PageRank 排名,**二分搜尋塞進固定 token 預算**(`aider/repomap.py:47`,預設 `map_tokens=1024`)。
  - **關鍵差異(缺席事實)**:Aider **從不宣稱 repo-map 省 X%** —— repo map 是用固定預算「買」能力的**開銷**,不是相對整檔餵入的節省。這跟 MCP 檢索工具的行銷框架正好相反。

### ③ 輸出側 diff-edit(與「輸出 token」最對口)

- **機制**:diff / SEARCH-REPLACE 只回傳變更的 hunk,取代整檔重寫(整檔 = "slow and costly because the LLM has to return the entire file")。
- **代表**:Aider 的 `diff`/`udiff`(unified diff,統一差異格式)格式、Claude Code 內建的 `old_string`/`new_string` Edit tool、OpenAI 的 `apply_patch` V4A。
- **具體正面例子**:Aider 的 unified diff 把 GPT-4 Turbo 的 lazy-coding benchmark 從 20% 拉到 61%(輸出壓縮**有時反而提升能力**)。
- **代價**:格式脆弱 —— 同文「關掉彈性 patching 時編輯錯誤增加 9 倍」;弱模型用 diff 準確率反而倒退(見〈使用者實證〉Aider)。

### ④ Context 中介層 / 長期記憶

- **mem0**(arXiv 2504.19413):跨 session 抽取重點事實進向量庫,取代重播全部對話史。宣稱 "saves more than 90% token cost"。
  - **口徑陷阱**:>90% 是 vs「每次重播 16k\~26k token 全史」這個 baseline —— 省 token 是機制上保送的,爭點全在準確率(見〈使用者實證〉)。
- **SuperClaude**(23.6k★):把 markdown 指令檔注入 context 的「行為框架」。v1 README 宣稱 "70% reduction",現版改口 "30-50% fewer tokens" —— **兩代皆零方法論**。
- **Anthropic 官方**:context editing + memory tool(宣稱 84% @ 100-turn web search)、Tool Search Tool(85% 省 input 側 tool definitions)。數字全部 internal、無獨立重現,但機制透明。

### ⑤ 模型路由 / 快取 proxy

- **RouteLLM**(arXiv 2406.18665):訓練強/弱模型路由器,簡單任務派便宜模型。宣稱「85% 省費用 + 保 95% GPT-4 能力」,但**只測 MT Bench/MMLU/GSM8K,無 coding**。
- **claude-code-router**(35.8k★):本地 proxy 把 Claude Code 各類流量路由到便宜模型。自身不宣稱 % 節省。
- **prompt caching**:省的是**費用不是 token**,且寫入有 1.25×/2× 溢價 —— 低命中率時反而更貴。**且它與大多數動態省 token 方法在很大程度上相衝突(見〈該不該用〉)。**

---

## 深入:SWE-Pruner —— 本次調查中最強的「省 token 且成功率不降」證據

(arXiv 2601.16746 v4,github.com/Ayanami1314/swe-pruner,299★)

這篇值得單獨拉出來,因為它**同時是 LLMLingua 型剪枝的最強反例、也是行級剪枝的最強正例**:

- **機制與 LLMLingua 的兩個關鍵分野**:
  - **task-aware**:agent 先給一個明確目標(如 "focus on error handling")引導剪枝,而非固定 perplexity 指標。
  - **行級而非 token 級**:0.6B 的 skimmer 整行保留或整行刪除,故**不產生 LLMLingua 式的字面損毀**。
- **正面數字(全量 500 題 SWE-bench Verified)**:
  - Claude Sonnet 4.5:成功率 70.6% → **72.0%(+1.4 pts)**,tokens −23.1%
  - GLM-4.6:55.4% → 56.6%(+1.2 pts),tokens −38.3%
- **機制性分野的鐵證(AST 正確率對照,paper Table 8)**:

  | 方法 | 壓縮後 AST 正確率 |
  |-|-|
  | Full Context(完整 context,不壓縮) | 98.5% |
  | LLMLingua-2(token 級) | **0.29%** |
  | LongCodeZip(行級) | 89.3% |
  | SWE-Pruner(行級) | 87.3% |

- **誠實邊界**:單輪 8x 極限壓縮時 EM(Exact Match,完全匹配率,輸出與標準答案逐字一致的比例)仍從 40.5 掉到 31.0(有實質退化);僅 Python;單一團隊、零第三方重現。
- **評級:很可能是可實際採用的方法、具參考價值。** 它是本次調查中唯一「省得多(−23\~38% token)又不掉能力(成功率反升)」有完整公開數據支撐的方向,機制(行級 task-aware)也站得住腳,值得作為設計參考。之所以還不列 🟢「放心採用」,純粹是外部驗證不足(單一團隊、零重現、僅 Python),不是機制或數據有硬傷 —— 建議先在自己的 codebase 上驗證(closure recipe #15)再導入,而非直接照搬或直接否定。

---

## 使用者實證(Q3:區分工程 bug vs 機制性退化)

以下引用皆附 issue 編號/連結,可回溯。分類標明是「機制性退化」還是「工程 bug」。

### 印證機制性退化

- **LLMLingua #89**(open):使用者在 LongBench qasper 實測,錯答/棄答率從 45.36% 升到 63.93%(+18 pts)。**#136**:dureader 準確率 0.68 → 0.02。【機制性】
- **Serena**:Reddit r/ClaudeCode「Feels like Serena MCP uses more tokens than without?」—— OP「用 Serena 反而更快撞 context 上限」,最高讚回覆證實「任何 MCP 都會增加 token」。另有 cursor devs 指出「語意檢索在 codebase 上表現比讓模型自己用 bash 搜還差」。【機制性:MCP schema 固定開銷】
- **Aider**:HN 使用者貼 log 指 repo-map 摘要「產生多個事實錯誤的描述、幻想出不存在的追蹤機制」;300-file monorepo 上「repo map 把 LLM 淹沒」。【機制性:壓縮表徵誤導 —— 正對「答非所問/agent 迷路」懷疑】
- **mem0 #5867**(open):偏好變更(Ronaldo→Messi)產生兩條並存的矛盾記憶,「檢索變得模稜兩可」。HN 使用者關掉記憶因為它「一直留著過時資訊、跨專案滲漏」。【機制性】

### 印證工程 bug(但多屬該機制的固有風險面)

- **Serena #1529**(open):`replace_symbol_body` 把 Go 的 `type Resampler struct` 改壞成 `type type Resampler struct`,**且工具回傳 `{"result": "OK"}` 完全不報錯** —— 靜默損毀,直接對應「改壞碼」懷疑。**#516**:Serena 自己的 `search_for_pattern` 單次回 32,204 tokens 爆掉 Claude Code 的 25k MCP 上限(省 token 工具反而超支)。
- **claude-context #145 / #226 / #232**:索引成功但隨即「codebase not indexed、從來搜不到」;狀態同步 bug;專案一改動就要整包重索引。
- **claude-mem #618**(confirmed bug):「Uses too much tokens —— claude code 在 10 則訊息內把我的 token 全吃光」(旗艦記憶插件的 token 膨脹 bug)。
- **claude-code-router #1378**(open):DeepSeek + thinking + tool calls 每輪必 400,「一碰到 tools 就完全無法用」。【機制性:協定不相容 → tool use 整段報廢,對 coding agent 是最高風險】

### 附帶:Claude Code /compact 是所有人比較的 baseline,自身即最大量負面實證

- **#10006**:「每次 auto-compact,它丟掉每一個細節」。**#13919**:skills 在 compaction 後全失效,任務時間從「\~1 小時」變「5-6+ 小時」。工程師實測:「compaction 後明顯變笨,不知道自己剛在看哪些檔案」。
- 分類:CLAUDE.md/skills 不回注是**工程 bug**(現版已部分修復);摘要丟中途決策/檔案清單是**機制性有損壓縮**。

### 平衡:檢索方向也有正面獨立證據

- **Cursor 官方 A/B**(2025-11):同模型下 semantic search vs grep-only,平均**準確率高 12.5%**(6.5%\~23.5%);大型 codebase code retention +2.6%。—— 反駁「檢索必然使 coding agent 變笨」的極端讀法。
- 但對照 Claude Code 官方立場:「Claude Code 目前不用 RAG(Retrieval-Augmented Generation,檢索增強生成,先向量檢索再把片段餵給模型);我們測下來 agentic search 在 Code 的使用情境優於 RAG」。

---

## 該不該用(實務建議)

- **想無腦省又不冒險**:優先用「機制上近無損」的手段 —— targeted/diff edits(Claude 4+ 級模型上格式順從稅可忽略)、噪音工具輸出壓縮(只在輸出髒時有效)、tool-definition 延遲載入、prompt caching(注意 cache-miss 反噬與寫入溢價)。Anthropic 內建的 token-efficient tool use 無需採用動作、已內建。
- **大 codebase(>20k LoC(lines of code,程式碼行數)級)且願付索引維運成本**:才值得上語意檢索工具;預期理論收益會被索引失同步、MCP 固定開銷部分吃掉。
- **最值得追的新方向**:SWE-Pruner 型的**行級 task-aware 剪枝**很可能是可用且值得參考的方法 —— 唯一「省得多又不掉能力」有完整公開數據者。導入前先自架 skimmer + 在自己 codebase 用 `ccusage` 實測(closure recipe #15),但不必因為「preprint、單一團隊」就直接排除。
- **⚠️ prompt caching 與動態省 token 方法在很大程度上相衝突,別無腦疊用**:prompt caching 靠「prompt 前綴逐位元組穩定、只在尾端 append」才能吃 0.1x 的 cache-read 折扣;但大多數動態縮 context 的方法(LLMLingua 壓縮、SWE-Pruner 剪枝、Anthropic context editing 清除舊 tool call、記憶/摘要改寫)**每輪都會改動前綴**,一改就讓該點之後的 cache 全部失效 → 退回全價 miss + 1.25×/2× 寫入溢價,兩者疊用可能比各自單用還貴。兩種策略方向相反:caching 要「穩定、只增不改」的 context,積極剪枝要「邊改邊縮」。依成本結構二選一 —— 成本主要來自重複的穩定長前綴就選 caching,來自無上限成長的 context 就選剪枝。
- **絕不要**:把 token 級 perplexity 剪枝(LLMLingua 型)用在 coding;把 2024 式分類器路由的 85%/95% 宣稱外推到 agentic coding。
- **通則**:分母不明的百分比不可比。引用任何「省 X%」前,先問**量什麼(input/output/費用/單一子任務)、什麼工作負載、baseline 是誰**。

---

## 🧪 Critical Assessment

### 這份稽核本身的邊界

- **沒有一個數字是在 Claude Code 上量的**:所有 vendor 數字都在各自的 harness、模型、工作負載上量出,外推到 Claude Code 都是推論。真正的閉環需要固定任務集在 `claude -p` headless 下用 `ccusage` 實測(見〈未解問題〉)。
- **Reddit 取證受限**:本環境對 Reddit 全面 403,多數 Reddit 證據改走 arctic-shift 存檔 API 或標為 secondary。
- **SWE-Pruner 的正面證據雖強,但單一團隊、零獨立重現、僅 Python** —— 它是「最值得驗證」而非「已證實」。其 paper Table 3 的 baseline(62.0%)與 Table 1(70.6%)不一致且未標明樣本數,是一個 paper 自己沒解釋的疑點。

### claim-laundering(宣稱洗白)現象普遍

不少「省 token」宣稱經二手部落格轉述後偷換了口徑,例如 LEANN 的「97% less storage」被誤傳成「省 97% token」(儲存 ≠ token);code-context-engine 的「94% fewer input tokens」baseline 是「整檔全讀」的稻草人。引用時務必回一手頁面確認量的維度。

### 證據強度極不均

從「有公開 benchmark + AST 檢核」(SWE-Pruner、Aider)到「官方明文拒絕 benchmark、用 LLM 自評」(Serena)到「零方法論的行銷數字」(SuperClaude、多數 MCP)都有。〈WRAPUP 逐工具總表〉的「證據強度」欄(🟢/🟡/🔴)已逐項標注,讀者不應把「有 star 數」當成「有證據」。

## 🔗 Related notes

- [Vector Database Comparison](../VectorDatabaseComparison/) — 語意檢索類工具(claude-context 等)背後的向量庫選型,同為市場稽核 survey 文類。
- [Shepherd](../Shepherd/) — agentic execution trace 的可程式化控制,與 context 管理/記憶中介層的設計取捨相關。

---
*本筆記為 read-only 市場稽核產出;所有引文抓取日期均為 2026-07-16,除標註者外皆來自實際開啟之頁面。完整逐條證據、11 項深查與 15 條未解問題的 closure recipe 見原始 dossier。*
