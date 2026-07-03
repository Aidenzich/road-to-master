# Spike Dossier：我們需要走進 MCP 這個 rabbit hole 嗎？（issue #80 主張稽核）

> 本文是對 [issue #80](https://github.com/Aidenzich/road-to-master/issues/80)(2025 年初撰寫)的逐條主張稽核,
> 由 read-only spike pipeline 產出(三模型 A/B/n 盲測勝出版本),證據規則:URL + 逐字引文 + 擷取日期,一手來源優先。
> 結案定調:issue 的分析層主張基本獲證實(可靠性優勢未經檢驗、幻覺風險只是轉移),被時間推翻的只有時效性事實(Claude 獨佔微調)。

- **調查對象**：GitHub issue https://github.com/Aidenzich/road-to-master/issues/80（撰於 2025 年初）
- **調查性質**：read-only spike，交付物為證據支撐的稽核報告，非實作計畫、非 PR
- **所有 web 來源擷取日期（retrieval date）**：2026-07-04
- **證據規則**：https-only；primary source 優先（modelcontextprotocol.io spec/changelog、Anthropic/OpenAI/Google/Microsoft 官方文件與公告、arXiv 論文）；每個 fact 附 URL + verbatim 引用 + 擷取日期；vendor 行銷語 / blog / 二手彙整明確標記為 secondary 或 opinion；未能存取的來源記為 unknown 而非猜測。
- **判決標記**：成立 ／ 已被推翻 ／ 部分成立 ／ 證據不足。

---

## 0. 執行摘要（先給結論）

**對標題問題「Do we need to go down the MCP rabbit hole?」的整體回答：**

到 2026 年，這個 issue 的**兩層主張要分開看**：

1. **關於「MCP 只是 Claude 專屬、其標準地位是空談」的部分 —— 已被時間推翻。** MCP 在 2025 年成為跨廠商 de-facto 標準：OpenAI、Google、Microsoft、AWS 全數落地官方支援，並在 2025-12-09 由 Anthropic 捐給 Linux Foundation 新成立的 Agentic AI Foundation（AAIF），OpenAI/Block 為共同創始、Google/Microsoft/AWS/Cloudflare/Bloomberg 為 supporter。作為「互通連接層（interop/connectivity layer）」，MCP 已經贏了，忽視它不再是理性選項。

2. **關於「MCP 的可靠性優勢是未經檢驗的假設、它不消除 hallucination 只是轉移風險」的部分 —— 大體上被 2026 年的證據證實（issue 這一層的懷疑是對的）。** 至今**沒有**任何嚴謹的 head-to-head benchmark 證明「MCP tool-calling 比讀 OpenAPI/REST 文件更可靠」；相反，多個 primary benchmark 顯示前沿模型在真實 MCP 任務上失敗率高達 40–70%，且工具數量一多，tool-selection 準確率崩到 13.6%。同時 MCP 沒有消除、反而**放大並改變了風險結構**——把「人寫的 tool description」從「誠實錯誤來源」變成「攻擊者可控的 injection 通道」（tool poisoning、rug pull、lethal trifecta、供應鏈後門、mcp-remote/Inspector 的 RCE CVE）。

**所以「要不要走進 rabbit hole」的精準答案是：**
- **要，但只走一半。** 把 MCP 當作「與外部／第三方連接、跨 client 可攜」的**標準邊界層**採用——這是它真正的、已被市場驗證的價值。
- **不要**掉進「把幾十上百個 MCP tool 定義塞進 context、讓模型裸選工具」這個真正的 rabbit hole——那正是 token bloat、選錯工具、與安全風險的集中點。2025–2026 的勝出模式是 **code-execution over MCP（Anthropic 官方宣稱 token 減少 98.7%）+ 動態工具檢索（RAG-over-tools）**，而非 raw tool-call fan-out。
- 對你自己完全掌控兩端的**內部信任工具**，plain function calling／直接 SDK 呼叫往往更簡單也更省——MCP 的邊際價值在「外部互通」與「跨廠商 client 可攜」，不在內部。

以下逐條稽核。

---

## 1. 主張一：「目前只有 Claude 針對 MCP 做過 fine-tune；OpenAI(GPT-4o)、Gemini 可能達不到同等效果」

**判決：已被推翻（作為標準化主張）；效能子主張為 證據不足。**

### 證據 A —— MCP 已成跨廠商 de-facto 標準（FACT）

OpenAI 於 2025-03-26 在 Agents SDK 落地 MCP 支援。Sam Altman 的原始 X 貼文因 HTTP 402 無法直接存取（記為 unknown），其原文由 TechCrunch 逐字轉述（**secondary，press**）：

> "People love MCP and we are excited to add support across our products. [It's] available today in the Agents SDK and support for [the] ChatGPT desktop app [and] Responses API [is] coming soon!"
> — https://techcrunch.com/2025/03/26/openai-adopts-rival-anthropics-standard-for-connecting-ai-models-to-data/（擷取 2026-07-04；方括號為 TechCrunch 編輯插入）

OpenAI Responses API 官方文件（**primary**）：

> "Remote MCP servers can be any server on the public Internet that implements a remote Model Context Protocol (MCP) server."
> — https://developers.openai.com/api/docs/guides/tools-connectors-mcp（擷取 2026-07-04）

Google Cloud 官方公告（2025-12-11，**primary**）：

> "Today we're announcing the release of fully-managed, remote MCP servers. Google's existing API infrastructure is now enhanced to support MCP, providing a unified layer across all Google and Google Cloud services."
> — https://cloud.google.com/blog/products/ai-machine-learning/announcing-official-mcp-support-for-google-services（擷取 2026-07-04）

Microsoft Copilot Studio GA（2025-05-29，**primary**）：

> "Model Context Protocol (MCP) is now generally available in Microsoft Copilot Studio!"
> — https://www.microsoft.com/en-us/microsoft-copilot/blog/copilot-studio/model-context-protocol-mcp-is-now-generally-available-in-microsoft-copilot-studio/（擷取 2026-07-04）

### 證據 B —— 多廠商治理正式化（FACT，已由我親自向 primary 覆核）

Anthropic 官方公告（2025-12-09，**primary**，我已用 WebFetch 覆核逐字）：

> "Anthropic is donating the Model Context Protocol to the Linux Foundation's new Agentic AI Foundation"
> Co-founders：**Anthropic, Block, OpenAI**；Supporters：**Google, Microsoft, AWS, Cloudflare, Bloomberg**
> — https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation（擷取 2026-07-04）

MCP 共同作者 David Soria Parra 在 TechCrunch（**secondary**）：

> "The main goal is to have enough adoption in the world that it's the de facto standard."
> — https://techcrunch.com/2025/12/09/openai-anthropic-and-block-join-new-linux-foundation-effort-to-standardize-the-ai-agent-era/（擷取 2026-07-04）

**小結：** 由 OpenAI/Google/Microsoft/AWS 共同背書、置於中立基金會治理，是「跨廠商標準」最強的證據。issue 寫作時（2025 初）此事尚未發生，故此主張作為標準化預測**已被推翻**。

### 效能子主張（Claude 是否真的比較會用 MCP）—— 證據不足

到 2026 年沒有 primary lab report 顯示 Claude 在 MCP tool-use 上獨佔優勢。唯一找到的是單一 secondary 彙整站（mcpplaygroundonline.com）給出跨模型分數（如 MCP-Atlas 上 Claude 領先、cross-server 上 Gemini 領先、single-server 上 GPT/GLM 領先）。**這些具體百分比與模型版本名（GPT-5.4、Opus 4.6/4.7、Gemini 3.1 Pro、GLM-5.1）僅來自單一 aggregator，我判定為 low-confidence，不足以支撐任何一方獨佔的結論。** 可確定的是：沒有證據支持「只有 Claude 有效」，該子主張**證據不足**（且方向上不利於原主張）。

---

## 2. 主張二：「MCP 本質上只是另一種 Function Calling」

**判決：已被推翻（此化約在當前 spec 下不成立）。**

我已親自向 primary spec（2025-11-25）用 WebFetch 逐字覆核。當前 spec 明列**六個 primitive**，Tools（≈傳統 function calling）只是其中一個 bullet：

> **Servers offer any of the following features to clients:**
> - **Resources**: Context and data, for the user or the AI model to use
> - **Prompts**: Templated messages and workflows for users
> - **Tools**: Functions for the AI model to execute
>
> **Clients may offer the following features to servers:**
> - **Sampling**: Server-initiated agentic behaviors and recursive LLM interactions
> - **Roots**: Server-initiated inquiries into URI or filesystem boundaries to operate in
> - **Elicitation**: Server-initiated requests for additional information from users
>
> — https://modelcontextprotocol.io/specification/2025-11-25（擷取 2026-07-04）

同頁 Base Protocol 明列 **"Stateful connections"** 與 capability negotiation，並列出 Progress tracking／Cancellation／Logging 等 utilities。逐一超出 function calling 的證據（**primary，均為 FACT**）：

- **Resources（資料/context 通道，非函式呼叫）**：「servers to expose resources… such as files, database schemas, or application-specific information. Each resource is uniquely identified by a URI.」— /specification/2025-11-25/server/resources
- **Prompts（user-controlled 模板，如 slash commands）**：「expose prompt templates… Prompts are designed to be **user-controlled**」— /specification/2025-11-25/server/prompts
- **Sampling（反向：server 回呼 host 的 LLM）**：「servers to request LLM sampling ("completions" or "generations") from language models via clients… with no server API keys necessary.」方向與 function calling **相反**。— /specification/2025-11-25/client/sampling
- **Elicitation（session 中向使用者索取輸入）**：於 **2025-06-18** 版新增（changelog：「Add support for **elicitation**… (PR #382)」）。— /specification/2025-06-18/changelog
- **Roots（filesystem/URI 邊界協商）**、**Notifications/logging/progress**（JSON-RPC one-way messages、list_changed 通知）。
- **三種 transport 與 stateful session**：**2025-03-26** 以 Streamable HTTP 取代 HTTP+SSE（changelog：「Replaced the previous HTTP+SSE transport with a more flexible **Streamable HTTP transport** (PR #206)」）。
- **OAuth 2.1 授權框架**：**2025-03-26** 新增（「Added a comprehensive **authorization framework** based on OAuth 2.1 (PR #133)」），**2025-06-18** 把 server 歸類為 OAuth Resource Server 並要求 RFC 8707 Resource Indicators。

**spec 版本演進（primary changelog，均逐字）：**

| 版本 | 主要新增（verbatim 摘要） |
|---|---|
| 2024-11-05 | 初版：Resources/Tools/Prompts + Sampling/Roots；無 auth；transport = stdio + HTTP+SSE |
| 2025-03-26 | OAuth 2.1 授權框架；Streamable HTTP 取代 HTTP+SSE；tool annotations；audio content |
| 2025-06-18 | server 歸類 OAuth Resource Server（RFC 8707/9728）；**elicitation**；structured tool output；移除 JSON-RPC batching |
| 2025-11-25（現行 stable） | OpenID Connect Discovery；incremental scope consent；tool icons；sampling 內可帶 tools/toolChoice；實驗性 **tasks** |
| 2026-07-28（Release Candidate，鎖定於 2026-05-21，尚未 final） | **stateless protocol core**、Extensions 框架、Tasks、正式 deprecation policy（見主張八） |

**小結：** function calling ≈ 只等於 Tools 這一個 primitive。當前 MCP 還標準化了 context 供給、user 模板、反向 LLM 呼叫、mid-session 使用者輸入、操作邊界、notification/logging 層、stateful session（三種 transport）與 OAuth 2.1 授權。「只是 function calling」的化約**已被推翻**。（但務實地說——對「只用 Tools primitive」的多數實務使用者，這個化約在**體感**上仍有部分道理；見第 6、第 8 節與「so what」。）

---

## 3. 主張三：「MCP 相對 (Open)API 文件的可靠性優勢，是未經檢驗的假設」

**判決：成立（此主張本身站得住——比較性證據至今不存在，且 MCP 自身在規模化下並不可靠）。**

### 證據 A —— 沒有 head-to-head「MCP vs OpenAPI 文件閱讀」的嚴謹 benchmark（關鍵發現）

跨多個 primary benchmark 與論文搜尋後，**找不到**任何「固定任務、對照 (a) MCP tools vs (b) 讀 OpenAPI/REST 文件 vs (c) code-execution」的受控研究。所有強力主張「MCP 比 REST 可靠」的來源皆為 **blog/vendor（secondary，非實證）**（如 buildbetter.ai「AI agents fail unpredictably on REST and succeed reliably on MCP」）。這**正好證實** issue 的框架：可靠性優勢是**被斷言、未被比較量測**的。

唯一相關 primary 研究是「OpenAPI→MCP 轉換」而非兩者對照（AutoMCP，arXiv 2507.16044）：

> "From a stratified sample of 1,023 tool calls, 76.5% succeeded out-of-the-box." … "After minor fixes, averaging just 19 lines of spec changes per API, AutoMCP achieved 99.9% success."
> — https://arxiv.org/html/2507.16044v3（擷取 2026-07-04）

其失敗多源於 **OpenAPI 契約瑕疵**（安全 scheme、base URL、header/token 前綴），這說明「由 OpenAPI 衍生的工具本來就能用得不錯」，並**未**證明 MCP 內在更可靠。

### 證據 B —— MCP 自身在真實任務上不可靠（primary，已覆核）

MCP-Universe（arXiv 2508.14704，2025-08-20，我已 WebFetch 逐字覆核）：

> "even SOTA models such as GPT-5 (43.72%), Grok-4 (33.33%) and Claude-4.0-Sonnet (29.44%) exhibit significant performance limitations"
> — https://arxiv.org/abs/2508.14704（擷取 2026-07-04）

LiveMCP-101（arXiv 2508.15760，**primary**）：

> "even frontier LLMs achieve a success rate below 60%"（GPT-5 領先，overall 58.42）
> — https://arxiv.org/html/2508.15760v1（擷取 2026-07-04）

MCP-Bench（arXiv 2508.20453，Accenture，**primary**）：schema compliance 近乎完美（多個模型 >98%）但 overall task 分數僅 ~0.69–0.75——**「產出格式正確的 tool call」≠「完成任務」**。（具體 0.749/0.715 分數來自 search 文字層，標為 medium-confidence。）

τ-bench / τ²-bench（Sierra，arXiv 2406.12045 / 2506.07982，**primary**）量測**一致性**：pass^k 隨 k 指數衰減，τ-bench retail「pass^8 values below 25%」；τ²-bench 雙控 telecom 域讓 GPT-4 從 74% 掉到 34%。（最新前沿模型已大幅飽和 telecom，故落差與世代/域相關。）

### 證據 C —— 工具數量一多，選擇準確率崩潰（primary）

RAG-MCP（arXiv 2505.03275）：baseline（全部工具塞 context）準確率 13.62%，檢索式選擇回到 43.13%（「more than triples… reduces prompt tokens by over 50%」）。「Skill Shadowing」（arXiv 2605.24050）引用並補充：「tool selection accuracy drops from above 90% with fewer than 30 candidates to 13.6% with 11,100」，且退化主因是**選擇干擾（skill shadowing，約 68%）**，而非單純 context 長度。

**小結：** issue 這條主張**成立**——「MCP 更可靠」的比較證據至今不存在，而 MCP 自身在規模化下的可靠性**很差**。

---

## 4. 主張四：「MCP 不消除 hallucination，只是轉移風險（tool 定義仍是人寫、轉換錯誤仍在）」

**判決：部分成立——方向正確，但嚴重低估（issue 大約只預見了約九類風險中的一類）。**

### 已被證實的「錯誤仍在」部分（FACT）

wrong-tool selection 與 parameter hallucination 在實證中確實持續：MCP-Bench 類評測把失敗區分為 malformed parameters／wrong tool selection vs cognitive failures；context bloat 讓 tool-selection 準確率「from ~95%… to ~71% with the full GitHub MCP server loaded」（secondary 測試）。**issue 的核心斷言成立。**

### issue 未預見的「新風險類別」（這才是重點）

**spec 本身現在承認 tool description 不可信**（我已向 primary 覆核，這是最有力的一句）：

> "descriptions of tool behavior such as annotations should be considered **untrusted**, unless obtained from a trusted server."
> — https://modelcontextprotocol.io/specification/2025-11-25（擷取 2026-07-04）

1. **Tool poisoning（透過 tool description 做 prompt injection，Invariant Labs 於 2025-04 提出）**：
   > "A Tool Poisoning Attack occurs when malicious instructions are embedded within MCP tool descriptions that are invisible to users but visible to AI models."
   > — https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks（擷取 2026-07-04）
   PoC 中被污染的工具誘導 agent 讀取 `~/.ssh/id_rsa` 並外送。這直接把 issue 口中「人寫的 tool 定義」變成攻擊者可控的 injection 通道。

2. **Rug pull / line jumping（核准後才變更定義）**：Invariant Labs「A malicious server can change the tool description after the client has already approved it.」；Elena Cross「MCP tools can mutate their own definitions after installation… by Day 7 it's quietly rerouted your API keys to an attacker.」spec 無強制 re-approval 機制。

3. **供應鏈：首個 in-the-wild 惡意 MCP server**：`postmark-mcp` v1.0.16（2025-09-17）植入 BCC 後門，每日外流 3,000–15,000 封企業信；週安裝約 1,500。— Snyk / The Hacker News（**secondary，但為真實事件**）。

4. **lethal trifecta（private data + untrusted content + 外送通道）**：Simon Willison（**expert opinion**）；Invariant Labs 的 GitHub MCP exfiltration PoC 指其「not a flaw in the GitHub MCP server code itself, but rather a fundamental architectural issue.」

5. **Confused deputy / OAuth token 竊取**：spec 2025-11-25 授權章節自承「Attackers can exploit MCP servers acting as intermediaries… By using stolen authorization codes, they can obtain access tokens without user consent.」並強制多項 MUST 級緩解。

6. **已指派 CVE（程式碼層，非模型層）**：`mcp-remote` **CVE-2025-6514**（CVSS 9.6，OS command injection RCE，惡意 server 回傳 crafted `authorization_endpoint`）；Anthropic MCP Inspector **CVE-2025-49596**（CVSS 9.4，DNS-rebinding RCE，0.14.1 修復）；官方 SQLite reference server 的 SQL injection（archived 前已被 fork 5,000+ 次）。

7. **學術攻擊面 survey**：MCPSecBench（arXiv 2508.13220）列 17 種攻擊；「Systematic Analysis of MCP Security」（arXiv 2508.12538）列 31 種；MCP-at-First-Glance（arXiv 2506.13538）掃 1,899 個 server，發現「8 distinct vulnerabilities -- only 3 of which overlap with traditional software vulnerabilities」「5.5% exhibit MCP-specific tool poisoning」。

**小結：** 「風險只是轉移」在**字面上成立**，但**精神上誤導**——風險從「低嚴重度的偶發錯誤」被推入「高嚴重度的對抗性領域」（RCE、資料外洩、供應鏈）。故判**部分成立**。

---

## 5. 主張五：「若 LLM 更會讀 API，MCP 價值可能下降」

**判決：部分成立——兩股力量並存；network effect 佔了上風，但 code-execution 確實在改寫 MCP 的「消費方式」（部分繞過裸 tool-call，但是包住 MCP 而非丟棄它）。**

### network effect（FACT）

官方 MCP Registry 於 2025-09-08 **preview** 上線（截至 2026-07-04 未見 GA）：「the Model Context Protocol (MCP) Registry—an open catalog and API… now available in preview.」目錄站規模已達數萬：Glama 站標題「Open-Source MCP Servers – 50,845 in the Glama Registry」、PulseMCP「18,240+」（**secondary，live directory**，擷取 2026-07-04）。

### 反向動能：code-execution 勝過裸 tool-call（primary vendor）

Anthropic「Code execution with MCP」（2025-11，**primary**）：

> "This reduces the token usage from 150,000 tokens to 2,000 tokens—a time and cost saving of 98.7%." … "developers should take advantage of this strength to build agents that interact with MCP servers more efficiently."
> — https://www.anthropic.com/engineering/code-execution-with-mcp（擷取 2026-07-04）

Cloudflare「Code Mode」（**primary vendor**）：

> "LLMs are better at writing code to call MCP, than at calling MCP directly."
> — https://blog.cloudflare.com/code-mode/（擷取 2026-07-04）

**關鍵細微差別（FACT）**：這兩篇都把 MCP server **當作 code API 來包裝**——是改善 MCP 的消費方式，而**非**拋棄 MCP。故原預測只**部分**成立：LLM 確實更會寫 code/讀 API，這侵蝕了「裸 tool-call」模式；但生態以「registry 網路效應 + 把 MCP 重新包成 code API」回應，**強化了作為標準/連接層的 MCP**。

---

## 6. 主張六：「少量 API → plain Function Calling 更有效率；工具多 → MCP 有價值，但 hallucination 風險仍在」

**判決：部分成立——精神正確，但需修正：工具一多時，把定義塞進 context（不論 function calling 或 MCP）都會退化；勝出模式是動態發現/RAG-over-tools/code execution，而非「上手就載入更多工具」。**

- context 成本（primary vendor）：Anthropic「as agents are connected to thousands of tools, they'll need to process hundreds of thousands of tokens before reading a request.」
- 退化實證（primary）：RAG-MCP baseline 13.62% → 檢索式 43.13%；工具多時準確率崩潰（見主張三證據 C）。
- 沒有權威「最大工具數 N」——primary 指引皆為**定性**：把**進入 context 的工具集**保持小，超過約數十個就改用檢索/code-execution。

**小結：** 「工具多 → MCP 有價值」只有在 MCP **搭配動態發現/RAG/code-execution** 時才成立；單純把大量 MCP tool 定義載入 context，會重演 plain function calling 的 prompt bloat 與選擇退化。

---

## 7. 主張七（issue 外，補做的替代方案地景調查）

**判決：非成立/推翻式判決，而是地景圖。到 2026 年，沒有任何標準在「tool-connectivity 層」取代 MCP；死掉的是 ChatGPT Plugins；A2A/AGNTCY/ACP 是「不同層、互補」而非競品。**

| 方案 | 維護者 | 2026 狀態 | 相對 MCP 的層級 |
|---|---|---|---|
| ChatGPT Plugins | OpenAI | **已死（2024-04-09 全面關閉）** | tool-connectivity（前身，後繼為 GPT Actions/OpenAPI） |
| A2A (Agent2Agent) | Linux Foundation（Google 捐贈） | **成熟/成長** | **不同層**（agent↔agent），互補 |
| agents.json | Wildcard AI | niche/存活（stateless、建於 OpenAPI 之上） | 同層，重疊/競爭 |
| llms.txt | Jeremy Howard | niche（約 10% 網域） | doc 檢索層，非執行層，鄰接 |
| 直接 OpenAPI function calling | 開放標準 | 成熟/普及 | 同層，常為 MCP 底層 |
| Vendor connectors（OpenAI/Google/AWS） | 各廠 | 成熟，且**日益改以 MCP 實作** | 同層，多半 built-on MCP |
| AGNTCY / ACP | Linux Foundation / IBM-BeeAI | 成熟/早期 | agent 基礎設施/agent↔agent，不同層 |

關鍵 primary 引用：
- ChatGPT Plugins 廢止：「On March 19, 2024, you will no longer be able to install new plugins… You will be able to continue existing conversations until April 9, 2024.」（OpenAI dev community 轉載，canonical help 頁 403，日期跨來源一致）。
- **A2A 與 MCP 互補（我已 WebFetch 逐字覆核）**：「Both the MCP and A2A protocols are essential… highly complementary needs.」「A2A is about agents _partnering_ on tasks, while MCP is more about agents _using_ capabilities.」— https://a2a-protocol.org/latest/topics/a2a-and-mcp/（擷取 2026-07-04）
- agents.json：「built on top of the OpenAPI standard… Agents.json is stateless.」（GitHub wild-card-ai/agents-json）——同層競品但採用 niche。

**小結：** MCP 在 tool-connectivity 層無人取代；對它的主要**壓力來自內部**（code-execution 改變消費方式），主要**「競品」其實是互補的上層**（agent 編排/agent↔agent，且假設底下用 MCP）。

---

## 8. 主張八（issue 外，補做的批評語料庫）

**判決：非成立/推翻式；下為分群、最強引用、以及「spec 是否已回應」。**

| # | 批評（分群） | 最強來源（類型） | spec 是否回應？ | 裁決 |
|---|---|---|---|---|
| 1 | 協定層無驗證（授權） | Elena Cross「S in MCP」（opinion） | **是**：OAuth 2.1(2025-03-26) → Resource Server/RFC 8707(2025-06-18) → OIDC(2025-11-25) | **大體已解**（限 transport auth） |
| 2 | prompt injection/lethal trifecta/tool poisoning | Willison（expert）；arXiv 2506.13538、2508.12538（實證） | 僅有 Security Best Practices 指引，無協定層防禦 | **仍成立**（最強、最被引） |
| 3 | rug pull（核准後變更定義） | Elena Cross（opinion） | 協定層未解，無強制完整性/pinning | **仍成立** |
| 4 | stateful session 過度設計 vs stateless HTTP | Degtyarev（opinion）；maintainer 自承 | **進行中**：2026-07-28 RC「MCP is now stateless at the protocol layer」「`Mcp-Session-Id`… removed」（我已覆核） | **在 2026-07-28 RC 中已解**（2025 全年仍成立） |
| 5 | 雙連線 SSE/transport 複雜 | Degtyarev（opinion） | **是**：2025-03-26 以 Streamable HTTP 取代 HTTP+SSE | **已解** |
| 6 | tool 定義 token bloat/context 經濟 | Degtyarev + vendor 量測（quasi-empirical） | 非 spec 修正，僅 client 端（Tool Search）緩解 | **spec 層仍成立** |
| 7 | 「早該用 REST/OpenAPI，重造輪子」 | GitHub Discussion #1093（primary 場域） | 設計分歧，無法用改版「解決」；arXiv 2507.16044 指 OpenAPI-gen 僅約 4.5% 採用 | **作為設計批評仍在**（但批評者偏好的路徑採用率也低） |
| 8 | 非決定性/可靠性 | agentbuild(opinion)；arXiv 2508.06418(實證) | 屬 LLM 層問題，spec 只能 nudge | **仍成立** |
| 9 | 版本churn/每 3 個月破壞性變更 | GitHub SEP-1400(primary)；DEV 文（opinion） | **進行中**：SEP-1400 語意化版本 + 2026-07-28 RC 正式 deprecation policy | **部分已解**（batching 於 2025-03-26 加、2025-06-18 移除，證實此抱怨） |

代表性最強引用：
- Willison（**expert opinion**，「prompt injection」一詞創造者）：「we've known about the issue for more than two and a half years and we still don't have convincing mitigations for handling it.」— https://simonwillison.net/2025/Apr/9/mcp-prompt-injection/
- maintainer 自承 stateful 是瓶頸：「The friction of stateful connections has become a bottleneck for managed services and load balancing.」— https://blog.modelcontextprotocol.io/posts/2025-12-19-mcp-transport-future/
- 2026-07-28 RC（我已覆核存在）：「a stateless core that scales on ordinary HTTP infrastructure」「MCP is now stateless at the protocol layer」——直接回應分群 4。

**小結：** spec 已明確修掉的：**授權**與**雙連線 SSE transport**；**stateful 過度設計**正在 2026-07-28 RC 修。**仍站得住**的：prompt injection/lethal trifecta/tool poisoning（最強、最多實證）、rug-pull 完整性缺口、token/context bloat（僅 client 端緩解）、非決定性。**無法靠改版解決**的哲學批評：「早該用 OpenAPI」。

---

## 9. 綜合回答：「Do we need to go down the MCP rabbit hole?」

嚴格依上述已稽核證據：

- **作為互通標準——是，已無懸念。** 跨廠商採用 + Linux Foundation 治理（主張一），使 MCP 成為與外部/第三方工具連接、且跨 client 可攜的預設邊界層。押注它會消失，已被證據否定。
- **作為「可靠性魔法」與「消除 hallucination」——issue 的懷疑大體被證實（主張三、四）。** 沒有比較性證據證明 MCP 比讀 API 文件更可靠；MCP 自身在規模化下可靠性差；它把風險推向對抗性領域。
- **真正該避開的 rabbit hole，不是 MCP 本身，而是「把大量 MCP tool 定義塞進 context、讓模型裸選」這個具體反模式**（主張五、六、八分群 6）——那是 token bloat、選錯工具、與安全風險的交集。2025–2026 的正解是 **code-execution over MCP + 動態工具檢索**。

---

## 10. 對本 repo（headless agent orchestration）的 so-what——嚴格由證據推導

1. **在「邊界」採用 MCP，不在「內核」。** 對接**外部/第三方/使用者自帶**的工具、需要跨廠商 client 可攜時 → 用 MCP（它贏了標準戰，主張一）。你**完全掌控兩端**的內部工具 → plain function calling／直接 SDK/code 呼叫通常更簡單省 token；MCP 的邊際價值在互通，不在內部（主張六、七）。

2. **以 code-execution + 檢索消費 MCP，而非 raw tool-call fan-out。** 直接證據：Anthropic 宣稱 token 減 98.7%、Cloudflare Code Mode、RAG-MCP 把準確率從 13.6% 拉回 43%。對 headless orchestrator，這同時解決 context 經濟與 tool-selection 退化兩個問題（主張五、六）。

3. **把每個 tool description 與 server response 視為 untrusted input——spec 自己這樣講。** 具體防線（主張四）：避免在單一 agent session 內湊齊 lethal trifecta（私有資料存取 + 不可信內容 + 外送通道）；對第三方 server 做 tool 定義 pin/verify（rug-pull）；稽核 `mcp-remote`(CVE-2025-6514) 與 MCP Inspector(CVE-2025-49596) 版本；不接受非本 server audience 的 token。

4. **不要因為「MCP 化」就期待 reliability 免費升級。** 可靠性瓶頸在 LLM 的 tool selection/parameterization 與工具數量，不在協定（主張三、六、八分群 8）。要靠 eval（如 τ²-bench 的 pass^k 一致性思路）與縮小 in-context 工具集來換可靠性。

5. **架構分層對齊生態。** tool 連接用 MCP，agent↔agent 編排用 A2A（互補、不同層，主張七）——不要把兩者當二選一。

---

## 11. 仍然真正未知（含 closure recipe）

1. **「MCP tool-calling vs OpenAPI 文件閱讀 vs code-execution」的受控對照**——公開文獻中**不存在**（主張三）。
   - *Closure recipe*：取 20–30 個固定任務，對每個 API 準備三種介面（MCP server／原始 OpenAPI spec+HTTP client／已封裝 SDK 供 code-exec），同一組模型各跑 n≥8 次，量 pass@1 與 pass^k 與 token 成本。這是尚無人做、但可決定「邊界層是否非 MCP 不可」的關鍵實驗。

2. **各廠商是否真的針對 MCP 做 fine-tune、以及跨模型 MCP 效能排名**——屬 vendor-internal，無 primary 佐證；現有跨模型分數僅單一 aggregator（主張一效能子主張）。
   - *Closure recipe*：以 MCP-Universe/LiveMCP-101 公開 harness，在你自己的 API 金鑰下對候選模型自跑，取 primary 數字，不採信 aggregator。

3. **官方 MCP Registry GA 日期**——截至 2026-07-04 仍為 preview，未見 GA 公告。
   - *Closure recipe*：追 https://blog.modelcontextprotocol.io 與 registry repo release，出現 「general availability」字樣即閉合。

4. **2026-07-28 spec（stateless core 等）是否已 final**——目前是 RC（鎖定 2026-05-21，預計 2026-07-28 final），本報告擷取於 2026-07-04，尚未 final。
   - *Closure recipe*：2026-07-28 後重讀 /specification 的 latest 與 changelog，確認 stateless core 是否落地為正式版。

5. **供應鏈風險量級**（如 Shodan 曝露數、postmark-mcp 下載量）——來自 vendor blog，未獨立核實，視為 indicative。

---

## 附錄：已親自向 primary 覆核的關鍵引用（WebFetch，2026-07-04）

- MCP spec 2025-11-25 六 primitive + 「Stateful connections」+ 「descriptions… should be considered untrusted」— modelcontextprotocol.io/specification/2025-11-25
- Anthropic 捐 MCP 給 Linux Foundation AAIF（co-founders/supporters 名單）— anthropic.com/news/donating-the-model-context-protocol-…
- MCP-Universe 分數 GPT-5 43.72% / Grok-4 33.33% / Claude-4.0-Sonnet 29.44% — arxiv.org/abs/2508.14704
- A2A 與 MCP「complementary / partnering vs using」— a2a-protocol.org/latest/topics/a2a-and-mcp/
- 2026-07-28 RC「stateless protocol core」「`Mcp-Session-Id`… removed」— blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/

（其餘引用來自平行 research agent 的 primary/secondary 蒐證，來源 URL 已逐條標於各主張中；secondary 與 medium/low-confidence 數字均已明確標記。）
