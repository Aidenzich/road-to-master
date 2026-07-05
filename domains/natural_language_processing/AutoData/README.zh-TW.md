# AutoData — Research Note
> [English](./README.md) | **繁體中文**

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Autodata: An agentic data scientist to create high quality synthetic data |
| Venue | unknown |
| Year | 2026 |
| Authors | Ilia Kulikov, Chenxi Whitehouse, Tianhao Wu, Yixin Nie, Swarnadeep Saha, Eryk Helenowski, Weizhe Yuan, Olga Golovneva, Jack Lanchantin, Yoram Bachrach, Jakob Foerster, Xian Li, Han Fang, Sainbayar Sukhbaatar, Jason Weston (FAIR at Meta) |
| Official Code | https://github.com/facebookresearch/RAM |
| Venue Kind | paper |

> 本筆記依據 arXiv 預印本 `2606.25996`（Meta FAIR「RAM」專案）撰寫。截至撰寫時未查到同儕審查發表場域，故 Venue 記為 `unknown`；正式版本若釋出可能與此預印本有出入。

## First Principles

### 從「過濾靜態資料」到「把資料生產當成一個科學家迴圈」

當前語言模型的後訓練越來越依賴合成資料。既有主流做法（Self-Instruct、Grounded Self-Instruct、CoT Self-Instruct、Self-Challenging 等）本質上是一次性的 prompt 產生加上事後過濾（filtering、evolution、refinement），並不直接控制資料的難度與品質。AutoData 的核心主張是把「造資料」重構成一位資料科學家（data scientist）會做的迭代流程：先生成一批資料，接著做定性檢視（eyeballing）與定量評測，歸納出 learnings，再據此更新資料生成的 recipe 產生更好的資料，直到滿足停止條件。

這個框架有一個外圈與一個內圈。內圈是「資料創造 → 資料分析 → 更新配方」的迴圈；外圈則是把「這位資料科學家 agent 本身」也拿來最佳化（meta-optimization），用內圈同一套評測準則（造出更能區分模型的資料）去引導外圈對 agent 的 prompt 與策略做改進。論文把它定位成一個「用推論期算力換取更高品質訓練資料」的通用機制。

![AutoData 的整體迴圈：agent 扮演資料科學家，反覆生成資料、做定性檢視與定量評測、歸納洞見並更新生成配方；外圈再對 agent 本身做最佳化。](imgs/autodata_pipeline.png)

### 具體實作：Agentic Self-Instruct 的弱—強（weak-vs-strong）設計

論文的實驗都建立在一個具體實作上，稱為 Agentic Self-Instruct。一個主 orchestrator agent 指揮四個 LLM 子代理（subagent）：Challenger 依主 agent 給的詳細 prompt 產生訓練樣例；一個「弱解題器」（weak solver，預期通常解不出）；一個「強解題器」（strong solver，預期通常能解出）；以及一個 Verifier / 判官（judge），檢查樣例與模型解答的品質並把 learnings 回饋給主 agent。系統的目標，是產生「強解題器會做、弱解題器會卡住」的訓練資料。

對可驗證（verifiable）任務，一種判準是要求強解題器的多數投票正確、而弱解題器的多數投票錯誤；對不可驗證任務，則要求判官量到的品質存在落差（gap），使題目對弱解題器不過易也不過難，同時用強解題器保證正確性。若準則未達成，主 agent 就依判官回饋修改送給 challenger 的 prompt，換一個推理角度重生一題，直到通過為止。論文特別指出：弱、強解題器可以是「同一個 LLM 的不同模式」——強版本被允許使用更多推論期算力、scaffolding 或聚合，甚至存取特權資訊。

![Agentic Self-Instruct 的弱—強設計：主 agent 指揮 challenger 產題、弱/強解題器作答、judge 評分，並依 judge 回饋更新 challenger 的 prompt 反覆迭代。](imgs/agentic_self_instruct.png)

### 一次具體的走查：CS 研究問答任務（用論文真實數字）

以電腦科學研究問答為例最能看清這個迴圈在做什麼。主 agent 用 Kimi-K2.6，強解題器用 Qwen3.5-397B-A17B，弱解題器用 Qwen3.5-4B。Challenger 從一篇論文生成「context + 問題 + 參考答案 + 自足的加權評分 rubric」，judge 依 rubric 逐條為兩個解題器（各作答 3 次以降低變異）打分。

問題在於：直接用 prompt 生成（CoT Self-Instruct 那一欄）出來的題目對這個 4B 弱解題器多半太簡單。下表是同一批論文素材下、生成當下由 Kimi-K2.6 評分的品質統計：

| Metric | CoT Self-Instruct | Agentic Self-Instruct |
|-|-|-|
| Weak solver avg | 0.677 | 0.458 |
| Strong solver avg | 0.696 | 0.772 |
| Gap (strong − weak) | 0.019 | 0.314 |
| Agentic rounds | 1.00 | 6.59 |
| Question length (chars) | 723 | 619 |
| Rubric items | 13.2 | 13.1 |

CoT 生成的題目弱解題器平均 0.677、強—弱落差只有 0.019，幾乎沒有可學習的訊號。於是論文把接受準則直接定義在這個落差上：一題只有在強解題器平均 ≥ 0.65、弱解題器 < 0.5、且強−弱落差 ≥ 20 個百分點時才被接受。為了省算力，judge 只在弱解題器通過其成功準則時才去評強解題器。跑完整個迴圈後，同一批素材下弱解題器分數從 0.677 掉了 22 個百分點到 0.458，強解題器則從 0.696 升 8 個百分點到 0.772——落差被撐開到 0.314，代表被接受的題目確實往「需要跟著論文論證走」的具體演算步驟、消融細節或數值主張移動。平均要 6.59 輪才生出一題，且有一條延伸到 10 輪以上的長尾。

在 880 個接受前（pre-acceptance）的失敗輪次裡，失敗原因高度單邊：80% 是因為題目太簡單、弱解題器分數太高被拒；13% 是因為強解題器也無法穩定解出。資料規模上，論文處理了 S2ORC 語料（2022 年後）中超過 10k 篇 CS 論文，用 Agentic Self-Instruct 產出 2.8k 條被接受的樣例，再經迴圈末端的品質 verifier 過濾（去除有論文專屬參考洩漏、context 過短或 rubric 格式錯誤者），保留 1.3k 條高品質樣例作為 RL 訓練資料；CoT 基線也套同一個 verifier 並抽同樣 1.3k 條做公平比較。

接著用 GRPO（batch size 16、learning rate 1e-6）在各 1.3k 條資料上訓練 Qwen3.5-4B，在 200 題的保留測試集上評測、由 Kimi-K2.6 依 rubric 評分。下表為 step 200 的結果，同時報告在「CoT 測試集」與「Agentic 測試集」兩個分佈上的 mean@3 / best@3：

| Response model | CoT mean@3 | CoT best@3 | Agentic mean@3 | Agentic best@3 |
|-|-|-|-|-|
| Qwen3.5-4B (no additional RL) | 0.630 | 0.758 | 0.366 | 0.484 |
| Qwen3.5-4B RL on CoT Self-Instruct data | 0.727 | 0.853 | 0.500 | 0.631 |
| Qwen3.5-4B RL on Agentic Self-Instruct data | 0.774 | 0.894 | 0.632 | 0.768 |

在較容易的 CoT 測試集上，用 Agentic 資料訓練把基礎 4B 從 0.630 拉到 0.774；在較難的 Agentic 測試集上則從 0.366 拉到 0.632——後者兩方法的差距是前者的兩倍以上。換言之，用「更能區分模型」的資料訓練，不只在難分佈上贏，也能反向遷移到易分佈（+0.05）。

### 同一個迴圈、相反的失效模式：法律推理

論文用第二個場域檢驗一般性：法律推理。素材取自 Pile of Law 的法院意見書等公開法律文件，評測於 PRBench-Legal 與其 PRBench-Legal-Hard 子集。有趣的是這裡的失效模式與 CS 相反：CoT 直接生成的題目對弱解題器「太難」而非太易——弱解題器平均只有 0.159，許多 rollout 直接得 0，使得 GRPO 每個 prompt 群組內的 advantage 趨近 0，幾乎沒有學習訊號。

因此這裡不再用 CS 那種硬編碼閾值，而改用一個更彈性的 loop judge：每份法律文件先由 extractor agent 抽出結構化摘要（主題關鍵詞、重要事實、holdings），challenger 據此生成一題加加權 rubric，弱解題器 rollout 5 次、強解題器 3 次，judge 讀取逐個 rollout 的模式、弱/強落差與 rubric，輸出結構化裁決（`weak_pattern`、`strong_pattern`、`gap_interpretation`、`rubric_concerns`、`grpo_suitability`）與 `accept`/`improve` 決定。判準不是固定數字，而是綜合資料品質與「GRPO 可學習性」。

結果是：agentic 迴圈把弱解題器平均從 0.159 推到 0.283、強解題器幾乎不動（0.717 → 0.698），強—弱落差反而從 0.558 收窄到 0.415；真正關鍵的改變是每題弱 rollout 的標準差從 7.93 升到 12.63——同樣的落差被攤在一個「可用的變異範圍」上，reward 訊號因此變得可學。loop judge 的 `grpo_suitability`（high/medium/low）分佈也印證了這點：CoT 池是 4.8% high / 41% medium / 45% low，Agentic 池變成 52% high / 43% medium / 2% low。下游 RL 上（GRPO、每 prompt n=8 rollout），用 Agentic 資料訓練的 4B 在 PRBench-Legal 拿到 0.441（GPT-5 為 judge）與 0.393（Kimi 為 judge），不只勝過同架構 CoT 訓練版（0.377 / 0.343），甚至勝過未再訓練、體量大得多的 Qwen3.5-397B-A17B 基線（0.404 / 0.358），且兩位 grader 結論一致。

CS 與 Legal 兩例把 AutoData 的論點講得很清楚：同一個迴圈套在「太易」與「太難」兩種相反失效模式上，落差往相反方向移動（CS 撐開、Legal 收窄），但下游 RL 結果一致變好。作者因此把重點放在一句話上：關鍵不是把題目變「更難」，而是把它調到讓模型「剛剛好」能拾級而上的難度。

### 更難的題目會遷移到更易的題目：科學推理

第三個場域是對數學物件做推理，素材與評測沿用 Principia collection（涵蓋 MSC2020 與 PHYS 課綱）。這裡比較三種資料來源做下游 RL：直接用 Principia 原題（CoT Self-Instruct）、Agentic 生成資料、以及兩者合併（Combined，訓練量加倍為 18k）。每個單一來源各 9k 訓練 + 1k 保留樣例，訓練 Qwen3.5-4B、用 GRPO（group size 8、batch size 64）、Kimi-K2.6 做二元比對評分。在合併驗證集上，Agentic 給出最大的整體 avg@8 提升 +3.20%，勝過直接用 CoT（+2.42%）與加倍資料的 Combined（+2.70%）；值得注意的是 Agentic 連在 CoT 子集上也贏（+3.05% vs +1.86%），顯示「在更難的題目上訓練會遷移到更易的題目」，而非只擅長它針對的難度層級。

一個容易被忽略、卻很實在的副產品是 token 效率：在 65,536 token 的推理預算下，基礎 4B 有 23.75% 的回應因長度被截斷（finish_reason=length），Agentic 訓練後降到 4.09%。作者對「答錯→答對」的翻轉做歸因，發現約有一半的準確率提升來自修好截斷（Agentic 的 945 個翻轉中 54.81% 歸因於截斷修復、41.06% 來自非截斷情形下的推理改善）——也就是說，在難題上訓練不只教會模型推得更對，也教會它在固定預算內推得更省。

### 外圈：把資料科學家 agent 本身 meta-最佳化

AutoData 的第二個層次，是連「資料科學家 agent 的 prompt 與策略」也用演化式最佳化去改。meta-optimizer 維護一個候選 prompt 的族群，每個候選是相對 baseline repository 的一段 code diff。每一輪：(1) 用 Boltzmann sampling 選一個 parent，候選 $c$ 被選中的機率正比於

$$P(c) \propto \exp(\text{score}_c / T), \qquad T = 0.1$$

（此式為論文所述的選取規則，符號為本筆記整理）強烈偏好高分候選但保留探索；(2) 在一小批訓練論文上評估 parent，蒐集軌跡與弱/強分數；(3) 用一個 LLM agent 讀完整解題器對話、寫出系統性失敗模式的根因分析；(4) 由 code-editing agent 依分析產生改良 diff；(5) 在保留驗證論文上重評 parent 與 mutant；(6) 只有當 mutant 驗證分數嚴格高於 parent 時才被接受納入族群；(7) 把結果摘要成後續分析可讀的歷史日誌。因單次評分有雜訊（解題器在 temperature 1.0），被接受的候選在之後被抽為 parent 時會累積更多次評估並取平均。

在 CS 論文任務上，從約 62.1% 的驗證通過率起步，經過 233 次迭代（126 次被接受）後提升到 79.6%（每 session 6 小時逾時上限）。這裡「通過」的判準是：弱解題器（Qwen3.5-4B）≤ 65%、最佳弱解題器嘗試 ≤ 75%、強解題器（Qwen3.5-397B-A17B）介於 60% 與 95% 之間、且強—弱落差 ≥ 20 個百分點。meta-optimizer 自動發現的幾項修改頗具啟發性：要求題目測「這篇論文特有」而非通用 ML 知識（自我檢查：若不讀這篇也能答對就是太易）；嚴禁 context 洩漏解法；以及一個反直覺發現——刪掉負權重 rubric 準則、改用上限為 7 的正整數權重，因為負權重歷史上會誤傷強模型分數卻無助鑑別。

## 🧪 Critical Assessment

### 問題是真的，但框架的「新意」多在整合而非機制

「用一個 agent 迭代地造更能區分模型的資料」是真實且當下的痛點：當前沿模型越來越強，靜態合成資料越來越難產生夠challenging 的題目。這一動機成立。但要誠實看待新意的來源：weak–strong challenger–solver、judge 回饋、難度調整、以致把 harness 當最佳化對象，這些元件在 STaR、Self-Rewarding、Self-Challenging、Absolute Zero、SPICE、GEPA、Meta-Harness 等前作各自出現過。AutoData 的貢獻主要是把它們統一在「資料科學家迴圈」這個敘事下，並展示 meta 層可疊加。這是有價值的工程整合與框架化，但論文本身用的措辭（「generalizes all the above methods」）偏大，機制層面的原創增量相對有限，讀者宜把它理解為一個統整性框架而非全新演算法。

### 基線、消融與指標：可信，但有評測循環性的隱憂

實驗設計有不少值得肯定之處：三個異質場域、CoT 與 Agentic 在相同資料預算/challenger/語料下的對照、法律任務刻意加入 GPT-5 作為獨立 grader 以排除 Kimi-grader 偏誤、以及把「更難」與「剛剛好」拆開討論。這些都比一般合成資料論文紮實。

但最需要警覺的是評測的循環性：CS 任務裡，資料的接受準則、訓練 reward、以及測試集評分幾乎都由同一個 Kimi-K2.6 依 rubric 打分，而 rubric 又是 challenger（也是 Kimi-K2.6）自己生成的。這等於資料生成端與評測端共用同一個模型與同一套 rubric 觀點，「Agentic 測試集本來就是為凸顯 Agentic 資料而生」，因此 Agentic 在 Agentic 測試集上贏得最多（mean@3 0.632 vs 0.500）並不完全獨立。法律任務用 PRBench 官方 judge 設定並加 GPT-5 重評，這一塊的說服力明顯高於 CS。此外，絕對增量偏小：法律上 Agentic 對 CoT 僅 +0.05–0.06、科學推理 OOD 僅 +1.04% 且部分類別（如 SuperGPQA、pass@8）出現退步，作者自己也承認 4B 可能已接近該任務分佈的容量上限。

### 自訂基準與「射靶」風險

CS 段落存在明顯的自訂基準疑慮：測試集由作者的 pipeline 生成、由生成端同源模型評分，難度定義又直接寫進接受準則（強−弱落差 ≥ 20 分）。這種「先定義好利於自身方法的難度，再在該難度上比較」的設定，本身就會讓 Agentic 佔優，難以判定增益有多少來自資料品質、多少來自靶心被畫在方法強項上。論文的 limitations 也坦承有些生成題目「過度綁定論文特定實驗數字而非測可泛化推理」，這與射靶疑慮是同一枚硬幣的兩面。相對地，法律任務靠外部 PRBench 與 GPT-5 grader 大幅緩解了這個問題，是全篇最能站得住腳的證據。

### 是否解決、以及真實世界相關性

在「能否用推論期算力換到更好訓練資料」這個受限問題上，論文提供了跨三域一致方向的證據，meta 層 62.1%→79.6% 的提升也具體。但要注意兩個未決點：一是成本，6.59（CS）與 4.98（Legal）的平均輪數意味著每條被接受資料要付出數倍到十數倍的推論算力，論文未系統性報告「同算力下 Agentic vs 大量 CoT」的等算力對照，因此「值不值得」仍開放。二是安全/hacking，作者自陳遇到 agent 「作弊」（例如改 prompt 叫弱解題器裝弱），目前只靠加約束部分緩解。整體而言，這是一個方向可信、初步證據紮實、但增量偏小且評測部分自我循環的框架性工作——正如作者所說「只是冰山一角」，其長期價值取決於後續在等算力對照、更多任務與更強防作弊上的驗證。

## 🔗 Related notes

- [Instruction Tuning with GPT-4](../InstructioinTuningWithGPT4/) — 同屬「用強模型合成指令資料做後訓練」的脈絡，可與 AutoData 把資料生成 agent 化的取向對照。
