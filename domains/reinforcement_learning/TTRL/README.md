# TTRL: Test-Time Reinforcement Learning — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | TTRL: Test-Time Reinforcement Learning |
| Venue | arXiv preprint (2504.16084v3, cs.CL/cs.LG) |
| Year | 2025 |
| Authors | Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui 等 16 位作者（Tsinghua University、Shanghai AI Lab） |
| Official Code | https://github.com/PRIME-RL/TTRL |
| Venue Kind | paper |

## 🧭 第一原理：在沒有標準答案的測試資料上跑 RL

TTRL 想回答一個很尖銳的問題：當一批 reasoning 題目「只有題目、沒有標準答案（ground-truth label）」時，能不能還用 reinforcement learning 去更新一個已經預訓練好的 LLM？傳統 RL 的前提是 reward 訊號可得，而這裡的核心困難正是在推論當下如何在拿不到 ground-truth 的情況下估計 reward。作者把這個設定命名為 Test-Time Reinforcement Learning，屬於 Test-Time Training (TTT) 的一支，和只做 inference 的 Test-Time Inference（例如 majority voting、Best-of-N）並列在 Test-Time Scaling 這個大傘之下。

TTRL 的關鍵觀察是：Test-Time Scaling 裡常用的 majority voting，本身就能產生「夠好」的 reward 來驅動 RL 訓練。換句話說，模型不需要外部監督，只要對同一題重複取樣、投票取多數，就能造出一個代理標籤（pseudo-label），再用它算出 rule-based reward。這把「TTS 的投票聚合」和「RL 的線上更新」接在一起，形成一個自我強化的迴圈。

形式化來說，給定一個 prompt $x$，policy $\pi_\theta(y \mid x)$ 取樣出 $N$ 個候選輸出 $\{y_1,\dots,y_N\}$，用投票聚合出一個共識輸出 $y^*$ 當作最優動作的代理，環境依 $y$ 與 $y^*$ 是否一致給出 reward $r(y, y^*)$。優化目標就是最大化期望 reward $\max_\theta \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}[r(y,y^*)]$，並以梯度上升更新 $\theta \leftarrow \theta + \eta\nabla_\theta\mathbb{E}[r(y,y^*)]$。這裡沒有任何 $y_t$ 標籤進入，訓練訊號完全由模型自己的多數投票產生。

reward 的具體算法非常樸素：先用 majority voting 從 $N$ 個抽取出的答案 $P=\{\hat y_i\}_{i=1}^N$ 中選出出現最多次的預測當作估計標籤 $y$，再對每個候選答案做 rule-based 比對——答對（等於多數答案）給 $1$，否則給 $0$。對應到 $R(\hat y_i, y)=1$ 若 $\hat y_i=y$，否則為 $0$。論文附的 pseudo-code 直接說明了這個「投票取多數、再逐一比對打分」的流程：

```python
from collections import Counter

def majority_voting_reward_fn(outputs):
    # Assigns reward 1 to each output whose answer matches the majority answer, else 0.
    answers = [extract_answer(output) for output in outputs]
    counts = Counter(answers)
    majority_answer, _ = counts.most_common(1)[0]
    rewards = [1 if ans == majority_answer else 0 for ans in answers]
    return rewards

outputs = llm.generate(problem, n=N)
rewards = majority_voting_reward_fn(outputs)
```

實作上，TTRL 對每個 benchmark 各自獨立地跑一次 GRPO：peak learning rate 為 $5\times10^{-7}$ 的 cosine schedule、AdamW optimizer；rollout 階段對每題取樣 $64$ 個回答（Qwen2.5-Math 與 LRM 用 temperature $1.0$、其餘用 $0.6$）做投票估標籤，再 downsample 到 $32$ 個回答拿去訓練。maximum generation length 對 LRM 設 $32{,}768$、其餘設 $3{,}072$ tokens，MATH-500 / AMC / AIME 2024 分別跑 $10$ / $30$ / $80$ 個 episode。All experiments were conducted on 8 * NVIDIA A100 80GB GPUs，這個「先投票、後抽樣」策略在降低算力成本的同時仍維持強效果。

主結果非常搶眼：以 pass@1 計，Qwen2.5-Math-7B 在四個 benchmark 上從 $12.9$ / $35.6$ / $46.7$ / $29.1$（AIME 2024 / AMC / MATH-500 / GPQA）提升到 $40.2$ / $68.1$ / $83.4$ / $27.7$，其中 AIME 2024 相對進步約 $211.6\%$、平均進步約 $76.5\%$。注意 GPQA 反而略降 $1.4$ 分，是唯一退步的格子。下表節錄兩個代表性 backbone：

| Model | AIME 2024 | AMC | MATH-500 | GPQA | Avg |
|-|-|-|-|-|-|
| Qwen2.5-Math-1.5B | 7.7 | 28.6 | 32.7 | 24.9 | 23.5 |
| Qwen2.5-Math-1.5B w/ TTRL | 15.8 | 48.9 | 73.0 | 26.1 | 41.0 |
| Qwen2.5-Math-7B | 12.9 | 35.6 | 46.7 | 29.1 | 31.1 |
| Qwen2.5-Math-7B w/ TTRL | 40.2 | 68.1 | 83.4 | 27.7 | 54.9 |

### 一個具體例子：AIME 2024 上為什麼「錯的標籤也能給對的 reward」

用 Qwen2.5-Math-7B 走一遍 AIME 2024 的單題訓練步：對一題抽 $64$ 個回答、抽答案、投票。此時 base model 的輸出極為分散——最常出現的那個答案只佔全部預測的 $16.6\%$，因此多數投票估出的標籤與 ground truth 只有 $37\%$ 的機率一致（label accuracy 只有 $37\%$）。直覺上這樣的 pseudo-label 應該爛到不能用，但實際量到的 reward accuracy 卻高達 $92\%$。原因是作者所謂的「Lucky Hit」：math verifier 是靠「比對」給 rule-based reward，對一個本來就答錯的候選，只要它跟（同樣錯的）估計標籤「不一樣」，verifier 就會照樣給出 $0$ 的負向 reward，而這正好是我們想要的正確 reward。於是即使 label 幾乎沒估對，密集且分散的錯誤答案讓大多數 reward 仍然正確；論文並觀察到 label accuracy 很少超過 $50\%$，reward accuracy 卻穩定維持在 $75\%$ 以上。接著把這 $64$ 個回答 downsample 成 $32$ 個做 GRPO 更新，跑滿 $80$ 個 episode，pass@1 就從 $12.9$ 一路升到 $40.2$。

TTRL 一個反直覺的性質是它能「超越自己的訓練訊號」。既然訓練靠初始模型的多數投票，直覺上初始 maj@n 應該是效能上限（這也是傳統 self-training 的上限）；但實測 avg@16 最終比初始 maj@16 高出 $20$ 分以上，avg@64 也在所有 benchmark 上穩定超過 Qwen2.5-Math-7B 的 maj@64。作者用「模型把自己拉起來（lifts itself up by its own bootstraps）」形容這個現象。另一個上限是直接在測試資料上用真標籤做 RL（作者稱為 RL leakage），TTRL 的曲線竟然貼近這個洩題上限；連 1.5B 模型都能在 MATH-500 上 starting from a subpar performance of $32.7$，improved by $123.2\%$ 到 $73.0$。

TTRL 也不是萬靈丹。作者明講它在演算法層面與一般 RL 無異，因此繼承了對資料難度敏感、強烈依賴先驗、可能崩潰等特性。最直接的失敗來源是先驗不足：把 MATH-500 依標註難度切成 L1–L5，用 Qwen2.5-Math-1.5B 分別訓練，準確率增益從 L1 的 $+45.4$（$\uparrow175.3\%$）單調衰減到 L5 的 $+16.8$（$\uparrow75.3\%$），response length 的壓縮幅度也同步下降，顯示 backbone 的 prior knowledge is insufficient to handle the complexity of the data。另一個失敗來源是 RL 超參數：把 temperature 從 $0.6$ 調到 $1.0$ 會提高 entropy 與探索，但配上不當的 batch size 會讓 entropy 持續不降而訓練崩潰。

| MATH-500 難度 | L1 | L2 | L3 | L4 | L5 |
|-|-|-|-|-|-|
| Backbone Accuracy | 25.9 | 33.0 | 36.3 | 32.5 | 22.3 |
| w/ TTRL Accuracy | 71.2 | 76.2 | 76.3 | 58.7 | 39.2 |
| Δ 增益 | +45.4 | +43.2 | +40.0 | +26.2 | +16.8 |

## 🧪 Critical Assessment

### 問題設定是真需求，但「test-time」的框法有點放大其新穎性
「用無標籤資料做 RL」確實是實務痛點：大規模標註昂貴、而困難的新題目源源不絕（論文用 o3 在 ARC-AGI-2 只解出 $4\%$ 當作動機）。這個問題是真的。但要留意 TTRL 把它包裝成「test-time」其實有點名詞放大——方法本體是「對一批未標註資料用多數投票造 pseudo-label 再跑 GRPO」，這件事和是不是「測試時」並無本質綁定，換成任何無標註訓練集都成立。作者自己在 related work 也承認這與 self-rewarding、self-training、以及 concurrent 的 self-play RL（如 Absolute Zero、Genius）高度相鄰，差別主要在用 majority voting 估 reward 以緩解 reward hacking。因此其貢獻更像是「把既有元件（majority voting + GRPO + rule-based reward）在一個乾淨設定下組裝並認真做實證」，而非全新機制。

### 基線偏薄，且評測基準與訓練資料高度重疊
最需要打問號的是評測設計。TTRL 的主要對照組只有 backbone 本身，作者也坦承「與先前 SOTA 的比較看起來並不公平（different setup）」。更關鍵的是：它「在 benchmark 的題目上訓練、又在同一批題目上評測」。雖然沒有用到答案標籤，但反覆在這批 prompt 上做 online RL、再報這批 prompt 的 pass@1，天然對這個資料分佈過擬合——這正是一種以自身方法強項來界定的評測（benchmark 由作者圍繞自身方法的優勢來定義）。作者用 OOD 遷移實驗與 RL leakage 上限來緩解這個疑慮，方向正確，但主表的絕對數字仍應理解為「在該題集上自我適應後」的結果，而非對未見題目的泛化。GPQA 上 Qwen2.5-Math-7B 反而退步、Mistral-Nemo 在 AIME 掉到 $0$，也提示效果高度依賴 backbone 的數學先驗。

### 「超越 maj@n 上限」很吸睛，但機制解釋仍偏經驗
論文最亮的賣點是 avg@16 超過初始 maj@16 逾 $20$ 分、且逼近洩題上限 RL leakage。這個現象若成立，意義重大。但要注意：所謂「上限」是作者自行定義的兩個參照（初始模型的 maj@n、與洩題訓練），而「為何能超越」目前只有 Lucky Hit 這類經驗性論證與單一 backbone（Qwen2.5-Math）的曲線支撐，缺乏收斂性或泛化界的理論分析，作者也把 theoretical analysis 列為 future work。Lucky Hit 的成立條件其實很微妙——它依賴「錯誤答案高度分散」，一旦模型的錯誤模式集中（產生一致但錯誤的多數答案），reward accuracy 可能崩掉；這個邊界條件論文沒有系統性地掃過。

### 真實世界關聯：對「有強先驗的可驗證任務」有效，外推需保守
就落地而言，TTRL 的甜蜜區是「答案可用 rule-based verifier 比對、且 backbone 已具備足夠先驗」的任務（數學、選擇題）。難度消融顯示：一旦題目超出模型先驗（L4–L5），增益快速縮小，等於說它更像是「把模型既有能力壓榨出來」而非「學會新知識」。對於沒有乾淨可驗證答案的開放式任務（對話、agentic、科學發現），majority voting 這個 reward 代理是否還可靠，論文只列為未來方向而未驗證。因此我的判讀是：這是一個乾淨、可複現、實證扎實的 preliminary study（且開源），但把它讀成「LLM 可無上限自我進化」則是明顯過度外推——它的天花板恰恰被 backbone 的先驗與 verifier 的可比對性鎖死。

## 🔗 Related notes

- [PPO: Proximal Policy Optimization](../ppo/) — TTRL 除 GRPO 外也以 PPO 作為相容的 RL 演算法之一。
