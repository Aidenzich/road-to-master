# LLM-Training-Types — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | The distinct training types/stages of modern LLMs and what each stage changes |
| Venue | unknown |
| Year | unknown |
| Authors | unknown |
| Official Code | unknown |
| Venue Kind | survey |

## 📚 Sources

本文以下列四篇論文為主要證據來源，全部取得全文並存入本地快取；下表僅記錄實際抓到的預印本版本，正式會議版本另在正文以推論標註。

| # | Title | Venue (fetched) | Year | arXiv id | Access |
|-|-|-|-|-|-|
| 1 | Training language models to follow instructions with human feedback (InstructGPT) | arXiv preprint | 2022 | arXiv:2203.02155 | fetched (arXiv 全文) |
| 2 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | arXiv preprint | 2023 | arXiv:2305.18290 | fetched (arXiv 全文) |
| 3 | LoRA: Low-Rank Adaptation of Large Language Models | arXiv preprint | 2021 | arXiv:2106.09685 | fetched (arXiv 全文) |
| 4 | Instruction Tuning for Large Language Models: A Survey | arXiv preprint | 2023 | arXiv:2308.10792 | fetched (arXiv 全文) |

依作者的先驗知識（未在抓取到的 arXiv 全文內逐字查證，屬合理推論）：#1 為 NeurIPS 2022、#3 為 ICLR 2022、#2 為 NeurIPS 2023、#4 為 instruction tuning 主題的綜述型 preprint；正式 camera-ready 版本的數字可能與 preprint 略有差異。

## 為什麼要拆開 seed 的四欄表

issue 的 seed 用「Pre-training / Post-Pretraining / Fine-tuning / Instruct-tuning」四欄，把每一格都當成一個並列的「階段」，並附上量級式的資料量與算力猜測。實際文獻並不支持這種平面的四欄切法：instruction tuning 在綜述裡與 SFT 幾乎是同義詞，preference optimization（RLHF/DPO）是 SFT 之後另一個獨立的對齊階段，而 LoRA 根本不是階段而是任一微調階段都能套用的方法。綜述把 instruction tuning 定義為在 (instruction, output) 配對上做監督式再訓練，用途是 bridges the gap between the next-word prediction objective of LLMs 與使用者要模型服從指令的目標——這一句就說明它調的是「照指令輸出」的行為介面，而非重新灌入知識。

下表是把 seed 依四篇文獻重新校正後的版本；量級猜測換成論文裡的真實數字，並把「方法」與「階段」分開：

| 訓練環節 | 屬性 | 目標函數 / 機制 | 典型資料量（本文文獻的真實數字） | 主要改變什麼 |
|-|-|-|-|-|
| Pre-training | 階段 | 自監督 next-token 預測 | 數千億～上兆 tokens（如綜述引用 T5 的 pre-training stage 為基準） | 知識與通用能力的主要來源 |
| Continued / continual pre-training | 階段（可選） | 同 pre-training 目標，換領域/時段語料 | 額外語料，量級小於初始 pre-training | 補充領域/時效語料，仍是同一種目標 |
| SFT ＝ instruction tuning | 階段 | 在 (instruction, output) 上做監督式 fine-tuning | InstructGPT 的 SFT dataset contains about 13k training prompts；綜述稱 fine-tuning costs 0.2% 的 T5 pre-training 算力 | 讓模型服從指令、輸出格式與風格對齊人類期待 |
| Preference optimization（RLHF / DPO） | 階段 | RLHF：RM＋PPO；DPO：single binary cross-entropy loss | InstructGPT：RM 33k、PPO 31k prompts、6B RM | 依人類偏好排序重排輸出分布 |
| LoRA / PEFT | 方法（與階段正交） | 凍結權重、注入低秩 BA 更新 | GPT-3 175B 上可 10,000 times 更少可訓練參數 | 不改階段語意，只改「怎麼便宜地做微調」 |

這張表的關鍵修正有二：其一，把 seed 併在一起的 Fine-tuning 與 Instruct-tuning 收斂成同一個 SFT 階段，因為綜述明說 SFT allows for a more controllable and predictable model behavior compared to standard LLMs，兩者調的是同一件事；其二，把 LoRA 從「階段欄」抽出來獨立成「方法」，因為它可以套在任何一個微調階段上。

## Pre-training 提供知識，對齊階段大多只改介面

要判斷「instruct-tuning 是否加入新知識」這個 folk claim，得先看知識從哪來。綜述整理的 superficial alignment hypothesis 主張 the knowledge and capabilities of a model are almost acquired in the pre-training stage，而後續的對齊訓練（含 instruction tuning）teaches models to generate responses under user-preferred formalizations。若這個假設成立，seed 那句「instruct-tuning 不加入新事實知識、只改善解析與回應能力」就大致站得住腳——但要注意它在論文裡是一個 hypothesis（LIMA 用約 1k 筆資料驗證），不是被廣泛複驗的定律。

## SFT / instruction tuning 改變的是行為，但綜述自己也有保留

綜述並非無批判地全盤肯定 instruction tuning：它明白記錄了一種強烈質疑，認為 SFT captures surface-level patterns and styles (e.g., the output format) rather than comprehending and learning——也就是模型可能只學到輸出的表面格式，而非真正理解任務。這一點同時支持又反向提醒 seed 的說法：支持「不加新知識」的部分，卻也暗示「改善解析與回應能力」可能被高估成「只學到格式」。

InstructGPT 提供了最乾淨的行為改變證據：在人類評比下 outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having over 100x fewer parameters。這兩個模型架構相同、只差在是否用人類資料微調，所以偏好差距來自對齊而非規模或新知識。同時 InstructGPT models show improvements in truthfulness and reductions in toxic output generation，顯示對齊確實動到了「真實性/毒性」這類行為面向，而非純粹的格式包裝。

## Preference optimization：從 RLHF 三步到 DPO 一步

InstructGPT 的對齊是三步：(1) supervised fine-tuning (SFT), (2) reward model~(RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)。它的資料規模在論文中寫得很具體——the SFT dataset contains about 13k training prompts；RM 與 PPO 各為 33k、31k prompts，且 In this paper we only use 6B RMs，用 6B 而非 175B 當 reward model 以省算力與穩定 RL。

DPO 把上面第 (2)(3) 步收成一步：它 optimize a policy using a simple binary cross entropy objective，而且是 without learning an explicit, standalone reward model or sampling from the policy during training。其損失函數把「偏好資料上的分類」寫成閉式：

$$
\mathcal{L}_\text{DPO}(\pi_{\theta}; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}\left[\log \sigma \left(\beta \log \frac{\pi_{\theta}(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta \log \frac{\pi_{\theta}(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

DPO 的理論賣點是它 implicitly optimizes the same objective as existing RLHF algorithms (reward maximization with a KL-divergence constraint)——這也是標題那句 Your Language Model Is Secretly a Reward Model 的意思：策略模型本身就隱含了 reward。實驗上在 TL;DR 摘要，DPO has a win rate of approximately 61 %（temperature 0.0），略勝 PPO 的約 57%，說明取消顯式 RL 並未犧牲品質。

## LoRA / PEFT 是正交的「方法」而非「階段」

把 LoRA 放進階段欄是 seed 最大的類別錯誤。LoRA 的做法是 freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture，用一個低秩更新近似全量微調：

$$
h = W_0 x + \Delta W x = W_0 x + BA x, \qquad W_0+\Delta W=W_0+BA
$$

它的效益是資源面的，不改變任何階段的語意：在 GPT-3 175B 上 LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times，checkpoint 從 350GB to 35MB，且 LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3。部署時把 BA 併回權重，we do not introduce any additional latency during inference。正因為 LoRA 可套在 SFT 或 preference 階段之上，它與「哪一個階段」是正交的兩個軸。

## 一個端到端的量化對照

把 InstructGPT 的招牌數字走一遍，最能看清「對齊階段到底改了什麼、又沒改什麼」。同一個 GPT-3 架構，175B 版本未對齊；1.3B 版本先用 the SFT dataset contains about 13k training prompts 做 SFT，再用 33k/31k 偏好與 PPO 資料對齊，最後 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3。表面上這像「小模型贏大模型」，但它只證明了在「照指令回答」這個偏好指標上對齊的效果，並不證明 1.3B 具備 175B 的知識廣度；相反地，論文承認對齊過程 comes at the cost of lower performance on certain tasks that we may care about（SQuAD、DROP、HellaSwag、WMT 都退步），並要靠 mixing PPO updates with updates that increase the log likelihood of the pretraining distribution 才壓得住。換算算力，綜述給的量級是 fine-tuning costs 0.2 %（相對 T5 pre-training）——所以「對齊很便宜、只改介面」大方向對，但「完全沒有代價」這半句被 alignment tax 直接推翻。

## 🧪 Critical Assessment

### 四個來源的可比較性

四篇跨了不同世代與任務，直接橫比很危險。InstructGPT 的 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3 是 2022 年、GPT-3 家族、OpenAI API prompt 分布上的人類偏好；LoRA 是 2021 年、以 RoBERTa/GPT-2/GPT-3 的任務式指標；DPO 是 2023 年、以 win rate 衡量。沒有任何一組是對「當代前沿模型」的 head-to-head，因此本文用它們拼出的階段圖譜，比較像「各世代各自的截面」而非同一把尺，這點在引用其數字時必須保留。

### 基準與指標是否足以支撐結論

偏好型指標本身就有靶心自訂的風險。InstructGPT 的偏好是在它自家 API 的 prompt 分布上、由自聘標註者評出；DPO 的 win rate of approximately 61 % 也依賴自動評比與參考完成度，而論文自陳 automatic evaluation metrics such as ROUGE can be poorly correlated with human preferences。也就是說，「對齊讓輸出更好」很大程度是相對於「評比者偏好的那種好」，把它讀成通用能力提升會過度外推。

### 新穎性 vs 重新包裝：方法與階段被混為一談

seed 的分類錯誤本質是把一個方法當階段。LoRA 的貢獻是資源面的——we do not introduce any additional latency during inference、10,000 times 更少參數——它不新增也不移除任何訓練階段的語意；同理 instruction tuning 與 SFT 在綜述裡是同一件事的兩個名字。把這些並列成四個對等欄位，會讓讀者誤以為它們是流水線上互斥的四步，這是命名與分類的重新包裝而非真實的階段劃分。

### 被主張的問題真的解決了嗎、對實務重要嗎

seed 最想驗證的那句「instruct-tuning 不加新知識、只改解析與回應」，證據是分裂的。一方面 superficial alignment hypothesis 說 the knowledge and capabilities of a model are almost acquired in the pre-training stage，支持前半句；另一方面 InstructGPT 的 alignment tax 顯示對齊 comes at the cost of lower performance on certain tasks that we may care about，直接否證「只改介面、零代價」的隱含結論。而且 superficial alignment 只是一個以約 1k 筆資料驗證的 hypothesis，尚未在當代大模型上被廣泛複驗，因此對實務的正確讀法是：對齊主要改行為與格式、成本低，但會有可量測的能力折損，且「知識全來自 pre-training」仍屬待證而非定論。

## 🔗 Related notes

- [InstructioinTuningWithGPT4](../InstructioinTuningWithGPT4/)
- [Lora](../Lora/)
- [ChatGPT](../ChatGPT/)
