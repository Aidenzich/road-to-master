# 第三章:三種切模型的方法——資料並行、張量並行、管線並行

> [回導讀](../README.zh-TW.md) | [證據與來源](08-evidence.md)

> 第二章建立了通訊字彙(rank、四種集體通訊)與成本模型(α+nβ)。這章回答:**當單卡放不下或不夠快,有哪幾種把工作分到多卡的方法?每一種把「資料放在哪張卡」怎麼改?逼出哪一種、多少集體通訊?** 這章只講「機制與逼出的通訊」;把這些通訊放到本平台的慢網路上會有多痛,是第四章的事。

三種基本切法,對應「切什麼」的三個維度:
- **資料並行**:每張卡放**整個模型**,切的是**資料**(不同卡吃不同批次)。
- **張量並行(TP)**:切的是**單一層內部的矩陣**——同一層被橫剖分到多卡。
- **管線並行(PP)**:切的是**層與層之間**——不同卡負責模型的不同「段」。

---

## 一、資料並行:DDP → ZeRO → FSDP

### 起點 DDP:每張卡一份完整模型,每步 AllReduce 梯度

最單純的多卡訓練叫 **DDP**(Distributed Data Parallel,分散式資料並行):**每張卡都放一份完整的模型**,四張卡各吃四分之一的批次資料、各自算出梯度,然後用一次 **AllReduce** 把四份梯度求平均(第二章的第一個小例子),大家拿到一致的平均梯度後同步更新。這樣等效於用四倍大的批次訓練,而且做得好的話通訊能藏在計算後面——DDP 把梯度分批(bucket)成塊,某塊一算完就非同步 AllReduce,與還在跑的反向計算重疊(「overlapping allreduce collectives with computations during backwards」[W8];「The AllReduce operation on gradients can start before the local backward pass finishes」[W9])。

DDP 的問題正是第一章的帳本:**每張卡都存完整的 16Ψ**。8B 模型 = 每卡 128 GB,單卡放不下,DDP 也救不了——因為 DDP 沒有分攤記憶體,只分攤資料。要突破記憶體牆,得把那 16Ψ 也切開。這就是 ZeRO 的動機。

### ZeRO 的三段分片:把 16Ψ 逐步切給 N 張卡

**ZeRO**(Zero Redundancy Optimizer)觀察到:DDP 讓每張卡存一份完全相同的參數/梯度/優化器狀態,是浪費。它分三個階段,逐步把這三樣切成碎片、每張卡只存 1/N,要用時再臨時湊齊:

| 做法 | 每張卡狀態記憶體 | 相對 DDP 的通訊量 | 用到的集體通訊 |
|---|---|---|---|
| **DDP** | $16\Psi$(不分片) | 基準(1×) | 每步一次梯度 **AllReduce** |
| **ZeRO-1**(切優化器狀態) | $2\Psi+2\Psi+12\Psi/N$ | 「no additional communication」[W10] | 效果同 AllReduce(≈ ReduceScatter+AllGather) |
| **ZeRO-2**(再切梯度) | $2\Psi+14\Psi/N$ | 「no additional communication using Pos and Pg」[W10] | ReduceScatter + AllGather |
| **ZeRO-3**(再切參數) | $16\Psi/N$ | **「a maximum of 1.5x communication」**[W10] | 前後向用 AllGather 臨時湊齊參數 + 反向 ReduceScatter |
| **FSDP Full Shard** | 同 ZeRO-3 量級 | 與 ZeRO-3 相近 | 同上 |

三段的直覺:ZeRO-1 先切最大的那塊(優化器狀態 12Ψ),ZeRO-2 連梯度也切,ZeRO-3 連參數本身都切。切得越多,單卡記憶體越省,但要用某一層前得先用 AllGather 把它的參數碎片湊回完整——通訊變多。ZeRO-3 因此付「最多 1.5 倍」於 DDP 的通訊量[W10]。

### FSDP:PyTorch 原生的 ZeRO-3,相近但不可畫等號

**FSDP**(Fully Sharded Data Parallel)是 PyTorch 內建、行為與 ZeRO-3 相近的分片方案:同樣把參數/梯度/優化器切片,前向進某層前 AllGather 湊回該層完整參數、算完就丟,反向再 AllGather 湊回 + 用 ReduceScatter 把梯度歸約並切回碎片。官方教學即稱 FSDP「can be considered a decomposition of DDP's all-reduce into reduce-scatter and all-gather operations」[W12]——正好呼應第二章「ReduceScatter + AllGather = AllReduce」的等式。

**但兩者不可直接等同(反面證據,信心:高):** FSDP 論文明言它是受 ZeRO 啟發但「a native solution」、「intrinsically different」的重新設計[W11],差別在 PyTorch 原生的單元包裝(unit wrapping)與反向預取——「issues the next AllGather before the current ReduceScatter」[W11],也就是在做某一層的梯度歸約時,就先預取下一層的參數,讓兩者重疊。行為量級相近,但包裝邊界、預取、offload 細節不同,調校與數值行為不能互相假設。

### 時間線與一個容易被忽略的差異:collective 的「粒度」由配置決定

把一步訓練攤成時間線:前向 → 反向 → 優化器更新。DDP 的通訊(梯度 AllReduce)能與反向計算重疊[W8][W9];FSDP/ZeRO-3 則在前後向湊回參數時發起 AllGather/ReduceScatter,並靠預取讓通訊與計算重疊[W11]。

這裡有個只看「總 byte 數」會漏掉的關鍵(信心:中,需 trace 或已知配置佐證)。第二章的成本模型是 $\alpha + n\beta$:總 byte 只決定 $n\beta$ 那一項,而 **collective 被切成幾次、每次多大,決定了要付幾次 α**。關鍵是——**這個「切幾次」不是模型層數的固定函數,而是由配置決定的粒度:**

- **DDP / ZeRO-1/2 的梯度 collective 粒度,由 gradient bucket(梯度桶)大小決定。** 一手文件只說「the Reducer organizes parameter gradients into buckets, and reduces one bucket at a time」[W8]——桶越大、次數越少、每次 payload 越大;桶越小則反之。ZeRO-2 只切梯度、參數全程常駐,前後向不需逐層湊參數。
- **ZeRO-3 / FSDP 的粒度,由 wrapping unit(包裝單元)決定。** 一手文件只說「only materializes unsharded parameters and gradients of one unit at a time」[W11]——一個 unit 可以是一層、也可以是數層合成一個包裝單元;unit 越細,前後向要湊參數的 AllGather 次數越多、每次越小,反之越少越大。所以「ZeRO-3 必然是逐層數十次小 collective」是**把某一種 wrapping 配置誤當通則**,一手來源並不支持固定次數。

**兩個推論,以及它們各自的條件:**
1. **總量差異有一手根據,可直接引用:** ZeRO-3 相對 DDP「a maximum of 1.5x communication」[W10]——這是 $n\beta$ 那一項的差。
2. **次數/α 差異是「配置相依」的,不能無條件斷言。** 若把 FSDP 的 wrapping unit 設得細(近乎每層一個),單步的跨主機 collective 次數會遠多於「DDP 用大桶只發幾次」的情形,α 累加後更難被預取重疊藏住;但把 unit 設粗、或把 DDP 桶設小,兩者的次數差就縮小。**另需更正一個常見誤解:α 是每則訊息的固定啟動延遲(第二章的定義),不是「每次 collective 都新建一條 TCP 連線」的握手成本**——NCCL 的通訊子在訓練期間長駐並重用連線,α 反映的是啟動/同步開銷,而非逐次重新握手。

**因此,在 α 相對昂貴的一般乙太網路上,「ZeRO-2 優於 ZeRO-3」是一個有方向、但須用配置或 trace 坐實的傾向,而非無條件結論。** 具體而言:唯有當 ZeRO-3/FSDP 的 wrapping 粒度細到單步發起明顯更多次跨主機 collective、且這些 α 無法被預取重疊藏住時,「ZeRO-2 + 大梯度累積」(少數幾次大 collective)才明確占優。第六章矩陣把它列為全參數微調的最小通訊基準,但同時標明這個排序的前提是「ZeRO-2 的狀態放得下(第三章帳本)、且 wrapping 粒度未被刻意放粗」,並以第七章 Phase 2 的 per-collective trace 作為坐實或翻轉此排序的封閉條件。

### 記憶體帳本套到 Llama-3-8B:分片後放得下嗎(worked example)

沿用第一章的 Ψ ≈ 8.0×10⁹,看各做法在 N=4 卡下每卡要多少狀態記憶體(未計 activation):

- **DDP:** 每卡 $16\Psi$ = **128 GB** → 任何單卡都放不下,8B 全參數 DDP 在本硬體不可行(信心:高,可核算)。
- **ZeRO-1(N=4):** $4\Psi + 12\Psi/4$ = 32+24 = **56 GB** → 仍 > 46 GB,不可行。
- **ZeRO-2(N=4):** $2\Psi + 14\Psi/4$ = 16+28 = **44 GB** → L40S 勉強放下狀態,但未計 activation,實務吃緊。
- **ZeRO-3 / FSDP(N=4):** $16\Psi/4$ = **32 GB/卡** → L40S 舒適,但恰好等於 5090 的 32 GB 總量,在 5090 上幾乎不留 activation 空間。

**兩個非顯而易見的結論:**
1. 8B 全參數微調在本平台**至少要 ZeRO-2、實務上要 ZeRO-3/FSDP**,且必須用滿四張卡——而這四張卡的 collective 全走一般網路。
2. **等量分片下,最小的那張卡(32 GB 的 5090)就是全隊的記憶體上界。** 把 5090 混入 8B 全參數的分片隊伍,它會先爆記憶體——這是「異質不宜混編同步訓練」的可核算根據之一(第四章展開)。

### 一個必須釐清的岔路:PEFT 不是「工作量相同」的替代基線

看到「128 GB 放不下」,很自然會想「那用 LoRA 就好了」。但這是**換了題目**,不是同一題的省事解法,必須分清楚(反面提醒):

- **全參數微調(full fine-tuning):** 更新模型**所有**旋鈕,付完整的 16Ψ 記憶體帳。深層對齊、持續預訓練等業務會硬性要求它。
- **PEFT / LoRA / QLoRA:** 凍結原模型,只在旁邊訓練一小塊「adapter」。它不付 12Ψ 的優化器狀態帳,是**完全不同的記憶體/計算 profile**。QLoRA 更進一步用 4-bit 存凍結的底模——論文摘要即「reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance」[W34];白話說就是**單張 48 GB 卡就能微調 65B 模型,且保持 16-bit 全量微調的任務表現**。

所以「用 LoRA 解決 8B 全參數微調」在字面上是矛盾的——它根本沒在做全參數更新。本研究因此把兩者當成**互斥的兩個維度**,不放在同一排序裡比較(第六章矩陣):

- **維度一:PEFT/LoRA/QLoRA** —— 單卡、跨卡通訊近乎零,是「領域/指令微調」題目的首選。
- **維度二:全參數分散式微調** —— 當業務硬性要求全量更新時,PEFT 不是替代方案;此時最小通訊的全參數**候選**是「3×L40S 上的 ZeRO-2 + 大梯度累積」。但這個候選能不能成立,取決於一個常被略過的帳:**上面算的只是「狀態」記憶體,activation 還沒算進去。** 可核算:7B 於 3 卡的 ZeRO-2 狀態 ≈ $2\Psi + 14\Psi/3$ = 14 + 32.7 ≈ **46.7 GB**。把它放到單張 L40S(46,068 MiB ≈ 48.3 GB)上,**狀態就吃掉幾乎整張卡,只剩約 1.6 GB 給 activation 與框架工作區**——對任何非極小序列長/批次的訓練前後向而言,這點餘裕根本不夠。
  - **關鍵澄清(反面提醒):大梯度累積並不會生出 activation 空間。** 梯度累積降的是**同步頻率**(少發幾次跨卡 collective),它不改變**單一 micro-batch 前後向所需的 activation 峰值**;拿它來補「activation 放不下」是文不對題。
  - **因此 ZeRO-2 於 7B 是一個「條件式」而非「已驗證」的基準:** 唯有在**同時**滿足 micro-batch=1、開啟 activation checkpointing(重算換記憶體)、並把序列長壓到一個明確上限、且經第七章 Phase 2 **實測每卡峰值 VRAM 確認 ≤ 46,068 MiB** 時,才談得上「放得下」。若這組 activation-safe 設定仍放不下,就必須降到 **ZeRO-3/FSDP(狀態 16Ψ/N=每卡 32 GB,留出更多 activation 餘裕)或啟用 offload**——這也是為什麼第六章矩陣把 ZeRO-2 標為條件式候選、把 activation 與峰值 VRAM 實測列為它的封閉條件。8B 的狀態更大(ZeRO-2 於 4 卡 44 GB),同樣的 activation 壓力只會更緊,需 4 卡且更可能落到 ZeRO-3 或 offload。

### 兩個減壓旋鈕:梯度累積與 wire dtype

在不換硬體的前提下,有兩個關鍵旋鈕能擴大資料並行的可行區間:

- **梯度累積(gradient accumulation):** 累積 k 個小批次才同步一次,**降低同步頻率**、攤薄跨卡通訊成本;代價是每次同步的有效批次變大、收斂特性改變。公平比較不同做法時必須固定「有效 global batch」,否則比 step time 沒有意義。
- **wire dtype(透過網路傳的精度):** 呼應第一章「算什麼精度」與「傳什麼精度」是兩回事。用更窄的 bf16/fp8 傳梯度可直接減少網路上的 byte 數[W14],是本平台最直接的減壓手段;但 fp8 的框架/collective/硬體支援與數值代價需逐一驗證(fp8 tensor 運算主要在 Hopper/Blackwell 架構;L40S 屬 Ada,其 fp8 tensor core 支援情形需以 Transformer Engine/硬體文件實測封閉——本次未取得逐句一手引文,標為待查)。

### 公平比較的量尺(分母定義)

後面談效能時會用到這些指標,先定義清楚:
- **step time:** 一個優化器 step 的 wall time(牆上時間)。
- **tokens/s:** 每秒處理的 token 數。
- **MFU(Model FLOPs Utilization):** 「the ratio of the observed throughput (tokens-per-second) relative to the theoretical maximum throughput of a system operating at peak FLOPs」[W15]——實測吞吐相對於「該系統以峰值算力運轉」的理論上限之比,且只計前後向必需的運算、不計重算,故可跨系統公平比較[W15]。

---

## 二、張量並行(TP):切開單一層,代價是每層四次 AllReduce

**張量並行**把「一層內部的大矩陣」橫剖分到多張卡:每張卡算矩陣的一部分,再合併。Megatron 式 TP 靠一對共軛算子 f/g 同步——「f is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity in the backward pass」[W17],結果是每個 Transformer 層在前向 2 次、反向 2 次 AllReduce,「4 total communication operations ... of a single ... transformer layer」[W17]。

關鍵在於這是**頻繁、同步、且難以與計算完全重疊**的 AllReduce,量正比於 activation 大小 × 層數。也就是說,**TP 幾乎在每一層都要打一次網路**。這也是為什麼官方一致建議把 TP 限制在有高速互連的範圍內:「tensor model parallelism should generally be used up to degree g when using g-GPU servers, and then pipeline model parallelism can be used to scale up ... across servers」[W19];序列並行論文更直白:「tensor-level model parallelism is typically limited to a relatively small group of GPUs that are connected with high speed bandwidth, such as GPUs connected with NVLink inside a DGX server」[W20]。175B 的官方範例即節點內 TP=8 + 跨節點 PP=16[W24]。

**對本平台的意涵已呼之欲出:** TP 要求的「高速互連的一小群卡」在本平台根本不存在(每主機一張卡、無 NVLink)。把 TP 的逐層 AllReduce 放到一般網路上會如何,第四章量化。

---

## 三、管線並行(PP):切開層與層,代價是「氣泡」

**管線並行**把模型沿層切成幾「段」(stage),每張卡負責一段;資料像在生產線上流動——第一張卡算完前幾層,把中間結果(activation)**點對點**傳給第二張卡算接下來幾層,依此類推。它的通訊量遠小於 TP:**只在段與段的交界傳一次 activation**,不是每層都打網路。

代價是**管線氣泡**(pipeline bubble):生產線剛啟動時後段的卡在空等第一批資料流過來,收尾時前段的卡也閒著。GPipe 給出「bubble time is O((K−1)/(M+K−1))」、「negligible when M ≥ 4×K」[W18];Megatron 給更精確的:

$$
\text{bubble 時間佔比} = \frac{t_{pb}}{t_{id}} = \frac{p-1}{m}
$$

其中 p = 段數、m = microbatch 數(把一批資料再切成 m 小份餵進生產線讓它保持滿載)[W19]。進入穩態後採 1F1B 排程(one forward pass followed by one backward pass)[W19];交錯式(每卡負責 v 個不連續 chunk)能把氣泡再降為 $\tfrac{1}{v}\cdot\tfrac{p-1}{m}$[W19]。

**走一個小例子(worked example,信心:高):** PP=4(p=4)、m=8 microbatch → 氣泡佔比 = (4−1)/8 = **37.5%** 的時間在空轉。要壓到 GPipe 說的「可忽略」,需要 m ≥ 4p = 16(此時氣泡 ≈ (4−1)/16 ≈ 19%,再靠 1F1B/交錯降低)。**推論:** 本平台卡數少(4),p 不大,氣泡的 (p−1) 項小;但要餵夠 microbatch(m≥4p)才划算,而餵大 m 需要足夠的批次資料與記憶體。

---

## 何時受通訊限、何時受計算或氣泡限(可核算判準)

把三種切法收斂成一組判準,供第四章套用本平台的實際頻寬:

- **資料並行:** 每步固定量的梯度 collective;當「collective 時間(byte÷busbw + 次數×α)」> 計算時間 → 通訊受限。
- **TP:** 每層打一次網路;當「層數 × 每層 AllReduce 時間」> 計算時間 → 通訊受限。本平台無 NVLink,且 TP 頻繁跨主機同步,因此通訊受限風險高;是否真的受限仍須用上述判準與實測頻寬判定。
- **PP:** 通訊小,主要損耗是氣泡與段不均衡;當 m 夠大、各段計算量均衡 → 計算受限(理想);m 太小 → 氣泡受限。

## 待檢驗的假設:「跨節點 TP 是反面教材、PP 是標準解法」

這個大方向成立,但不能當成無條件通則——主動找出反例與限定(信心:高):

1. **反例一(推論用 TP 優先):** vLLM 對放不下單卡的模型**先用 TP**、甚至用滿整個節點,只有跨節點才疊 PP[W21]。所以「TP 一律差」是錯的——它在**有 NVLink 的節點內**是推論首選。但本平台每主機一張卡、連「節點內 TP」的高速前提都不存在,因此**本平台**跨卡 TP 應作為高通訊風險的對照候選,不能僅憑沒有 NVLink 就斷言一定輸給 PP;仍須比較實測延遲、頻寬、負載與 PP 氣泡。
2. **反例二(序列並行):** 序列並行以 4 次 AllGather + 4 次 ReduceScatter 取代 4 次 AllReduce,頻寬「the same」、「does not introduce any communication overhead」[W20],可在同頻寬下延伸 TP 的範圍。這不是「必須有 NVLink 才能用」的限制;但它也不會憑空增加本平台的網路頻寬,不能據此宣稱跨節點瓶頸已消失。
3. **反例三(PP 非必勝):** PP 有氣泡 (p−1)/m 與段不均問題[W18][W19];不預設 PP 必勝。**當模型放得下單卡時,根本不該用任何模型並行**——單卡最省。
4. **反例四(低並行即時串流下 PP 反而可能輸給 TP):** 要把「高吞吐批次服務」與「低延遲互動串流(批次=1)」分開。串流推論常是批次=1、m=1;此時 PP 的氣泡佔比退化為 (p−1)/p,PP=4 即 **75%** 的瞬間有三張卡閒置,且每產生一個 token 都要**串列跨越 p 台主機的網路躍點**,逐字延遲成倍放大。反觀 TP=2 在批次=1 時每次 AllReduce 的 payload 極小(n 小),通訊時間由 α 主導而非 β(第二章的模型),兩卡又能同時並行算矩陣。所以在**小 n、低並行**這個角落,TP=2 的逐字延遲可能優於 PP。這不翻轉本平台大方向(高吞吐時 n 大、β 主導,TP 的逐層 AllReduce 昂貴),而是界定了 TP 成為反面教材的邊界——需第七章 Phase 3 的 m=1 探針定量。

**最後一個必須守住的方法論:** 若未來實測 PP 在 5090+3×L40S 上較 TP 快,不能直接歸因於「PP 較好」,因為 5090 與 L40S 算力/VRAM 不同會**獨立**影響段平衡與氣泡。要把「策略效應」與「GPU 型號效應」分開:先用同型號子集(3×L40S)做同質基線,再引入 5090 觀察型號效應(第七章 Phase 3)。

**至此,三種切法的機制與逼出的通訊都清楚了。** 但每一個判斷都帶著一句「在一般網路上」「無 NVLink 使…」的但書。下一章就把這些但書兌現:把這些通訊放到本平台真實的慢網路 + 異質卡上,代價到底變成多大,以及平台的排程系統能幫上什麼、幫不上什麼。
