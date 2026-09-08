# 第六章:平台取捨、策略適配矩陣與建議

> [回導讀](../README.zh-TW.md) | [證據與來源](08-evidence.md)

> 前五章逐一拆解了每種做法的機制與在本平台的代價。這章把它們放在一起,回答:**綜合來看,哪種做法在本平台可行、哪種是反面教材?最強的反面證據是什麼、怎麼處置?最終建議為何、什麼情況會翻轉它?**

## 先定量尺:怎麼算「一個策略在本平台可行」

用三個可量的門檻衡量,任一不過即淘汰(這三條正是前幾章的判準收斂):

1. **記憶體放得下嗎(可核算):** 以第三章的帳本估每卡峰值,對照 L40S 46,068 MiB(`docs/l40s-production-runbook.md`)與 5090 32 GB。放不下就靠切分,以最小卡(5090 32 GB)為準。
2. **每步通訊時間 vs 計算時間(理論推估 + 待實測):** 以第二章 α+nβ 模型與各 collective 的過線 byte 估算,除以本平台實測 busbw(**目前未知**)。通訊時間顯著大於計算即「通訊受限」,策略失去意義。
3. **同步 barrier 是否被最慢卡/最小卡綁住(定性 + 待 trace):** 同步 collective 等最慢卡[W32];異質卡使此惡化(第四章)。

真實後果:若一個策略讓每步 30 秒都在等網路、只有 5 秒在算(第四章敏感度表),那同樣的卡改成「各自跑單卡工作」反而總吞吐更高——這就是全研究反覆回到「單卡放得下優先」的原因。

## 策略適配矩陣(候選 × 本地約束)

共同基線:每主機一張卡、無 NVLink、跨卡走一般乙太網路、HAMi 單卡一 workload、GPU 型號逐型號驗證(非排程禁令)。**注意:乙太網路的具體後端在 repo 是條件式的**(直連路由成立才選明文 host-gw,否則走加密且 MTU 夾到 1200 的 wireguard-native,兩者 busbw 特性不同),是第七章要先量出的未知數,**不是既定的 host-gw 基線**。評分是**本平台**的相對適配,非策略本身優劣。**PEFT 與全參數微調是兩個互斥維度**,不在同一排序內競爭。

| 候選策略 | 主要 collective | 本平台記憶體適配 | 通訊負擔(本平台) | 整合接點(gpu-cluster-iac) | 可逆性 | 主要風險 | 證據強度 | 本平台適配 |
|---|---|---|---|---|---|---|---|---|
| **單卡推論(現況)** | 無 | 需放得下單卡 | 無跨卡通訊 | 既有:Deployment+Service | 高 | 模型須 ≤ 卡容量 | 高(repo 現況 + [W21]) | ★★★★★ 首選 |
| **單卡 4-bit 量化推論(含 MoE)** | 無 | 量化外推單卡邊界;Mixtral 純權重下界≈23 GB[W30],**實載峰值須含 KV cache/工作區,取決於序列長與並行數,待指定預算實測** | 無跨卡通訊 | 既有單卡 Job/Deployment | 高 | 量化精度損失須業務可接受;序列長/並行過大時單卡放不下 | 中(純權重下界可核算 + [W30];實載峰值待 Phase 5 實測) | ★★★★☆ 大模型/MoE 首選候選(須指定 KV/序列預算並實測放得下) |
| **管線並行 PP(含非對稱)** | 點對點 send/recv | 沿層切,適配大模型;非對稱可給 5090 較少層 | 低(僅段邊界 activation) | vLLM `pipeline_parallel_size`+Ray;Megatron flag | 中 | 氣泡 (p−1)/m;m=1 串流氣泡 75%;段不均;奇數卡 | 高([W18][W19][W21][W33]) | ★★★☆☆ 高吞吐大模型時本平台最佳模型並行 |
| **張量並行 TP** | 每層 4× AllReduce | 可切大模型 | 極高(每層打網路,無 NVLink) | vLLM `tensor_parallel_size`;Megatron flag | 中 | 頻繁跨主機同步 → 通訊受限風險高;須依實測判定,非缺少 NVLink 即必然 | 高([W17][W19][W20][W33]) | ★☆☆☆☆ 本平台反面教材(m=1 串流為條件例外) |
| **MoE 跨節點專家並行** | 2× AlltoAll/層 | 總參數需多卡常駐 | 極高(AlltoAll 對慢鏈路最敏感;lockstep 空跑) | 未實作;需 gang + NCCL | 中 | 跨節點 AlltoAll 瓶頸 + 專家傾斜 + 動態路由不均 | 高([W25][W27][W37]) | ★☆☆☆☆ 跨節點反面教材 |
| **全參數:ZeRO-2(3×L40S 候選)** | ReduceScatter(+AllGather) | 2Ψ+14Ψ/N;7B/3≈46.7 GB **僅為狀態**,佔滿 46,068 MiB L40S 後**僅剩≈1.6 GB 給 activation/工作區** | 中高(僅梯度 collective,粒度隨 bucket 大小;無前後向逐層湊參數) | training gang(未實作);NCCL over 選定後端 | 高 | **狀態幾乎佔滿卡,activation 餘裕不足**;須 micro-batch=1 + activation checkpointing + 受限序列長並實測峰值 VRAM;通訊受限待實測 busbw | 中(狀態可核算[W10];activation-safe 放得下待 Phase 2 實測) | ★★☆☆☆ 條件式候選:須實測 activation-safe 峰值 ≤46,068 MiB(否則落 ZeRO-3/offload,見下註) |
| **全參數:ZeRO-3 / FSDP** | 前後向 AllGather+ReduceScatter | 16Ψ/N;8B/4=32 GB(5090 卡死) | 高(總量 1.5× DDP[W10];湊參數 collective 次數/α 隨 wrapping 粒度上升,待 trace) | training gang;NCCL over 選定後端 | 高 | 記憶體被最小卡綁死;細 wrapping 下 α 累加,待 Phase 2 trace 坐實 | 高([W10][W11][W12]) | ★★☆☆☆ 記憶體放不下 ZeRO-2 時採用 |
| **全參數:DDP** | AllReduce | 每卡 16Ψ,8B=128 GB 放不下 | 極高(每步 all-reduce) | training Volcano Job + gang(未實作) | 高 | 記憶體直接爆 + 通訊受限 | 高(可核算 [W10]) | ★☆☆☆☆ 不可行(大模型) |
| **PEFT/LoRA/QLoRA(獨立維度)** | 少量/無跨卡 | 凍結底模,只訓 adapter;QLoRA 單卡 48GB 可微調 65B[W34] | 低 | training Job,單卡常可行 | 高 | 非全量微調題目,不與全參數列比較 | 高([W34] QLoRA 一手) | ★★★★★ 領域/指令微調首選(非全參數) |

**ZeRO-2 列在 ZeRO-3 之前是「條件式」而非既定(誠實界定,信心:中,待實測):** 這個順序有兩個各自獨立的前提,任一不成立都會翻轉它。
- **前提一(記憶體,關鍵):ZeRO-2 的 46.7 GB 只是「狀態」,不含 activation。** 它佔滿 46,068 MiB L40S 後僅剩約 1.6 GB,對真實訓練前後向的 activation 遠遠不夠(第三章)。**大梯度累積降的是同步頻率、不是 activation 峰值**,補不了這個缺口。因此 ZeRO-2 於 7B 只有在 **micro-batch=1 + activation checkpointing + 明確上限序列長、且第七章 Phase 2 實測每卡峰值 VRAM ≤46,068 MiB** 時才算「放得下」;否則就該落到 **ZeRO-3/FSDP(每卡 32 GB 狀態、留更多 activation 餘裕)或 offload**——這一條就足以讓 ZeRO-2 讓位給 ZeRO-3。
- **前提二(通訊,次要):** 排序**不是**基於「ZeRO-3 一定逐層數十次小 collective」的固定次數斷言——一手來源只支持粒度由 DDP 梯度 bucket 與 FSDP wrapping unit 的**配置**決定(第三章)。只在前提一成立(activation-safe 下 ZeRO-2 確實放得下)後,才輪到比通訊:此時只要 ZeRO-3/FSDP 的 wrapping 未被刻意放粗,細粒度下前後向湊參數的 α 通常較難被重疊藏住,ZeRO-2 才較省。
- **綜合:** 只有「activation-safe 峰值實測放得下」且「ZeRO-3 未用粗 wrapping 壓低次數」兩者同時成立,ZeRO-2 才排在前;**任一不成立即翻轉**。第七章 Phase 2 的每卡峰值 VRAM 與 per-collective trace 是坐實或翻轉它的封閉條件。

**為什麼「跨節點大模型訓練」的強候選少於三個:** 本平台的物理限制(無 NVLink、一般網路)使跨節點 DP/TP/EP 都受通訊限;真正可行的候選集中在「單卡放得下」(推論、LoRA、單卡 MoE)與「PP(大模型且能餵夠 microbatch)」。這不是候選不足,而是**約束把候選集收斂到少數——這本身就是結論**。

## 最強的反面證據與處置

一份研究若只找支持自己的證據就不可信。以下主動列出最能推翻本研究結論的論點及處置:

1. **「busbw 高 = 快」** → busbw 是換算慣例非線速[W1],且無 IB 時 NCCL 走 TCP socket 只達線速一部分[W4]。**處置:** 不以廠商 busbw 數字下結論,改以第七章 Phase 1 實測本平台 busbw 為準。
2. **「PP 必勝」** → PP 有氣泡 (p−1)/m 與段不均[W18][W19]。**處置:** 限定「模型放不下單卡且能餵 m≥4p microbatch」時 PP 才划算;放得下就別用模型並行。
3. **「TP 一律差」** → 推論在 NVLink 節點內 TP 是首選[W21];序列並行同頻寬延伸 TP[W20]。**處置:** 明確區分「TP 本身」與「本平台無 NVLink」——結論限定於本平台。
4. **「FSDP = ZeRO-3」** → FSDP 是原生重設計、包裝/預取不同[W11]。**處置:** 視為相近但不可互相假設調校。
5. **「異質就是浪費 5090」** → 排除 5090 於同步群確有機會成本與奇數卡不整除的代價(第四章反例)。**處置:** 保留條件——若出現「非四卡不可」的模型,以非對稱 PP 保留 5090。
6. **「MoE 跨節點沒救」** → hierarchical/topology-aware AlltoAll 能把躍點從 O(p) 降到 O(G+p/G)[W27];dropless 改變取捨[W28]。**處置:** 承認可緩解但不可消除;本平台無 NVLink 使緩解上限受限,故仍優先單卡 MoE。
7. **「用 ZeRO++ 就能解低頻寬訓練」(考慮後排除)** → ZeRO++ 的 hpZ 核心是「in hpZ, we eliminate the inter-node all-gather during the backward pass by holding secondary FP16 weights partition within each node」,前提是「intra-node communication bandwidth is significantly higher than inter-node」[W35]。**處置:排除。** 本平台每主機一張卡,無「節點內」階層:secondary partition=1 等於每主機持全量權重(記憶體暴增、直接 OOM),=N 則退化為一般跨節點 ZeRO-3、失去 hpZ 節省。ZeRO++ 為單機多卡階層網路而設,在單卡節點失效。
8. **「用 zero-bubble / DualPipe 消掉 PP 氣泡」(考慮後排除)** → zero-bubble 排程需更多 in-flight microbatch 才能填滿管線,一手來源明證「ZB-2p ... comes at the cost of doubling the memory consumption compared to 1F1B」[W36]。**處置:排除為預設。** 此排除**僅**建立在 W36 已證實的「zero-bubble 記憶體約 2× 1F1B」之上:在 32/46 GB 的卡上,慢速網路使填管線所需的活化值暫存更吃緊,故標準 1F1B 是記憶體上界最可預測的實務選擇。(誠實標注:①「峰值活化記憶體隨通訊時間線性暴增」的具體公式未能在一手來源逐字證實,不採;②本研究**未**取得「DualPipe 硬性要求全雙工 IB+NVLink」的逐字一手引文,故不以此作排除依據——排除 DualPipe 的可靠理由只是上述 zero-bubble 家族共通的記憶體代價。)
9. **「引入 BytePS 參數伺服器改善通訊」(考慮後排除)** → 官方 repo 橫幅明證「This repository was archived by the owner on Dec 8, 2025. It is now read-only」[W38]。**處置:排除。** 此排除**僅**建立在該封存事實上:引入一個已封存唯讀、不再維護的通訊層即是採用停止維護的相依。(誠實標注:「不相容現代 PyTorch 2.x 生態」是作者對維護停滯後果的推論,未逐字查證,不作為決定性排除理由。)

## 建議與會翻轉建議的條件

**根因判斷:** 本平台的分散式 ML 能力上限由**卡間互連**決定,而非 GPU 算力或框架選擇。在「每主機一張卡、無 NVLink、一般乙太網路」下:

1. **推論(平台主場):** 模型能放單卡就單卡;放不下就**先 4-bit 量化**,再不行才 PP;**避免跨節點 TP 與跨節點 MoE 專家並行**。低並行即時串流(m=1)是唯一要回頭考慮 TP=2 的角落。
2. **微調(兩個互斥維度):** 領域/指令微調優先 **PEFT/LoRA/QLoRA 單卡**(QLoRA 單張 48GB 卡可微調 65B[W34]);業務硬性要求**全參數更新**時,最小通訊**候選**是 **3×L40S 的 ZeRO-2 + 大梯度累積 + bf16/fp8 wire dtype**——但它是**條件式**的:7B 的 46.7 GB 只是狀態,佔滿 L40S 後 activation 餘裕不足,**唯有在 micro-batch=1 + activation checkpointing + 受限序列長、且第七章 Phase 2 實測每卡峰值 VRAM ≤46,068 MiB 時才成立**(梯度累積不補 activation 缺口)。若該 activation-safe 設定仍放不下,即改用 **ZeRO-3/FSDP(每卡 32 GB 狀態、更多 activation 餘裕)或 offload**;無論 ZeRO-2 或 ZeRO-3,都**必須先實測本平台 busbw 確認非通訊受限**,否則不值得。
3. **異質:** 採「5090 獨立低延遲推論 + 3×L40S 同質群」。精確界定:**嚴禁把 5090 混入等量分片同步訓練(DDP/ZeRO/FSDP)**;但 5090 仍可作**非對稱 PP 節點**(承載較少層)參與大模型推論——隔離同質群是效能決策,非平台排程禁令。
4. **MoE:** 總參數 ≲ 50B 者**首選「單卡 4-bit 量化」作為最先驗證的候選**(如 Mixtral-8x7B AWQ、DeepSeek-V2-Lite bf16 於 L40S)。**但「放得下」須條件式界定,不是純權重下界就成立:** Mixtral 4-bit 純權重下界僅 ≈23 GB[W30],實際載入峰值還要加 KV cache(隨 `max-model-len` 與並行數成長)與執行期工作區;因此須先指定量化方案/執行期、上限序列長、並行數與 KV 保留比例,並由第七章 Phase 5 **實測峰值 VRAM 確認落在 L40S 46,068 MiB 內**,方能定案。跨節點專家並行的 AlltoAll(+ vLLM lockstep 空跑[W37])是本平台最不利型態,MoE 的跨節點 PP 又有動態路由不均放大氣泡的陷阱,故僅在指定預算下單卡確認放不下時才採用。

**會翻轉上述的條件(明確列出):**
- **加裝 RDMA/RoCE 或 ≥100 GbE 且實測 NCCL busbw 達線速高比例** → 跨節點 DP/PP/EP 的可行區間大幅擴大,ZeRO-2/ZeRO-3 全量微調可能變可行。
- **第七章 Phase 1 的 MTU=9000 + socket 調校使 busbw 躍升至線速 ≥75%** → 重估 ZeRO-2 全參數微調可行邊界(排除 socket 造成的偽陰性)。
- **更換為支援 NVLink 的 GPU 與相容多卡／NVLink Switch 系統** → 在實際高速互連群組內重新評估 TP,以及 MoE hierarchical AlltoAll 的收益[W27][W42]。現有 RTX 5090／L40S 均不支援 NVLink[W40][W41],單純搬到同一台主機或購買橋接器無法達成;這是更換硬體平台,不是軟體設定調整。
- **Phase 1 實測顯示 busbw 遠高於預期**(例如生效後端為明文 host-gw、NIC 為 25 GbE 而意外接近線速)→ 放寬「通訊受限」判斷。
- **模型出現「非量化、非四卡不可」的 VRAM 需求** → 以**非對稱 PP** 保留 5090(而非排除),重估四卡總 VRAM(170 GB)方案。

**這些「會翻轉」的條件,幾乎全都指向同一個尚未量測的數字。** 下一章就把「如何量出它、以及如何驗證前六章每一個判斷」設計成具體的實驗矩陣。
