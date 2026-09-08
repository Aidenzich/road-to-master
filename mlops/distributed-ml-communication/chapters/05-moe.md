# 第五章:MoE 與它獨有的通訊

> [回導讀](../README.zh-TW.md) | [證據與來源](08-evidence.md)

> 前四章的模型都是「密集」的——每個 token 都會流過所有參數。這章處理一類不同的模型:**混合專家(MoE,Mixture of Experts)**。它回答:**MoE 的 token 路由如何引入第三種通訊型態(AlltoAll)?為什麼這種通訊在本平台特別棘手?哪一種 MoE 部署才適合本平台?**

## 先搞懂 MoE 在幹嘛:一個 token 的旅程

密集模型裡,每一層是同一組權重,所有 token 都用它。MoE 把某些層換成「一群專家 + 一個路由器」:一層裡有好幾個平行的「專家」(expert,各自是一組獨立權重),外加一個小小的**路由器**(router);每個 token 進來,路由器決定把它送去哪幾個專家,只有被選中的專家會處理它。這樣模型的**總參數**可以很大(專家很多),但**每個 token 實際用到的參數**(active parameters)只有被選中的那幾個——用大容量換算力效率。

用一個具體 token 走 Mixtral 式 top-2-of-8 的旅程(每層 8 個專家、每個 token 選 2 個)[W30]:

1. **token 進入某 MoE 層**,路由器算出這個 token 最該去的 2 個專家(top-2)。
2. **dispatch(派送):** 若這 2 個專家分佈在不同卡上,就得把這個 token 的資料**送到專家所在的那張卡**。全體 token 同時做這件事,就是一次 **AlltoAll**(第二章的第四種:每張卡都有要給每張卡的份,徹底重新洗牌)。
3. **expert compute:** 每張卡上的專家處理落到自己這裡的 token。
4. **combine(收回):** 把專家的輸出**送回 token 原本的卡**,又是一次 **AlltoAll**。

所以 **MoE 專家並行(Expert Parallelism)每一個 MoE 層要來回兩次 AlltoAll**。GShard 的 dispatch/combine 在跨裝置時正是以 AlltoAll 重新分片實現——「AllToAll ... used to reshard a sharded tensor from one dimension to another」[W25];DeepSpeed-MoE 直言「Expert parallelism requires all-to-all communication between all expert parallel devices」[W27]。top-k 光譜:Switch 用 top-1(「route to only a single expert」[W26]),GShard/Mixtral 用 top-2[W25][W30]。

**為什麼 AlltoAll 在本平台特別棘手:** 它是四種集體通訊裡最「散」的——不是規律地把資料加總或收集,而是每張卡都要和每張卡交換一小份,對慢鏈路與延遲最敏感。DeepSpeed-MoE 明言「it is difficult to scale expert parallelism to many devices as the latency increases linearly with the increase in devices」[W27]。在無 NVLink、只有一般網路的本平台,兩次 AlltoAll 全走網路,是最不利的通訊型態。

## 影響傳輸與品質的四個旋鈕:capacity、drop、dropless、topology-aware

MoE 不是鐵板一塊,不同實作的傳輸量與品質行為差很多(不可一概而論):

- **capacity factor 與 token drop:** 每個專家能收的 token 有上限——`expert capacity = (tokens per batch / number of experts) × capacity factor`[W26]。超過上限的 token 被**丟棄**:「computation is skipped and the token representation is passed directly to the next layer through the residual connection」[W26];GShard 亦同,overflow token 一樣走 residual[W25]。capacity factor 越大 → padding 與計算/記憶體越多,但丟棄越少[W28]。這決定了 AlltoAll 的傳輸形狀是被夾成固定大小、還是隨實際路由浮動。
- **dropless(反面做法):** MegaBlocks 用 block-sparse 計算「never drops tokens」,消除「dropping tokens ... or wasting computation and memory on padding」的兩難,端到端加速達 40%[W28]。dropless 下 AlltoAll 的量由實際路由決定(可能不均),而非被 capacity 夾成固定形狀。
- **expert skew(專家負載傾斜)與尾延遲:** 路由可能讓某些專家拿到不成比例的 token → 該專家成為拖住整次 AlltoAll 的掉隊者。Switch 用可微的 auxiliary load-balancing loss 緩解[W26];DeepSeek-V3 改用無輔助損失的 per-expert bias(「decrease the bias term by γ if ... overloaded」),因為「too large an auxiliary loss will impair the model performance」[W29]。
- **topology-aware / hierarchical dispatch:** 把 AlltoAll 的躍點數從 O(p) 降到 O(G+p/G)(G = 節點內 GPU 數)[W27]。但本平台每主機一張卡,沒有「節點內」階層可用,這條緩解幾乎失效。

## 案例:總參數 vs active 參數,以及「單卡放得下」的邊界

MoE 的記憶體與算力要分兩個數字看:**總參數決定要不要多卡(權重容量),active 參數決定單 token 的算力**。

- **Mixtral-8x7B:** 「46.7B total parameters but only uses 12.9B parameters per token」,top-2-of-8[W30]。46.7B 權重(bf16 ≈ 93 GB)放不下任一單卡,**若用 bf16 推論就需模型並行或多卡**;但單 token 只算 12.9B。
- **DeepSeek-V2-Lite:** 「15.7B total parameters, of which 2.4B are activated for each token」[W31]。bf16 權重 ≈ 31 GB,**恰好落在單張 L40S(46 GB)可放、單張 5090(32 GB)吃緊**的區間——是本平台**單卡 MoE 推論**的理想案例。

**本平台推論(信心:高):**

- **若 MoE 權重放得下單卡(如 DeepSeek-V2-Lite 於 L40S)→ 單卡推論,完全避開 AlltoAll,是本平台最佳解。**
- **4-bit 量化把「放得下單卡」的邊界大幅外推,使單卡成為 MoE 應當「最先驗證」的候選,而非跨節點的附註。** 這裡要嚴格區分「可核算的下界」與「已驗證的實載足跡」:
  - **可核算的下界(信心:高,作者計算):** AWQ/GPTQ(activation-aware weight quantization 與 GPTQ,兩種常見的**訓後(post-training)4-bit 權重量化**方法,把每個權重從 16-bit 壓到約 4-bit 存放)下,把 Mixtral-8x7B 的 46.7B 總參數[W30]乘以 0.5 byte,得**純權重下界 ≈ 23.35 GB**。這是一個**下界**,只證明「4-bit 後純權重遠小於 bf16 的 93 GB,單卡有機會」。
  - **它為什麼還不是「放得下」的證明(信心:高,反面提醒):** 實際載入後的峰值 VRAM 還要加上量化格式本身的 scale/zero-point 中繼資料、**維持較高精度的 embedding/norm 層**、以及**隨 `max-model-len` 序列長度與並行請求數(concurrency)線性成長的 KV cache**,再加上執行期(如 vLLM)的工作區與碎片。這些量**沒有任何一手來源或本平台實測數字**支撐,序列長與並行數一大,單卡就可能放不下。因此先前的「24–28 GB」是**未定參數下的粗估,不作為已驗證事實**;同理「總參數 ≲ 50B 一律單卡放得下」是**推斷傾向,不是通則**——真正的邊界取決於量化方案與 KV 預算。
  - **要把它變成可下的結論,需要一份指定預算 + 一次實測封閉:** 指定量化方案與執行期(如 vLLM + AWQ 4-bit)、上限序列長 `max-model-len`、目標並行數、為 KV cache 保留的 VRAM 比例,然後**實測峰值 VRAM**(第七章 Phase 5)確認落在 L40S 46,068 MiB 之內。在該預算被指定並實測前,結論只能是**條件式**的。
  - **在此前提下的方向性判斷(信心:中,待 Phase 5 封閉):** 單卡 4-bit 量化推論一旦在指定預算下放得下,就同時消除了跨節點 AlltoAll、PP 氣泡、跨節點 activation 傳輸與多節點失效風險,因此**應列為 ≲50B 級 MoE 的首選「候選」並最先驗證**;除非量化後精度損失業務不可接受、或指定的序列長/並行預算使單卡放不下,才轉向跨節點專家並行或 PP。
- **若必須跨卡(量化仍放不下的更大 MoE)→** 兩次 AlltoAll 全走一般網路,是本平台最不利型態;沿層切的 PP 只傳 activation、通常優於跨節點專家並行,**但 MoE 的 PP 另有陷阱:** MoE 的動態 top-k 路由使各層/各 token 分到各段的專家計算量隨時波動(密集模型各層計算量固定,MoE 不然),動態段不均衡會讓實際氣泡大於密集模型的理論值 (p−1)/m[W19]。

## 一個框架層的實測反面教材:vLLM 跨節點 MoE

vLLM 的跨節點資料並行 + 專家並行要求各 rank 的前向**嚴格對齊**——「Forward passes must be aligned, and expert layers across all ranks are required to synchronize during every forward pass, even when there are fewer requests to be processed than DP ranks」;且「For MoE models, when any requests are in progress in any rank, we must ensure that empty 'dummy' forward passes are performed in all ranks that don't currently have any requests scheduled」[W37]。

翻成白話:即使某張卡此刻沒有請求要處理,只要別的卡有,它也得**跑一次空的前向**來陪跑對齊。在無 NVLink、只走一般網路的本平台,這種 lockstep + 空跑會讓最慢節點或網路抖動直接綁死全叢集的逐字延遲——坐實「MoE 跨節點是本平台反面教材」。

**至此五種模型形態(資料並行、TP、PP、異質同步、MoE)在本平台的機制與代價都齊了。** 下一章把它們放進同一張「候選 × 本地約束」的對照表,連同最強的反面證據一起,收斂成明確建議。
