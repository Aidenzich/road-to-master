# 第七章:未來實驗設計(五面向)

> [回導讀](../README.zh-TW.md) | [證據與來源](08-evidence.md)

> 前六章的判斷分三種:一手來源支持的**事實**、可核算的**理論推估**、以及仍待實測的**假設**。這章回答:**要把「假設」與「理論推估」變成「實測確認」,該設計哪些實驗?** 每個矩陣列出:假設、環境前提、變因、工具/範例命令、指標、預期可能結果、以及每個結果會改變哪個決策。**本次只設計、不執行;所有命令僅為範例,執行需另行授權並遵守唯讀、不留殘留原則。**

全部五個矩陣有一個共同的、最高優先的前置未知:**本平台實際選定後端路徑的 NIC 線速與 NCCL busbw**——其餘四個面向的結論全都掛在它上面。**首要待確認項:實際生效的 flannel 後端是 host-gw(明文、MTU 依 NIC)還是 wireguard-native(加密、MTU 夾到 1200),以及對應的實體介面。** 第四章已說明這是條件式選擇,**不能預設 host-gw/eno2**。

---

## Phase 1 — 網路/collective baseline matrix(最高優先)

- **假設:** 本平台選定後端的實際 collective busbw 遠低於 NVLink,且 NCCL over TCP 只達 NIC 線速一部分;若生效的是 wireguard-native,加密與 MTU 1200 會進一步壓低 busbw。
- **環境前提:** 2–4 台 GPU 節點皆 Ready、NCCL 版本固定、無背景 GPU 負載;取得 NIC 線速(需求標為未知)。**先讀出實際生效的後端與介面**,而非假設 host-gw/eno2:`kubectl get nodes`(節點數與 Ready)、讀各節點 `/etc/rancher/k3s/config.yaml` 的 `flannel-backend`/`flannel-iface`(對照第四章的 host-gw vs wireguard-native 選擇邏輯)。**兩條後端各量一次(若可切換),或至少記錄現況屬哪一條**,因為兩者 busbw 上界不同。
- **變因(含決定性 socket 調校,避免偽上限):** 訊息大小 4KB–1GB(`-b 4K -e 1G -f 2`)、rank 數(2/3/4)、單向/雙向、同質(3×L40S)/異質(含 5090)rank group、**後端(host-gw 明文 vs wireguard-native 加密)**、`NCCL_ALGO`(Ring/Tree)、`NCCL_PROTO`(Simple/LL/LL128)、`NCCL_SOCKET_IFNAME=<實際生效介面>`(由前一步讀出,勿硬編)。**必須納入的強制控制變因(否則會誤報硬體上限):MTU(host-gw:1500 vs Jumbo 9000;wireguard-native 固定 1200)、Linux TCP window(`net.ipv4.tcp_rmem/tcp_wmem`)、`NCCL_NSOCKS_PERTHREAD`(1 vs 4)、`NCCL_SOCKET_NTHREADS`(1 vs 4)、`NCCL_BUFFSIZE`(4 MiB 預設 vs 16 MiB)。** 理由:無 RDMA 時 NCCL 退回 IP sockets[W4],預設 1500 MTU + 未調校 socket 常只達 NIC 線速一小部分;若只記錄預設值,可能把「可調校到接近線速」誤判成「網路徹底不足」(偽陰性)。NCCL 文件亦載 `NCCL_NSOCKS_PERTHREAD` 於「100G networks, can be set to 4」、與 `NCCL_SOCKET_NTHREADS` 乘積不得超過 64[W39]。
- **工具/範例命令:** `iperf3 -c <peer> -P 8`(NIC 線速上界)、`iperf3 -c <peer> -R`(反向)、獨立 RTT(α);先在 MTU=1500 與 9000 各量一次 iperf3(wireguard-native 時 MTU 固定 1200)。**nccl-tests 的跨節點多 rank 必須由 MPI 啟動、且二進位須以 `MPI=1` 編譯**——README 明言「NCCL tests rely on MPI to work on multiple processes, hence multiple nodes」、「The number of process is managed by MPI and is therefore not passed to the tests as argument」,總 rank 數 =「(number of processes)×(number of threads)×(number of GPUs per thread)」[W6]。因本平台**每主機一張卡**,正確形式是「每節點一個 MPI 行程、每行程一 GPU」——先 `make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl`,再對 N 個節點以 hostfile 啟動,例如 3 節點:
  ```
  NCCL_SOCKET_IFNAME=<實際生效介面> NCCL_IB_DISABLE=1 NCCL_BUFFSIZE=16777216 \
    mpirun -np 3 -N 1 --hostfile hosts ./build/all_reduce_perf -b 4K -e 1G -f 2 -g 1
  ```
  其中 `-np 3`=3 個 MPI 行程、`-N 1`=每節點一行程、`-g 1`=每行程一 GPU → 共 3 個跨主機 rank(對照 README 的 8 節點範例 `mpirun -np 64 -N 8 ... -g 1`[W6])。**單機單行程 `./build/all_reduce_perf ... -g 1` 只有一個本機 rank,量不到跨節點 collective。** 同理跑 `all_gather_perf`、`reduce_scatter_perf`、`all_to_all_perf`;每組 warm-up ≥5、重複 ≥20 取中位/尾值、correctness check、背景負載監控。這些 NCCL 環境變數為 100% 宣告式(Pod template / ConfigMap 注入),無程式碼污染、可回滾;`NCCL_SOCKET_IFNAME` 需對齊**實際讀出**的介面,而非假設 `eno2`。
- **指標:** algbw、busbw(逐 collective 用 [W1] 因子)、α(小訊息)、有效 β、iperf3 線速、NCCL/iperf 比值。
- **預期可能結果 → 決策:** ①busbw ≪ NIC 線速(TCP 瓶頸)→ 確認通訊受限,跨節點 DP/TP/EP 皆保守,優先單卡與 PP;②busbw 接近 NIC 線速且 NIC ≥100 GbE → 放寬第三/四章通訊受限判斷;③異質 rank group busbw 明顯低於同質 → 支持第四章同質群建議;④**MTU=9000 + socket 調校後 busbw 達線速 ≥75%(相對 1500 預設明顯躍升)→ 重估 3×L40S ZeRO-2 全參數微調可行邊界**(排除 socket 偽陰性);⑤調校前後幾無差異 → 瓶頸在實體線速而非 socket,結論①更穩固。

## Phase 2 — DDP/ZeRO/FSDP 記憶體與 step-time 對照

- **假設:** 8B 全量微調在本平台通訊受限;梯度累積與 bf16/fp8 wire dtype 可顯著減壓。
- **環境前提:** Phase 1 busbw 已知;3–4 卡;固定模型(如 Llama-3-8B)、dtype、optimizer、seed、有效 global batch。**通訊粒度必須被固定並記錄**:因為「一次收多少 byte、切成幾則 collective」不是方法(DDP/ZeRO/FSDP)本身決定的,而是由設定決定——DDP/ZeRO 是把**梯度分成設定好的 bucket** 逐桶處理 [W8],FSDP 則是按 **wrapping unit(每次 all-gather/reduce-scatter 的權重單位)** 逐單位處理 [W11]。若不固定這兩個旋鈕,量到的 collective 次數差可能只是「桶/單位切得不同」,而非方法差。
- **變因:** 方法(DDP/ZeRO-1/2/3/FSDP,**重點對照 ZeRO-2 vs ZeRO-3 於固定全參數目標**)、N(2/3/4)、梯度累積步數、wire dtype(fp32/bf16/fp8)、activation checkpointing 開關、CPU offload 開關;**並把通訊粒度本身當成受控變因逐一掃描**:DDP/ZeRO 的 gradient bucket 大小(如 DeepSpeed `reduce_bucket_size`/PyTorch DDP `bucket_cap_mb`)、FSDP/ZeRO-3 的 wrapping 單位(逐層 wrap vs 整塊 wrap),每組實驗都記下實際生效值。
- **工具/範例命令:** PyTorch FSDP / DeepSpeed ZeRO 設定(明確寫定 bucket 大小與 wrapping policy 並回讀);Nsight Systems 抓 timeline 看 compute/comm 重疊;記錄每卡峰值 VRAM。
- **指標:** 每卡峰值 VRAM(對照 32/46 GB)、step time、tokens/s、MFU([W15] 分母)、通訊/計算時間比;**Nsight 實測**每 step 的 collective 次數/大小分布與 GPU idle 氣泡數——這是「觀測值」,要對齊上面記錄的 bucket/wrapping 設定一起解讀,而非預設某方法一定「逐層屏障」或「少數大 collective」。
- **預期可能結果 → 決策:** ①DDP OOM(如預測)→ 確認需分片;②**在相同記憶體可行前提、且 bucket/wrapping 粒度受控並記錄下**,若 Nsight trace 顯示 ZeRO-3 的細粒度 all-gather 使 idle 氣泡明顯多於 ZeRO-2、有效運算佔比較低 → 全參數微調以 ZeRO-2 為基準,ZeRO-3 僅在 ZeRO-2 放不下時才用;若把 ZeRO-3 的 wrapping 併大後氣泡消失 → 說明先前差異來自粒度設定而非方法,排名須改依調校後 trace;③大梯度累積把通訊/計算比壓到 <1 → 全量微調在特定 batch 下可行;④5090 在等量分片先 OOM → 支持把 5090 排除於等量分片訓練群(但保留其非對稱 PP 位置)。

## Phase 3 — 跨節點 TP vs PP 對照(訓練與推論分開)

- **假設:** 本平台跨節點 TP 通訊受限、劣於 PP;但需與 GPU 型號/rank 效應分離。
- **環境前提:** 先以 3×L40S 同質做基線,再引入 5090;固定模型/序列長/batch。
- **變因:** 策略(TP=2/4、PP=2/4、TP×PP 組合)、同質(3×L40S)/異質(+5090)、microbatch 數 m(測氣泡,**含 m=1 串流角落**)、對稱 vs **非對稱 PP 層切**(如 5090 承載 6 層、各 L40S 8–10 層,測到達差是否被切分消除)、服務型態(高吞吐批次 vs 低延遲互動串流)、vLLM(推論)vs Megatron-LM(訓練)。
- **工具/範例命令:** 訓練 `--tensor-model-parallel-size`/`--pipeline-model-parallel-size`[W24];推論 vLLM `tensor_parallel_size`/`pipeline_parallel_size` + Ray[W21];Nsight Systems 抓 all-reduce vs send/recv 佔比與氣泡。
- **指標:** 訓練 throughput(tokens/s)、推論 TTFT/ITL/TPOT[W23]、通訊佔比、氣泡佔比(對照 (p−1)/m[W19])。
- **預期可能結果 → 決策:** ①TP 的 all-reduce 佔比高、throughput 低 → 確認跨節點 TP 反面教材;②PP 在 m≥4p 時氣泡可忽略、throughput 佳 → 確立 PP 為本平台模型並行首選;③同質與異質差異大 → 分離型號效應,支持第四章;④**m=1 串流下 TP=2 的 ITL 優於 PP → 確認低並行串流是 TP 的條件例外,服務層據 batch 大小選策略**;⑤**非對稱 PP 使各段計算時間趨齊、到達差消除 → 5090 可作非對稱 PP 節點,不必排除於大模型推論**。各 trace 能回答:TTFT→首 token 延遲(PP warm-up/TP 同步);ITL→穩態逐 token(氣泡/同步);training throughput→整體效率;Nsight→compute/comm/氣泡分解。

## Phase 4 — 異質掉隊者分解與排程隔離

- **假設:** 異質同步群的每步被最慢卡/最小卡綁住;同質群顯著較穩。
- **環境前提:** 3×L40S 同質群 vs 加入 5090 的異質群;固定 workload。
- **變因:** 群組組成(同質/異質)、per-rank micro-batch(等量/依 VRAM 調整)、HAMi shared 鄰居有無(測背景共享抖動)、Node Affinity/Taint 隔離開關。
- **工具/範例命令:** 唯讀 `kubectl get node -o json` 讀 `volcano.sh/vgpu-*` allocatable;Nsight Systems 或框架 profiler 把每步分解為 compute / 到達差(barrier 等待)/ transfer(collective)/ overlap;`nvidia-smi` 逐秒抓 util/mem。
- **指標:** 每步四段時間分解、barrier 等待時間、p95/p99 step time、掉隊者頻率、5090 vs L40S 每步計算時間差。
- **預期可能結果 → 決策:** ①異質群到達差段明顯大於同質 → 證實掉隊者效應,採同質群 + 5090 獨立;②HAMi 鄰居使 transfer/compute 抖動 → shared 模式訓練需退為 exclusive;③某工作 5090 反較慢 → 反駁「5090 恆快」,強化不可由型號推定。

## Phase 5 — MoE 由 microbenchmark 到真模型的分層驗證

- **假設:** 跨節點專家並行的 AlltoAll 是瓶頸;單卡 MoE 可完全避開。
- **環境前提:** 先 collective microbenchmark,再合成小 MoE,再真模型;不部署到生產。
- **變因:** top-k(1/2)、capacity factor(1.0/1.25/2.0)、dropless vs drop、專家分佈(單卡/跨卡)、hierarchical vs 平面 AlltoAll、模型與部署(合成小 MoE → DeepSeek-V2-Lite 單卡 bf16 → **Mixtral-8x7B 單卡 AWQ/GPTQ 4-bit**(純權重下界≈23 GB;**實載峰值 VRAM 為本 Phase 待量的關鍵數字**,須固定量化方案/執行期、`max-model-len`、並行數與 KV 保留比例)→ Mixtral-8x7B 跨節點 PP-3 bf16)。
- **工具/範例命令:** `all_to_all_perf -b 4K -e 256M -f 2`(microbenchmark);合成 MoE 量 dispatch/combine 佔比;真模型量 skew 與尾延遲;Nsight 抓 AlltoAll。
- **指標:** AlltoAll algbw/busbw、專家負載分佈(skew)、尾延遲、padding/drop 率、端到端 tokens/s、單卡 vs 跨卡對比。
- **每層能/不能支持的結論:** microbenchmark 只能支持「AlltoAll 頻寬與訊息大小關係」,**不能**支持真模型 skew;合成 MoE 能支持「dispatch/combine 佔比與 capacity 效應」,**不能**支持真實路由分佈;真模型才能支持「單卡 vs 跨節點專家並行的實際取捨」。
- **預期可能結果 → 決策:** ①AlltoAll 佔比隨跨節點線性上升[W27]→ 確認避免跨節點專家並行;②DeepSeek-V2-Lite 單卡吞吐佳 → 確立單卡 MoE 為本平台 MoE 首選;③hierarchical AlltoAll 顯著改善 → 若必須跨節點則採之;④**在指定 `max-model-len`/並行/KV 預算下實測單卡 AWQ-4bit Mixtral 峰值 VRAM 落在 L40S 46,068 MiB 內、且 tokens/s 與端到端延遲優於跨節點 PP-3(量化精度損失亦業務可接受)→ 確立「≲50B MoE 於該預算首選單卡量化」;若實測峰值超出單卡或精度不可接受 → 才轉跨節點 PP/EP**;⑤跨節點 PP-3 的實測氣泡因動態路由不均大於密集理論值 (p−1)/m → 佐證 MoE 不宜跨節點 PP。

---

完成這五個 Phase,前六章的每一個「假設」都會有對應的實測封閉。所有一手來源的精確位置與逐字引文,見**[證據附錄](08-evidence.md)**。
