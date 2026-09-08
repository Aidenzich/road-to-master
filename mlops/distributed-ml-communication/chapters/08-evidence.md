# 證據附錄

> [回導讀](../README.zh-TW.md) | [證據與來源](08-evidence.md)

> 正文各章只留必要引用與其含義;這裡收錄可追溯的完整證據——本地程式碼的精確位置(檔案:行號)、外部來源的網址與逐字引文、以及明確標注的證據缺口。

## 來源固定慣例(可重現性)

- **本地來源:** `gpu-cluster-iac`,git HEAD `ae6bba2`,2026-09-08 唯讀讀取,以 `path:line` 標定。
- **外部論文:** 一律指向 **arxiv.org 官方全文 PDF**(`https://arxiv.org/pdf/<id>`,同時含摘要與正文,可用 Ctrl-F 核對逐字引文),並註明引文所在段落(摘要 / 正文某節)。本輪已將先前使用的 ar5iv 第三方渲染代理全面改為 arxiv.org 官方 URL。**重要更正(本輪):** 多數載入結論的逐字引文(記憶體公式、通訊機制、氣泡佔比等)出自論文**正文**,不在 `https://arxiv.org/abs/` 摘要頁;故改用 `/pdf/` 全文連結以確保讀者能在所引 URL 核對到該逐字句。
- **官方 repo 檔案:** 釘選到**不可變 commit SHA**(URL 內嵌的 40 字元 hash),而非會移動的 `main`/`master` 分支。
- **滾動維護的官方文件 / model card / blog:** 無版本化 URL 者,以查閱日 2026-09-08 標示。
- **URL 格式:** 本附錄每一條外部參考都附**完整 https:// 網址**,讀者可直接複製取用、無需查閱任何內部支援檔。

---

## 本地錨點(gpu-cluster-iac,git HEAD `ae6bba2`,2026-09-08 唯讀)

以下為原研究當時唯讀檢視的基礎設施專案相對路徑,不是本公開專案內的檔案。該來源快照未隨本文發布,外部讀者無法僅憑本 checkout 獨立重驗;應將這些項目視為作者記錄的本地觀察,而非公開可重現的部署證據。私人節點名稱與位址已去識別化;對外可查的技術來源列於後文。

- **平台分層與 HAMi 政策:** `docs/architecture.md:35-56`;GPU 型號標籤 `manifests/k3s/config.yaml:6-11`(5090,`gpu.arch=blackwell / compute-capability=120 / vram-class=32gb / tier=consumer`)、`manifests/k3s/agent.yaml:5-10`(L40S,capability 89);`docs/architecture.md:6`「RTX 5090 is onboarded first」。
- **節點清單(未納 git 追蹤的操作者庫存,非部署證據):** `config/nodes.env:4-11`(私人節點名稱與位址已省略)被 `.gitignore:1` 忽略;HEAD `ae6bba2` 追蹤的是佔位範例 `config/nodes.example.env:4-11,19-22`(示例節點與佔位位址,自述「local operator configuration ... intentionally ignored by Git」、後端 `auto`)。
- **網路 backend 選擇(條件式):** `scripts/lib/dual-node-runtime.sh:239-289`(`auto`→沿用既有控制面後端;否則 host-gw 需雙向直連 `:277-284`,否則 wireguard-native `:282`);唯一硬選 host-gw 的是**單節點**生產設定 `manifests/l40s-production/server-config.yaml:4-5`(`flannel-iface: eno2`、`flannel-backend: host-gw`),`scripts/l40s-production.sh:614` 斷言 `kubectl get nodes | wc -l == 1`;WireGuard MTU 1280/1200 `manifests/k3s/flannel-net-conf.json:7`、`scripts/install-k3s.sh:50`;NVLink/RDMA/線速全 repo 零命中只代表未找到宣告,不能證明硬體不支援或現場沒有;NVLink 型號支援改以官方規格 [W40][W41] 為準。
- **Volcano:** `manifests/volcano/gpu-job.yaml:16-44`(Job/minAvailable/nodeSelector/vgpu-*);scheduler plugins `manifests/volcano/values.yaml:6-23`(gang + deviceshare binpack);版本 `versions.env:6-12`。
- **HAMi:** `manifests/hami/values-exclusive.yaml:2-17`(資源名、`deviceSplitCount:3`、`gpuMemoryFactor:2`);容量 `docs/l40s-production-runbook.md:56`(46,068 MiB → 23,034 units,`23034×2=46068`);斷言 `scripts/l40s-production.sh:283`;配額 annotation `manifests/canary/canary.yaml:66`;版本 `versions.env:13-15`。
- **契約與單卡結構:** `docs/workload-contract.md:9-61`(WorkloadProfile 欄位)、`:17-25`(**同一 profile 宣告 `["89","120"]` 雙型號相容,證明允許多型號**)、`:63-64`(**「This CRD does not exist yet」——profile→Job 是未實作整合縫**)、`:117-118`(逐型號驗證,非排程禁令)、`:152-155`(核心 % 非硬隔離)、`:144-146`(capacity receipt best-effort、不保證免 OOM);能力標籤非排程/非資源帳 `docs/architecture.md:74-89`;admission 硬鎖單卡 `scripts/l40s-production-admit-workload.py:58,72`;分散式訓練為未來 `docs/architecture.md:60-68`;不遷移 `:156-158`;CUDA 雙架構 `docs/architecture.md:98-108`。
- (支援材料)完整逐行走查:`.loop-manager/spike/evidence/local-gpu-cluster-iac.md`。

---

## 外部參考(一手來源;查閱日 2026-09-08;每條均附完整 https:// 網址)

**第一章(單卡記憶體與精度)**
- **[W10]** ZeRO,`https://arxiv.org/pdf/1910.02054`(正文 記憶體/通訊分析節)——「an fp32 copy of the parameters, momentum and variance, with memory requirements of 4Ψ, 4Ψ, and 4Ψ bytes, respectively.」;「In total, this results in 2Ψ+2Ψ+KΨ=16Ψ bytes of memory requirement.」(混合精度 Adam K=12);「ZeRO-DP incurs no additional communication using Pos and Pg」;「ZeRO-DP incurs a maximum of 1.5x communication when using Pp in addition to Pos and Pg」。(作者已核。)
- **[W13]** PyTorch AMP 2.14,`https://docs.pytorch.org/docs/2.14/amp.html`——reductions「often require the dynamic range of float32」;fp16 underflow →「gradient scaling」。
- **[W14]** FP8 Formats,`https://arxiv.org/pdf/2209.05433`(正文 formats/實驗節)——「E4M3 for weight and activation tensors, and E5M2 for gradient tensors.」;「FP8 training matches FP16 or bfloat16 training results ... without changing any ... hyperparameters.」
- **[W16]** Llama-3-8B model card,`https://huggingface.co/meta-llama/Meta-Llama-3-8B`——「Llama 3 comes in two sizes — 8B and 70B parameters」。(8.03B 為 HF 側欄值,未逐字擷取;計算以 Ψ≈8.0×10⁹。)

**第二章(網路與 collective)**
- **[W1]** nccl-tests PERFORMANCE.md,`https://raw.githubusercontent.com/NVIDIA/nccl-tests/c6eb15875f508076f3f26de4f7da3899701bc4db/doc/PERFORMANCE.md`——「Algorithm bandwidth is using the most commonly used formula for bandwidth : size (S) / time (t).」;AllReduce busbw 因子 `2*(n-1)/n`,AllGather/ReduceScatter/AllToAll `(n-1)/n`,Broadcast/Reduce = 1;「The bus bandwidth should reflect the speed of the hardware bottleneck : NVLink, PCI, QPI, or network.」
- **[W2]** Thakur/Rabenseifner/Gropp, "Optimization of Collective Communication Operations in MPICH," IJHPCA 19(1):49-66, 2005,`https://fs.hlrs.de/projects/rabenseifner/publ/HPCA_collectives.pdf`(PDF 全文)——「the time taken to send a message between any two nodes can be modeled as α + nβ ...」;ring allgather `T=(p−1)α + (p−1)/p nβ`;Rabenseifner allreduce `T=2 lg p α + 2(p−1)/p nβ + (p−1)/p nγ`;「minimizing latency for short messages and minimizing bandwidth use for long messages.」
- **[W3]** NVIDIA blog, "Massively Scale Deep Learning Training: NCCL 2.4," 2019-02-04,`https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4`——ring「to achieve full bandwidth」但「latency scales linearly with the number of GPUs」;double binary trees「full bandwidth and a logarithmic latency」。
- **[W4]** NCCL 環境變數,`https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html`——`NCCL_IB_DISABLE`「NCCL will fall back to using IP sockets.」;`NCCL_ALGO`/`NCCL_PROTO`/`NCCL_SOCKET_IFNAME`。
- **[W5]** NCCL collectives,`https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html`——「Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation.」
- **[W6]** nccl-tests README(釘選 commit b52f9a1),`https://github.com/NVIDIA/nccl-tests/blob/b52f9a1efbe7c810853be922531e5e0345c2e7b1/README.md`——binaries 與 `-b/-e/-f/-g` 旗標;單機例 `./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8`;**多節點(載入 Phase 1):**「NCCL tests rely on MPI to work on multiple processes, hence multiple nodes.」、「make MPI=1 MPI_HOME=... CUDA_HOME=... NCCL_HOME=...」、「The number of process is managed by MPI and is therefore not passed to the tests as argument. The total number of ranks (=CUDA devices) will be equal to (number of processes)*(number of threads)*(number of GPUs per thread).」、8 節點例 `mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1`(NB: 須 `MPI=1` 編譯)→ 本平台每主機一卡的類比形式 `mpirun -np N -N 1 ... -g 1`。GAP:in-place/out-of-place 為執行期 stdout,非文件本文,不作引文。
- **[W7]** iperf3 文件,`https://software.es.net/iperf/invoking.html`——「iperf3 is a tool for performing network throughput measurements. It can test TCP, UDP, or SCTP throughput.」;`-P` 多流、`-R` 反向。限制(推論,非引文):只量主機 TCP/UDP,不量 GPU collective。

**第三章(資料/張量/管線並行)**
- **[W8]** PyTorch DDP notes 2.14,`https://docs.pytorch.org/docs/2.14/notes/ddp.html`——「DDP's performance advantage comes from overlapping allreduce collectives with computations during backwards.」;「the Reducer organizes parameter gradients into buckets, and reduces one bucket at a time.」
- **[W9]** DDP 論文,`https://arxiv.org/pdf/2006.15704`(正文 system-design 節)——「The AllReduce operation on gradients can start before the local backward pass finishes.」
- **[W11]** FSDP 論文,`https://arxiv.org/pdf/2304.11277`(正文 §3 設計節)——「During forward and backward computation, FSDP only materializes unsharded parameters and gradients of one unit at a time」;反向「launches ReduceScatter to reduce and shard gradients.」;「FSDP is intrinsically different ... a native solution」;「FSDP backward prefetching issues the next AllGather before the current ReduceScatter.」
- **[W12]** PyTorch FSDP2 tutorial,`https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html`——「FSDP can be considered a decomposition of DDP's all-reduce into reduce-scatter and all-gather operations.」
- **[W15]** PaLM,`https://arxiv.org/pdf/2204.02311`(正文 training-efficiency/MFU 節)——MFU =「the ratio of the observed throughput (tokens-per-second) relative to the theoretical maximum throughput of a system operating at peak FLOPs.」
- **[W17]** Megatron-LM,`https://arxiv.org/pdf/1909.08053`(正文 model-parallel transformer 節)——「f is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity in the backward pass.」;「two all-reduces in the forward path and two in the backward path」;「There are 4 total communication operations in the forward and backward pass of a single model parallel transformer layer.」
- **[W18]** GPipe,`https://arxiv.org/pdf/1811.06965`(正文 performance 節)——「This bubble time is O(K−1 / M+K−1) amortized over the number of micro-steps M.」;「negligible when M ≥ 4×K.」
- **[W19]** Megatron pipeline,`https://arxiv.org/pdf/2104.04473`(正文 §2 排程/bubble 節)——「Bubble time fraction (pipeline bubble size) = t_pb / t_id = (p−1)/m」,`t_id = m·(t_f + t_b)`;「one forward pass followed by one backward pass (1F1B for short)」;interleaved「= 1/v · (p−1)/m ... reduces the bubble time by v」;「tensor model parallelism should generally be used up to degree g when using g-GPU servers, and then pipeline model parallelism can be used to scale up to larger models across servers.」(作者已核。)
- **[W20]** 序列並行,`https://arxiv.org/pdf/2205.05198`(正文 introduction/背景節)——「tensor-level model parallelism is typically limited to a relatively small group of GPUs that are connected with high speed bandwidth, such as GPUs connected with NVLink inside a DGX server.」;序列並行頻寬「the same」、「does not introduce any communication overhead.」
- **[W21]** vLLM 平行化文件,`https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/docs/serving/parallelism_scaling.md`——「if the model is too large for a single GPU but fits on a single node ... use tensor parallelism.」;「The common practice is to set the tensor parallel size to the number of GPUs in each node, and the pipeline parallel size to the number of nodes.」;「Ray for multi-node inference」。
- **[W22]** vLLM 論文,`https://arxiv.org/pdf/2309.06180`(正文 method 節)——「Megatron-LM style tensor model parallelism ... GPUs constantly synchronize intermediate results via an all-reduce operation.」
- **[W23]** vLLM benchmarking,`https://raw.githubusercontent.com/vllm-project/vllm/9e905f7450fb556c4f43e49d4ece0f728cbfda2b/docs/benchmarking/cli.md`——TTFT「time from sending a request to receiving its first streamed output.」;ITL「time between consecutive streamed outputs.」;TPOT「(end-to-end latency − TTFT) / (number of output tokens − 1)」。
- **[W24]** Megatron-LM 175B 範例(釘選 commit 779c5b7),`https://raw.githubusercontent.com/NVIDIA/Megatron-LM/779c5b748dbcf00ad9e36d539c576b404ab4abe9/examples/gpt3/train_gpt3_175b_distributed.sh`——`--tensor-model-parallel-size 8`、`--pipeline-model-parallel-size 16`。
- **[W33]** vLLM 平行化文件(同 [W21] 釘選版),`https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/docs/serving/parallelism_scaling.md`——「Furthermore, if the GPUs on the node do not have NVLINK interconnect (e.g. L40S), leverage pipeline parallelism instead of tensor parallelism for higher throughput and lower communication overhead.」

**第四章(異質/掉隊者)**
- **[W32]** PyTorch straggler mitigation blog, 2024-11-14,`https://pytorch.org/blog/straggler-mitigation/`——DDP「by default runs synchronous SGD」;「all the processes have to wait for the stragglers before synchronizing gradients ... bottlenecks distributed performance to the slowest worker.」;「persistent stragglers, which can be caused by hardware degradation or a network issue ... leading to nearly no straggler mitigation.」;「unstable network I/O」;「some input examples can be outliers in terms of the data size」。(「異質 VRAM→不同 micro-batch→到達差」為作者推論,非本文引文。)

**第五章(MoE)**
- **[W25]** GShard,`https://arxiv.org/pdf/2006.16668`(正文 algorithm/einsum 節)——dispatch/combine einsum;「AllToAll ... used to reshard a sharded tensor from one dimension to another.」;「each token dispatched to at most two experts.」
- **[W26]** Switch Transformer,`https://arxiv.org/pdf/2101.03961`(正文 routing/capacity 節)——「route to only a single expert」;「expert capacity = (tokens per batch / number of experts) × capacity factor」;drop:「computation is skipped and the token representation is passed directly to the next layer through the residual connection.」;aux loss `α·N·Σ f_i·P_i`,α=1e-2。
- **[W27]** DeepSpeed-MoE,`https://arxiv.org/pdf/2201.05596`(正文 expert-parallel/communication 節)——「Expert parallelism requires all-to-all communication between all expert parallel devices.」;「it is difficult to scale expert parallelism to many devices as the latency increases linearly with the increase in devices.」;hierarchical「reduces the communication hops from O(p) to O(G+p/G), where G is the number of GPUs in a node and p is the total number of GPU devices.」(作者已核。)
- **[W28]** MegaBlocks,`https://arxiv.org/pdf/2211.15841`(正文/摘要)——「Our approach never drops tokens ... enabling end-to-end training speedups of up to 40%.」;移除的兩難:「choose between dropping tokens ... or wasting computation and memory on padding.」
- **[W29]** DeepSeek-V3,`https://arxiv.org/pdf/2412.19437`(正文 load-balance 節)——bias-term balancing「decrease the bias term by γ if ... overloaded, and increase it by γ if ... underloaded.」;「too large an auxiliary loss will impair the model performance.」
- **[W30]** Mixtral——`https://mistral.ai/news/mixtral-of-experts`(blog)「Mixtral has 46.7B total parameters but only uses 12.9B parameters per token.」、「a router network chooses two of these groups ('experts')」;論文 `https://arxiv.org/pdf/2401.04088`(摘要)「access to 47B parameters, but only uses 13B active」。(作者已核。)
- **[W31]** DeepSeek-V2-Lite model card,`https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite`——「DeepSeek-V2-Lite comprises 15.7B total parameters, of which 2.4B are activated for each token.」;「2 shared experts and 64 routed experts ... 6 experts will be activated for each token.」
- **[W37]** vLLM data-parallel 部署文件(釘選 commit cd64c2d),`https://raw.githubusercontent.com/vllm-project/vllm/cd64c2dea9c72af333de7ec05293d54bbf1bd128/docs/serving/data_parallel_deployment.md`——「Forward passes must be aligned, and expert layers across all ranks are required to synchronize during every forward pass, even when there are fewer requests to be processed than DP ranks.」;「For MoE models, when any requests are in progress in any rank, we must ensure that empty 'dummy' forward passes are performed in all ranks that don't currently have any requests scheduled.」

**第三章 PEFT 維度 / 第六章反面證據**
- **[W34]** QLoRA,`https://arxiv.org/pdf/2305.14314`(**摘要**)——「We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance.」(**本輪更正:** 先前引用的「average memory requirements ... from >780GB ... to <48GB」一句出自論文**正文/引言**、不在摘要頁;為確保讀者能在所引 URL 核對到逐字句,改用上述**摘要原句**,正文相依敘述亦改為「單張 48GB 卡可微調 65B、保持 16-bit 全量微調任務表現」。此為 4-bit 凍結底模 + adapter 的 profile,**非**全參數更新。)
- **[W35]** ZeRO++,`https://arxiv.org/pdf/2306.10209`(正文 hpZ 節)——「in hpZ, we eliminate the inter-node all-gather during the backward pass by holding secondary FP16 weights partition within each node.」;「on modern GPU clusters, intra-node communication bandwidth is significantly higher than inter-node communication bandwidth.」
- **[W36]** Zero-Bubble PP,`https://arxiv.org/pdf/2401.10241`(正文 memory 節)——「ZB-2p ... comes at the cost of doubling the memory consumption compared to 1F1B」;「higher memory consumption compared to the 1F1B baseline」。(註:「峰值活化隨 T_comm 線性」之公式未於一手頁逐字證實,不採。)
- **[W38]** BytePS repo,`https://github.com/bytedance/byteps`——橫幅「This repository was archived by the owner on Dec 8, 2025. It is now read-only.」

**第七章(實驗設計)**
- **[W39]** NCCL 環境變數,`https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html`——`NCCL_BUFFSIZE` 預設 4194304(4 MiB);`NCCL_NSOCKS_PERTHREAD`「For 100G networks, can be set to 4」,與 `NCCL_SOCKET_NTHREADS` 乘積「cannot exceed 64」;`NCCL_SOCKET_IFNAME`「specifies which IP interfaces to use for communication.」

---

## NVLink 概念與型號支援補查(2026-09-08)

- **[W40]** [NVIDIA RTX 5090 官方規格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)——完整規格的 `NVIDIA NVLink™ (SLI-Ready)` 欄為 **No**。支持本型號不能直接加裝 NVLink 的判斷,不代表已探測現場裝置。
- **[W41]** [NVIDIA L40S 官方規格](https://www.nvidia.com/en-us/data-center/l40s/)——`NVIDIA® NVLink® Support` 欄為 **No**;互連介面列 PCIe Gen4 x16。GPU 本地記憶體頻寬不是跨主機網路頻寬。
- **[W42]** [NVIDIA NVLink and NVLink Switch 概念介紹](https://www.nvidia.com/en-us/data-center/nvlink/)——介紹 GPU 間專用互連與交換硬體,並說明專用 Switch 系統可延伸跨節點至機架。用於界定 NVLink 並非通用軟體、也非永遠限於單主機;不能把頁面的系統頻寬套到 RTX 5090 或 L40S。本研究不以此產品介紹代替模型效能實測。

## 明確標注的證據缺口 / 待實測

- 本平台 NIC 線速與 NCCL busbw:repo 未宣告,為最高優先未知(第七章 Phase 1 封閉)。
- 實際生效的 flannel 後端(host-gw 明文 vs wireguard-native 加密)與實體介面:source 顯示為條件式選擇,現況需唯讀讀出(第七章 Phase 1),不可預設 host-gw/eno2。
- FP8 tensor 在 L40S(Ada)之實際支援與代價:未取得 Transformer Engine 逐句一手引文,標為待查。
- Llama-3-8B 精確參數(8,030,261,248):官方 card 標「8B」,子欄 8.03B 未逐字擷取;計算以 Ψ≈8.0×10⁹。
- nccl-tests in-place/out-of-place:僅執行期 stdout,官方本文未敘明,不作引文。
- 「異質 VRAM→不同 micro-batch→到達差」為作者推論,非 [W32] 直接引文。
- min-member>1 真實 gang、binpack 實際放置、RTX 5090 之 HAMi 通告容量:repo 未見,為 runtime-only,需授權唯讀探測封閉。
