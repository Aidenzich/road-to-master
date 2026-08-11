# vLLM 效能工程：從顯存預算、KV cache 到量化與壓測

> 更新日期：2026-08-11

把 vLLM 啟動並不等於完成部署。真正的效能工程，是先定義 workload 與 SLO，
再用可重現的實驗找出品質、延遲、吞吐量和顯存成本之間的 Pareto frontier。
本文以 vLLM 官方文件、模型卡及原始論文為依據，建立可驗證、可重現的調校方法。

## 先釐清：這是不是 MLOps 的工作？

是，但組織分工沒有唯一答案。推論 runtime 調校通常落在 MLOps、LLMOps、ML
platform 或 inference engineering 的交集：模型團隊定義品質門檻，平台團隊維護
映像、容量、觀測與回滾，服務擁有者則決定 latency／throughput SLO。重要的不是
職稱，而是以下責任有明確 owner：

- 固定模型、runtime、driver 與 kernel 版本，確保結果可重現。
- 以真實 prompt/output 長度和到達率壓測，而不是只跑單筆 demo。
- 同時守住品質、錯誤率、p95/p99 latency 與單位 token 成本。
- 用 metrics 判斷瓶頸，經過 canary 與 rollback 才變更 production 設定。

## 推論的兩個階段，以及 embedding 的差異

生成式 decoder-only LLM 的一次請求可以拆成：

1. **Prefill**：平行處理 prompt tokens，通常較偏 compute-bound，決定 TTFT 的
   重要部分。
2. **Decode**：每一步產生少量新 token，反覆讀取模型權重與既有 KV cache，
   常較偏 memory-bandwidth-bound，影響 ITL／TPOT。

vLLM 的 continuous batching 會在每次 scheduler iteration 重新組合工作，不需要
等一個靜態 batch 全部完成。V1 的 chunked prefill 在可用時預設開啟，會優先排
decode，再把長 prefill 切塊塞入剩餘 token budget。因此 `max_num_batched_tokens`
是「每次 iteration 最多處理多少 tokens」，`max_num_seqs` 才是「每次 iteration
最多處理多少 sequences」；兩者都不是 API 同時連線數。

Attention kernel 的 I/O 最佳化與 KV cache 的分頁管理是兩個不同問題；可先閱讀
[FlashAttention 與 PagedAttention 的架構比較](../../domains/utils/transformer-family/flash_attention/flash-attn-vs-paged-attn.md)，
再回到本文觀察它們如何影響 scheduler、cache capacity 與 latency。

Embedding／pooling 模型只有輸入前向與 pooling，沒有自迴歸 decode，所以不應把
生成服務的 ITL、output tokens/s 或長時間保留 decode KV 的直覺直接套用上去。
然而這不代表 vLLM pooling runtime 必然完全沒有 cache 配置：vLLM V1 已為部分
last-pooling 模型提供 prefix caching 與 chunked prefill，實際配置仍依模型與版本
而定。評估 embedding 應改看 batch tokens/s、requests/s、p95 latency、輸入長度
分佈與 embedding 品質。

## VRAM 不是只有模型權重

可用以下預算式建立心智模型：

```text
runtime VRAM
  = model weights
  + KV / recurrent-state cache
  + temporary activations and workspaces
  + CUDA graphs and communication buffers
  + allocator / CUDA context overhead
```

對一般 multi-head 或 grouped-query attention，未計 block padding 與實作額外成本
時，每個 sequence token 的 KV cache 約為：

```text
KV bytes/token
  = 2 × number_of_layers × number_of_kv_heads × head_dim × bytes_per_element
      ^ K 與 V
```

總 KV 用量再乘上所有同時駐留 sequences 的 cached tokens。這解釋了為何 context
長度、併發量、GQA/MQA 結構與 cache dtype 都很重要；只知道「模型是 8B」不能
推算 KV cache。

### `gpu_memory_utilization` 到底控制什麼？

截至本文更新時，vLLM 官方文件把 `--gpu-memory-utilization` 定義為該 instance
的 **model executor 總 GPU memory 比例**，最新文件預設值是 `0.92`。啟動時 vLLM
會 profile 非 KV 記憶體，再把預算中的可用部分配置給 GPU cache。它不是「將
GPU 的 92% 全部設為 KV cache」。若要精確控制 cache bytes，可用
`--kv-cache-memory-bytes`；一旦指定，該值會覆蓋以 utilization 推導 cache 大小的
方式。

因此看到 Qwen3-Embedding-8B 使用約 16 GB，不能直接推論「16 GB 都是 KV
cache」。官方模型卡標示模型為 8B parameters；若權重以 BF16/FP16 載入，僅權重
的理論下限就約為 16 GB，尚未包含 allocator、activation 與 runtime buffer。

另一個常見誤解是：降低 `max_model_len` 就會等比例縮小 vLLM 的 KV pool。它首先
限制單一 sequence 的可接受長度與容量可行性；若總 GPU 預算不變，runtime 仍可能
把剩餘空間建立成 cache blocks。要為同卡其他服務明確留空間，應調整
`gpu_memory_utilization` 或 `kv_cache_memory_bytes`，再由啟動日誌和
`nvidia-smi` 驗證，而不是只改 context 上限。

## 何時應增加或減少 KV cache 預算？

先用資料證明是 cache 壓力，不要看到 queue 就盲目加顯存。等待可能來自 KV
不足，也可能是 scheduler token budget、模型計算、CPU tokenization 或上游限流。

| 訊號與 workload | 判讀 | 優先實驗 |
|---|---|---|
| `num_preemptions` 持續增加，且 cache 接近滿載 | 高度可能是 KV 壓力 | 增加 cache 預算；或降低 `max_num_seqs`／`max_num_batched_tokens` |
| 長 context、同時生成 sequences 多 | 每個 request 的 cached tokens 高 | 增加 cache、量化 KV、縮短 token budget，或做 workload 分池 |
| `num_requests_waiting{reason="capacity"}` 上升，但 cache 不滿 | 可能是 scheduler／compute 容量 | sweep token budget 與 concurrency，不先增加 cache |
| 高 cache 使用率但無排隊、preemption，SLO 正常 | 資源正在被有效利用 | 不因百分比高就變更 |
| 同卡要 co-locate 多個服務 | 必須提供硬邊界 | 降低 utilization，最好用明確 bytes、獨立程序與 admission control |
| pooling／embedding，短輸入為主 | decode cache 收益有限，但 batching 仍重要 | 降低預算做 A/B，增加 batch token budget，觀察 OOM 與吞吐 |
| CUDA OOM | 不一定是 KV；也可能是 graph、activation 或 workspace peak | 先讀 stack/log，保留 headroom，再分別調低總預算或 batch |

vLLM 官方 optimization guide 建議：頻繁 preemption 時可增加
`gpu_memory_utilization`，或降低 `max_num_seqs`／`max_num_batched_tokens`。前者增加
cache 容量，後兩者降低同一 iteration 的並行記憶體壓力。這是診斷後的選項，
不是所有機器都適用的固定值。

## 調度參數的真正 trade-off

| 參數 | 主要限制 | 調高通常帶來 | 主要風險 |
|---|---|---|---|
| `max_num_seqs` | 每 iteration sequences 數 | 更多並行機會 | KV／activation 壓力與 tail latency |
| `max_num_batched_tokens` | 每 iteration token budget | 更大的 prefill batch、較高吞吐或較佳 TTFT | decode 被大 prefill 影響，ITL 可能變差 |
| `max_model_len` | 單 sequence 最大長度 | 接受更長請求 | 啟動容量門檻與單請求最壞資源需求上升 |
| `gpu_memory_utilization` | instance 總顯存預算 | 通常留下更多 KV blocks | 給 graph、driver 或同卡程序的 headroom 下降 |
| `kv_cache_memory_bytes` | 每 GPU cache 的明確 bytes | 容量可預測、便於 co-location | 設太大會擠壓其他 runtime 記憶體 |
| `kv_cache_dtype` | KV 儲存精度 | FP8 可容納更多 tokens | 需要支援 kernel；應以資料校準並驗證品質 |

官方 tuning guide 對 chunked prefill 的方向是：較小的 token budget 通常有利
ITL，較大的值通常有利 TTFT 與吞吐；但分界會受模型、GPU 和長度分佈影響。
不要把範例中的 `2048` 或 `>8192` 當成跨硬體常數。

## 量化：權重大小、速度與品質是三個問題

### 檔案和顯存會縮小，但不是完全相等

只計算量化後的主要權重，`P` 個參數、每參數 `b` bits 的理想下限是：

```text
weight payload bytes ≈ P × b / 8
```

實際 checkpoint 還包含 scale、zero-point、metadata，以及可能保持高精度的
embedding、normalization 或其他 tensors。載入 VRAM 後更要加上前述 cache、
activation、graph 與 workspace。因此「檔案 18 GB，所以 runtime 只用 18 GB」
並不成立。

以 31B～32B dense model 為例，只看理想 weight payload：

| 權重精度 | 理論 payload | 部署解讀 |
|---|---:|---|
| BF16／FP16 | 約 62～64 GB | 單張 32 GB GPU 放不下權重 |
| FP8／INT8 | 約 31～32 GB | 幾乎沒有 runtime、activation 與 KV headroom，通常不適合直接塞入 32 GB |
| 4-bit | 約 15.5～16 GB | checkpoint 常因 scales／高精度 tensors 更大，但通常可留下可用 headroom |

這只是容量初篩，不是「需要多少 VRAM」的最終答案。最終值還取決於模型架構、
量化格式、context、concurrency、backend kernel 與是否 offload。

### 量化不保證更快

Decode 經常受 memory bandwidth 限制，較小權重可能減少每步搬運量；但 prefill
可能較偏 compute-bound，而低位元 kernel 支援、dequantization、batch shape 與
硬體世代都會改變結果。某個 AWQ checkpoint 在一張 GPU 上快，不代表 GPTQ、
GGUF 或同一格式在另一張 GPU 也快。速度必須在目標 runtime 與 workload 實測。

### 大模型低位元，不是永遠勝過小模型高精度

「永遠優先大模型 + 4-bit」不是可靠規則。模型家族、訓練資料、task、instruction
following、安全性、長 context 與量化敏感度都可能改變排名。GPTQ 與 AWQ 論文
證明特定模型在其評測設定下可大幅壓縮且維持接近原模型的品質；這不等於任何
70B 4-bit 都必然勝過任何 8B BF16。

正確方法是先定義任務品質 gate，再比較同一 VRAM／成本預算下的候選：

1. 在未量化 checkpoint 建立 task-specific baseline。
2. 對每個候選固定 prompt template、sampling、資料集與評分器。
3. 同時量測品質、TTFT、ITL、throughput、p95/p99、VRAM 與錯誤率。
4. 選擇通過品質門檻後成本最低或 SLO 最佳的版本。

## AWQ、GPTQ 與 Q4_K_M 不只是三個「4-bit 副檔名」

| 名稱 | 核心概念 | 常見 artifact／runtime | 注意事項 |
|---|---|---|---|
| AWQ | 用 activation statistics 找出 salient channels，透過 scaling 降低 weight-only quantization error | 常見為 Safetensors；vLLM 等 GPU runtimes | 原論文不是把「1% 權重另存高精度」這麼簡單；kernel 與硬體支援決定速度 |
| GPTQ | 使用 approximate second-order information，逐層／逐 block 補償量化誤差 | 常見為 Safetensors；vLLM 等 GPU runtimes | bit width、group size、act-order 與 backend 都是 artifact contract 的一部分 |
| Q4_K_M | GGUF 生態中的 K-quant 組合；可對不同 tensor 使用不同量化型別 | llama.cpp／GGUF 工具鏈 | `_M` 的實際 tensor 配置應讀 artifact metadata／quantizer，不應只靠名稱猜品質 |

格式和演算法也不能完全混為一談：GGUF 是容器格式，Q4_K_M 是其中一種
quantization recipe；AWQ、GPTQ 是量化方法，發布時仍需搭配實際 checkpoint 格式
和相容 kernel。選型順序應是「模型支援 → artifact contract → runtime/kernel →
實測」，不能只看到 `4-bit` 就視為等價。

## 單張 32 GB GPU 的可重現練習路線

以下場景用一張 32 GB GPU 就能完成，重點是每次只改一個因素。不要以刻意把
production 打到 OOM 作為學習方法；在隔離環境設定 request timeout、concurrency
上限與自動復原。

### 實驗 0：固定基線

保存 runtime image digest、模型 revision、dtype、driver、所有 engine arguments
及 workload seed。準備至少三種長度分佈：短／短、長／短、長／長
（input/output），先跑 warmup，再記錄冷啟動與穩態結果。

### 實驗 1：顯存與 KV 邊界

固定模型與流量，sweep 明確的 KV cache bytes 或 utilization。記錄啟動日誌中的
weight、non-KV、cache 容量，以及 `kv_cache_usage_perc`、preemptions、waiting、
TTFT/ITL。預期產物不是一個「最大值」，而是一條 concurrency × context 的容量
曲線。

### 實驗 2：低延遲與高吞吐 Pareto frontier

對 `max_num_seqs`、`max_num_batched_tokens` 做小型 grid search；每組同時測閉環
concurrency 與開環 request rate。使用 `vllm bench serve` 固定 input/output length、
request rate、burstiness 與 max concurrency，保存詳細 JSON，而非只截終端平均值。

### 實驗 3：prefix caching 與量化

Prefix cache A/B 應包含兩組流量：高比例相同且 block-aligned 的長前綴，以及完全
不同前綴。若只測重複 prompt，會高估真實收益。量化 A/B 則必須加入 task-specific
品質 gate；不能只比較 tokens/s 和 VRAM。

### 實驗 4：embedding 專用 workload

對 pooling 模型 sweep 輸入 token 長度、batch token budget、concurrency 與
utilization。比較 vLLM 與專用 embedding runtime 時，使用完全相同的 tokenizer、
normalization、instruction 與輸出維度，否則速度數據和向量品質都不可比。

## 生產觀測與告警

vLLM 官方 `/metrics` 暴露的核心訊號包括：

- `vllm:num_requests_running` 與 `vllm:num_requests_waiting`。
- `vllm:num_requests_waiting_by_reason`，可區分 capacity 與 deferred。
- `vllm:kv_cache_usage_perc` 與 `vllm:num_preemptions`。
- `vllm:time_to_first_token_seconds`、`vllm:inter_token_latency_seconds`、
  `vllm:e2e_request_latency_seconds` 與 queue time。
- prompt／generation tokens counters，用來正規化吞吐與成本。

不要用「cache > 90% 且 waiting > 5」這類無 workload 背景的固定數字直接決定
擴容。告警應基於持續時間與使用者 SLO，例如 p95 queue time 超標、preemption
rate 持續上升、錯誤率或 timeout 超標；容量動作則要先區分 cache、compute、CPU
與上游 admission control 瓶頸。

## 一份可靠的調校順序

1. 固定版本、模型 artifact、品質 gate 與 workload trace。
2. 先確定權重能載入並保留安全 headroom。
3. 以 context × concurrency sweep 找出 cache／preemption 邊界。
4. 再調 scheduler token/sequence budget，畫 latency-throughput frontier。
5. Prefix cache、KV quantization、weight quantization 每次只開一項並重跑品質 gate。
6. 用 production metrics 驗證實驗結論，而不是把 benchmark 最快組合直接上線。

## 結論

vLLM 不會只憑 GPU 型號替服務自動找到「最佳 batch 或 concurrency」。它會依可用
顯存 profile cache 容量，scheduler 也會動態組 batch；但 SLO、流量分佈與品質
門檻仍必須由部署者提供。超出當下執行容量的請求通常會等待、被後續 iteration
排入，KV 壓力下也可能 preempt/recompute；這和「所有請求固定 FIFO、一批跑完
才跑下一批」不同。

最可靠的能力不是背下某組參數，而是能從模型結構和顯存預算提出假設，用可重現
壓測與 metrics 證偽，再把結果轉成有回滾能力的部署設定。

## 參考資料

- [vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM Production Metrics](https://docs.vllm.ai/en/latest/usage/metrics/)
- [vLLM Pooling Models](https://docs.vllm.ai/en/latest/models/pooling_models/)
- [vLLM V1 User Guide](https://docs.vllm.ai/en/latest/usage/v1_guide/)
- [vLLM Benchmark Serve CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/)
- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [llama.cpp Quantization README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Qwen3-Embedding-8B model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
