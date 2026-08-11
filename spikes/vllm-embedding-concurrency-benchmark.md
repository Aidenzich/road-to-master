# Qwen3-Embedding-8B 在單 GPU vLLM 上的併發與 Batch 實驗

## 摘要

本實驗測量 `Qwen/Qwen3-Embedding-8B` 透過 vLLM OpenAI-compatible embeddings API
提供服務時的端到端效能，重點回答三個問題：

1. 併發請求增加時，吞吐量與延遲如何變化？
2. 單一 `/v1/embeddings` request 的 `input[]` 可以容納多大的 batch？
3. vLLM 的 dynamic batching 在目前設定下如何排程？

在約 990 tokens 的文章輸入上，吞吐量於 concurrency 4 左右飽和在約
`13.7 requests/s`。繼續增加至 concurrency 64 沒有提高吞吐量，但 p95 latency
由 `84.8 ms` 增加至 `4.67 s`。單一 request 的 `input[]` 已成功驗證至 256 筆，
但它會被 scheduler 分波處理，並不代表 256 筆會同時進入 GPU。

## 實驗目的

- 建立 embedding API 的單機容量基準。
- 找出適合線上服務的 concurrency 範圍。
- 量測大 batch 對延遲、吞吐量、response size 與排隊的影響。
- 釐清 `MAX_NUM_SEQS`、`MAX_NUM_BATCHED_TOKENS` 與 API `input[]` 大小之間的差異。

## 設備與軟體規格

為避免揭露環境識別資訊，僅保留與效能分析直接相關的數值。

| 項目 | 規格 |
|---|---:|
| GPU 數量 | 1 |
| GPU VRAM | 32,607 MiB |
| Host RAM | 約 94 GiB |
| vLLM GPU memory utilization | 0.60 |
| 模型規模 | 8B |
| 模型精度 | FP16 |
| 最大輸入長度 | 8,192 tokens |
| Embedding 維度 | 4,096 |
| 模型載入後 GPU memory | 約 19,732 MiB |
| 壓測期間 GPU memory 峰值 | 約 19,880 MiB |
| vLLM 版本 | 0.26.0 |

## Dynamic batching 設定

| 設定 | 值 |
|---|---:|
| Scheduler policy | FCFS |
| `MAX_NUM_SEQS` | 32 |
| `MAX_NUM_BATCHED_TOKENS` | 2,048 |
| `MAX_MODEL_LEN` | 8,192 |
| `prefill_schedule_interval` | 1 engine step |
| Chunked prefill | Enabled |
| Prefix caching | Enabled |
| Asynchronous scheduling | Disabled |

`MAX_NUM_BATCHED_TOKENS` 並未由部署參數明確指定；2,048 是 vLLM 0.26.0 在此硬體容量與
OpenAI API server usage context 下推導出的預設值。

目前沒有固定的毫秒級 batch collection window。vLLM 使用 continuous batching：新請求在
下一個 engine step 參與排程，每一步同時受 active sequences 與 token budget 限制。對本實驗
約 990-token 的輸入而言，2,048-token budget 比 32-sequence 上限更早成為限制。

## 實驗設計

### 測試資料

- 輸入是一篇完整的 Markdown 工程文章。
- 原文長度為 3,670 characters。
- 加入測試標記後，每筆約 987--993 prompt tokens。
- 每個 input 的最前方加入不同 sample identifier，避免相同 prefix cache 使結果過度樂觀。
- 使用完整的 4,096-dimensional embedding，未降低輸出維度。

### 測試路徑

- Client 與 API server 位於同一台主機，經 loopback HTTP 呼叫。
- 每次測試前先執行一次 warm-up。
- latency 包含 HTTP、tokenization、model inference、pooling、JSON serialization 與 response parsing。
- 測試期間同步採樣 GPU utilization、GPU memory，以及 vLLM running/waiting metrics。
- 所有測試均使用相同模型、精度與 scheduler 設定。

### Concurrency 測試

測試 concurrency 為 `1, 2, 4, 8, 16, 32, 48, 64`。每個 worker 每次送出一篇文章：

- concurrency 1、2、4：各執行 8 requests。
- concurrency 8 以上：各執行兩波 requests，即 `2 × concurrency`。
- 每個 concurrency level 的第一波以 barrier 同時開始。

記錄 successful requests、wall-clock throughput、input token throughput、p50、p95、最大延遲、
GPU 峰值與 waiting queue 峰值。

### 單一 request batch 測試

依序測試 `input[]` 大小 `1, 8, 16, 32, 64, 128, 256`。每個 input 都是加入唯一前綴的
完整文章，記錄整個 HTTP request 的 latency、回傳 embedding 數量、response size、items/s
以及 waiting queue 峰值。

## 實驗數據

### Concurrency scaling

| Concurrency | Requests | Success | Throughput (req/s) | Input tokens/s | p50 (ms) | p95 (ms) | Max (ms) | Peak waiting | Peak GPU util. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 12.10 | 11,964.9 | 81.69 | 84.81 | 84.81 | 0 | 90% |
| 2 | 8 | 8 | 13.24 | 13,095.7 | 149.10 | 155.45 | 155.45 | 0 | 98% |
| 4 | 8 | 8 | 13.78 | 13,625.7 | 253.64 | 367.27 | 367.27 | 0 | 99% |
| 8 | 16 | 16 | 13.73 | 13,579.5 | 562.79 | 680.18 | 680.18 | 4 | 100% |
| 16 | 32 | 32 | 13.78 | 13,655.7 | 1,153.04 | 1,191.67 | 1,270.36 | 12 | 100% |
| 32 | 64 | 64 | 13.80 | 13,672.0 | 2,238.13 | 2,385.91 | 2,388.23 | 27 | 100% |
| 48 | 96 | 96 | 13.75 | 13,624.6 | 3,437.48 | 3,593.48 | 3,607.58 | 43 | 100% |
| 64 | 128 | 128 | 13.68 | 13,558.0 | 4,604.55 | 4,666.01 | 4,750.77 | 59 | 100% |

所有 concurrency levels 的 HTTP 與 vLLM processing error 均為 0。

### 單一 request 的 `input[]` batch

| Batch size | Returned | Prompt tokens | Latency (ms) | Throughput (items/s) | Response size (MiB) | Peak waiting | Result |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 1 | 989 | 85.22 | 11.73 | 0.08 | 0 | Success |
| 8 | 8 | 7,912 | 641.35 | 12.47 | 0.67 | 4 | Success |
| 16 | 16 | 15,846 | 1,264.87 | 12.65 | 1.34 | 11 | Success |
| 32 | 32 | 31,702 | 2,482.04 | 12.89 | 2.69 | 28 | Success |
| 64 | 64 | 63,414 | 4,957.86 | 12.91 | 5.37 | 59 | Success |
| 128 | 128 | 126,994 | 9,970.65 | 12.84 | 10.75 | 123 | Success |
| 256 | 256 | 254,098 | 20,128.63 | 12.72 | 21.49 | 251 | Success |

## 結果分析

### 1. Throughput 在低 concurrency 即飽和

Concurrency 從 1 增加至 4 時，吞吐量由 12.10 增加至 13.78 requests/s；之後即使提高到
64，吞吐量仍維持在約 13.7 requests/s。Concurrency 4 時 GPU utilization 已達 99%，顯示
此 workload 已接近單 GPU 的運算吞吐上限。

### 2. 高 concurrency 主要增加 queue latency

Concurrency 8 開始出現明顯 waiting queue。從 concurrency 8 增加到 64，p95 latency 由
680 ms 增加至 4.67 s，但 throughput 沒有相應提升。因此，在類似的約 1,000-token
線上 embedding workload 中，concurrency 4--8 是較合理的起始 operating range。

### 3. API batch size 不等於 GPU batch size

單一 request 放入 256 個 inputs 可以成功，但 scheduler 會依 `MAX_NUM_SEQS` 與
`MAX_NUM_BATCHED_TOKENS` 分波執行。Batch 256 時觀察到 waiting queue 峰值 251，且吞吐量仍約
12.7 items/s。大 `input[]` 的主要效果是把等待時間與 response payload 集中在單一 HTTP
request，而不是讓 256 筆同時進入 GPU。

### 4. 大 batch 的 response payload 不可忽略

4,096-dimensional FP16 model output 經 JSON 數字序列化後，batch 256 的 response 約
21.49 MiB。即使 server 能接受，gateway body-size、client memory、timeout、retry amplification
與 head-of-line blocking 都可能先成為限制。

## 建議配置

針對相近長度的線上文章 embedding workload，可先採用以下界線，再依 SLO 重測：

| 項目 | 建議起始值 |
|---|---:|
| Client concurrency | 4--8 |
| 一般線上 `input[]` batch | 8--16 |
| 離線 `input[]` batch | 32--64 |
| 已實證可完成的 `input[]` batch | 256 |

256 是本次 bounded experiment 驗證成功的大小，不是 API 或 vLLM 的理論最大值。若要調高
`MAX_NUM_BATCHED_TOKENS`，應在 IaC 中明確指定候選值，逐一比較 throughput、p95、GPU memory
與錯誤率，不應依賴版本相關的預設值。

## 限制

- 每個 concurrency level 僅執行一個固定大小的測試區段，未進行多輪隨機化重複實驗。
- 測試使用單一文章結構，不能直接代表短句、多語言或不同 token-length distributions。
- 測試經 loopback 執行，不包含外部網路、TLS、API gateway 或 load balancer latency。
- 結果包含 JSON serialization 與 client parsing，適合 API capacity planning，但不是純 GPU kernel benchmark。
- 只測試至 batch 256；未尋找會造成 OOM 或服務不穩定的破壞性上限。

## 結論

在本實驗環境中，約 990-token 的 Qwen3-Embedding-8B 請求於 concurrency 4 左右已充分使用
單 GPU，穩定吞吐約為 13.7 requests/s。vLLM 能接受至少 64 個同時 HTTP requests，也能完成
包含 256 個 inputs 的單一 request，但超過有效運算 concurrency 後，新增工作主要進入 queue。

容量規劃時應分別管理 HTTP concurrency、API `input[]` batch size、scheduler sequence 上限與
token budget；它們是四個不同層次的限制，不能把「request 接受成功」解讀為「同時完成推論」。

## 參考資料

- [vLLM 0.26.0 configuration API](https://docs.vllm.ai/en/v0.26.0/api/vllm/config/vllm/)
- [vLLM optimization and tuning](https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md)
- [Qwen3-Embedding-8B model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
