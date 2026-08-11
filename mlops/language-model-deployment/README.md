# 語言模型部署筆記：vLLM、llama.cpp 與 TEI 的取捨

> 更新日期：2026-08-11

把模型「跑起來」通常不難；真正困難的是選到與模型格式、硬體容量、API
負載及維運方式相符的 runtime。框架名氣不能取代相容性驗證：同一個 30B
模型，可能只有 GGUF 量化檔，也可能只有 AWQ/GPTQ checkpoint；這會直接
決定部署路線。

這篇筆記整理自一次 32 GB 消費級 NVIDIA GPU 的部署實作。目標模型是
[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)，
服務需提供 OpenAI-compatible API，並以可攜、可重建、可回滾的 IaC 管理。

## 先看 artifact，再選 runtime

部署前至少要確認以下四件事：

1. 模型架構是否被 runtime 原生支援，或至少有經過驗證的相容 backend。
2. 量化格式是 AWQ、GPTQ、GGUF、BitsAndBytes，還是廠商自有格式。
3. 權重、KV cache、vision encoder 與 speculative drafter 的總記憶體需求。
4. API 的預期負載是單人互動、固定少量 slots，還是大量非同步請求。

Muse Glimmer 的 BF16 權重約 60 GB，無法直接放進 32 GB VRAM。官方 4-bit
版本則發布在
[`Muse-Glimmer-30B-GGUF`](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)：

| Artifact | 約略大小 | 用途 |
|---|---:|---|
| K-Quant-Dynamic | 19.7 GB | 32 GB GPU 的主要語言模型 |
| K-Quant-17GB | 16.8 GB | 24 GB GPU 或需要更多記憶體餘裕 |
| Perception encoder | 1.4 GB | 圖片輸入 |
| DFlash drafter | 1.6 GB | Speculative decoding |

因此，這不是單純在 vLLM 啟動參數加上 `--quantization` 就能解決的問題；
官方 4-bit artifact 本身就是為 llama.cpp 準備的 GGUF。

## 三種常見 serving runtime

### vLLM

vLLM 適合已受支援的生成模型，以及對高併發、吞吐量與 GPU 記憶體調度有
要求的服務。核心優勢包括 PagedAttention、continuous batching、prefix
caching，以及成熟的 OpenAI-compatible server。請求超過當下可執行容量時，
scheduler 會保留在等待佇列，並在 token budget 與 KV block 可用時持續併入
後續 engine step；不是每次湊滿一個固定 batch 才開始。

PagedAttention 管理 KV cache 的 block 佈局與生命週期，FlashAttention 則最佳化
attention kernel 的資料搬運；兩者可以在 serving engine 中協作，但不是同一層的
替代方案。詳細區分見
[FlashAttention 與 PagedAttention：算子最佳化和 KV cache 管理](../../domains/utils/transformer-family/flash_attention/flash-attn-vs-paged-attn.md)。

但「vLLM 支援 GGUF」不代表「vLLM 支援所有 GGUF 模型」。GGUF loader 仍需
理解模型架構；而且官方文件將 GGUF 支援標示為 experimental、under-optimized。
截至 2026-08-11，vLLM v0.27.0 與當日 main 都沒有 Muse Glimmer 的原生
registry entry，官方 supported-model list 也尚未列入此架構。

### llama.cpp

llama.cpp 適合 GGUF、本地或邊緣部署、單 GPU，以及需要可攜二進位與較小
維運面積的情境。`llama-server` 已支援 OpenAI-compatible chat API、平行
decoding、continuous batching、multimodal input、function calling、metrics
與 speculative decoding。

它同樣有 KV cache 與 Flash Attention：

- 每個 processing slot 保存 KV cache，預設會依 prompt 相似度嘗試重用前綴。
- KV cache 可使用 F16，也可量化為 Q8/Q4；選擇量化 cache 時要確認相應的
  CUDA Flash Attention kernel，否則可能退回較慢路徑。
- `--flash-attn on` 可以明確啟用 CUDA Flash Attention。
- `--parallel N` 決定同時工作的 slots；超出的 request 會 deferred。
- `--ctx-size` 是 server 總 context，會分配給 slots。例如
  `--ctx-size 131072 --parallel 4` 代表每個 slot 約 32K tokens，而不是每個
  request 都有 128K。

llama.cpp 支援 continuous batching，但其 unified KV cache 與 slots 模型不像
vLLM 的 paged KV block scheduler 那麼適合高併發、多種長度混合的工作負載。
它的優勢是簡潔與模型相容性，不是全面取代 vLLM 的 production scheduler。

### Text Embeddings Inference（TEI）

TEI 是專門服務 embedding model 的 runtime。它提供 tokenization、動態
batching、Prometheus metrics 與常見 embedding API；若機器只服務 embedding，
通常比通用生成框架更聚焦。相對地，它不是用來提供一般 chat completion 的
小型 LLM runtime。若同一台主機同時規劃 embedding 與生成模型，應先決定是否
接受兩套 runtime，不能只為了「統一」而忽略各模型的官方量化格式與支援度。

## 通用情境：先按 workload 選框架

先不考慮任何特定模型，llama.cpp 與 vLLM 解決的是部分重疊、但重心不同的
問題。前者優先處理可攜性、廣泛硬體與 GGUF 生態；後者優先處理 GPU serving
的吞吐量、動態排程與叢集擴展。

| 決策面向 | llama.cpp | vLLM |
|---|---|---|
| 主要目標 | 輕量、本地、邊緣與單機推論 | 高吞吐、多租戶 GPU API serving |
| 常見模型格式 | GGUF 與 K/I-quants | Hugging Face Safetensors、AWQ、GPTQ、FP8 等 |
| 硬體範圍 | CPU、Apple Metal、CUDA、ROCm、Vulkan 等 | 以資料中心 GPU/accelerator 為主 |
| Request scheduling | Slots + continuous batching | Token-level continuous batching + scheduler |
| KV cache 管理 | Unified KV cache；容量與 slots/context 配置關係直接 | PagedAttention；以 block 動態配置，較能降低碎片 |
| 高併發與混合長度 | 可服務多使用者，但固定 slots 較容易成為容量邊界 | 通常更擅長大量、長短不一的併發 request |
| Prefix cache | 依 slot/prompt 相似度重用，也可保存與還原 slot | Block-based prefix caching，較適合共享前綴 workload |
| 分散式推論 | 支援多 GPU，但不是主要強項 | Tensor/Data/Pipeline/Expert parallel 生態較完整 |
| OpenAI API | 常用 chat/completions 可用；仍有實作差異 | API 相容性與 production integration 通常較完整 |
| 維運成本 | Binary/container 輕量，容易攜帶與離線部署 | Python/CUDA stack 較重，但 observability 與擴展工具成熟 |
| 典型選擇 | 個人助理、edge/offline、單機 GGUF、CPU/GPU 混合 offload | 團隊共用 API、高 QPS、批次推論、多 GPU 服務 |

這張表描述的是設計傾向，不是絕對效能排名。低併發時，模型 kernel 與量化格式
可能比 scheduler 更影響速度；高併發時，TTFT、inter-token latency、輸入長度
分佈與 KV cache 壓力又可能比單筆 tokens/s 更重要。最終仍要用實際 workload
做 concurrency sweep。

## 特定案例：Muse Glimmer 的 llama.cpp 與 vLLM 決策表

將通用決策套用到 Muse Glimmer 後，artifact 與 runtime support 形成更強的
限制。下表只描述 2026-08-11 當下的 Muse Glimmer 部署情境，不應直接外推到
其他模型：

| 面向 | llama.cpp | vLLM |
|---|---|---|
| Muse Glimmer 支援 | 官方、可直接跑 | 尚未正式支援 |
| 官方 4-bit | K-Quant GGUF | 沒有 AWQ/GPTQ checkpoint |
| 單機部署 | 輕量、可攜性高 | Python/CUDA stack 較重 |
| 高併發排程 | Continuous batching，但以 slots 為主 | PagedAttention、動態排程通常更強 |
| KV 記憶體利用 | Unified/fixed slot 設計 | Paged KV cache，碎片與高併發管理更佳 |
| OpenAI API | 支援，但相容性不如 vLLM 完整 | 相容性、metrics、production tooling 較成熟 |
| 這個模型的成熟度 | 官方針對 RTX 5090 驗證 | 尚未驗證 |

這張 model-specific 表不是一般性的「框架排名」。對 Llama、Qwen 等 vLLM 原生支援且具有
AWQ/GPTQ checkpoint 的模型，結論可能相反；對 Muse Glimmer，官方 artifact
與已驗證實作使 llama.cpp 成為較低風險的選擇。

## IaC 應該鎖定什麼

一鍵啟動不能只是一份 `docker-compose.yml`。可重建部署至少應保存：

- 模型 repo、revision、精確檔名與 SHA-256。
- runtime source tag/commit，或 container manifest digest。
- GPU architecture、build flags、context、parallel slots、batch 與 KV cache 型別。
- loopback bind、API authentication、TLS gateway 與來源 allowlist。
- 健康檢查、未授權 401、模型 discovery、實際 generation smoke test。
- 切換前驗證與 rollback；新模型未通過 smoke test 時自動恢復舊服務。
- 權重、Docker/containerd、compile cache 都放在資料碟，不占用系統碟。

浮動 container tag 是常見陷阱。本次實作時，官方 `server-cuda` image 的實際
build 是 10335，但 Muse Glimmer 至少要求 llama.cpp b10353。若只看到 image
成功 pull 就停下來，模型會在切換後才拒絕載入。因此部署改為鎖定 b10355
source commit，自建 CUDA image，並在停掉舊服務前驗證 runtime build number。

API key 也不應出現在 CLI arguments。可優先使用 runtime 提供的 environment
variable 或唯讀 secret file，避免 command line、process list 或啟動日誌外洩。
服務本身只監聽 loopback，再由 TLS gateway 暴露必要的 `/v1/` 路由。

## 建議的決策流程

1. 先從官方 model card 找到受支援的 artifact 與推薦 runtime。
2. 在不占用 GPU 的情況下完成下載、checksum、image build 與設定驗證。
3. 以預期的 context、slots 與 KV precision 估算記憶體，不以模型檔大小代替
   runtime VRAM。
4. 停止舊模型後啟動候選服務；檢查 authentication、API response、GPU allocation
   與 container health。
5. 執行 restart recovery，才算完成可維運部署。
6. 最後再做 concurrency sweep；用吞吐量、TTFT、inter-token latency、p95/p99
   與錯誤率選擇 slots，而不是只看單筆 tokens/s。

## 結論

如果模型已被 vLLM 原生支援，且目標是多使用者高併發 API，vLLM 通常是較好的
起點；如果官方只發布 GGUF、部署在單張消費級 GPU，或模型首先在 llama.cpp
落地，llama.cpp 通常更可靠。純 embedding 節點則值得優先評估 TEI。

框架統一可以降低維運成本，但相容性、可回滾與可驗證性應優先於形式上的統一。
在 Muse Glimmer 的案例中，採用 llama.cpp 不是因為它普遍優於 vLLM，而是因為
它是目前唯一有官方 4-bit artifact 與硬體驗證的路線。

## 參考資料

- [Muse Glimmer 30B model card](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [Muse Glimmer 30B GGUF model card](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)
- [llama.cpp HTTP server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [vLLM GGUF documentation](https://docs.vllm.ai/en/latest/features/quantization/gguf/)
- [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
