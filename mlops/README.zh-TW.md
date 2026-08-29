# MLOps 實務筆記

> [English](./README.md) | **繁體中文**

這個目錄收錄模型部署、資源管理與效能工程的實務筆記。每篇文章描述可重現的
診斷方法與決策邊界，不把單一模型或單一叢集的參數當成通用預設值。

| 主題 | 重點 |
| --- | --- |
| [容器內 ML workload 的資源感知](./container-resource-awareness/README.zh-TW.md)（[English](./container-resource-awareness/)） | Host-visible 資源與 Pod cgroup 實際預算的落差、thread oversubscription、GPU sharing、NUMA、`/dev/shm` 與儲存容量 |
| [逐層調教 TTS 推論](./tts-inference-performance-tuning/README.zh-TW.md)（[English](./tts-inference-performance-tuning/)） | BERT device transfer／batching、內部分段 shape、TTS batch 與外層 checkpoint 邊界的 production-shaped A/B |
| [語言模型部署](./language-model-deployment/) | 依 artifact、硬體與 workload 選擇 vLLM、llama.cpp 或 TEI |
| [vLLM 效能工程](./vllm-performance-engineering/) | 併發、KV cache 與 serving benchmark 的量測方法 |
