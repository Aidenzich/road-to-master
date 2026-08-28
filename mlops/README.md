# MLOps Field Notes

> **English** | [繁體中文](./README.zh-TW.md)

This directory collects field notes on model deployment, resource management, and performance engineering. Each note documents reproducible diagnostic methods and decision boundaries instead of treating settings from one model or cluster as universal defaults.

| Topic | Focus |
| --- | --- |
| [Resource-aware ML workloads in containers](./container-resource-awareness/) ([繁體中文](./container-resource-awareness/README.zh-TW.md)) | Gaps between host-visible resources and the actual Pod cgroup budget, thread oversubscription, GPU sharing, NUMA, `/dev/shm`, and storage capacity |
| [Language model deployment](./language-model-deployment/) | Choosing vLLM, llama.cpp, or TEI based on artifacts, hardware, and workload |
| [vLLM performance engineering](./vllm-performance-engineering/) | Measuring concurrency, KV cache behavior, and serving benchmarks |
