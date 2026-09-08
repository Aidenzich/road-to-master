# Distributed AI/ML computation and communication bottlenecks

> [繁體中文：完整筆記](README.zh-TW.md) | [MLOps index](../README.md)

A concept-first technical spike on distributed training and inference using one RTX 5090 and three L40S GPUs across Ethernet-connected hosts, with K3s, Volcano, and HAMi.

The full note is in **Traditional Chinese**. This page is an English overview, not a full translation. The note covers:

1. NVLink, PCIe, Ethernet, and single-GPU memory limits.
2. Collective communication and its latency/bandwidth cost model.
3. DDP, ZeRO/FSDP, tensor parallelism, and pipeline parallelism.
4. Network constraints, heterogeneous GPUs, and stragglers.
5. MoE routing and AlltoAll.
6. Platform-specific trade-offs and decision boundaries.
7. Future experiments, followed by an evidence appendix.

**Scope:** technical research and experimental design only, dated 2026-09-08. No GPU benchmarks, model deployments, or cluster changes were performed. Numerical examples and hypotheses must not be read as measured results.

The complete note includes the NVLink clarification: the official RTX 5090 and L40S specifications both list no NVLink support. Moving these cards into one host or adding a bridge cannot supply that missing capability; lack of NVLink alone does not prove a parallelism strategy will lose.

Public-export changes are limited to navigation and removal of local absolute paths and private node identifiers. Source-relative infrastructure references describe a historical, locally inspected repository snapshot; that private repository is not bundled here, so those references are not independently verifiable from this public checkout. Public technical sources remain in the [evidence appendix](chapters/08-evidence.md).
