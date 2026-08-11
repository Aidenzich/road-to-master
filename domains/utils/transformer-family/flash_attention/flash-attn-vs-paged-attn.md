# FlashAttention 與 PagedAttention：算子最佳化和 KV cache 管理

FlashAttention 與 PagedAttention 經常一起出現在 LLM serving 系統，卻不是互相
替代的技術。FlashAttention 主要減少一次 attention 計算在 GPU 記憶體階層間的
資料搬運；PagedAttention 則讓長度動態變化的 KV cache 能以固定大小 blocks
配置、共享與回收。

## 核心差異

| 面向 | FlashAttention | PagedAttention |
|---|---|---|
| 主要問題 | Attention kernel 的 HBM I/O 與中間張量 | Serving 時 KV cache 的動態配置、碎片與共享 |
| 抽象層級 | GPU attention algorithm／kernel | KV block manager + 能讀取 paged KV 的 attention algorithm |
| 主要資料 | 當次計算的 Q、K、V、softmax statistics 與 output | 跨 decoding steps 保留的 K/V tensors |
| 核心方法 | Tiling、online softmax、kernel fusion，避免物化完整 attention matrix | Logical blocks、physical blocks 與 block table；按需配置 physical blocks |
| 直接收益 | 降低記憶體流量與 attention 暫存空間，改善 kernel latency | 降低 KV cache 浪費、提高可容納 sequences 數，支援 prefix／beam sharing |
| 仍無法消除 | Attention 的數學計算量；Q/K/V 與 output 仍需讀寫 HBM | 最後一個 block 的內部碎片、block metadata 與 scheduler 成本 |
| 常見受益階段 | 長序列 prefill／training；decode 也需相容的最佳化 kernel | 自迴歸 serving 的 prefill 與 decode，decode 階段持續成長最明顯 |

## FlashAttention：避免物化二次方中間矩陣

Naive attention 會把 $S = QK^T$ 與 softmax probabilities 寫入 HBM。對序列長度
$N$，這些中間矩陣具有 $O(N^2)$ 空間需求。FlashAttention 將 Q、K、V 分塊，
在 on-chip memory 中以 online softmax 累積結果，不把完整 $N \times N$ 矩陣
寫回 HBM。

它仍然計算精確 attention，計算複雜度仍是 $O(N^2d)$；改善的是 I/O 複雜度與
額外儲存，不是把 dense attention 變成線性時間。Q、K、V 仍要從 HBM 載入，
output 也要寫回 HBM，因此「完全不碰 HBM」並不正確。

## PagedAttention：按 block 管理 KV cache

自迴歸推論需要保存每一層過去 tokens 的 K/V。每個 request 的實際長度不同，
而且在 decode 過程中逐 token 成長；若為最大長度預留連續區域，容易造成過度
預留與碎片。

PagedAttention 把 request 的 KV cache 表示為 logical blocks，並透過 block table
映射到不必連續的 physical blocks：

1. Prefill 根據已處理的 prompt tokens 配置所需 blocks。
2. Decode 填入最後一個 block；容量不足時再取得新的 physical block。
3. Request 完成後，physical blocks 回到 free pool。
4. 多個 sequences 共享相同 prompt 時，可透過 reference counting 與
   copy-on-write 共享既有 blocks。

這個設計把單一 sequence 的 block-level 內部浪費限制在最後一個未填滿 block，
但不代表顯存可以達到 100% 利用率。模型權重、activations、CUDA graphs、allocator
與 kernel workspaces 都需要 VRAM；block size 也在 kernel 效率、metadata 與內部
碎片之間形成 trade-off。原始 vLLM 論文報告的低浪費與吞吐提升是其測試設定下
的實驗結果，不是所有模型與 workload 的固定保證。

## 兩者如何一起工作

Serving engine 可以同時使用 paged KV cache 與 FlashAttention-family kernel：

```text
request scheduler
      │
      ├─ KV block manager ── logical blocks → physical KV blocks
      │
      └─ attention backend ── block table + Q/K/V → attention output
```

「Paged」描述 KV 的佈局與生命週期；「Flash」描述 attention 計算如何降低 I/O。
實際 backend 必須同時支援模型 dtype、head size、KV dtype、block size、GPU compute
capability 與所需 attention pattern。框架能管理 paged KV，不代表任意
FlashAttention build 都能讀取該佈局。

以 vLLM 為例，backend 會依硬體與模型配置自動驗證和選擇；目前官方 feature
matrix 同時列出 FlashAttention、FlashInfer、Triton Attention 等選項，而且不同
GPU 世代的優先序不同。因此不要依賴手寫的假想 log 字串判定功能已啟用，應：

1. 固定 vLLM image/version 與啟動參數。
2. 讀取實際啟動日誌中的 selected attention backend。
3. 對照該版本的 backend feature matrix。
4. 用目標 prompt/context/concurrency 實測 TTFT、ITL、throughput、VRAM 與錯誤率。

## 選擇問題時的快速判斷

- 單筆長 prompt 的 attention 很慢或暫存空間過高：先檢查 attention backend、
  dtype、head dimension 與 FlashAttention 相容性。
- 高併發或長 context 下 KV cache 接近滿載、發生 preemption：先檢查 cache
  bytes/token、block 容量、scheduler budget 與 PagedAttention/KV manager metrics。
- 兩種現象同時存在：分別量測 kernel latency 與 request-level queue/cache 指標，
  不要把所有改善都歸因於其中一項技術。

## 延伸閱讀

- [FlashAttention 深度解析](./README.md)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180)
- [vLLM attention backend feature matrix](https://docs.vllm.ai/en/latest/design/attention_backends/)
