# 容器內 ML workload 的資源感知：別把 host-visible 當成 Pod budget

> [English](./README.md) | **繁體中文**

> 更新日期：2026-08-28

在 Kubernetes 裡，模型明明只被配置 8 CPU，PyTorch、OpenMP、MKL、ONNX Runtime
或資料前處理程序卻可能建立遠多於 8 個執行緒。節點監控甚至可能顯示大量 CPU
idle，但 Pod 內的推論仍斷斷續續、延遲暴增:
- **資源偵測（discovery）**可能看到 host CPU、physical cores、CPU affinity 或套件自己的
  預設值，並據此建立 thread pool。
- **資源執法（enforcement）**仍由 Linux cgroup 執行 Kubernetes 的 CPU、memory 等限制。
- discovery 與 enforcement 對同一 workload 的預算理解不同，就會發生 oversubscription。

Kubernetes [明確說明 CPU limit 是由 kernel throttling 強制執行](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#requests-and-limits)；它不是 thread pool
大小，也不會自動替每個 ML runtime 選出合理的平行度。因此，`resources.limits.cpu: 8`
不能取代 `OMP_NUM_THREADS=8`、`torch.set_num_threads(8)` 或相應 runtime 設定。

## 為什麼全機 CPU idle，Pod 還是會被 throttle

假設節點有 128 cores，Pod 的 CPU limit 是 8。以常見的 100 ms CFS period 為例，
該 Pod 在每個 period 共有約 800 ms 的 aggregate CPU time：

- 8 個持續工作的 threads 可以把預算均勻用完整個 100 ms。
- 64 個 threads 理論上約 12.5 ms 就能一起用完 800 ms，之後整個 cgroup 等待下一個 period。
- 其餘約 120 個 host cores 即使保持 idle，也不能借給已碰到 hard limit 的 Pod。

對 BERT 這種包含許多小型 operator、barrier 與逐句呼叫的路徑，quota burst 還會疊加：

- OpenMP、MKL、OpenBLAS、PyTorch intra-op/inter-op 各自建立或喚醒 worker。
- 多個 Python worker、DataLoader 或 serving process 再乘上一層 thread pool。
- context switch、cache thrashing 與 thread-pool spinning 消耗預算。
- barrier 只要等到一個被延後的 worker，整個 operator 就無法完成。

所以限制 thread 數後變快，不代表「thread 越少越好」；它代表**有效平行度終於和
cgroup 可用預算對齊**。

一個實際案例中，約 128-core 節點上的 TTS Pod 限制為 8 CPU，程序總 thread 數約
273。cgroup 觀察到 82 個 throttled periods，以及約 417 秒的累積 throttled time；舊路徑
處理 85 句 BERT 曾耗時約 14 分 54 秒。將 PyTorch/BLAS 的數值計算 thread pools 對齊
8 CPU 後，另一筆真實 workload 的 73 句前處理約 7 秒，單次 BERT forward 多數約
25–40 ms。

這不是嚴格控制所有變因的 benchmark，不能據此宣稱每個模型都應固定設成 8；但它證明
了兩件事：全機 idle 不能排除 Pod throttling，而且 thread budget 必須是 workload contract
的一部分。歷史慢請求發生時尚無 phase-level watchdog，因此不能回溯宣稱某一個 BERT
operator 是唯一根因。

## 業界常見資源認知落差比較表

| 落差 | 套件／程序常看到什麼 | 實際限制或競爭點 | 常見症狀 | 優先處理方式 | 應觀測的證據 |
| --- | --- | --- | --- | --- | --- |
| CPU topology vs CPU quota | Host logical/physical CPU、affinity，或 runtime 預設 thread 數 | cgroup CPU quota/period | 節點 idle 但 Pod latency 呈週期性尖峰、`throttled` 持續增加 | 以 effective CPU budget 設定 intra-op、inter-op、OMP、MKL、OpenBLAS；先量測再調整 | [`cpu.stat` 與 CPU limit](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#how-kubernetes-applies-resource-requests-and-limits)、throttled ratio/time、run queue、phase latency、runtime thread 數 |
| Process 數 × thread pool | 每個 worker 都認為自己可使用完整 CPU | 所有 workers 共用同一 Pod/cgroup quota | worker 增加後吞吐不升反降，context switches 與 load 暴增 | 先決定 process concurrency，再把每個 process 的 thread budget 分配到總 CPU 預算內 | process/thread 數、context switches、CPU PSI、每 worker throughput |
| CPU request vs limit vs exclusive cores | Application 通常只知道可執行 CPU，不理解 Kubernetes QoS | request 影響排程與競爭權重；limit 是 hard ceiling；exclusive CPU 還需要 static CPU Manager、Guaranteed Pod 與整數 CPU request | 同規格 Pod 在不同節點 latency 差異大，與 sidecar/host process 互相干擾 | latency-sensitive workload 才評估 Guaranteed + static CPU Manager；一般 workload 先正確配置 request/limit | Pod QoS、cpuset、CPU Manager policy、steal/throttling、node noise |
| Host RAM vs cgroup memory | Runtime、cache 或 allocator 可能依 host RAM 規劃 | Pod memory limit、tmpfs `emptyDir` 也可能計入 container memory | Node 看似有 RAM，但 Pod `OOMKilled`；page cache、模型 cache 或 tmpfs 推高 working set | 以 cgroup memory budget 設 cache、batch、prefetch；為 memory-backed volume 設 `sizeLimit` | working set/RSS、cgroup memory events、OOM reason、tmpfs/cache bytes |
| `/dev/shm` vs DataLoader/IPC 需求 | multiprocessing 假設有足夠 POSIX shared memory | [Docker 預設 `/dev/shm` 只有 64 MiB](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)；Kubernetes volume 配置另行決定 | DataLoader bus error、NCCL/IPC 初始化失敗、大 batch 才偶發 | 明確配置 memory-backed `emptyDir` 或 runtime shm size，並把它納入 memory capacity | `df -h /dev/shm`、shared-memory 使用量、worker exit reason |
| GPU device count vs 可用算力／VRAM | `CUDA_VISIBLE_DEVICES` 顯示一張 GPU | [NVIDIA time-slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html#understanding-time-slicing-gpus) 只共享時間，沒有 MIG 等級的 memory/fault isolation；其他 sharing layer 也可能另有 VRAM/core contract | Pod 看得到 GPU 卻 OOM、latency 抖動，或兩個 workload 互相拖慢 | 明確選擇 exclusive、MIG、time-slicing 或 vendor vGPU；不要把「1 GPU」當成固定算力與 VRAM | per-process VRAM、SM utilization、allocator peak、Xid/OOM、sharing mode |
| CPU/NUMA/GPU locality | Runtime 看到所有 CPUs 與一張可見 GPU | CPU、RAM 與 PCIe GPU 可能跨 NUMA node | Host utilization 正常但 H2D、tokenization 或 input pipeline latency 高且不穩 | 需要時使用 Topology Manager、CPU Manager/cpuset，並驗證 memory locality；不要只做 CPU pinning | NUMA miss、PCIe throughput、CPU affinity、GPU topology、phase latency |
| Container writable layer vs 模型／checkpoint 大小 | Application 把 `/tmp`、HF cache、Torch cache 視為一般磁碟 | ephemeral-storage limit、node disk pressure、image pull/unpack 與 log 也競爭本機磁碟 | `Evicted`、`ENOSPC`、啟動或 checkpoint 變慢，重啟後 cache 消失 | 模型與大型 cache 放明確 volume；配置 ephemeral request/limit、清理與水位告警 | volume/rootfs bytes、inode、ephemeral-storage events、pull/unpack latency |
| Runtime spinning vs 真實工作量 | Runtime 為降低單次延遲讓 idle worker spin | 多個模型／session 共用 CPU 時，spinning 會消耗 quota 與電力 | 沒有請求仍有固定 CPU，其他服務 p95 上升 | 依 latency/throughput 目標關閉或縮短 spinning；不要僅看單 request benchmark | idle CPU、power、session/thread 數、p50/p99、throttled time |

表中的控制方式不是可以直接複製的固定模板。例如 Kubernetes static CPU Manager 只有在
節點正確設定，且 container 位於 Guaranteed Pod、具有整數 CPU request 時，才會分配
exclusive CPUs（見 [Kubernetes CPU Manager](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/#static-policy-configuration)）。GPU time-slicing 也不等同 VRAM 隔離；NVIDIA 文件明確指出它沒有
memory 或 fault isolation。

## 常見 ML runtime 的 CPU 控制面

| Runtime／library | 主要控制方式 | 常見陷阱 |
| --- | --- | --- |
| PyTorch | `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`torch.set_num_threads()`、`torch.set_num_interop_threads()` | 必須在 eager/JIT/autograd 工作開始前設定；總 process threads 不會等於數值計算 threads |
| NumPy + BLAS | `OMP_NUM_THREADS`、OpenBLAS/MKL 對應變數，或 `threadpoolctl` | NumPy 本身常是單 thread，但背後 BLAS 可以另外開多 thread |
| TensorFlow | `tf.config.threading.set_intra_op_parallelism_threads()`、`set_inter_op_parallelism_threads()` | `tf.data`、serving workers 與 operator thread pools 可能形成多層平行 |
| ONNX Runtime | `intra_op_num_threads`、`inter_op_num_threads`、execution mode、thread spinning 設定 | [預設 intra-op pool 可依 physical cores 建立](https://onnxruntime.ai/docs/performance/tune-performance/threading.html#intra-thread-count)；多個 session 會重複建立 pool |
| DataLoader／multiprocessing | process/worker 數、prefetch、每個 child 的 BLAS/PyTorch threads | `workers × threads` 才是總 CPU 壓力；fork/spawn 行為與共享記憶體也要納入 |
| Serving engine | request concurrency、batch/token budget、CPU preprocess workers | GPU batch 調大不代表 CPU tokenizer、音訊編碼、publisher 有足夠容量 |

設定環境變數要早於 NumPy、PyTorch 或會載入 BLAS/OpenMP 的套件 import。只在模型載入後
修改，可能已經建立 thread pool，結果依 runtime 而異。

```python
import os

# 這個數字應由 deployment 的 effective CPU budget 注入，不應硬編碼成所有服務共用值。
threads = int(os.environ["ML_CPU_THREADS"])
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = str(threads)

import torch

torch.set_num_threads(threads)
torch.set_num_interop_threads(min(2, threads))
```

Kubernetes 可以用 [`resourceFieldRef`](https://kubernetes.io/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/#the-downward-api) 把 container CPU limit 以 millicores 注入 entrypoint，
再由 entrypoint 轉成有上下界的 thread budget：

```yaml
resources:
  requests:
    cpu: "8"
    memory: 32Gi
  limits:
    cpu: "8"
    memory: 32Gi
env:
  - name: CONTAINER_CPU_LIMIT_MILLICORES
    valueFrom:
      resourceFieldRef:
        containerName: inference
        resource: limits.cpu
        divisor: 1m
```

應用仍要驗證值存在、可解析並落在 release 支援的範圍；不要讓缺少 limit 變成 0 threads，
也不要看到 64 host CPUs 就自動採用 64。若 Pod 有 sidecar，每個 container 的 budget 與整個
Pod 的 QoS/資源總和都必須分別核對。

## 診斷順序：從「哪一層在等」開始

1. **拆 phase timer。** 將 normalization、tokenizer、H2D、model forward、postprocess、encode、
   publish 分開量測；只有 end-to-end latency 無法區分 CPU、GPU、I/O 或 queue。
2. **同時看 node 與 cgroup。** Node idle 不會否定 container throttling；Node busy 也不代表
   該 request 一定在 CPU 上。
3. **讀 effective CPU。** 比較 affinity/cpuset 與 quota/period。可用的平行度上界概念上是：

   ```text
   effective_cpu = min(cpuset_cpu_count, cpu_quota / cpu_period)
   ```

   quota 若為 unlimited，才只看 cpuset/affinity。fractional CPU 需要以 benchmark 決定 thread
   數，不能直接把小數無條件進位。
4. **列出所有 thread pools。** 除了 framework，還要包含 BLAS、tokenizer、DataLoader、web
   server、Celery/Ray workers、ffmpeg 與 sidecar。
5. **做相同輸入 A/B。** 固定 model、input、batch、GPU profile 與同機競爭 workload，只改
   thread budget；比較 throughput、p50/p95/p99、throttling 與輸出品質。
6. **保留 watchdog 與 stack。** phase 超時時輸出所有 thread stack，再由 orchestrator 重啟；
   restart 是止血，不是根因證明。

對 cgroup v2，`cpu.stat` 的 `nr_periods`、`nr_throttled`、`throttled_usec` 與 `cpu.max` 是重要
證據；cgroup v1 使用對應的 `cpu.stat`、`cpu.cfs_quota_us` 與 `cpu.cfs_period_us`。監控系統可能
以不同 metric 名稱暴露相同資料，runbook 應保存實際查詢，而不是只寫 dashboard 截圖。

## Production checklist

- [ ] Deployment 明確宣告 CPU/memory requests 與 limits。
- [ ] Application thread budget 來自 effective CPU，而不是 host CPU count。
- [ ] `process concurrency × threads per process` 不超出經 benchmark 證明的容量。
- [ ] Phase metrics、cgroup throttling、OOM、GPU allocated/reserved 與 queue age 可觀測。
- [ ] `/dev/shm`、tmpfs、ephemeral storage、model cache 與 checkpoint 有容量上限。
- [ ] GPU sharing 模式、VRAM/fault isolation 與 workload request 語意一致。
- [ ] Latency-sensitive Pod 若要求 exclusive CPU，已驗證 QoS、整數 request、CPU Manager 與 NUMA。
- [ ] 相同輸入做過冷啟動、穩態、並發與同機 noisy-neighbor A/B。
- [ ] watchdog 能指出停滯 phase；重啟後任務具 checkpoint、冪等與安全重試。

## 參考資料

- [Kubernetes：Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Kubernetes：Control CPU Management Policies on the Node](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/)
- [PyTorch：Threading Environment Variables](https://docs.pytorch.org/docs/stable/threading_environment_variables.html)
- [PyTorch：CPU threading and TorchScript inference](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
- [PyTorch：`torch.set_num_threads`](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html)
- [NumPy：Number of threads used for linear algebra](https://numpy.org/doc/stable/reference/global_state.html#number-of-threads-used-for-linear-algebra)
- [TensorFlow：Configure intra-op parallelism](https://www.tensorflow.org/api_docs/python/tf/config/threading/set_intra_op_parallelism_threads)
- [TensorFlow：Configure inter-op parallelism](https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads)
- [ONNX Runtime：Thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [Docker：Runtime constraints on resources](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)
- [NVIDIA GPU Operator：Time-Slicing GPUs in Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)
