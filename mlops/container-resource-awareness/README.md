# Resource-Aware ML Workloads in Containers: Host-Visible Is Not the Pod Budget

> **English** | [繁體中文](./README.zh-TW.md)

> Updated: 2026-08-28

In Kubernetes, a model may be assigned only 8 CPUs while PyTorch, OpenMP, MKL, ONNX Runtime, or data-preprocessing code creates far more than 8 threads. Node monitoring may even show substantial idle CPU while inference inside the Pod remains intermittent and latency spikes.
- **Resource discovery** may observe host CPUs, physical cores, CPU affinity, or a library's own defaults and size a thread pool from that information.
- **Resource enforcement** still applies Kubernetes CPU and memory limits through Linux cgroups.
- Oversubscription occurs when discovery and enforcement disagree about the budget available to the same workload.

Kubernetes [explicitly states that CPU limits are enforced through kernel throttling](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#requests-and-limits). A CPU limit is neither a thread-pool size nor a mechanism that automatically chooses sensible parallelism for every ML runtime. Therefore, `resources.limits.cpu: 8` does not replace `OMP_NUM_THREADS=8`, `torch.set_num_threads(8)`, or the corresponding runtime controls.

## Why a Pod Can Be Throttled While the Host CPU Is Idle

Suppose a node has 128 cores and a Pod has a CPU limit of 8. With a typical 100 ms CFS period, the Pod receives about 800 ms of aggregate CPU time per period:

- 8 continuously busy threads can spread that budget across the full 100 ms.
- 64 threads can theoretically consume all 800 ms together in about 12.5 ms, after which the entire cgroup waits for the next period.
- Even if the other 120 host cores remain idle, they cannot be borrowed by a Pod that has reached its hard limit.

For BERT paths containing many small operators, barriers, and per-sentence calls, quota bursts compound with other effects:

- OpenMP, MKL, OpenBLAS, and PyTorch intra-op/inter-op execution may each create or wake workers.
- Multiple Python workers, DataLoaders, or serving processes can multiply another layer of thread pools.
- Context switching, cache thrashing, and thread-pool spinning consume the budget.
- If a barrier waits for even one delayed worker, the entire operator cannot finish.

The speedup after limiting threads does not mean that fewer threads are always better. It means that **effective parallelism finally matches the budget available to the cgroup**.

In one real incident, a TTS Pod on a roughly 128-core node had an 8-CPU limit while the process had about 273 threads. The cgroup recorded 82 throttled periods and about 417 seconds of cumulative throttled time. On the old path, processing 85 BERT sentences once took about 14 minutes 54 seconds. After aligning the PyTorch and BLAS numerical thread pools with the 8-CPU budget, preprocessing another real workload of 73 sentences took about 7 seconds, and most individual BERT forwards took about 25–40 ms.

This was not a benchmark with every variable strictly controlled, so it does not prove that every model should be fixed at 8 threads. It does establish two points: host-level idle capacity does not rule out Pod throttling, and the thread budget must be part of the workload contract. The historical slow request had no phase-level watchdog, so the evidence cannot retrospectively prove that one BERT operator was the sole root cause.

## Common Resource-Perception Gaps in Production

| Gap | What the library or process may see | Actual constraint or contention point | Common symptom | First response | Evidence to observe |
| --- | --- | --- | --- | --- | --- |
| CPU topology vs CPU quota | Host logical/physical CPUs, affinity, or the runtime's default thread count | cgroup CPU quota/period | Node is idle, but Pod latency spikes periodically and `throttled` keeps increasing | Set intra-op, inter-op, OMP, MKL, and OpenBLAS from the effective CPU budget; measure before tuning | [`cpu.stat` and CPU limit](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#how-kubernetes-applies-resource-requests-and-limits), throttled ratio/time, run queue, phase latency, runtime thread count |
| Process count × thread pool | Every worker assumes it can use the full CPU | All workers share one Pod/cgroup quota | More workers reduce throughput while context switches and load surge | Choose process concurrency first, then divide the total CPU budget among per-process thread pools | Process/thread count, context switches, CPU PSI, per-worker throughput |
| CPU request vs limit vs exclusive cores | The application usually sees runnable CPUs but does not understand Kubernetes QoS | Request affects scheduling and contention weight; limit is a hard ceiling; exclusive CPUs also require static CPU Manager, a Guaranteed Pod, and an integer CPU request | Identical Pods have different latency across nodes or interfere with sidecars and host processes | Consider Guaranteed QoS plus static CPU Manager only for latency-sensitive workloads; otherwise first set correct requests and limits | Pod QoS, cpuset, CPU Manager policy, steal/throttling, node noise |
| Host RAM vs cgroup memory | A runtime, cache, or allocator may plan from host RAM | Pod memory limit; tmpfs `emptyDir` may also count toward container memory | Node appears to have RAM, but the Pod is `OOMKilled`; page cache, model cache, or tmpfs expands the working set | Size cache, batch, and prefetch from the cgroup memory budget; set `sizeLimit` on memory-backed volumes | Working set/RSS, cgroup memory events, OOM reason, tmpfs/cache bytes |
| `/dev/shm` vs DataLoader/IPC demand | Multiprocessing assumes sufficient POSIX shared memory | [Docker defaults `/dev/shm` to 64 MiB](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources); Kubernetes volume configuration is separate | DataLoader bus errors, NCCL/IPC initialization failures, or failures only at large batch sizes | Explicitly configure a memory-backed `emptyDir` or runtime shm size and include it in memory capacity planning | `df -h /dev/shm`, shared-memory usage, worker exit reason |
| GPU device count vs available compute/VRAM | `CUDA_VISIBLE_DEVICES` exposes one GPU | [NVIDIA time-slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html#understanding-time-slicing-gpus) shares time only and lacks MIG-level memory/fault isolation; other sharing layers may define separate VRAM/core contracts | A Pod sees a GPU but OOMs, has unstable latency, or slows another workload | Explicitly choose exclusive GPU, MIG, time-slicing, or vendor vGPU; do not treat “1 GPU” as fixed compute and VRAM | Per-process VRAM, SM utilization, allocator peak, Xid/OOM, sharing mode |
| CPU/NUMA/GPU locality | The runtime sees all CPUs and one visible GPU | CPU, RAM, and PCIe GPU may reside on different NUMA nodes | Host utilization looks normal, but H2D, tokenization, or input-pipeline latency is high and unstable | Use Topology Manager and CPU Manager/cpuset when needed, then verify memory locality; CPU pinning alone is insufficient | NUMA misses, PCIe throughput, CPU affinity, GPU topology, phase latency |
| Container writable layer vs model/checkpoint size | The application treats `/tmp`, HF cache, and Torch cache as ordinary disk | Ephemeral-storage limit, node disk pressure, image pulls/unpacking, and logs compete for local disk | `Evicted`, `ENOSPC`, slow startup/checkpointing, or cache loss after restart | Put models and large caches on explicit volumes; set ephemeral requests/limits, cleanup, and watermarks | Volume/rootfs bytes, inodes, ephemeral-storage events, pull/unpack latency |
| Runtime spinning vs real workload | The runtime keeps idle workers spinning to reduce single-request latency | With multiple models or sessions sharing CPU, spinning consumes quota and power | Fixed CPU usage without requests and increased p95 for other services | Disable or shorten spinning according to latency/throughput goals; do not rely on a single-request benchmark | Idle CPU, power, session/thread count, p50/p99, throttled time |

These controls are not fixed templates that can be copied unchanged. For example, Kubernetes static CPU Manager allocates exclusive CPUs only when the node is configured correctly and the container belongs to a Guaranteed Pod with an integer CPU request; see [Kubernetes CPU Manager](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/#static-policy-configuration). GPU time-slicing likewise does not provide VRAM isolation; NVIDIA explicitly documents the absence of memory and fault isolation.

## CPU Controls for Common ML Runtimes

| Runtime/library | Primary controls | Common trap |
| --- | --- | --- |
| PyTorch | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `torch.set_num_threads()`, `torch.set_num_interop_threads()` | Configure them before eager, JIT, or autograd work begins; total process threads do not equal numerical-compute threads |
| NumPy + BLAS | `OMP_NUM_THREADS`, the corresponding OpenBLAS/MKL variables, or `threadpoolctl` | NumPy itself is often single-threaded while the underlying BLAS opens additional threads |
| TensorFlow | `tf.config.threading.set_intra_op_parallelism_threads()`, `set_inter_op_parallelism_threads()` | `tf.data`, serving workers, and operator pools may create multiple layers of parallelism |
| ONNX Runtime | `intra_op_num_threads`, `inter_op_num_threads`, execution mode, thread-spinning settings | [The default intra-op pool may be sized from physical cores](https://onnxruntime.ai/docs/performance/tune-performance/threading.html#intra-thread-count); multiple sessions create repeated pools |
| DataLoader/multiprocessing | Process/worker count, prefetch, per-child BLAS/PyTorch threads | `workers × threads` is the total CPU pressure; fork/spawn behavior and shared memory also matter |
| Serving engine | Request concurrency, batch/token budget, CPU preprocessing workers | A larger GPU batch does not guarantee sufficient capacity in CPU tokenization, audio encoding, or publishing |

Set environment variables before importing NumPy, PyTorch, or any package that loads BLAS/OpenMP. Changing them only after the model loads may be too late because the runtime may already have created its thread pools.

```python
import os

# The deployment should inject this effective CPU budget; do not hard-code one value for every service.
threads = int(os.environ["ML_CPU_THREADS"])
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = str(threads)

import torch

torch.set_num_threads(threads)
torch.set_num_interop_threads(min(2, threads))
```

Kubernetes can inject the container CPU limit in millicores through [`resourceFieldRef`](https://kubernetes.io/docs/tasks/inject-data-application/downward-api-volume-expose-pod-information/#the-downward-api). The entrypoint can then convert it into a bounded thread budget:

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

The application must still validate that the value exists, parses correctly, and falls within a release-supported range. A missing limit must not become zero threads, and seeing 64 host CPUs must not automatically select 64 threads. If a Pod has sidecars, verify both each container's budget and the total Pod QoS/resources.

## Diagnostic Order: Start with Which Phase Is Waiting

1. **Instrument phase timers.** Measure normalization, tokenization, H2D, model forward, postprocessing, encoding, and publishing separately. End-to-end latency alone cannot distinguish CPU, GPU, I/O, or queue delay.
2. **Observe node and cgroup together.** Node idle does not disprove container throttling; node busy does not prove that a request is executing on CPU.
3. **Read the effective CPU budget.** Compare affinity/cpuset with quota/period. Conceptually, the upper bound on parallelism is:

   ```text
   effective_cpu = min(cpuset_cpu_count, cpu_quota / cpu_period)
   ```

   Only when quota is unlimited should cpuset/affinity be used alone. Fractional CPU needs benchmarking to choose a thread count; do not unconditionally round it up.
4. **Inventory every thread pool.** Include BLAS, tokenizers, DataLoaders, web servers, Celery/Ray workers, ffmpeg, and sidecars in addition to the framework.
5. **Run a same-input A/B test.** Hold model, input, batch, GPU profile, and colocated competing workloads constant. Change only the thread budget, then compare throughput, p50/p95/p99, throttling, and output quality.
6. **Preserve watchdog evidence and stacks.** On a phase timeout, emit all thread stacks before the orchestrator restarts the process. Restarting is mitigation, not proof of root cause.

For cgroup v2, `nr_periods`, `nr_throttled`, and `throttled_usec` in `cpu.stat`, plus `cpu.max`, are important evidence. cgroup v1 exposes the corresponding `cpu.stat`, `cpu.cfs_quota_us`, and `cpu.cfs_period_us`. Monitoring systems may publish the same data under different metric names, so a runbook should preserve the actual queries rather than only dashboard screenshots.

## Production Checklist

- [ ] The Deployment explicitly declares CPU/memory requests and limits.
- [ ] The application thread budget comes from effective CPU, not the host CPU count.
- [ ] `process concurrency × threads per process` stays within benchmarked capacity.
- [ ] Phase metrics, cgroup throttling, OOMs, GPU allocated/reserved memory, and queue age are observable.
- [ ] `/dev/shm`, tmpfs, ephemeral storage, model cache, and checkpoints have capacity limits.
- [ ] GPU sharing mode, VRAM/fault isolation, and workload request semantics agree.
- [ ] A latency-sensitive Pod requesting exclusive CPU has verified QoS, integer request, CPU Manager, and NUMA configuration.
- [ ] Same-input A/B tests cover cold start, steady state, concurrency, and colocated noisy neighbors.
- [ ] The watchdog identifies the stalled phase; restarted tasks support checkpoints, idempotency, and safe retries.

## References

- [Kubernetes: Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [Kubernetes: Control CPU Management Policies on the Node](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/)
- [PyTorch: Threading Environment Variables](https://docs.pytorch.org/docs/stable/threading_environment_variables.html)
- [PyTorch: CPU threading and TorchScript inference](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
- [PyTorch: `torch.set_num_threads`](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html)
- [NumPy: Number of threads used for linear algebra](https://numpy.org/doc/stable/reference/global_state.html#number-of-threads-used-for-linear-algebra)
- [TensorFlow: Configure intra-op parallelism](https://www.tensorflow.org/api_docs/python/tf/config/threading/set_intra_op_parallelism_threads)
- [TensorFlow: Configure inter-op parallelism](https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads)
- [ONNX Runtime: Thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [Docker: Runtime constraints on resources](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)
- [NVIDIA GPU Operator: Time-Slicing GPUs in Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)
