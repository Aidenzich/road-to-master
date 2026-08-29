# Layer-by-Layer TTS Inference Tuning: BERT Transfers, Batching, and Checkpoint Boundaries

> **English** | [繁體中文](./README.zh-TW.md)

Updated: 2026-08-29

This field note records a sequence of production-shaped GPT-SoVITS experiments on an NVIDIA L40S. It is not a universal tuning recipe. Its main result is that TTS throughput is controlled by several different boundaries, and changing the wrong one can trade recoverability or VRAM for little speed:

```text
business task
  -> outer request/checkpoint chunks
    -> sentence-aware internal segments
      -> inference batches
        -> text normalization and BERT
          -> GPT + VITS
            -> MP3 parts and bounded-memory join
```

The best measured configuration in this case combined roughly 50-character sentence-aware internal segments, batch 32, and a single outer request for inputs that fit a measured VRAM budget. Long inputs still need outer checkpoints. Separately, keeping BERT feature expansion on GPU was a low-risk micro-optimization, while batching BERT forwards produced a smaller end-to-end gain and a higher observed peak.

## Four Controls That Must Not Be Confused

| Control | What it changes | Primary benefit | Primary risk |
| --- | --- | --- | --- |
| Outer request/checkpoint size | How much business text is sent through one API request | Fewer round trips, repeated prompt work, cache cleanup, MP3 joins | Higher failure replay cost and higher single-request VRAM |
| Internal segment target | Sentence-aware text pieces seen by the inference engine | More uniform sequence shapes and less padding waste | Too many tiny segments add frontend overhead; large targets create uneven batches |
| TTS inference batch | Number of internal segments processed together | Fewer GPU rounds and higher GPU utilization | Larger activation peak and diminishing returns in tail batches |
| BERT batch | Number of cleaned text segments tokenized and forwarded together | Fewer tokenizer/BERT calls | Padding and a higher transient peak; mixed-language paths need explicit fallback |

`batch_size=32` does not mean 32 characters, 32 business tasks, or 32 outer chunks. In these experiments it meant up to 32 internal text segments in one inference round. The production worker still used one business task at a time; it did not dynamically combine segments from unrelated requests.

## Experimental Contract

The experiments used completed real inputs but did not reclaim or update their database rows. Before each intrusive A/B:

1. The production consumer stopped claiming new work and drained the current task naturally.
2. Input text, replacement rules, voice/reference audio, fine-tuned models, generation parameters, and the inference runtime were held constant unless named as the independent variable.
3. The probe called the inference endpoint directly and wrote only private test outputs.
4. Wall time, target-process VRAM, total GPU VRAM, output duration, clipping/impulse metrics, and service restarts were recorded.
5. The temporary configuration was removed and the production consumer was shown to complete real work again.

Some early runs reused an already-warm legacy process, so raw `nvidia-smi` values include model state and allocator cache. Peak values are therefore reported as observed process working sets, not model size. Values from different experiment windows should not be subtracted unless they share the same baseline and sampling method.

## Experiment 1: Internal Segment Targets of 50, 100, and 500 Characters

A 4,611-character input was run with batch 32. Each variant was repeated twice. The target is sentence-aware: it accumulates complete sentences around the target instead of blindly cutting at exactly the Nth character.

| Internal target | Median latency | Maximum observed TTS-process peak | Output duration |
| ---: | ---: | ---: | ---: |
| 50 characters | 31.423 s | 12,616 MiB | 808.416 s |
| 100 characters | 33.630 s | 11,240 MiB | 808.956 s |
| 500 characters | 33.220 s | 23,544 MiB | 787.356 s |

The 500-character target did not improve latency, nearly doubled the observed peak relative to target 50, and changed output duration much more. The likely mechanism is sequence-shape imbalance: a batch waits for its longest sequence and pads shorter members, while autoregressive and attention work grows with sequence length. More characters per segment do not automatically mean better GPU saturation.

Target 50 was retained for this workload. This is an empirical value, not a generic default for every voice model, language, punctuation distribution, or GPU.

## Experiment 2: TTS Batch 4, 16, and 32

The next A/B used the same 4,109-character input and the same three sequential outer chunks of 1,987, 1,999, and 123 characters. Internal splitting remained at roughly 50 characters and `parallel_infer=true`; only TTS batch changed.

Before the batch sweep, enabling per-request parallel inference at batch 4 reduced the complete probe from approximately 201.042 to 101.421 seconds: 49.55% less time. The three outer requests were still sequential; parallelism existed inside each request, not across checkpoint chunks.

| Batch | End-to-end inference + join | Relative to batch 4 | Observed process peak | Notes |
| ---: | ---: | ---: | ---: | --- |
| 4 | 101.421 s | baseline | 17,080 MiB | Parallel inference already enabled |
| 16 | 36.960 s | 2.744× faster | 17,592 MiB | 63.56% less time |
| 32 | 31.325 s | 3.238× faster | 15,704 MiB | 15.25% faster than batch 16 |

The batch-32 raw peak appears lower because the warm process reused cached allocator blocks and one-second `nvidia-smi` sampling can miss sub-second allocations. It is not evidence that batch 32 intrinsically needs less VRAM than batch 16.

For this input, 123 short segments required ten GPU rounds at batch 16 and six rounds at batch 32. The final 123-character outer chunk contained only four internal segments, so its tail batch could not use the larger capacity. This explains the diminishing gain from 16 to 32.

A direct comparison against the legacy parameters used one outer request, batch 35, and a four-sentence split:

| Path | Latency | Observed process peak | Internal batch shape |
| --- | ---: | ---: | --- |
| Legacy-shaped request | 21.097 s | 24,036 MiB | approximately `[35, 35, 35, 3]` |
| 50-character segments, batch 32 | 18.586 s | 24,870 MiB | `[32, 32, 32, 31]` |

The smaller batch was 11.90% faster because the segments were more uniform and the tail batch was fuller. Batch capacity is only useful when the sequence shapes and final occupancy allow it to be used.

## Experiment 3: Outer 2,000-Character Checkpoints vs One Request

Outer chunks provide durable progress: after each successful chunk, an implementation can persist an MP3 part and checksum so a later failure does not replay the whole document. However, each boundary repeats request setup, prompt/reference handling, frontend work, response transfer, and final MP3 joining.

Two independent real-input A/Bs measured this cost.

| Input | Before | After | Result | VRAM evidence |
| --- | --- | --- | --- | --- |
| 4,109 characters | three outer requests, batch 32: 31.325 s | one outer request, batch 32: 18.586 s | 40.66% less time | single request peaked at 24,870 MiB |
| 4,611 characters, fixed seed, two repetitions | three outer requests: median 33.174 s | one outer request: median 24.123 s | 27.3% less time | observed peak rose from 7,684 to 13,038 MiB |

The second experiment used an interleaved order—outer, single, single, outer—to reduce warm-up and ordering bias. All four outputs passed the offline clipping and impulse detector. The outer and single outputs differed in duration by 0.90%, so fixed seed did not make the two execution graphs byte-identical.

The production candidate policy from this evidence is adaptive:

- Inputs no larger than about 5,000 characters may use one outer request only when the selected GPU profile has measured headroom.
- Longer inputs retain bounded outer chunks and durable per-chunk checkpoints.
- A failed chunk resumes from a verified checkpoint; it does not concatenate every decoded MP3 into Python memory.
- The threshold belongs to the application release profile and corpus benchmark, not to cluster IaC.

This preserves fast execution for common medium-length inputs without giving up bounded replay for genuinely long documents.

## Experiment 4: Keep BERT Phone Expansion on GPU

The inherited BERT path first moved character-level hidden states to CPU, expanded every character in a Python loop, and later moved the result back to GPU:

```python
# Before: device transfer, Python list growth, then another transfer downstream.
res = hidden_states.cpu()
phone_level = torch.cat([
    res[index].repeat(repeat_count, 1)
    for index, repeat_count in enumerate(word2ph)
])
```

The revised path keeps the tensor on its existing device and performs one vectorized expansion:

```python
# After: preserve order and dtype on the current device.
repeat_counts = torch.as_tensor(word2ph, device=res.device, dtype=torch.long)
phone_level = torch.repeat_interleave(
    res,
    repeat_counts,
    dim=0,
    output_size=sum(word2ph),
)
```

| Microbenchmark input | CPU/Python expansion | Device-local `repeat_interleave` | Interpretation |
| --- | ---: | ---: | --- |
| Representative 50-character segment, p50 | 0.4945 ms | 0.3232 ms | about 0.17 ms saved per segment |
| 2,000-character synthetic expansion | 16.04 ms | 0.4146 ms | removes Python-loop scaling and transfer overhead |

For roughly 70 short segments, the direct saving was only about 12 ms, so this change alone cannot explain multi-second end-to-end gains. Its value is low-risk cleanup of an unnecessary device round trip and a poor long-sequence scaling path. It should be validated for exact row order, dtype, device, and phone count before deployment.

## Experiment 5: Batch BERT Forwards

The sequential path still invoked tokenizer and BERT once per cleaned segment. An experimental path grouped up to 32 plain-Chinese segments, padded them, ran one BERT forward, then reconstructed each row using its attention mask and `word2ph`. Mixed-language or unsupported input fell back to the existing sequential path.

The original 4,611-character test contained Latin letters and correctly took the fallback, so it was not used to claim a batching result. A separate completed 4,144-character all-Chinese input was tested twice with the same model, voice, seed, batch 32, and one outer request.

| Metric | Sequential BERT | Batched BERT, max 32 | Change |
| --- | ---: | ---: | ---: |
| Two runs | 22.002 / 22.055 s | 21.625 / 19.659 s | — |
| Median latency | 22.029 s | 20.642 s | 6.3% faster |
| Maximum observed TTS-process peak | 20,628 MiB | 22,798 MiB | +2,170 MiB, or 10.5% |
| Output duration | 743.256 s | 741.744 s | -0.20% |
| Integrated loudness | -25.4 LUFS | -25.4 LUFS | unchanged |

Runtime phase logs proved that the request entered `batch-model-forward` rather than fallback. It formed five BERT forwards per request. Both outputs had zero hard clips, near clips, or excessive adjacent sample steps.

This is a promising but smaller gain than the outer-boundary and TTS-batch changes. It should remain configurable until a corpus covers short/long text, punctuation distributions, mixed languages, cold/warm state, and colocated GPU load.

## Post-Experiment Production Snapshot

After removing the temporary BERT override and restoring the worker and scheduler, the TTS Pod had restart count zero and completed real database work again. A one-hour snapshot recorded:

| Completed tasks | Characters | Mean processing time | P50 | P90 |
| ---: | ---: | ---: | ---: | ---: |
| 89 | 381,773 | 35.80 s | 37.00 s | 41.00 s |

The direct A/B requests were read-only and did not update business rows, so they were not included in these production counts. This snapshot proves recovery and useful throughput after the experiment; it does not by itself attribute all production throughput to one optimization.

## What the Experiments Support

| Decision | Current evidence | Recommended status |
| --- | --- | --- |
| Keep BERT phone expansion on GPU | Exact operation replacement plus focused microbenchmark | Low-risk default after source and tensor-contract tests |
| Use sentence-aware target near 50 characters | Faster and much lower peak than target 500 in the measured input | Workload default, still configurable |
| Use TTS batch 32 | Strong gain from 4→16 and smaller gain from 16→32 | Default only with a matching VRAM profile |
| Disable parallel inference after any retry | Historical retry count did not identify GPU pressure | Do not do this; downgrade only for release-scoped OOM/timeout evidence |
| One outer request for every document | Fast for 4–5K characters but raises replay and VRAM risk | Reject as a universal rule |
| Adaptive single request below a measured threshold | Reproduced 27–41% latency reduction | Good candidate; keep checkpoints above the threshold |
| Batch BERT forwards | 6.3% median gain with 10.5% higher observed peak in one all-Chinese case | Opt-in experiment pending broader regression |

## A Reproducible Tuning Order

1. **Instrument phases first.** Separate language detection, normalization, tokenizer, BERT, GPT, VITS, download, encode, join, and publish time.
2. **Fix correctness and hidden I/O.** Model downloads, cache misses, language-model loading, and retry races can dominate any kernel optimization.
3. **Bound CPU threads.** Match PyTorch/BLAS thread pools to the effective Pod CPU budget; see [Resource-Aware ML Workloads in Containers](../container-resource-awareness/).
4. **Remove unnecessary transfers.** Prove device, dtype, order, and shape equivalence with a focused test.
5. **Tune internal segment shape.** Compare latency, padding/round count, peak VRAM, duration, and acoustic artifacts.
6. **Sweep TTS batch within a fixed outer plan.** Do not compare batch values while also changing chunking or concurrency.
7. **Then tune the outer boundary.** Measure the speed/replay/VRAM trade-off and retain checkpoints above a validated threshold.
8. **Experiment with frontend batching.** BERT batching is a separate optimization with separate padding and language-routing risks.
9. **Run production-shaped validation.** Include cold/warm state, retry, OOM recovery, mixed workloads, audio quality, DB publication, and rollback.

## Metrics That Prevent Misleading Conclusions

- Report per-run values and medians, not only the fastest run.
- Identify whether GPU memory is process working set, allocator allocated/reserved, or scheduler reservation.
- Record sampling interval; one-second `nvidia-smi` can miss transient peaks.
- Hold outer chunks, internal split, batch, parallel mode, model, seed, and colocated load constant unless they are the tested variable.
- Validate duration, clipping, impulse/join artifacts, loudness, F0, and a spectral proxy; use listening and task-specific intelligibility checks as well.
- Treat MFCC cosine and pitch similarity as diagnostics, not speaker identity or semantic equivalence.
- Verify that the consumer resumes real work with no OOM, timeout, traceback, or restart after the experiment.

## Limitations

- The strongest tables use two real inputs of about 4K–4.6K characters on one L40S and one voice/model family.
- Some historical runs were randomized; fixed-seed repetitions were added later but do not cover the whole matrix.
- Different VRAM tables came from different warm allocator states and sampling rates, so only within-window comparisons are valid.
- The BERT-batching result covers an all-Chinese path. Mixed-language batching was intentionally not claimed.
- These experiments did not implement cross-request continuous batching. That would require queue fairness, cancellation, per-request reconstruction, and latency-SLO design.

The reusable lesson is not “always choose 50, 32, and 5,000.” It is to tune each layer independently, preserve failure recovery where it matters, and promote a setting only with same-input latency, resource, quality, and recovery evidence.
