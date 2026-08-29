# 逐層調教 TTS 推論：BERT 搬運、batch 與 checkpoint 邊界

> [English](./README.md) | **繁體中文**

更新日期：2026-08-29

這篇實務筆記整理一系列在 NVIDIA L40S 上、貼近 production 的 GPT-SoVITS 實驗；它不是所有
TTS 都能直接複製的參數表。核心結論是：TTS 吞吐同時受好幾種邊界控制，改錯旋鈕可能犧牲
可恢復性或 VRAM，卻只換到很小的速度收益：

```text
業務任務
  -> 外層 request／checkpoint 分片
    -> 保留句界的內部分段
      -> inference batches
        -> 文字正規化與 BERT
          -> GPT + VITS
            -> MP3 parts 與 bounded-memory 合併
```

這個案例目前量到的最佳組合，是約 50 字且保留完整句界的內部分段、batch 32，以及在實測
VRAM 預算容許時，讓中等長度輸入只走一個外層 request。真正的長文仍需保留外層 checkpoint。
另一條線上，將 BERT feature expansion 留在 GPU 是低風險微調；批次執行 BERT forward 則有較小
的端到端收益，也觀察到較高峰值。

## 四個不能混為一談的控制面

| 控制面 | 改變什麼 | 主要收益 | 主要風險 |
| --- | --- | --- | --- |
| 外層 request／checkpoint 大小 | 一次 API request 放多少業務文字 | 減少 round trip、重複 prompt 工作、cache cleanup 與 MP3 join | 單次失敗重算更多，單 request VRAM 較高 |
| 內部分段目標 | 推論引擎看到的句界感知文字片段 | sequence shape 更均勻、減少 padding 浪費 | 太碎會增加 frontend overhead；太長會形成不均勻 batch |
| TTS inference batch | 同時處理多少內部分段 | 減少 GPU rounds、提高利用率 | activation peak 上升，尾批效益遞減 |
| BERT batch | 同時 tokenize 並 forward 多少 clean text segments | 減少 tokenizer／BERT 呼叫次數 | padding 與 transient peak；混合語言必須明確 fallback |

`batch_size=32` 不代表 32 個字、32 筆業務任務或 32 個外層 chunk；在這些實驗裡，它是一次
inference round 最多處理 32 個內部文字片段。Production worker 仍一次處理一筆業務任務，沒有
把不同 request 的片段動態混成 continuous batch。

## 實驗契約

實驗使用已完成的真實輸入，但不重新 claim 或更新其 DB row。每次會影響 runtime 的 A/B 前：

1. Production consumer 先停止領新工作，讓當前任務自然 drain。
2. 除了被測變因，固定輸入、replace rules、voice/reference、fine-tuned models、生成參數與 runtime。
3. Probe 直接呼叫 inference endpoint，只寫入私有測試輸出。
4. 記錄 wall time、目標 process VRAM、整卡 VRAM、音訊長度、clipping／impulse 與 service restart。
5. 移除暫時配置，再確認 production consumer 能繼續完成真實工作。

早期部分實驗重用了已 warm 的 legacy process，因此 `nvidia-smi` raw value 包含模型 working set
與 allocator cache。本文將它稱為觀察到的 process working set，不稱為模型大小。不同測試窗口
若沒有相同 baseline 與採樣方法，不能直接互相相減。

## 實驗一：內部分段目標 50、100、500 字

使用一筆 4,611 字輸入與 batch 32，每個變體重複兩次。這個目標會累積完整句子到門檻附近，
不是機械式在第 N 個字硬切。

| 內部分段目標 | 延遲中位數 | TTS process 觀察最大峰值 | 輸出長度 |
| ---: | ---: | ---: | ---: |
| 50 字 | 31.423 s | 12,616 MiB | 808.416 s |
| 100 字 | 33.630 s | 11,240 MiB | 808.956 s |
| 500 字 | 33.220 s | 23,544 MiB | 787.356 s |

500 字不但沒有變快，觀察峰值接近 50 字版本的兩倍，輸出長度差異也大得多。最合理的機制是
sequence shape 不均：一個 batch 會等最長序列，短序列還要 padding；autoregressive 與 attention
成本也會隨序列長度增加。每段字數增加，不等於 GPU 一定更飽和。

因此這個 workload 保留 50 字目標；它是實驗值，不是所有 voice model、語言、標點分布或 GPU
的通用預設。

## 實驗二：TTS batch 4、16、32

使用相同 4,109 字輸入，外層固定為依序執行的 1,987／1,999／123 字三段；內部分段保持約
50 字、`parallel_infer=true`，只改 TTS batch。

在 batch sweep 前，先固定 batch 4 比較 per-request parallel inference；完整 probe 由約
201.042 秒降至 101.421 秒，時間減少 49.55%。三個外層 requests 仍是串行；parallelism 發生在
每個 request 內部，不會跨 checkpoint chunks 同時執行。

| Batch | 推論加合併端到端時間 | 相對 batch 4 | Process 觀察峰值 | 備註 |
| ---: | ---: | ---: | ---: | --- |
| 4 | 101.421 s | baseline | 17,080 MiB | 已啟用 parallel inference |
| 16 | 36.960 s | 快 2.744× | 17,592 MiB | 時間減少 63.56% |
| 32 | 31.325 s | 快 3.238× | 15,704 MiB | 比 batch 16 再快 15.25% |

Batch 32 的 raw peak 看起來反而較低，是因為 warm process 重用了 allocator blocks，而且一秒一次
的 `nvidia-smi` 可能漏掉次秒配置；這不是 batch 32 天生比 batch 16 省 VRAM 的證據。

該輸入形成 123 個短片段：batch 16 需要十輪 GPU batch，batch 32 需要六輪。最後 123 字的
外層 chunk 只有四個短片段，再大的 batch 也用不到，這解釋 16→32 的邊際效益已明顯下降。

另一輪直接重現 legacy 參數：單一外層 request、batch 35、每四句切一次：

| 路徑 | 延遲 | Process 觀察峰值 | 內部 batch shape |
| --- | ---: | ---: | --- |
| Legacy-shaped request | 21.097 s | 24,036 MiB | 約 `[35, 35, 35, 3]` |
| 50 字分段、batch 32 | 18.586 s | 24,870 MiB | `[32, 32, 32, 31]` |

較小的 batch 反而快 11.90%，因為片段 shape 更均勻，尾批也更滿。Batch 容量只有在 sequence
shape 與最後一批 occupancy 允許時才有價值。

## 實驗三：外層 2,000 字 checkpoint vs 單一 request

外層分片可保存 durable progress：每完成一段就保存 MP3 part 與 checksum，後段失敗不用全文
重產。但每個邊界都會重複 request setup、prompt/reference 處理、frontend、response transfer 與
最終 MP3 join。

兩組獨立真實輸入 A/B 量出這個成本：

| 輸入 | 前測 | 後測 | 結果 | VRAM 證據 |
| --- | --- | --- | --- | --- |
| 4,109 字 | 三個外層 requests、batch 32：31.325 s | 一個外層 request、batch 32：18.586 s | 時間減少 40.66% | 單一 request 峰值 24,870 MiB |
| 4,611 字、固定 seed、各兩次 | 三個外層 requests：中位 33.174 s | 一個外層 request：中位 24.123 s | 時間減少 27.3% | 觀察峰值由 7,684 升至 13,038 MiB |

第二組採用 outer、single、single、outer 的交錯順序，降低 warm-up 與順序偏差。四個輸出都通過
離線 clipping／impulse detector；outer 與 single 的音訊長度差 0.90%，所以 fixed seed 也不代表
兩種 execution graph 會 byte-identical。

目前證據支持的 production candidate policy 是自適應：

- 約 5,000 字內，且所選 GPU profile 有實測 headroom 時，可走單一外層 request。
- 更長輸入保留 bounded outer chunks 與 durable per-chunk checkpoints。
- 某段失敗時從驗證過的 checkpoint 繼續；不要把所有 decoded MP3 累積在 Python memory。
- 門檻屬於 application release profile 與 corpus benchmark，不屬於 cluster IaC。

這能讓常見中等長度輸入走快路徑，又不犧牲真正長文的 bounded replay。

## 實驗四：讓 BERT phone expansion 留在 GPU

繼承的 BERT 路徑先把 character-level hidden states 搬到 CPU，在 Python loop 逐字展開，後面又搬回
GPU：

```python
# Before：device transfer、Python list growth，後面還要再 transfer。
res = hidden_states.cpu()
phone_level = torch.cat([
    res[index].repeat(repeat_count, 1)
    for index, repeat_count in enumerate(word2ph)
])
```

修正版把 tensor 留在原 device，用一次向量化操作展開：

```python
# After：在原 device 保留順序與 dtype。
repeat_counts = torch.as_tensor(word2ph, device=res.device, dtype=torch.long)
phone_level = torch.repeat_interleave(
    res,
    repeat_counts,
    dim=0,
    output_size=sum(word2ph),
)
```

| Microbenchmark 輸入 | CPU／Python 展開 | Device-local `repeat_interleave` | 解讀 |
| --- | ---: | ---: | --- |
| 代表性 50 字片段，p50 | 0.4945 ms | 0.3232 ms | 每段約省 0.17 ms |
| 2,000 字 synthetic expansion | 16.04 ms | 0.4146 ms | 移除 Python-loop scaling 與 transfer overhead |

約 70 個短片段的直接收益只有約 12 ms，所以這項修改本身不能解釋數秒的端到端差異。它的價值
是用低風險方式移除不必要的 device round trip 與長序列下很差的 Python scaling。部署前仍要驗證
row order、dtype、device 與 phone count 完全一致。

## 實驗五：批次 BERT forward

逐句版本仍會為每個 clean segment 各做一次 tokenizer 與 BERT forward。實驗版把最多 32 個
純中文 segments 一起 padding、做一次 BERT forward，再依 attention mask 與 `word2ph` 重建每列；
混合語言或不支援的輸入會 fallback 到既有逐句路徑。

原本 4,611 字測試含拉丁字母，正確走 fallback，因此沒有拿它冒充 batch 結果。另選一筆已完成的
4,144 字全中文輸入，在相同 model、voice、seed、batch 32 與單一外層 request 下各跑兩次：

| 指標 | 逐句 BERT | 批次 BERT，上限 32 | 差異 |
| --- | ---: | ---: | ---: |
| 兩次耗時 | 22.002 / 22.055 s | 21.625 / 19.659 s | — |
| 延遲中位數 | 22.029 s | 20.642 s | 快 6.3% |
| TTS process 最大觀察峰值 | 20,628 MiB | 22,798 MiB | +2,170 MiB，或 10.5% |
| 音訊長度 | 743.256 s | 741.744 s | -0.20% |
| Integrated loudness | -25.4 LUFS | -25.4 LUFS | 相同 |

Runtime phase log 證明請求真的進入 `batch-model-forward`，而不是 fallback；每個 request 形成五次
BERT forwards。兩側都沒有 hard clip、near clip 或超門檻相鄰 sample step。

這個收益有潛力，但小於外層邊界與 TTS batch 的收益。正式設為 default 前應維持可配置，並用
短／長文、不同標點、混合語言、cold／warm 狀態與 colocated GPU load 做 corpus regression。

## 實驗後的 production 快照

移除暫時 BERT override 並恢復 worker／scheduler 後，TTS Pod restart count 為 0，且重新完成真實
DB 工作。最近一小時快照為：

| 完成任務 | 字數 | 平均處理時間 | P50 | P90 |
| ---: | ---: | ---: | ---: | ---: |
| 89 | 381,773 | 35.80 s | 37.00 s | 41.00 s |

直接 A/B request 只讀且不更新業務 row，因此不計入 production 完成量。這份快照證明實驗後已
恢復並具有實際吞吐；它本身不能把全部 production throughput 歸因給某一項優化。

## 實驗目前支持哪些決策

| 決策 | 當前證據 | 建議狀態 |
| --- | --- | --- |
| BERT phone expansion 留在 GPU | Exact operation replacement 與 focused microbenchmark | 通過 source/tensor contract 測試後可作低風險預設 |
| 句界感知目標約 50 字 | 此輸入比 500 字快，峰值也低很多 | Workload 預設，但保持可配置 |
| TTS batch 32 | 4→16 大幅提升，16→32 仍有較小收益 | 只在匹配的 VRAM profile 下作預設 |
| 任一 retry 就關掉 parallel inference | 歷史 retry count 無法代表 GPU 壓力 | 不應採用；只依 release-scoped OOM／timeout 證據降級 |
| 所有文件都走一個外層 request | 4–5K 字很快，但 replay 與 VRAM 風險上升 | 拒絕作通用規則 |
| 實測門檻以下走自適應單 request | 兩組實驗重現 27–41% latency reduction | 好的候選；門檻以上保留 checkpoints |
| 批次 BERT forward | 一筆全中文案例快 6.3%，觀察峰值高 10.5% | 擴大回歸前維持 opt-in |

## 可重現的調教順序

1. **先拆 phase timer。** 分開量 language detection、normalization、tokenizer、BERT、GPT、VITS、
   download、encode、join 與 publish。
2. **先修 correctness 與 hidden I/O。** 模型下載、cache miss、language-model loading 與 retry race
   足以淹沒任何 kernel optimization。
3. **限制 CPU threads。** 讓 PyTorch／BLAS thread pools 對齊 Pod effective CPU budget；見
   [容器內 ML workload 的資源感知](../container-resource-awareness/README.zh-TW.md)。
4. **移除不必要的 transfer。** 用 focused test 證明 device、dtype、order、shape 等價。
5. **調整內部分段 shape。** 同時比較 latency、padding／round 數、peak VRAM、duration 與音訊 artifact。
6. **固定外層計畫後掃 TTS batch。** 不要一邊改 batch、一邊改 chunking 或 concurrency。
7. **最後才調外層邊界。** 量化 speed／replay／VRAM trade-off，在門檻以上保留 checkpoints。
8. **實驗 frontend batching。** BERT batching 是獨立優化，有自己的 padding 與語言 routing 風險。
9. **跑 production-shaped validation。** 包含 cold／warm、retry、OOM recovery、混合 workload、
   音訊品質、DB publication 與 rollback。

## 避免錯誤結論的量測項目

- 報告每輪數值與中位數，不只挑最快的一輪。
- 說清楚 GPU memory 是 process working set、allocator allocated/reserved，還是 scheduler reservation。
- 記錄採樣間隔；一秒一次的 `nvidia-smi` 可能漏掉 transient peak。
- 除被測變因外，固定 outer chunks、internal split、batch、parallel mode、model、seed 與 colocated load。
- 驗證 duration、clipping、impulse／join artifact、loudness、F0 與頻譜代理，也要人工耳聽與做任務專屬
  intelligibility check。
- MFCC cosine 與 pitch similarity 只是診斷，不等於聲紋身分或語意等價。
- 實驗後確認 consumer 能繼續完成真實工作，且沒有 OOM、timeout、traceback 或 restart。

## 限制

- 最完整的表格只涵蓋兩筆約 4K–4.6K 字真實輸入、一張 L40S 與一個 voice/model family。
- 部分歷史實驗仍有隨機性；後續補了 fixed-seed repetitions，但沒有覆蓋整個矩陣。
- 各 VRAM 表格來自不同 allocator warm state 與採樣頻率，只有同一窗口內的比較有效。
- BERT batching 只驗證全中文路徑；本文刻意不宣稱混合語言已通過。
- 實驗尚未實作 cross-request continuous batching；那需要 queue fairness、cancellation、逐 request
  reconstruction 與 latency SLO 設計。

真正可複用的 lesson 不是「永遠選 50、32、5,000」，而是分層隔離變因、在需要的地方保留失敗
恢復，並只用相同輸入的 latency、資源、品質與 recovery 證據提升某個設定。
