| Property  | Data |
|-|-|
| Created | 2026-01-23 |
| Updated | 2026-08-11 |
| Author | @Aiden |
| Tags | #study #optimization #attention |

# FlashAttention 深度解析

FlashAttention 是由 Tri Dao 等人於 2022 年提出的一種 **IO-aware** 的精確注意力演算法，通過優化 GPU 記憶體存取模式，在不犧牲精度的情況下顯著提升 Transformer 的訓練和推理速度。

## 目錄
- [動機：為什麼需要 FlashAttention？](#動機為什麼需要-flashattention)
- [GPU 記憶體層次結構](#gpu-記憶體層次結構)
- [標準 Attention 的問題](#標準-attention-的問題)
- [FlashAttention 核心原理](#flashattention-核心原理)
- [Tiling 分塊策略](#tiling-分塊策略)
- [Online Softmax 技巧](#online-softmax-技巧)
- [Recomputation 重計算策略](#recomputation-重計算策略)
- [FlashAttention-2 改進](#flashattention-2-改進)
- [FlashAttention-3 改進](#flashattention-3-改進)
- [實作範例](#實作範例)
- [效能分析](#效能分析)

---

## 動機：為什麼需要 FlashAttention？

Naive Self-Attention 具有 $O(N^2)$ 的計算量與中間張量空間，其中 $N$ 是序列長度。下表只計算**單一 attention head 的一張 FP16 score matrix**，不代表模型的總 VRAM：

| 序列長度 | 注意力矩陣大小 | 記憶體需求 (FP16) |
|---------|--------------|------------------|
| 1K | 1M | 2 MB |
| 4K | 16M | 32 MB |
| 16K | 256M | 512 MB |
| 64K | 4B | 8 GB |
| 128K | 16B | 32 GB |

這導致了兩個關鍵問題：
1. **記憶體瓶頸**：無法處理長序列
2. **速度瓶頸**：大量的 HBM 讀寫操作成為效能瓶頸

---

## GPU 記憶體層次結構
想像 GPU 是一間工廠：registers 與 shared memory 是運算單元附近、容量小但快速的
工作區；HBM/global memory 是容量大但資料搬運成本較高的倉庫。實際容量與頻寬取決
於 GPU 世代和型號，因此不應把某張 GPU 的數字當成 FlashAttention 的固定規格。

Naive Attention 會產生 $N \times N$ scores/probabilities，並在不同 kernels 之間
寫入、讀回 HBM。FlashAttention 的解法是把 Q、K、V 分塊載入 on-chip memory，
用 online softmax 累積結果，避免把完整的二次方中間矩陣寫入 HBM。Q、K、V 仍需
從 HBM 載入，最終 output 也要寫回；省掉的是大量中間資料搬運，而不是完全避開 HBM。

---

## 標準 Attention 的問題

### 標準實作流程

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

標準實作需要多次 HBM 讀寫：

```python
# 標準 Attention 實作 (偽代碼)
def standard_attention(Q, K, V):
    # Step 1: 計算 S = QK^T (寫入 HBM)
    S = Q @ K.T / sqrt(d_k)        # HBM write: O(N²)

    # Step 2: 計算 P = softmax(S) (讀取 S，寫入 P)
    P = softmax(S, dim=-1)          # HBM read: O(N²), write: O(N²)

    # Step 3: 計算 O = PV (讀取 P 和 V)
    O = P @ V                       # HBM read: O(N²)

    return O
```

### HBM 存取分析

| 操作 | HBM 讀取 | HBM 寫入 |
|-----|---------|---------|
| $S = QK^T$ | $O(Nd)$ | $O(N^2)$ |
| $P = \text{softmax}(S)$ | $O(N^2)$ | $O(N^2)$ |
| $O = PV$ | $O(N^2 + Nd)$ | $O(Nd)$ |
| **總計** | $O(N^2 + Nd)$ | $O(N^2 + Nd)$ |

當 $N >> d$ 時，HBM 存取量為 $O(N^2)$，這是主要的效能瓶頸。

---

## FlashAttention 核心原理

FlashAttention 的核心思想是讓每個 tile 的主要運算與 softmax statistics 留在
on-chip memory，避免將完整 $N \times N$ 中間矩陣寫入 HBM。

### 一句話總結

> **FlashAttention = 分塊處理 + 邊算邊修正 + 不存中間結果**

傳統 Attention 像是「先把整張大圖畫完存起來，之後再用」；FlashAttention 則是「畫一小塊、用一小塊、丟一小塊」——雖然畫的動作多了，但省下的存取時間遠超過多畫的時間。

### 三大核心技術

| 技術 | 一句話解釋 | 解決的問題 |
|-----|-----------|-----------|
| **Tiling** | 把大矩陣切成小塊，在快取裡處理 | 中間結果太大放不進 SRAM |
| **Online Softmax** | 邊處理邊修正，不用看完全部才能算 | 分塊後無法直接計算 softmax |
| **Recomputation** | 反向傳播時重算而非儲存 | 訓練時需要保存中間結果 |

```
┌─────────────────────────────────────────────────────────────┐
│                   FlashAttention 核心技術                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Tiling (分塊)                                            │
│     ├── 將 Q, K, V 分成小塊                                  │
│     ├── 每個小塊可以放入 SRAM                                │
│     └── 分塊計算 attention                                   │
│                                                              │
│  2. Online Softmax                                           │
│     ├── 逐塊計算 softmax                                     │
│     ├── 動態更新 normalization factor                        │
│     └── 保證數值穩定性                                       │
│                                                              │
│  3. Recomputation (重計算)                                   │
│     ├── 反向傳播時重新計算 attention matrix                  │
│     ├── 用計算換記憶體                                       │
│     └── 總體仍然更快 (減少 HBM 存取)                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Tiling 分塊策略

想像你要計算一個超大的乘法表（例如 10000 × 10000），但你的計算紙（SRAM）只能容納 100 × 100 的小表格。

**傳統做法**：硬著頭皮把整張大表算出來，存到倉庫（HBM）裡，之後再從倉庫搬出來用。問題是：倉庫雖然大，但搬運速度很慢。

**Tiling 的做法**：把大表格切成很多 100 × 100 的小塊，每次只在計算紙上處理一小塊。處理完一塊就直接用掉，不用存到倉庫。這樣雖然要分很多次計算，但因為省去了大量的「搬運時間」，總體反而更快。

**關鍵洞察**：現代 GPU 的計算速度遠超記憶體傳輸速度。與其花時間搬運資料，不如多算幾次。

### 分塊示意圖

```
        K (N×d)                           V (N×d)
    ┌───┬───┬───┬───┐                 ┌───┬───┬───┬───┐
    │K₁ │K₂ │K₃ │K₄ │                 │V₁ │V₂ │V₃ │V₄ │
    ├───┼───┼───┼───┤                 ├───┼───┼───┼───┤
    │   │   │   │   │                 │   │   │   │   │
    └───┴───┴───┴───┘                 └───┴───┴───┴───┘
          ▲                                 ▲
          │                                 │
          └────────────┬────────────────────┘
                       │
              Load blocks to SRAM
                       │
                       ▼
Q (N×d)          ┌─────────────┐
┌───┐            │             │
│Q₁ │──────────► │   SRAM      │ ──► O₁
├───┤            │  Compute    │
│Q₂ │──────────► │  Attention  │ ──► O₂
├───┤            │  Block by   │
│Q₃ │──────────► │  Block      │ ──► O₃
├───┤            │             │
│Q₄ │──────────► │             │ ──► O₄
└───┘            └─────────────┘
```

### 分塊大小選擇

設 SRAM 大小為 $M$，塊大小為 $B_r \times B_c$，需要滿足：

$$
B_r \cdot d + B_c \cdot d + B_r \cdot B_c \leq M
$$

最優塊大小約為：
- $B_c = \frac{M}{4d}$
- $B_r = \min\left(\frac{M}{4d}, d\right)$

### Causal Masking 的 Block-wise 優化

對於 GPT 這類 Decoder-only 模型，需要使用 Causal Mask（遮蔽未來 token）。
Naive dense 實作可能先計算右上三角區域再套 mask；其他 fused／triangular kernels
也可能跳過這些工作，所以這不是所有「標準 Attention」實作的必然行為。

FlashAttention 的 Tiling 策略帶來了一個額外優勢：**直接跳過無效的 Block**。

- 因為是分塊計算，如果某個 Block 完全落在 Mask 區域內（例如 $Q$ 的時間步小於 $K$ 的時間步），該 Block **完全不需要讀取和計算**。
- 對完整的方形 causal attention，跳過整個被遮蔽的 tiles 可省去接近一半的 score
  計算；邊界 tiles、kernel 排程與非方形 Q/K 長度會使實際加速不同。

---

## Online Softmax 技巧

Softmax 的計算有個麻煩：你需要知道「全班的成績」才能算出每個人的「相對排名」。具體來說，softmax 需要：
1. 找出全班最高分（用來做數值穩定）
2. 算出所有人分數的總和（用來做正規化）

**問題**：如果我們把資料分塊處理，處理第一塊時還不知道後面塊的資料，怎麼算出正確的 softmax？

**Online Softmax 的解法**：採用「邊走邊修正」的策略。
1. 先用目前已知的資料算出一個「暫時答案」
2. 當看到新的資料塊時，檢查是否需要更新最大值
3. 如果最大值變了，就用一個「修正係數」把之前的答案調整過來
4. 最後所有塊都處理完時，答案就是正確的

這就像改考卷時，你可以邊改邊算平均分，每次有新的分數進來就更新平均值，不需要等所有考卷都改完才開始算。

### 問題：如何分塊計算 Softmax？

標準 softmax 需要知道整行的最大值和總和：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j(x_j)
$$

### Online Softmax 演算法

FlashAttention 使用 **增量更新** 的方式計算 softmax：

```python
def online_softmax_attention():
    """
    Online Softmax 的核心思想：
    1. 維護運行中的 max 和 sum
    2. 當處理新的 block 時，更新這些統計量
    3. 使用 rescaling 來修正之前的計算
    """
    # 初始化
    m_i = -inf      # running max
    l_i = 0         # running sum of exp
    O_i = 0         # running output

    for j in range(num_blocks):
        # 載入 K_j, V_j 到 SRAM
        S_ij = Q_i @ K_j.T / sqrt(d)

        # 計算當前 block 的 max
        m_ij = max(S_ij)

        # 更新 running max
        m_new = max(m_i, m_ij)

        # Rescale 之前的結果
        l_i = l_i * exp(m_i - m_new) + sum(exp(S_ij - m_new))
        O_i = O_i * exp(m_i - m_new) + exp(S_ij - m_new) @ V_j

        # 更新 max
        m_i = m_new

    # 最終 normalize
    O_i = O_i / l_i
    return O_i
```

### 數學推導

對於兩個 block $x^{(1)}$ 和 $x^{(2)}$，設：
- $m^{(1)} = \max(x^{(1)})$, $m^{(2)} = \max(x^{(2)})$
- $m = \max(m^{(1)}, m^{(2)})$

則合併後的 softmax 為：

$$
\text{softmax}([x^{(1)}, x^{(2)}])_i = \frac{e^{x_i - m}}{\sum_j e^{x_j^{(1)} - m} + \sum_k e^{x_k^{(2)} - m}}
$$

可以透過 rescaling 從各自的 softmax 結果計算得到：

$$
\ell^{(1)} \cdot e^{m^{(1)} - m} + \ell^{(2)} \cdot e^{m^{(2)} - m}
$$

其中 $\ell^{(i)} = \sum_j e^{x_j^{(i)} - m^{(i)}}$

---

## Recomputation 重計算策略
在訓練神經網路時，反向傳播需要用到前向傳播的中間結果。傳統做法是把這些中間結果都存起來，但 Attention 的中間矩陣是 $N \times N$ 的，非常佔記憶體。

**FlashAttention 的策略**：乾脆不存這些中間結果，等到反向傳播需要時再重新算一遍。

**這樣不是更慢嗎？** 乍看之下，多算一遍應該更慢。但實際上：

1. **省下的 HBM 讀寫時間 > 多花的計算時間**
   - 儲存 $N \times N$ 矩陣需要大量 HBM 寫入
   - 讀取這些矩陣又需要大量 HBM 讀取
   - 這些 I/O 操作比重新計算還要慢

2. **記憶體省下來可以做更多事**
   - 可以用更大的 batch size
   - 可以處理更長的序列
   - 這些都能提升整體訓練效率

**類比**：與其把筆記存到遠處的倉庫再搬回來，不如當場重新寫一遍——因為搬運的時間比寫字還久。

### 為什麼需要重計算？

在反向傳播中，需要 $S = QK^T$ 和 $P = \text{softmax}(S)$ 來計算梯度。

**傳統方法**：儲存 $S$ 和 $P$（$O(N^2)$ 記憶體）

**FlashAttention**：只儲存 $O$、$\ell$（softmax normalizer）、$m$（max values）
- $m$ 和 $\ell$ 足以在反向傳播時還原出前向傳播的 Softmax 分佈 $P$，而無需完整儲存 $P$。
- 反向傳播時重新計算 $S$ 和 $P$
- 用額外的計算換取記憶體節省

### 記憶體需求對比

| 方法 | Attention 額外中間儲存 | 包含 Q/K/V/O 的量級 |
|-----|---------|---------|
| Naive Attention | $O(N^2)$ | $O(N^2 + Nd)$ |
| FlashAttention | $O(N)$ softmax statistics | $O(Nd)$ |

---

## FlashAttention-2 改進

FlashAttention-2 論文在其 A100 kernel benchmarks 中，將相對 theoretical maximum
FLOPs/s 的效率由 FlashAttention 的 **25～40%** 提升至 **50～73%**。這不是整張
GPU 的通用 utilization 指標，也不是任意模型的端到端提升。

### 主要改進

#### 改進 1：減少非矩陣乘法的 FLOPs
GPU 上有專門的硬體（Tensor Core）來加速矩陣乘法，速度極快。但其他運算（如 softmax 中的 exp、除法、比較大小）就只能用普通的 CUDA core，速度慢很多。

FlashAttention-1 的 online softmax 實作中，每處理一個新的 block 都要做一次 rescaling（乘以修正係數）。FlashAttention-2 重新設計了演算法：

- **延遲 rescaling**：不是每個 block 都修正，而是累積起來最後一次修正
- **減少中間計算**：重構公式，讓非矩陣乘法的運算量降低

這樣 Tensor Core 就能更充分地發揮作用，不會被其他慢速運算拖累。

---

#### 改進 2：更好的並行化策略
GPU 有很多「工人」（thread blocks）可以同時工作。問題是：怎麼分配工作才能讓所有工人都忙起來？

**FlashAttention-1 的問題**：
- 並行維度是 batch size × number of heads
- 如果 batch size 小（例如推理時只有 1），很多工人會閒著
- 長序列時效率特別差

**FlashAttention-2 的改進**：
- 額外沿著序列長度維度並行
- 即使 batch size = 1，只要序列夠長，就能讓所有工人都有事做
- 特別適合現在流行的長上下文模型（32K、128K tokens）

這就像工廠只有 4 條產品線（4 heads），但有 100 個工人。與其讓 96 個工人閒著，不如把每條產品線的工作再細分，讓大家都能幫忙。

---

#### 改進 3：優化 Work Partitioning（工作分配）

GPU 的工人分成小組（warp，32 個 thread 一組），同組的工人必須做一樣的事。如何在小組內分配工作，會大幅影響效率。

**FlashAttention-1 的問題**：
- 一個 warp 內的 threads 需要頻繁交換資料（透過 shared memory）
- 這些讀寫和同步操作會造成等待

**FlashAttention-2 的改進**：
- 重新設計資料分配方式，讓每個 thread 能獨立完成更多工作
- 減少 shared memory 的讀寫次數
- 減少 threads 之間需要「對答案」（同步）的次數

這就像團隊合作時，與其每做一步就開會討論，不如事先分好工，各自獨立完成後再彙整。

---

## FlashAttention-3 改進

FlashAttention-3 針對 NVIDIA Hopper GPU (H100) 進行了深度優化。

### 主要特性

#### 特性 1：WGMMA (Warpgroup Matrix Multiply-Accumulate)
Hopper 引入 warpgroup-level asynchronous matrix multiply instructions。FlashAttention-3
利用 WGMMA 與 Tensor Memory Accelerator（TMA），讓 GPU warpgroups 的矩陣運算
和 global/shared memory 資料搬運重疊。這是 GPU kernel 內部的非同步 pipeline，
不是「CPU 發一條指令後等待 GPU」的流程。

---

#### 特性 2：Ping-pong Scheduling

Kernel 以 warp specialization 分配 producer 與 consumer 工作，並讓 consumer
warpgroups 交錯執行 matmul 和 softmax。這可隱藏部分資料搬運與非 matmul 運算
延遲，但收益取決於 shape、dtype 與硬體，不能理解為 GPU 永遠沒有 idle cycles。

---

#### 特性 3：FP8 支援
FP8 是一種只用 8 位元表示的浮點數（相比 FP16 的 16 位元或 FP32 的 32 位元）。

**潛在優勢**：FP8 operand payload 約為 FP16 的一半，且 Hopper Tensor Cores 有
相應低精度吞吐能力；這不表示 attention 的數學 FLOPs 減半，也不保證端到端速度
或可用 context 必然翻倍。

**挑戰**：FP8 精度很低，容易造成數值問題

**FlashAttention-3 的做法**：
- 矩陣乘法用 FP8（這是計算密集的部分）
- Softmax 等敏感操作仍用 FP32
- **Block-wise Quantization**：不像傳統量化是對整個矩陣用同一組參數，FlashAttention 對每個小 block 單獨計算量化參數。這樣能大幅減少量化誤差，因為每個 block 的數值分佈比較集中，量化後的失真較小。

這就像省錢策略：大宗採購用折扣價，但關鍵食材還是買好的。而且每個部門（Block）有自己的預算控制（Block-wise Quantization），比全公司統一預算更精準。

---

#### 特性 4：Incoherent Processing
這裡的 incoherent processing 不是取消 $QK^T \rightarrow softmax \rightarrow PV$
的資料依賴。FlashAttention-3 對輸入套用隨機正交變換，使 outliers 分散後再做
FP8 block quantization；正交變換保持 attention 所需的數學關係，目的是降低低精度
量化誤差。計算與資料搬運的重疊則屬於前述 asynchrony／warp specialization。

### 效能提升

FlashAttention-3 論文在其 H100 測試設定中，報告 FP16 相對 FlashAttention-2
有 1.5～2.0 倍速度提升，最高約 740 TFLOPs/s；FP8 接近 1.2 PFLOPs/s。這是指定
GPU、kernel shapes 與 benchmark 的結果，不應直接外推成任意模型的端到端加速。

---

## 實作範例

### 使用 FlashAttention

```python
# 安裝
# pip install flash-attn --no-build-isolation

import torch
from flash_attn import flash_attn_func

# 基本用法
batch_size = 2
seq_len = 2048
num_heads = 32
head_dim = 64

# 輸入格式: (batch, seqlen, nheads, headdim)
q = torch.randn(batch_size, seq_len, num_heads, head_dim,
                dtype=torch.float16, device='cuda')
k = torch.randn(batch_size, seq_len, num_heads, head_dim,
                dtype=torch.float16, device='cuda')
v = torch.randn(batch_size, seq_len, num_heads, head_dim,
                dtype=torch.float16, device='cuda')

# FlashAttention 計算
output = flash_attn_func(q, k, v, causal=True)
# output shape: (batch, seqlen, nheads, headdim)
```

### 使用 xFormers

```python
# pip install xformers

import torch
from xformers.ops import memory_efficient_attention

# 輸入格式: (batch, seqlen, nheads, headdim)
q = torch.randn(2, 2048, 32, 64, dtype=torch.float16, device='cuda')
k = torch.randn(2, 2048, 32, 64, dtype=torch.float16, device='cuda')
v = torch.randn(2, 2048, 32, 64, dtype=torch.float16, device='cuda')

# Memory efficient attention (包含 FlashAttention backend)
output = memory_efficient_attention(q, k, v)
```

### PyTorch 原生 SDPA

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# PyTorch 2.0+ 內建 scaled_dot_product_attention
# 會自動選擇最佳後端 (包括 FlashAttention)

q = torch.randn(2, 32, 2048, 64, dtype=torch.float16, device='cuda')
k = torch.randn(2, 32, 2048, 64, dtype=torch.float16, device='cuda')
v = torch.randn(2, 32, 2048, 64, dtype=torch.float16, device='cuda')

# 僅允許 FlashAttention backend；不相容時會直接報錯，適合驗證 backend。
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

## 效能分析

### 時間複雜度

| 操作 | 標準 Attention | FlashAttention |
|-----|---------------|----------------|
| 計算 FLOPs | $O(N^2 d)$ | $O(N^2 d)$ |
| HBM 存取 | $O(N^2 + Nd)$ | $O(N^2 d^2 M^{-1})$ |

當 $M = \Theta(Nd)$ 時，上式化簡為 $O(Nd)$；若把 head dimension $d$ 視為常數，
才可簡寫成對序列長度 $N$ 的線性 I/O。

### 記憶體複雜度

| 方法 | Attention 額外中間儲存 | 包含 Q/K/V/O 的量級 |
|-----|---------|---------|
| Naive Attention | $O(N^2)$ | $O(N^2 + Nd)$ |
| FlashAttention | $O(N)$ softmax statistics | $O(Nd)$ |

實際加速不應套用固定倍數。應在相同 dtype、shape、causal mask、dropout、warmup
與同步方式下，比較 kernel latency、peak allocated memory 與端到端模型吞吐；短序列
或不相容 shape 可能讓 launch／dispatch overhead 主導。

---

## 限制與注意事項

1. **硬體要求**
   - 支援範圍取決於套件與版本。官方 FlashAttention-2 CUDA 實作目前主要支援
     Ampere、Ada、Hopper；Turing 使用另一個功能子集 repo，另有 ROCm backends。
   - FlashAttention-3 針對 Hopper；不能把 FA3 的 H100 結果外推到其他 GPU。

2. **精度要求**
   - 官方 CUDA FlashAttention-2 主要支援 FP16/BF16；FA3 另有 Hopper FP8 forward。
   - Backend 支援不等於模型品質不變，低精度路徑仍需數值驗證。

3. **注意力變體支援**
   - 標準 attention ✓
   - Causal attention ✓
   - 某些複雜的 attention mask 可能不支援

4. **Head dimension 限制**
   - 官方 CUDA FlashAttention-2 目前支援 head dimension 至 256；其他 backend、
     backward、dropout 與 GPU 組合可能有不同限制，應查當下版本 feature matrix。

---

## 參考資料

1. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
2. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
3. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
4. [FlashAttention official implementation and support matrix](https://github.com/Dao-AILab/flash-attention)

## 延伸閱讀

- [FlashAttention 與 PagedAttention：算子最佳化和 KV cache 管理](./flash-attn-vs-paged-attn.md)
