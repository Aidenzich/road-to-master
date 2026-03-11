# Flash Attention vs. Paged Attention 架構解析





## 一、 核心對比與技術定位

| 對比維度 | Flash Attention (運算加速/IO最佳化) | Paged Attention (記憶體管理) |
| :--- | :--- | :--- |
| **解決痛點** | GPU 記憶體頻寬瓶頸 (Memory Wall / IO-Bound) | KV Cache 顯存碎片化與利用率低下 |
| **核心目標** | 讓運算「跑得更快」 (降低延遲，提升計算極限) | 讓顯存「裝得下更多」 (提升 Batch Size 與吞吐量) |
| **作用焦點** | Attention 矩陣機制的底層算子最佳化 (Kernel 層級)| LLM 生成過程中的上下文 (KV Cache) 分配機制 |
| **發揮優勢階段**| Prefill (預填充/編碼) 階段效率極高 | Decode (解碼/生成) 階段顯存管線的核心 |
| **核心技術手段**| Tiling (分塊計算)、Kernel Fusion (算子融合) | OS 分頁機制 (Paging)、Block Table 虛擬映射 |

---

## 二、 Flash Attention：突破讀寫瓶頸 (IO-Aware)

**核心概念**：把「草稿」留在腦袋裡算完，不寫在紙上。避免 GPU 運算單元（SRAM）頻繁去慢速的大容量顯存（HBM）讀寫短暫的過程數據。

1. **算子融合與分塊計算 (Tiling)**：
   傳統 Attention 會將龐大的 $N \times N$ 注意力分數矩陣物化（Materialize）並寫入 HBM，再讀出來做 Softmax。Flash Attention 將 $Q, K, V$ 切成小塊 (Blocks) 載入高速但極小的 **SRAM**，在內部直接完成所有運算，**只將最終結果寫回 HBM**。
2. **記憶體複雜度降維**：
   將 Attention 的顯存佔用從暴衝的 $O(N^2)$ 降到了線性 $O(N)$，同時極大地減少了 HBM 的讀寫次數 (Memory Accesses)，有效解決 IO-Bound 瓶頸。

---

## 三、 Paged Attention：解放顯存碎片化

**核心概念**：借鑑作業系統的「虛擬記憶體分頁」，動態為無法預知長度的對話分配空間。

1. **痛點：未知的生成長度**：
   傳統 LLM 生成文字時，需為每筆請求預先分配「最大可能長度」的連續顯存（KV Cache）。這會產生巨大的內部與外部碎片，導致實際顯存利用率通常不到 30%。
2. **非連續存儲 (Paging Mechanisms)**：
   將 KV Cache 切分成固定大小的單位 **Blocks**（例如每個 Block 存 16 個 Token 的 K 與 V）。
3. **邏輯與物理映射 (Block Table)**：
   系統在 CPU 記憶體中維護一張 **Block Table（類似 OS 的分頁表）**，負責記錄 Key-Value 映射：將對話的「邏輯 Block 編號」對應到 GPU 顯存中的「物理 Block 索引」。**注意：表內只存指標，真正的張量數據（KV Cache）皆存放於顯存中。**
   * **分配時機**：在 Prefill（預填充）階段，如果 Prompt 有 32 個 Token，系統會**一次性**分配 2 個 Block（假設 1 Block = 16 Token）並記錄在 Block Table 中。進入 Decode（生成）階段後，每生成一個新 Token 就塞進最後一個 Block，<mark>直到最後一個 Block 塞滿了，系統才會去顯存池再要一個新的 Block</mark>。最糟的情況也只會浪費「最後一個 Block 沒裝滿」的零頭空格（浪費 < 4%）。
   * **註銷時機（釋放與回收）**：當該筆 Request 徹底結束時（例如模型生成了 `<EOS>` 結束符號、達到長度上限，或客戶端主動中斷），vLLM 的調度器 (Scheduler) 就會將這張 Block Table 裡所有記錄的「物理 Block 指標」，歸還給 GPU 的「空閒池 (Free Pool)」。完成指標釋放後，這張專屬該 Request 的 Block Table 就會從 CPU 記憶體中被註銷刪除。
4. **極致的顯存利用率（消滅碎片化）**：
   * **消滅內部碎片 (Internal Fragmentation)**：傳統為了防範生成過長，必須一開始就為每個 Request 預先切出一大塊（例如 2048 token）的連續顯存。如果最後只生成了 100 個 token，剩下 1948 的空間就被白白鎖死（這就是巨大浪費）。有了 Block 和 Block Table 後，變成**用多少拿多少**。
   * **消滅外部碎片 (External Fragmentation)**：傳統的連續分配，會在顯存池中留下許多不規則大小的「空隙」（像是裝箱時剩下的畸零空間），這些空隙太小放不下新 Request。但在 Paged Attention 中，所有的 Block 都是**一模一樣大**的標準規格。任何一個被釋放的 Block，都能立刻被其他 Request 拿去拼圖，顯存空間可以被 100% 嚴密榨乾。

---

## 四、 兩者在 vLLM 中的協作 (The Big Picture)

在先進的 LLM 推理框架（如 vLLM）中，這兩者並非競爭，而是**上下層的緊密結合**：

* **資源調度 (Paged Attention)**：
  負責「資源調度」。框架會決定每個 Request 的 KV Cache 該放在物理顯存的哪些**不連續 Blocks** 中，並維護 Block Table。
* **運算加速 (Flash Attention / Paged-Flash Attention)**：
  負責「極速運算」。當 Decode 階段需要計算 Attention 時，會拿著 Block Table 把這些分散的 KV Cache 讀入 SRAM，並套用 Flash Attention 的 Tiling 技巧快速算出結果。

---

## 五、 如何在日誌中確認生效

在啟動 vLLM 或其他推理引擎時，可透過日誌觀察後端算子的調用情況。若硬體支援，系統會自動選用最佳算子組合：

```bash
# 查看啟動日誌中是否包含以下字樣
INFO: vLLM is using PagedAttention.
INFO: Using FlashAttention-2 backend.
# 此時代表你的框架正在完美運作：PA 管理顯存 + FA 加速算子
```