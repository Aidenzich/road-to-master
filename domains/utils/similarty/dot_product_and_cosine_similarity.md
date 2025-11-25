# Dot-Product vs Cosine Similarty
## 數學公式定義

我們假設有兩個 $n$ 維向量 $A$ 和 $B$：
* $A = [a_1, a_2, ..., a_n]$
* $B = [b_1, b_2, ..., b_n]$

### 內積 (Dot Product)

內積，也稱為點積，有兩種定義方式：

**a) 代數定義 (計算方式):**
將兩個向量對應的維度相乘後全部加總。

$$
A \cdot B = \sum_{i=1}^{n} (a_i \times b_i) = a_1b_1 + a_2b_2 + ... + a_nb_n
$$

**b) 幾何定義 (物理意義):**
一個向量在另一個向量上的投影長度，再乘以另一個向量的長度。

$$
A \cdot B = \|A\| \|B\| \cos(\theta)
$$

* $\|A\|$ 和 $\|B\|$ 是 $A$ 和 $B$ 的**L2 範數 (L2 Norm)**，也就是它們各自的「歐幾里得長度」。
* $\theta$ (theta) 是 $A$ 和 $B$ 之間的夾角。

### 餘弦相似度 (Cosine Similarity)

餘弦相似度就是上述幾何定義中的 $\cos(\theta)$。我們只需要對公式進行移項：

$$
\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
$$

**將 (a) 和 (b) 結合，我們得到最完整的計算公式：**

$$
\text{Cosine Similarity}(A, B) = \frac{\sum_{i=1}^{n} (a_i \times b_i)}{\sqrt{\sum_{i=1}^{n} a_i^2} \times \sqrt{\sum_{i=1}^{n} b_i^2}}
$$

* **分子 ($A \cdot B$)：** 就是內積。
* **分母 ($\|A\| \|B\|$)：** 是兩個向量的 L2 範數 (長度) 的乘積。



## 為什麼 L2 正規化後，內積 = 餘弦相似度？

這個推導非常簡單且重要：

1.  **什麼是 L2 正規化 (L2 Normalization)？**
    L2 正規化是**將一個向量 $A$ 轉換為一個新的「單位向量」$A'$**，使得 $A'$ 的方向與 $A$ 相同，但其 L2 範數 (長度) **恰好等於 1**。

    操作方法是：將向量 $A$ 除以它自己的 L2 範數 $\|A\|$：
    $$A' = \frac{A}{\|A\|}$$

    *因此，L2 正規化後的向量 $A'$，其長度 $\|A'\|$ 必為 1。*

2.  **將正規化後的向量 $A'$ 和 $B'$ 進行內積：**
    我們有兩個新向量：
    * $A' = \frac{A}{\|A\|}$
    * $B' = \frac{B}{\|B\|}$

    現在，我們計算 $A' \cdot B'$ (新向量的內積)：
    $$A' \cdot B' = \left( \frac{A}{\|A\|} \right) \cdot \left( \frac{B}{\|B\|} \right)$$

    因為 $\|A\|$ 和 $\|B\|$ 只是純數字 (純量)，我們可以把它們提出來：
    $$A' \cdot B' = \frac{A \cdot B}{\|A\| \|B\|}$$

3.  **結論：**
    請看上一步的結果：$\frac{A \cdot B}{\|A\| \|B\|}$
    這**不就等於**我們在第一點中定義的 $\text{Cosine Similarity}(A, B)$ 嗎？

**因此，我們證明了：**
$$\text{DotProduct}(A_{normalized}, B_{normalized}) = \text{CosineSimilarity}(A_{original}, B_{original})$$

**用幾何定義來理解更直觀：**
* 根據定義，$\text{DotProduct}(A', B') = \|A'\| \|B'\| \cos(\theta)$
* 因為 $A'$ 和 $B'$ 都經過 L2 正規化，它們的長度 $\|A'\| = 1$ 且 $\|B'\| = 1$。
* 所以，$\text{DotProduct}(A', B') = 1 \times 1 \times \cos(\theta) = \cos(\theta)$
* 而 $\cos(\theta)$ 就是 $A$ 和 $B$ 之間的餘弦相似度。


## 補充. 比較 L2 Regularization vs. L2 Normalization

| 概念 | L2 Regularization (L2 懲罰項 / 權重衰減) | L2 Normalization (L2 正規化 / 單位向量化) |
| :--- | :--- | :--- |
| **目 的** | **訓練 (Training) 時**，防止模型過擬合 (Overfitting)。 | **推論 (Inference) 或儲存時**，使向量長度為 1。 |
| **應 用** | 作為一個**懲罰項**加入 **Loss Function** 中。 | 一個**資料預處理**步驟，應用在模型的**輸出 (Embeddings)**上。 |
| **對 象** | 模型的**權重 (Weights)**。 | 模型的**輸出向量 (Embeddings)**。 |
| **公 式** | $Loss_{new} = Loss_{original} + \lambda \|W\|^2$ | $V_{new} = \frac{V_{original}}{\|V_{original}\|}$ |
| **RAG 應用** | 用於 *訓練* Embedding 模型本身，使其更強健。 | 用於 *處理* Embedding 模型的 *輸出*，以便存入向量 DB。 |

