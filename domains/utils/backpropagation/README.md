# Backpropagation
Backpropagation 是一種「歸咎 (Attribution) 機制」，它利用微積分的**連鎖律 (Chain Rule)**，將最終的總誤差 (Loss) 一層層往回推，算出每一個參數該對這個錯誤「負多少責任 (Gradient)」，並據此要求它們進行修正。


## Chain Rule Review
連鎖律 (Chain Rule) 是微積分中用來計算「複合函數」導數的規則，在深度學習中，它是 Backpropagation 的數學靈魂: 如果你想知道 A 對 C 的影響，你可以先把『A 對 B 的影響』乘以『B 對 C 的影響』。

假設有三個變數像骨牌一樣連鎖反應：
$$x \to y \to z$$
( $x$ 影響 $y$， $y$ 影響 $z$ )

我們想知道 **$x$ 變動一點點，會導致 $z$ 變動多少？** ($\frac{\partial z}{\partial x}$)

連鎖律告訴我們：
$$
\frac{\partial z}{\partial x} = \underbrace{\frac{\partial z}{\partial y}}_{\text{z 對 y 的敏感度}} \times \underbrace{\frac{\partial y}{\partial x}}_{\text{y 對 x 的敏感度}}
$$


1.  **第一層 (x -> y):**
    * 規則：$x$ 每增加 **1**，$y$ 就會增加 **2**。
    * $\frac{\partial y}{\partial x} = 2$

2.  **第二層 (y -> z):**
    * 規則：$y$ 每增加 **1**，$z$ 就會增加 **3**。
    * $\frac{\partial z}{\partial y} = 3$

3.  **連鎖結果 (x -> z):**
    * 如果 $x$ 增加 **1**，那麼 $y$ 變 **2**。
    * 因為 $y$ 變了 **2**，而 $z$ 是 $y$ 的 3 倍放大，所以 $z$ 最終增加了 $2 \times 3 = \mathbf{6}$。

**結論：** $x$ 對 $z$ 的影響力（梯度）就是 $2 \times 3 = 6$。
這就是連鎖律：**把路徑上所有的「倍率（導數）」乘起來。**


## Backpropagation 的數學公式
我們用數學公式把這個梯度(Gradient)怎麼一層一層往回跑的過程寫出來。

這會讓我們非常清楚地看到 **殘差連結 (Residual Connection)** 裡的那個 $\mathbf{1}$ 是如何保護梯度的。


### 1. 定義符號 (Setup)

假設我們有一個 **$n$ 層** 的神經網路。

* **$x_i$**：第 $i$ 層的**輸入**（也就是第 $i-1$ 層的輸出）。
* **$F_i$**：第 $i$ 層的變換函數（包含 Attention 或 FFN）。
* **殘差公式**：每一層的輸出是 $x_{i+1} = x_i + F_i(x_i)$。
* **$L$**：最終的 Loss（總誤差）。

我們的目標是算出 **第 $i$ 層的梯度** $\frac{\partial L}{\partial x_i}$，也就是 Loss 對這一層輸入的梯度

### 2. 從最後一層開始 (The Start)

假設最後一層輸出是 $x_{n+1}$。我們算出它跟標準答案的誤差梯度：
$$
g_{n+1} = \frac{\partial L}{\partial x_{n+1}}
$$
這是我們反向傳播的**起點**。

### 3. 往回推一層 (Step $n \to n-1$)

現在我們要算第 $n$ 層的輸入梯度 $g_n = \frac{\partial L}{\partial x_n}$。

根據連鎖律 (Chain Rule)：
$$
g_n = \underbrace{g_{n+1}}_{\text{從} n+1 \text{層 }} \cdot \frac{\partial x_{n+1}}{\partial x_n}
$$

代入殘差公式 $x_{n+1} = x_n + F_n(x_n)$：
$$
\frac{\partial x_{n+1}}{\partial x_n} = \frac{\partial (x_n + F_n(x_n))}{\partial x_n} = \mathbf{I} + F'_n(x_n)
$$
*(註：$\mathbf{I}$ 是單位矩陣，也就是 Residual Connection 的那個 1. 它確保了梯度不會消失)*

所以，第 $n$ 層的梯度是：
$$
g_n = g_{n+1} \cdot (\mathbf{I} + F'_n)
$$

---

### 4. 展開到任意第 $i$ 層 (General Case)

如果我們繼續往回推，推到第 $i$ 層，會發現這是一個連乘的過程。

第 $i$ 層的梯度 $g_i$ 會等於 **最後的梯度 $g_{n+1}$** 乘上 **沿途每一層的 Jacobian 矩陣**：

$$
g_i = g_{n+1} \cdot \underbrace{(\mathbf{I} + F'_n) \cdot (\mathbf{I} + F'_{n-1}) \cdots (\mathbf{I} + F'_i)}_{\text{從第 } n \text{ 層一路乘回第 } i \text{ 層}}
$$

用數學連乘符號 ($\prod$) 表示：

$$
g_i = g_{n+1} \cdot \prod_{k=i}^{n} (\mathbf{I} + F'_k(x_k))
$$


### 5. 為什麼這條公式證明了 Residual Connection 的「高速公路」理論？
假設只有兩層 ($k=1, 2$)，則傳播路徑如下：
$$
\text{Loss} \to x_3 \to x_2 \to x_1
$$



梯度傳回第 1 層時，$g_1$ 等於：
$$
g_1 = g_{3} \cdot (\mathbf{I} + F'_2) \cdot (\mathbf{I} + F'_1)
$$

把它乘開（分配律）：
$$
g_1 = g_{3} \cdot (\mathbf{I} \cdot \mathbf{I} + \mathbf{I} \cdot F'_1 + F'_2 \cdot \mathbf{I} + F'_2 \cdot F'_1)
$$

整理後得到：
$$
g_1 = \underbrace{g_{3}}_{\text{1. 直達車}} + \underbrace{g_{3} F'_1 + g_{3} F'_2}_{\text{2. 短程路徑}} + \underbrace{g_{3} F'_2 F'_1}_{\text{3. 長程路徑}}
$$

#### 這裡揭示了 Residual Connection 的數學結構：
1.  **直達車項 ($g_3$)：**
    公式裡直接包含了一個原始的 $g_{3}$。
    這意味著，最後一層的誤差訊號，完全沒有乘以任何 $F'$（權重變化），**原封不動地 (Losslessly)** 傳到了第 1 層。這就是那條「高速公路」。即便 $F'$ 很小（梯度消失），這一項保證了 $g_1$ 絕對不會是 0。

2.  **短程路徑 ($g_3 F'_1 + g_3 F'_2$)：**
    這代表梯度在傳遞過程中，**有的層走了捷徑（High Way），有的層走了普通道路（Function）**。

    * **$g_3 F'_2$**：梯度經過第 2 層的變換 ($F'_2$)，但在第 1 層走了捷徑 ($\mathbf{I}$)。這相當於一個「只有第 2 層」的淺層網路。
    * **$g_3 F'_1$**：梯度在第 2 層走了捷徑 ($\mathbf{I}$)，但在第 1 層經過了變換 ($F'_1$)。這相當於一個「只有第 1 層」的淺層網路。
    
    這揭示了 ResNet 的真正強大之處——**它不只是一個深層網路，它是無數個「淺層網路」的集成 (Ensemble)**。
    透過殘差連結，一個 100 層的網路，其實同時包含了 1 層、2 層、...99 層的各種路徑組合。這讓模型即使某幾層壞掉了，其他路徑依然能運作，極大增加了訓練的穩定性。

3.  **長程路徑項 ($g_3 F'_2 F'_1$)：**
    這就是傳統神經網路（沒有殘差連結）唯一的路徑。如果 $F'_2$ 和 $F'_1$ 都小於 1（例如 0.1），乘起來就變成 $0.01$，梯度就消失了。但在殘差網路裡，這只是梯度的一小部分而已。





## $g_i$ 是如何用來**更新權重 $W_i$** 的。
雖然我們計算的是 $g_i = \frac{\partial L}{\partial x_i}$（傳給上一層的梯度），但我們真正的目標是要更新**這一層**的權重 $W_i$。

**權重更新的梯度**是 $\frac{\partial L}{\partial W_i}$，它是透過 $g_{i+1}$ 與這一層的輸入 $x_i$ 結合，經過另一次 Chain Rule 得到的：
$$
\frac{\partial L}{\partial W_i} = g_{i+1} \cdot \frac{\partial F_i(x_i)}{\partial W_i}
$$

最終就是利用這個 $\frac{\partial L}{\partial W_i}$ 來執行 
**梯度下降 (Gradient Descent)**：
$$
W_i^{\text{new}} = W_i^{\text{old}} - \eta \cdot \frac{\partial L}{\partial W_i}
$$
- $\eta$ 是學習率

### 梯度的計算與權重的更新是在同一層同時完成的

**在第 $i$ 層 (從 $n \to 1$ 往回走) 執行的動作：**

| 順序 | 動作 | 目的 |
| :--- | :--- | :--- |
| **I.** | **接收** $g_{i+1}$ (已知) | 收到來自後面的回傳梯度 |
| **II.** | **計算並更新** $W_i$ | 處理自己的權重更新 ($\frac{\partial L}{\partial W_i}$) |
| **III.** | **計算並傳遞** $g_i$ | 算出要給前一層的回傳梯度 ($\frac{\partial L}{\partial x_i}$) |
| **IV.** | **結束** | 前往 $i-1$ 層 |

**數學步驟如下：**
1.  **計算 Weight 梯度 ($\frac{\partial L}{\partial W_i}$) 與執行更新:**
    $$\frac{\partial L}{\partial W_i} = \underbrace{g_{i+1}}_{\text{後一層傳回的梯度}} \cdot \underbrace{\frac{\partial x_{i+1}}{\partial W_i}}_{\text{自己對輸出的敏感度}}$$
    - 這一步算出 $W_i$ 應變動的方向和程度
    $$
    W_i^{\text{new}} = W_i^{\text{old}} - \eta \cdot \frac{\partial L}{\partial W_i}
    $$
    - 接著[執行梯度下降 (Gradient Descent)](../optimizer/gradient_descent.md)
    - $\eta$ 是學習率 (Learning Rate)。
    - 這一步是將計算出的梯度乘上步幅 $\eta$，然後從舊權重中減去，完成參數的修正。



2.  **計算要傳遞給上一層的梯度 ($g_i$):**
    $$g_i = \underbrace{g_{i+1}}_{\text{後一層傳回的梯度}} \cdot \underbrace{\frac{\partial x_{i+1}}{\partial x_i}}_{\text{這一層的傳遞倍率}}$$
    *（這一步算出要給 $i-1$ 層的梯度 $g_i$。）*

**小結：** Backpropagation 是一個單次的連續過程。站在第 $i$ 層時，同時用 **後一層回傳的梯度 ($g_{i+1}$)** 完成了 **Weight 修正** 和 **Output 傳遞 ($g_i$)** 兩項任務。


**雙重職責 (Dual Responsibility)**：

| 職責 | 核心目的 | 數學表達 |
| :--- | :--- | :--- |
| **更新權重** | **內部學習 (Local Learning)** | 計算 $\frac{\partial L}{\partial W_i}$，用於**改變**該層的知識。 |
| **傳遞梯度** | **全域傳播 (Global Propagation)** | 計算 $\frac{\partial L}{\partial x_i}$，用於**通知**前一層。 |



## 常見考題
> **「請精確描述在神經網路的訓練過程中，反向傳播 (Backpropagation) 是如何更新某一單層 $i$ 的權重 $W_i$ 的？」**

### 🚨 典型的錯誤答案 (證明理解不完整)

| 錯誤類型 | 回答內容 | 缺失的環節 |
| :--- | :--- | :--- |
| **只答階段 2** | 「Backpropagation 就是直接把算出來的梯度 $g$ 乘上學習率 $\eta$，然後從 $W$ 減掉。」 | 忽略了階段 1：**沒有解釋梯度 $g$ (即 $\frac{\partial L}{\partial W_i}$) 是怎麼來的**，也沒有提到 Chain Rule。 |
| **只答階段 1** | 「Backpropagation 是用連鎖律算出 $\frac{\partial L}{\partial W}$。」 | 忽略了階段 2：**沒有將梯度應用到 $W$ 上**，等於「找到了問題，但沒有解決問題」。 |
| **流程混淆** | 「它先算出 $\frac{\partial L}{\partial x}$，然後把 $\frac{\partial L}{\partial x}$ 乘上 $x$ 來更新 $W$。」 | **混淆了 $\frac{\partial L}{\partial x}$ 和 $\frac{\partial L}{\partial W}$** 的職責。 $\frac{\partial L}{\partial x}$ 是給前一層的，不是給 $W$ 用的。 |




