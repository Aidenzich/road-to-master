# Derivative of Softmax (Jacobian Matrix) when i = j

$$
\frac{\partial y_i}{\partial x_i} = y_i (1 - y_i)
$$

## 推導過程
### 1\. Softmax 公式
$$
y_i = \frac{e^{x_i}}{\sum_{k} e^{x_k}} = \frac{e^{x_i}}{\textcolor{cyan}{\Sigma}}
$$
為方便呈現，我們用 $\Sigma$ 代表 $\sum_{k} e^{x_k}$

### 2\. 使用除法法則
我們要對 $x_i$ 求導數。回憶微積分公式： 
$$
(\frac{u}{v})' = \frac{u'v - uv'}{v^2}
$$

代入：
* 分子 $u = e^{x_i}$
* 分母 $v = \Sigma$

### 3\. 開始微分
針對 $x_i$ 微分：
1.  **分子的微分 ($u'$)：**
    $$\frac{\partial}{\partial x_i}(e^{x_i}) = e^{x_i}$$
2.  **分母的微分 ($v'$)：**
    $$\frac{\partial}{\partial x_i}(\Sigma) = \frac{\partial}{\partial x_i}(e^{x_1} + ... + e^{x_i} + ...) = e^{x_i}$$
    *(注意：分母雖然是一堆東西相加，但除了 $e^{x_i}$ 這項之外，其他項對 $x_i$ 來說都是常數，微分後為 0)*

### 4\. 套入公式計算
$$
\frac{\partial y_i}{\partial x_i} = \frac{(\textcolor{cyan}{e^{x_i}})(\Sigma) - (\textcolor{cyan}{e^{x_i}})(e^{x_i})}{\Sigma^2}
$$

$$
= \frac{ \textcolor{cyan}{e^{x_i}}(\Sigma - e^{x_i})}{\Sigma^2}
$$

$$
= \frac{e^{x_i}}{\textcolor{cyan}{\Sigma}} \cdot \frac{\Sigma - e^{x_i}}{\textcolor{cyan}\Sigma}
$$

### 5\. 替換回 $y_i$
$$
= \textcolor{yellow}{\frac{e^{x_i}}{\Sigma}} \cdot \textcolor{magenta}{\frac{\Sigma - e^{x_i}}{\Sigma}}
$$
* 第一部分 $\frac{e^{x_i}}{\Sigma}$ 就是 **$y_i$**。
* 第二部分 $\frac{\Sigma - e^{x_i}}{\Sigma} = 1 - \frac{e^{x_i}}{\Sigma}$，也就是 **$1 - y_i$**。

所以得到最終公式：
$$
\frac{\partial y_i}{\partial x_i} = y_i (1 - y_i)
$$



### 補充：這個公式的物理意義 (為什麼會飽和？)

這個導數公式 $y_i(1-y_i)$ 是一個開口向下的拋物線：

1.  **當 $y_i = 0.5$ 時：**
    梯度 $= 0.5 \times (1 - 0.5) = 0.25$ (這是梯度的最大值，更新最快)。
2.  **當 $y_i$ 接近 1 時 (例如 0.99)：**
    梯度 $= 0.99 \times 0.01 = 0.0099$ (梯度非常小)。
3.  **當 $y_i$ 接近 0 時 (例如 0.01)：**
    梯度 $= 0.01 \times 0.99 = 0.0099$ (梯度非常小)。

**這證明了：**
如果你的輸入 $x_i$ 太大，導致輸出 $y_i$ 變成極端的 0 或 1，**梯度就會自動變成 0**。這就是為什麼我們必須做 Scaling，把數值拉回中間，讓 $y_i$ 盡量不要太早衝到 0 或 1 的位置。