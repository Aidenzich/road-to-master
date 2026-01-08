# Gradient Descent

這篇文章將介紹梯度下降 (Gradient Descent) 的基本概念，並展示如何使用 Python 實現一個簡單的梯度下降演算法。

**梯度下降 (Gradient Descent)** 是一種優化演算法，它通過計算**損失函數（Loss）的負梯度**，指導參數 $W$ 和 $B$ 朝著錯誤下降最快的方向，以特定的步幅（學習率）進行迭代修正。

**下降 (Descent)** 這個詞特指：將計算出來的梯度 (方向是 Loss 增加最快的方向) 乘上 $-1$，然後用這個「負梯度」來修正權重。



## 1\. 數學模型 (線性迴歸)

我們的目標是找到一條直線 $y = wx + b$ 來擬合 (fit) 我們的資料。

* **模型 (Model / Hypothesis):**
$$y_{pred} = w \cdot x + b$$

* **參數 (Parameters):** 
我們要學習的權重是 $w$ (斜率) 和 $b$ (截距)。

* **損失函數 (Loss Function):**
我們使用**均方誤差 (Mean Squared Error, MSE)** 來衡量預測的好壞。我們的目標是最小化這個 $L(w, b)$。
$$L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (y_{pred}^{(i)} - y_{true}^{(i)})^2$$
將模型代入，得到：
$$L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (\textcolor{red}{(w \cdot x^{(i)} + b)}  - y_{true}^{(i)})^2$$



## 2\. 梯度的數學表達

「梯度」就是損失函數 $L$ 對所有參數 ($w$ 和 $b$) 的偏微分。這告訴我們在目前的位置，往哪個方向調整參數能讓 Loss 下降最快。

#### A. 對 $w$ 的偏微分 ($\frac{\partial L}{\partial w}$)


把線性回歸推導出的數學模型帶入 $\frac{\partial L}{\partial w}$：
$$
\frac{\partial L}{\partial w} = \frac{\partial}{\partial w} \textcolor{cyan}{ \left[ \frac{1}{N} \sum ((w \cdot x + b) - y)^2 \right]}
$$

利用微分的線性性（linearity of differentiation) 把 $\textcolor{cyan}{\frac{1}{N} \sum }$ 移出:
$$
= \textcolor{cyan}{\frac{1}{N} \sum } \frac{\partial}{\partial w} \left[ \textcolor{lime}{(w \cdot x + b - y)^2} \right]
$$

把 $((w \cdot x + b) - y)^2$ 拆成兩個 functioin $f(x) = u^2, g(w)=(w \cdot x + b - y)$ , 根據 Chain Rule $\frac{\partial f}{\partial w} = f'(g(w)) \cdot g'(w)$, 則:
$$
\frac{\partial f}{\partial w} = f'(g(w)) \cdot g'(w) = 2 g(w) + \frac{\partial (w \cdot x + b - y)}{\partial w}
$$

$$
= \frac{1}{N} \sum \textcolor{cyan}{2 \cdot ((w \cdot x + b) - y) \cdot \frac{\partial}{\partial w} [w \cdot x + b - y]}
$$

$$= \frac{1}{N} \sum 2 \cdot (y_{pred} - y) \cdot x$$
$$\frac{\partial L}{\partial w} = \frac{2}{N} \sum_{i=1}^{N} x^{(i)} \cdot (y_{pred}^{(i)} - y_{true}^{(i)})$$

#### B. 對 $b$ 的偏微分 ($\frac{\partial L}{\partial b}$)

同樣使用連鎖律：

$$
\begin{aligned}
\frac{\partial L}{\partial b} &= \frac{\partial}{\partial b} \left[ \frac{1}{N} \sum ((w \cdot x + b) - y)^2 \right] \\
&= \frac{1}{N} \sum \frac{\partial}{\partial b} \left[ ((w \cdot x + b) - y)^2 \right] \\
&= \frac{1}{N} \sum 2 \cdot ((w \cdot x + b) - y) \cdot \frac{\partial}{\partial b} [w \cdot x + b - y] \\
&= \frac{1}{N} \sum 2 \cdot (y_{pred} - y) \cdot 1 \\
\frac{\partial L}{\partial b} &= \frac{2}{N} \sum_{i=1}^{N} (y_{pred}^{(i)} - y_{true}^{(i)})
\end{aligned}
$$

## 3\. Gradient Descent

有了梯度之後，我們就可以利用它來更新參數。梯度的方向是指向函數值「增加最快」的方向，因此我們要往梯度的**反方向**走，才能讓 Loss 變小。

更新公式如下：

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

其中 $\eta$ (eta) 是 **學習率 (Learning Rate)**，它決定了我們每一步走多遠。
*   如果 $\eta$ 太大，可能會跨過最低點，甚至發散。
*   如果 $\eta$ 太小，收斂速度會非常慢。



具體的程式碼實作請參閱 [compute_gradient.ipynb](compute_gradient.ipynb)。