# Gradient Descent

1.  定義數學模型與損失函數。
2.  手動推導梯度
3.  撰寫 Python 程式碼，將數學公式轉化為可執行的 `gradient_descent` 函式。


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

1.  $\frac{\partial L}{\partial b} = \frac{\partial}{\partial b} \left[ \frac{1}{N} \sum ((w \cdot x + b) - y)^2 \right]$
2.  $= \frac{1}{N} \sum \frac{\partial}{\partial b} \left[ ((w \cdot x + b) - y)^2 \right]$
3.  $= \frac{1}{N} \sum 2 \cdot ((w \cdot x + b) - y) \cdot \frac{\partial}{\partial b} [w \cdot x + b - y]$
4.  $= \frac{1}{N} \sum 2 \cdot (y_{pred} - y) \cdot 1$
5.  **$\frac{\partial L}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (y_{pred}^{(i)} - y_{true}^{(i)})$**

-----

## 3\. Python 完整實作 (使用 NumPy)

現在我們可以把上面的數學公式寫成程式碼。我們將嚴格按照您的要求，**將梯度計算「函式化」** (即 `compute_gradient`)。

這份程式碼實作的是「**批次梯度下降 (Batch Gradient Descent)**」，因為它在每一步都使用了**所有**的訓練樣本 ( $N$ 個) 來計算梯度。

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_gradient(X, y_true, w, b):
    """
    計算線性迴歸的梯度。
    
    Args:
        X (np.array): 訓練資料 (特徵)
        y_true (np.array): 訓練資料 (真實標籤)
        w (float): 目前的權重 (斜率)
        b (float): 目前的權重 (截距)
        
    Returns:
        tuple: (grad_w, grad_b) 梯度的值
    """
    num_samples = len(y_true)
    
    # 1. 計算預測值 (Forward pass)
    # y_pred = w * X + b
    y_pred = (w * X) + b
    
    # 2. 計算誤差
    # error = y_pred - y_true
    error = y_pred - y_true
    
    # 3. 根據數學公式計算梯度 (Backward pass)
    # grad_w = (2/N) * sum(X * error)
    grad_w = (2 / num_samples) * np.sum(X * error)
    
    # grad_b = (2/N) * sum(error)
    grad_b = (2 / num_samples) * np.sum(error)
    
    return grad_w, grad_b

def gradient_descent(X, y_true, learning_rate, num_epochs):
    """
    執行完整的梯度下降訓練迴圈。
    
    Args:
        X (np.array): 訓練資料 (特徵)
        y_true (np.array): 訓練資料 (真實標籤)
        learning_rate (float): 學習率 (alpha)
        num_epochs (int): 迭代次數 (epochs)
        
    Returns:
        tuple: (w_final, b_final, loss_history) 訓練後的權重與損失紀錄
    """
    # 初始化權重
    w = 0.0  # 隨機初始化或設為 0
    b = 0.0
    
    loss_history = []
    
    print(f"開始訓練... Learning Rate: {learning_rate}, Epochs: {num_epochs}")
    
    for i in range(num_epochs):
        # ----------------------------------------------------
        # 1. 計算梯度 (呼叫您要求的 gradient 函式)
        grad_w, grad_b = compute_gradient(X, y_true, w, b)
        
        # 2. 更新權重 (核心步驟)
        # w = w - alpha * grad_w
        # b = b - alpha * grad_b
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        # ----------------------------------------------------
        
        # (可選) 計算並記錄當前的 Loss
        y_pred = (w * X) + b
        loss = np.mean((y_pred - y_true) ** 2)
        loss_history.append(loss)
        
        if (i + 1) % 100 == 0:
            print(f"Epoch [{i+1}/{num_epochs}], Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}")
            
    print("訓練完成！")
    return w, b, loss_history

# --- 主程式：執行範例 ---

# 1. 產生模擬資料
# 讓我們假裝真實的 w = 3, b = 4
np.random.seed(42)
X_train = 2 * np.random.rand(100) # 100 個 0~2 之間的點
y_train = 4 + 3 * X_train + np.random.randn(100) * 0.5 # y = 4 + 3x + 雜訊

# 2. 設定超參數
learning_rate = 0.01
num_epochs = 1000

# 3. 執行梯度下降
w_final, b_final, loss_history = gradient_descent(X_train, y_train, learning_rate, num_epochs)

# 4. 輸出最終結果
print(f"\n--- 最終結果 ---")
print(f"真實模型: y = 3.0 * x + 4.0")
print(f"學到模型: y = {w_final:.4f} * x + {b_final:.4f}")

# 5. 繪製結果
plt.figure(figsize=(15, 5))

# 圖一：Loss 下降曲線
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Loss Function Convergence")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (Loss)")
plt.grid(True)

# 圖二：模型擬合結果
plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, color='blue', label='Data Points', alpha=0.6)
y_pred_final = w_final * X_train + b_final
plt.plot(X_train, y_pred_final, color='red', linewidth=2, label='Fitted Line (Our Model)')
plt.title("Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```
