# L1 and L2 Normalization
L1 和 L2 的主要目的都是**防止模型過度擬合 (Overfitting)**。它們透過在損失函數 (Loss Function) 中加入一個「懲罰項」(Penalty Term) 來**限制參數 W 的大小**。

## 為什麼限制 W 就能防止過度擬合？

一個「過度擬合」的模型，往往是因為它對訓練資料中的雜訊（noise）過於敏感。這在數學上通常表現為**參數 W 的值非常大**。

> **想像一個情境：**
> 假設 $y = w_1 x_1 + w_2 x_2$。
> 如果模型過度擬合 $x_1$，它可能會學到一個非常大的 $w_1$（比如 $w_1 = 1000$）。
> 這意味著 $x_1$ 只要有 0.01 的微小變動， $y$ 就會劇烈變化 10。
> 這種「劇烈變化」就是「不平滑」的表現。

L1 和 L2 懲罰「過大的 W」。通過限制 W 的值，模型被迫變得「遲鈍」一些，不能對單一特徵反應過度。這使得模型的決策邊界（Decision Boundary）或它所代表的函數**更「平滑」、更「簡單」**，從而有更好的泛化能力 (Generalization)。



## 1. L2 正規化 (Ridge Regression)：「圓滑」曲線

L2 懲罰的是**權重的平方和**。

* **懲罰項：** $\lambda ||\mathbf{w}||_2^2 = \lambda \sum w_i^2$
* **它的做法：** L2 傾向於讓**所有**的參數 $w_i$ 都 **「小一點」**，但**不傾向於讓它們變為 0**。
* **效果：**
    * 這被稱為「權重衰減」(Weight Decay)，因為它會把所有 W 的值都往 0 的方向拉。
    * 它會讓參數值W更平均地分佈。
    * 這非常符合您說的「圓滑」：**它讓模型的決策邊界更平滑**，不會有太「尖銳」的轉折。

## 2. L1 正規化 (Lasso Regression)：帶來「稀疏性」

L1 懲罰的是**權重的絕對值總和**。

* **懲罰項：** $\lambda ||\mathbf{w}||_1 = \lambda \sum |w_i|$
* **它的做法：** L1 在把參數 $w_i$ 推向 0 的過程中，**非常容易將許多 $w_i$ 直接變成 0**。
* **效果：**
    * 這會產生「稀疏矩陣」(Sparse Matrix)，也就是 W 向量中有很多 0。
    * 這等於是**自動進行了特徵選取 (Feature Selection)**。如果 $w_i$ 變成 0，就等於模型認為第 $i$ 個特徵 ($x_i$) 根本不重要，可以直接丟棄。
    * 所以 L1 簡化模型的方式不是「圓滑」，而是**「刪減」**。


## 視覺化呈現

L2 是在做「圓滑」（讓模型更平滑、更簡單）。而 L1 則是透過「稀疏性」（移除不重要的特徵）來簡化模型。

我們可以透過以下圖例來了解 L1, L2 具體的運作方式，常見的損失函數在 3維空間($W_1$, $W_2$, $Loss$)的呈現如下：
![alt text](imgs/1_sJaq79557XxTOucSTKJEwA@2x.jpg)

我們假定存在一個 Loss 值最低的過擬合點 $\theta_{opt}$ (谷底), 這時, 我們的「漣漪」（橢圓等高線）從谷底開始擴散。
它一定會在某個時刻「接觸」到圖中顏色區域的邊界。
第一個接觸點（相切點），就是我們的最佳解(圖中綠點)，這是正規化起作用了的典型情況。

![alt text](imgs/1__EZjUXI05ZDoCW6qGjxOFA.png)

圖中的顏色區域的大小不是天生固定的，而是由超參數(Hyperparameter)決定的。
$\lambda$ 越大，藍色區域越小，模型參數被更強烈的拉向 `0`. 相反，模型可以「跑得更遠」去接近「原始損失的谷底」（橢圓中心）。

## Pytorch Example
```python
import torch
import torch.optim as optim

# 1. 實例化您的模型
model = YourModel()

# 2. 建立優化器 (例如 AdamW 或 SGD)
# 這裡的 weight_decay=0.01 就是 L2 的 "lambda"
# lr (learning rate) 學習率
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```


## Wrap Up

| 特性 | L2 (Ridge) | L1 (Lasso) |
| :--- | :--- | :--- |
| **懲罰項** | 權重的**平方**和 ($\sum w_i^2$) | 權重的**絕對值**和 ($\sum |w_i|$) |
| **對 W 的影響** | 使 W **趨近**於 0 (但通常不等於 0) | 使**許多** W **等於** 0 |
| **主要效果** | 權重衰減、**平滑**決策邊界 | **稀疏性**、特徵選取 |
| **比喻** | 讓所有特徵都出「一點力」 | 只挑出「最有力」的幾個特徵 |


## Reference
- https://medium.com/codex/understanding-l1-and-l2-regularization-the-guardians-against-overfitting-175fa69263dd
