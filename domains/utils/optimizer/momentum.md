# Grediant Descent with Momentum

![alt text](<imgs/Screenshot 2025-11-12 at 4.28.57 PM.png>)
![alt text](<imgs/Screenshot 2025-11-12 at 4.29.29 PM.png>)
## 1. Momentum 跟 loss function 的關係是什麼？

Momentum (動量 $m$) 與 Loss Function (損失函數 $\mathcal{L}$) 的關係是**間接**的，並且是**透過「梯度」($\nabla\mathcal{L}$) 來聯繫的**。

* Loss Function $\mathcal{L}$ 本身是一個「計分器」，它告訴我們模型現在的表現有多差。
* 為了更新模型，我們計算 Loss Function 的**梯度 $\nabla\mathcal{L}$**（即您所說的「斜率」）。
* Momentum $m$ 的計算，**直接依賴於這個梯度**。

[cite_start]根據論文中的公式 (Equation 17 或 229)，新的動量 $m_{i+1}$ 是由「舊的動量 $m_i$」和「**當前的梯度 $\nabla\mathcal{L}$**」共同決定的 [cite: 229]。

所以，這個關係鏈是：
**Loss Function ($\mathcal{L}$) $\rightarrow$ 產生 $\rightarrow$ Gradient ($\nabla\mathcal{L}$) $\rightarrow$ 用來計算 $\rightarrow$ Momentum ($m$)**


## 2. 增加每一步的 loss 跳躍距離，導致更快接近局部最小值
可以將 Momentum 想像成一顆「**滾下山的球**」，而標準的梯度下降 (GD) 則是一個「**每走一步都會停下來重新看地圖**」的登山者。

* **加速收斂 (更快接近最小值)：**
    * 當您處在一個長而陡峭的斜坡上時，**梯度會持續指向同一個方向**。
    * 標準 GD 登山者每一步都只走一小步（由學習率決定）。
    * [cite_start]而 Momentum 這顆球，因為**持續**接收到同方向的梯度，它會**累積速度**（即 $m$ 的值會越加越大）[cite: 229]。
    * [cite_start]因為 $m$ 的值變大了，所以 $W_{i+1} = W_i + m_{i+1}$ 這一步的「跳躍距離」**確實會變大** [cite: 228]。
    * 這使得 Momentum 比標準 GD **更快地**衝向山谷（即局部最小值）。
* **克服次優解 (不只更快，還更穩)：**
    * [cite_start]Momentum 還有另一個好處。如果遇到梯度非常小但還沒到最低點的「平坦區域」（鞍點），標準 GD 可能會因為梯度接近 0 而「卡住」[cite: 52]。
    * 但 Momentum 因為**攜帶著過去累積的速度**，它有足夠的「動能」**衝過**這些平坦區域或非常淺的局部最小值，繼續尋找更深的山谷。
    * 同時，如果梯度在一個狹窄的山谷中來回震盪，Momentum 會將這些震盪（一正一負）平均掉，使得步伐更加穩定，專注於「下山」的主方向。


## Reference
https://kenndanielso.github.io/mlrefined/blog_posts/13_Multilayer_perceptrons/13_4_Momentum_methods.html