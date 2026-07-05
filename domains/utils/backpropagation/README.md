# Backpropagation
> **English** | [繁體中文](./README.zh-TW.md)

Backpropagation is an "Attribution mechanism". Using the **Chain Rule** of calculus, it pushes the final total error (Loss) back layer by layer, computes how much "responsibility (Gradient)" each parameter should bear for this error, and accordingly asks them to make corrections.


## Chain Rule Review
The Chain Rule is the rule in calculus for computing the derivative of a "composite function". In deep learning it is the mathematical soul of Backpropagation: if you want to know the influence of A on C, you can first multiply "the influence of A on B" by "the influence of B on C".

Suppose three variables cascade like dominoes:
$$x \to y \to z$$
( $x$ influences $y$, $y$ influences $z$ )

We want to know **how much a tiny change in $x$ causes $z$ to change** ($\frac{\partial z}{\partial x}$).

The Chain Rule tells us:
$$
\frac{\partial z}{\partial x} = \underbrace{\frac{\partial z}{\partial y}}_{\text{sensitivity of z to y}} \times \underbrace{\frac{\partial y}{\partial x}}_{\text{sensitivity of y to x}}
$$


1.  **First layer (x -> y):**
    * Rule: for every increase of **1** in $x$, $y$ increases by **2**.
    * $\frac{\partial y}{\partial x} = 2$

2.  **Second layer (y -> z):**
    * Rule: for every increase of **1** in $y$, $z$ increases by **3**.
    * $\frac{\partial z}{\partial y} = 3$

3.  **Chained result (x -> z):**
    * If $x$ increases by **1**, then $y$ becomes **2**.
    * Because $y$ changed by **2**, and $z$ is a 3× amplification of $y$, $z$ ends up increasing by $2 \times 3 = \mathbf{6}$.

**Conclusion:** the influence of $x$ on $z$ (the gradient) is $2 \times 3 = 6$.
This is the Chain Rule: **multiply together all the "multipliers (derivatives)" along the path.**


## The Mathematical Formula of Backpropagation
We write out, with mathematical formulas, how this gradient runs back layer by layer.

This lets us see very clearly how the $\mathbf{1}$ inside the **Residual Connection** protects the gradient.


### 1. Define the Notation (Setup)

Suppose we have an **$n$-layer** neural network.

* **$x_i$**: the **input** of the $i$-th layer (i.e. the output of the $i-1$-th layer).
* **$F_i$**: the transformation function of the $i$-th layer (containing Attention or FFN).
* **Residual formula**: the output of each layer is $x_{i+1} = x_i + F_i(x_i)$.
* **$L$**: the final Loss (total error).

Our goal is to compute the **gradient of the $i$-th layer** $\frac{\partial L}{\partial x_i}$, i.e. the gradient of the Loss with respect to this layer's input.

### 2. Start from the Last Layer (The Start)

Suppose the output of the last layer is $x_{n+1}$. We compute the error gradient between it and the ground-truth answer:
$$
g_{n+1} = \frac{\partial L}{\partial x_{n+1}}
$$
This is the **starting point** of our backpropagation.

### 3. Push Back One Layer (Step $n \to n-1$)

Now we want to compute the input gradient of the $n$-th layer $g_n = \frac{\partial L}{\partial x_n}$.

By the Chain Rule:
$$
g_n = \underbrace{g_{n+1}}_{\text{from layer } n+1 } \cdot \frac{\partial x_{n+1}}{\partial x_n}
$$

Substituting the residual formula $x_{n+1} = x_n + F_n(x_n)$:
$$
\frac{\partial x_{n+1}}{\partial x_n} = \frac{\partial (x_n + F_n(x_n))}{\partial x_n} = \mathbf{I} + F'_n(x_n)
$$
*(Note: $\mathbf{I}$ is the identity matrix, i.e. the 1 of the Residual Connection. It ensures the gradient does not vanish.)*

So the gradient of the $n$-th layer is:
$$
g_n = g_{n+1} \cdot (\mathbf{I} + F'_n)
$$

---

### 4. Expanding to an Arbitrary $i$-th Layer (General Case)

If we keep pushing back to the $i$-th layer, we find this is a chained-multiplication process.

The gradient $g_i$ of the $i$-th layer equals the **final gradient $g_{n+1}$** multiplied by the **Jacobian matrix of every layer along the way**:

$$
g_i = g_{n+1} \cdot \underbrace{(\mathbf{I} + F'_n) \cdot (\mathbf{I} + F'_{n-1}) \cdots (\mathbf{I} + F'_i)}_{\text{multiplied all the way from layer } n \text{ back to layer } i}
$$

Expressed with the product notation ($\prod$):

$$
g_i = g_{n+1} \cdot \prod_{k=i}^{n} (\mathbf{I} + F'_k(x_k))
$$


### 5. Why Does This Formula Prove the "Highway" Theory of the Residual Connection?
Suppose there are only two layers ($k=1, 2$); then the propagation path is:
$$
\text{Loss} \to x_3 \to x_2 \to x_1
$$



When the gradient propagates back to layer 1, $g_1$ equals:
$$
g_1 = g_{3} \cdot (\mathbf{I} + F'_2) \cdot (\mathbf{I} + F'_1)
$$

Multiplying it out (distributive law):
$$
g_1 = g_{3} \cdot (\mathbf{I} \cdot \mathbf{I} + \mathbf{I} \cdot F'_1 + F'_2 \cdot \mathbf{I} + F'_2 \cdot F'_1)
$$

After simplifying we get:
$$
g_1 = \underbrace{g_{3}}_{\text{1. express train}} + \underbrace{g_{3} F'_1 + g_{3} F'_2}_{\text{2. short-range paths}} + \underbrace{g_{3} F'_2 F'_1}_{\text{3. long-range path}}
$$

#### This reveals the mathematical structure of the Residual Connection:
1.  **Express-train term ($g_3$):**
    The formula directly contains an original $g_{3}$.
    This means the error signal of the last layer, without being multiplied by any $F'$ (weight change), reaches layer 1 **losslessly**. This is that "highway". Even if $F'$ is very small (vanishing gradient), this term guarantees that $g_1$ is absolutely never 0.

2.  **Short-range paths ($g_3 F'_1 + g_3 F'_2$):**
    This means that during propagation, **some layers took the shortcut (the highway), and some layers took the ordinary road (the function)**.

    * **$g_3 F'_2$**: the gradient passes through the transformation of layer 2 ($F'_2$), but takes the shortcut ($\mathbf{I}$) at layer 1. This is equivalent to a shallow network with "only layer 2".
    * **$g_3 F'_1$**: the gradient takes the shortcut ($\mathbf{I}$) at layer 2, but passes through the transformation ($F'_1$) at layer 1. This is equivalent to a shallow network with "only layer 1".
    
    This reveals the real power of ResNet — **it is not merely a deep network, it is an ensemble of countless "shallow networks"**.
    Through the residual connection, a 100-layer network actually contains, at the same time, all sorts of path combinations of 1 layer, 2 layers, ...99 layers. This lets the model keep working through other paths even if a few layers break, greatly increasing training stability.

3.  **Long-range path term ($g_3 F'_2 F'_1$):**
    This is the only path of a traditional neural network (without residual connections). If both $F'_2$ and $F'_1$ are smaller than 1 (say 0.1), multiplying them gives $0.01$, and the gradient vanishes. But in a residual network, this is only a small part of the gradient.




## How $g_i$ Is Used to **Update the Weight $W_i$**.
Although we compute $g_i = \frac{\partial L}{\partial x_i}$ (the gradient passed to the previous layer), our real goal is to update the weight $W_i$ of **this** layer.

**The gradient for the weight update** is $\frac{\partial L}{\partial W_i}$, obtained by combining $g_{i+1}$ with this layer's input $x_i$ through another Chain Rule:
$$
\frac{\partial L}{\partial W_i} = g_{i+1} \cdot \frac{\partial F_i(x_i)}{\partial W_i}
$$

Finally, this $\frac{\partial L}{\partial W_i}$ is used to perform
**Gradient Descent**:
$$
W_i^{\text{new}} = W_i^{\text{old}} - \eta \cdot \frac{\partial L}{\partial W_i}
$$
- $\eta$ is the learning rate

### Gradient Computation and Weight Update Happen Simultaneously in the Same Layer

**Actions performed at the $i$-th layer (walking back from $n \to 1$):**

| Order | Action | Purpose |
| :--- | :--- | :--- |
| **I.** | **Receive** $g_{i+1}$ (known) | Receive the returned gradient from behind |
| **II.** | **Compute and update** $W_i$ | Handle this layer's own weight update ($\frac{\partial L}{\partial W_i}$) |
| **III.** | **Compute and pass** $g_i$ | Compute the returned gradient to give to the previous layer ($\frac{\partial L}{\partial x_i}$) |
| **IV.** | **Finish** | Move on to layer $i-1$ |

**The mathematical steps are as follows:**
1.  **Compute the Weight gradient ($\frac{\partial L}{\partial W_i}$) and perform the update:**
    $$\frac{\partial L}{\partial W_i} = \underbrace{g_{i+1}}_{\text{gradient returned from the next layer}} \cdot \underbrace{\frac{\partial x_{i+1}}{\partial W_i}}_{\text{sensitivity of itself to the output}}$$
    - This step computes the direction and magnitude by which $W_i$ should change
    $$
    W_i^{\text{new}} = W_i^{\text{old}} - \eta \cdot \frac{\partial L}{\partial W_i}
    $$
    - Then [perform Gradient Descent](../optimizer/gradient_descent.md)
    - $\eta$ is the learning rate.
    - This step multiplies the computed gradient by the step size $\eta$, then subtracts it from the old weight, completing the correction of the parameter.



2.  **Compute the gradient to pass to the previous layer ($g_i$):**
    $$g_i = \underbrace{g_{i+1}}_{\text{gradient returned from the next layer}} \cdot \underbrace{\frac{\partial x_{i+1}}{\partial x_i}}_{\text{the pass-through multiplier of this layer}}$$
    *(This step computes the gradient $g_i$ to give to layer $i-1$.)*

**Summary:** Backpropagation is a single continuous process. Standing at the $i$-th layer, it uses the **gradient returned from the next layer ($g_{i+1}$)** to complete both tasks of **Weight correction** and **Output passing ($g_i$)**.


**Dual Responsibility**:

| Responsibility | Core Purpose | Mathematical Expression |
| :--- | :--- | :--- |
| **Update weights** | **Local Learning** | Compute $\frac{\partial L}{\partial W_i}$, used to **change** this layer's knowledge. |
| **Pass gradients** | **Global Propagation** | Compute $\frac{\partial L}{\partial x_i}$, used to **notify** the previous layer. |



## Common Exam Question
> **"Please describe precisely how, during the training of a neural network, Backpropagation updates the weights $W_i$ of a single layer $i$?"**

### 🚨 Typical Wrong Answers (Proving Incomplete Understanding)

| Type of error | Answer content | Missing piece |
| :--- | :--- | :--- |
| **Answering only stage 2** | "Backpropagation just directly multiplies the computed gradient $g$ by the learning rate $\eta$, then subtracts it from $W$." | Ignores stage 1: **does not explain how the gradient $g$ (i.e. $\frac{\partial L}{\partial W_i}$) came about**, nor mention the Chain Rule. |
| **Answering only stage 1** | "Backpropagation uses the Chain Rule to compute $\frac{\partial L}{\partial W}$." | Ignores stage 2: **does not apply the gradient to $W$**, which is like "finding the problem but not solving it". |
| **Process confusion** | "It first computes $\frac{\partial L}{\partial x}$, then multiplies $\frac{\partial L}{\partial x}$ by $x$ to update $W$." | **Confuses the responsibilities of $\frac{\partial L}{\partial x}$ and $\frac{\partial L}{\partial W}$.** $\frac{\partial L}{\partial x}$ is for the previous layer, not for $W$. |