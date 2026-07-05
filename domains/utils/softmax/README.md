# Derivative of Softmax (Jacobian Matrix) when i = j
> **English** | [繁體中文](./README.zh-TW.md)

$$
\frac{\partial y_i}{\partial x_i} = y_i (1 - y_i)
$$

## Derivation
### 1\. Softmax Formula
$$
y_i = \frac{e^{x_i}}{\sum_{k} e^{x_k}} = \frac{e^{x_i}}{\textcolor{cyan}{\Sigma}}
$$
For convenience, we use $\Sigma$ to denote $\sum_{k} e^{x_k}$

### 2\. Using the Quotient Rule
We want to differentiate with respect to $x_i$. Recall the calculus formula:
$$
(\frac{u}{v})' = \frac{u'v - uv'}{v^2}
$$

Substituting:
* Numerator $u = e^{x_i}$
* Denominator $v = \Sigma$

### 3\. Start Differentiating
Differentiating with respect to $x_i$:
1.  **Derivative of the numerator ($u'$):**
    $$\frac{\partial}{\partial x_i}(e^{x_i}) = e^{x_i}$$
2.  **Derivative of the denominator ($v'$):**
    $$\frac{\partial}{\partial x_i}(\Sigma) = \frac{\partial}{\partial x_i}(e^{x_1} + ... + e^{x_i} + ...) = e^{x_i}$$
    *(Note: although the denominator is a sum of many terms, every term other than $e^{x_i}$ is a constant with respect to $x_i$, so its derivative is 0.)*

### 4\. Plugging into the Formula
$$
\frac{\partial y_i}{\partial x_i} = \frac{(\textcolor{cyan}{e^{x_i}})(\Sigma) - (\textcolor{cyan}{e^{x_i}})(e^{x_i})}{\Sigma^2}
$$

$$
= \frac{ \textcolor{cyan}{e^{x_i}}(\Sigma - e^{x_i})}{\Sigma^2}
$$

$$
= \frac{e^{x_i}}{\textcolor{cyan}{\Sigma}} \cdot \frac{\Sigma - e^{x_i}}{\textcolor{cyan}\Sigma}
$$

### 5\. Substituting $y_i$ Back
$$
= \textcolor{yellow}{\frac{e^{x_i}}{\Sigma}} \cdot \textcolor{magenta}{\frac{\Sigma - e^{x_i}}{\Sigma}}
$$
* The first part $\frac{e^{x_i}}{\Sigma}$ is exactly **$y_i$**.
* The second part $\frac{\Sigma - e^{x_i}}{\Sigma} = 1 - \frac{e^{x_i}}{\Sigma}$, which is **$1 - y_i$**.

So we obtain the final formula:
$$
\frac{\partial y_i}{\partial x_i} = y_i (1 - y_i)
$$



### Supplement: The Physical Meaning of This Formula (Why Does It Saturate?)

This derivative formula $y_i(1-y_i)$ is a downward-opening parabola:

1.  **When $y_i = 0.5$:**
    Gradient $= 0.5 \times (1 - 0.5) = 0.25$ (this is the maximum value of the gradient, updates fastest).
2.  **When $y_i$ is close to 1 (e.g. 0.99):**
    Gradient $= 0.99 \times 0.01 = 0.0099$ (the gradient is very small).
3.  **When $y_i$ is close to 0 (e.g. 0.01):**
    Gradient $= 0.01 \times 0.99 = 0.0099$ (the gradient is very small).

**This proves that:**
If your input $x_i$ is too large, causing the output $y_i$ to become an extreme 0 or 1, **the gradient automatically becomes 0**. This is why we must do Scaling, pulling the values back toward the middle so that $y_i$ does not rush to the 0 or 1 position too early.