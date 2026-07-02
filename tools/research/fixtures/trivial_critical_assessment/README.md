# Transformer (Attention Is All You Need) — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Attention Is All You Need |
| Venue | NIPS |
| Year | 2017 |
| Authors | Vaswani et al. |
| Official Code | unknown |
| Venue Kind | paper |

## First Principles

The core mechanism is scaled dot-product attention. Given queries $Q$, keys $K$ and
values $V$ the layer computes:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

| Symbol | Shape | Meaning |
|-|-|-|
| $Q$ | $(L \times d_k)$ | what a token is looking for |
| $K$ | $(L \times d_k)$ | what a token exposes |
| $V$ | $(L \times d_v)$ | the content carried |

A minimal reference implementation of one attention head reads:

```python
def head(q, k, v, d_k):
    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
    return softmax(scores, dim=-1) @ v
```

![scaled dot-product attention](imgs/fig1.png)

## 🧪 Critical Assessment

The baseline looks fine.

## 🔗 Related notes

- BERT and later encoder-only models build directly on this stack.
