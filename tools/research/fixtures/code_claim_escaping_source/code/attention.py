def head(q, k, v, d_k):
    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
    return softmax(scores, dim=-1) @ v
