# Attention is All You Need
https://arxiv.org/abs/1706.03762

## Keywords
### Attention
#### Illustration
![self-attention](self-attention.gif)
#### Python Code
```python
import math
import torch
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```
#### Attention vs Fully Connected Network
- 單純使用 full connected network 會有極限：
- If we use a window covers the whole sequence for fully connected network to condsider the whole swquence...
    - FC requires a lot of parameters, the amount of calculation is large, and it may also be overfitting.
- 因此使用 Attention 讓模型能夠考慮到整個 sequence 但是又不把所有的資訊，所以我們有一個特別的機制。這個機制是根據 a 換個系統，
### Multi-Head Attention
### Positional Encoding
### Transformer Encoder
### Transformer Decoder