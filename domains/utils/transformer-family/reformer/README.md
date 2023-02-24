| Property  | Data |
|-|-|
| Created | 2023-02-22 |
| Updated | 2023-02-22 |
| Author | [@YiTing](https://github.com/yiting-tom), [@Aiden](https://github.com/Aidenzich) |
| Tags | #study |


# Reformer

- [Slide with note made with ⭐️ by YiTing](/present/yt/Reformer.pdf)

## Pain point in vanilla Transformer
- Quadratic time and memory complexity within self-attention module. 
- Memory in a model with $N$ layers is $N$-times larger than in a single-layer model because we need to store activations for back-propagation.
- The intermediate FF layers are often quite large.

## Reformer proposed 2 main changes:
| Change | Result |
|-|-|
| Replace the `dot-product` attention with `locality-sensitive hashing (LSH) attention` | reducing the complexity from $O(L^2)$ to $O(L \log L)$ |
| Replace the standard `residual blocks` with `reversible residual layers` | allows storing activations only once during training instead of $N$ times, i.e. proportional to the number of layers |

### Locality-Sensitive Hashing Attention
- wip.

### Reversible Residual Network
- wip