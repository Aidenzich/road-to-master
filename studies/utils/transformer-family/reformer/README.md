# Reformer
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
### Reversible Residual Network