| Property  | Data |
|-|-|
| Created | 2023-02-21 |
| Updated | 2023-03-01 |
| Author | [@Aiden](https://github.com/Aidenzich) |
| Tags | #study |

# Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation
| Title | Venue | Year | Code |
|-|-|-|-|
| [Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation](https://dl.acm.org/doi/pdf/10.1145/3488560.3498433)  | WSDM | '22 | [✓](https://github.com/RuihongQiu/DuoRec) |

## Abstract

| Component | Definition | Example |
|-|-|-|
| Problem Define | `Representation degeneration problem` : Item embeddings tend to degenerate into an [anisotropic]((https://www.google.com/search?q=anisotropic&rlz=1C5CHFA_enTW974TW974&oq=anisotropic&aqs=chrome..69i57j0i512l5j69i60l2.2546j0j7&sourceid=chrome&ie=UTF-8)) shape, resulting in high semantic similarities | High similarity among embeddings |
| Proposed Solution | `DuoRec`, a recommender model designed to reshape the distribution of sequence representations and implicitly apply regularization to the item embedding distribution | Contrastive regularization, model-level augmentation based on Dropout, novel sampling strategy |
| Experiment | Visualization results validate that DuoRec can largely alleviate the representation degeneration problem | - | 

## Proposed Solution
### Attention-Based
#### 1. Embedding Layer

| Property | Definition |
|-|-|
| $s = [v_1, v_2, ..., v_t]$  | Input sequence |
| $V \in \mathbb{R}^{\|\mathcal{V}\| \times d}$  | Embedding maxtrix |
| $\nu = [ \nu_1, \nu_2, ..., \nu_t]$ | Token embedding |
| $P \in \mathbb{R}^{N \times d}$ | Positional encoding maxtrix |
| $p_t$ | The positional encoding of the time step $t$ |
| $h^0_t = \nu_t + p_t$ | Embedded vector, 0-layer means not entered into the module yet |

#### 2. Self-attention Module
$$
H^L = \text{Trm}(H^0)
$$

| Property | Definition |
|-|-|
| $L$ | Layer number of mutlti-head attention mechanism |
| $\text{Trm}$ | $L$-layer multi-head Transformer encoder | 
| $H^0 = [h_0^0, h_1^0, ..., h_t^0]$ | The hidden representation of the sequence as input to transformer |
| $H^L = [h_0^L, h_1^L, ..., h_t^L]$ | The last hidden representation of the user representation of the user sequence |


### Contrastive Regularization
| Type | Definition |
|-|-|
| `Unsupervised positive augmentation` | same input, same encoder(Transformer) with different Dropout masks. |
| `Supervised positive augmentation` | different sequences with the *same target item* are semantically similar. |
| `Negative pairs` | sampling negative pair from the same batch |
#### 1. Unsupervised Augmentation
Therefore, we choose a different Dropout mask for the unsupervised augmentation of the input sequence $s$, which is firstly operated on the `input embedding` to the Transformer encoder in Equation (8) to obtain an $h^{0'}_t$. 

Afterward, the augmented input sequence embedding is fed into the Transformer encoder with the same weight but a different Dropout mask:

$$
\begin{aligned}
H^{L'} &= \text{Trm}(H^{0'}) \\ 
h' &= h_t^{L'} = H^{L'} [-1]
\end{aligned}
$$

| Property | Definition |
|-|-|
| $h'$ | It's the augmented sequence representation |
| $[-1]$ | The last element in the list |


#### 2. Supervised Positive Sampling
...