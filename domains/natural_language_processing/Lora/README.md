| Property  | Data |
|-|-|
| Created | 2023-05-10 |
| Updated | 2023-10-06 |
| Author | @Aiden |
| Tags | #study |

# LoRA
| Title | Venue | Year | Code |
|-|-|-|-|
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | ICLR | '22 | [✓](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py) |

## What is Low-Rank?
The term `low rank` in mathematics and machine learning, especially in matrix theory and linear algebra, refers to a matrix having a relatively low rank. The rank of a matrix is the maximum number of linearly independent rows or columns in the matrix. A matrix being described as `low rank` implies that there is a strong linear relationship among its rows or columns, leading to them not being completely linearly independent.

In machine learning and data mining, low-rank matrices are often used to simplify models, reduce noise and overfitting, and improve computational efficiency. For instance, low-rank matrix factorization techniques, such as `Singular Value Decomposition (SVD)` or `Principal Component Analysis (PCA)`, are commonly used for feature reduction, data compression, and information extraction.

In the context of the Low-Rank Adaptation (LoRA) method mentioned, `low rank` refers to the additional parameter matrices introduced during model fine-tuning having a low rank. This approach effectively adjusts and optimizes model parameters while limiting model complexity and the risk of overfitting.

## Low-Rank-Parametrized Update Matrices
- The pre-trained language models have a low `instrisic dimension` and can still learn efficiently despite a random projection to a smaller subspace. 
    - `instrisic dimension` refers to the number of dimensions needed to represent the statistical patterns it has learned. In other words, it represents the **complexity of the model**.
    - This paper hypothesizes the updates to the weights also have a `low intrinsic rank` during adaptation.

![lora](./assets/lora.png)

$$
h = W_0 x + \Delta W_x  = W_0 + BA_x
$$


| Property | Definition |
|-|-|
| $W_0$ | The pre-trained language models's weight, $W_0 \in \mathbb{R}^{d \times k}$ |
| $A$   | $A$ contain trainable parameters. Initialize random Gaussian.  |
| $B$   | $B$ contain trainable parameters. Initialize with 0. |

- $\Delta W = BA$ is zero at the beginning.

## LoRA in attention
As figure below, LoRA is applied to the attention layer only in the q and v matrices. Trained components are colored in shades of magenta. The original paper only adapts the attention weights of the Transformer self-attention sub-layer with LoRA.
![lora_attention](./assets/lora_attention.png)


## Merge LoRA weights into the base model
LoRA is smaller and quicker to train, but sometimes it can be slow when you start using it because it has to load two models separately. 
To make it work faster, you can use a special function called `merge_and_unload()` from huggingface. This combines the two parts into one, making it work as a single model and eliminating slow times.
![merge](./assets/lora_merge.png)
Imagine you have two separate puzzle pieces (called A and B here) while training. Once training is done, you can merge them into one piece which is identical to having them separately.
Model merging is essentially simple, formed by purely linear combinations to create new weights \( W_{\text{merge}} \).


