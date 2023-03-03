| Property  | Data |
|-|-|
| Created | 2023-02-21 |
| Updated | 2023-03-03 |
| Author | [@Aiden](https://github.com/Aidenzich) |
| Tags | #study |

# Self-Attentive Sequential Recommendation
| Title | Venue | Year | Code |
|-|-|-|-|
| [Self-Attentive Sequential Recommendation](https://ieeexplore.ieee.org/abstract/document/8594844?casa_token=KSghig8Awq4AAAAA:jd_bRp3qNTzU-E_L0h_l1bCBQMaUL3MgDhUKpu1FbspTD0UMPZNVVh8BElcQ2_733hId9DNC3A) | ICDM | ['18](https://icdm2018.org/program/list-of-accepted-papers/) | [code](https://github.com/kang205/SASRec) |

## Abstract

| Component |  Definition |  Example |
|-|-|-|
| Problem Definition |  Markov Chains (MCs) and Recurrent Neural Networks (RNNs) are two common approaches for capturing sequential dynamics in recommender systems, but each has its limitations in terms of handling extremely sparse or dense datasets. |  MC-based methods perform best in extremely sparse datasets, while RNNs perform better in denser datasets. |
| Proposed Solution | `SASRec`, a self-attention based sequential model that balances the goals of capturing `long-term semantics` (like an RNN) and making predictions based on relatively few actions (like an MC) using an `attention mechanism`. |  SASRec uses self-attention to identify relevant items from a user's action history to predict the next item. |
| Experiment Result |  SASRec outperforms various state-of-the-art sequential models (including MC/CNN/RNN-based approaches) on both `sparse` and `dense` datasets, and is an order of magnitude more efficient than comparable CNN/RNN-based models. Visualizations on attention weights also show how the model adaptively handles datasets with various density and uncovers meaningful patterns in activity sequences. |  |
