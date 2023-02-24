| Property  | Data |
|-|-|
| Created | 2023-02-21 |
| Updated | 2023-02-21 |
| Author | [@Aiden](https://github.com/Aidenzich) |
| Tags | #study |

# Self-Attentive Sequential Recommendation
| Title | Venue | Year | Code |
|-|-|-|-|
| [Self-Attentive Sequential Recommendation](https://ieeexplore.ieee.org/abstract/document/8594844?casa_token=KSghig8Awq4AAAAA:jd_bRp3qNTzU-E_L0h_l1bCBQMaUL3MgDhUKpu1FbspTD0UMPZNVVh8BElcQ2_733hId9DNC3A) | ICDM | ['18](https://icdm2018.org/program/list-of-accepted-papers/) | [code](https://github.com/kang205/SASRec) |
## TBC
Sequential dynamics are a key feature of many modern recommender systems, 
which seek to capture the ‘context’ of users’ activities on the basis of actions they have performed recently. 
To capture such patterns, two approaches have proliferated: Markov Chains (MCs) and Recurrent Neural Networks (RNNs). 

Markov Chains assume that a user’s next action can be predicted on the basis of just their last (or last few) actions, while RNNs in principle allow for longer-term semantics to be uncovered. 

Generally speaking, MC-based methods perform best in extremely sparse datasets, where model parsimony is critical, while RNNs perform better in denser datasets where higher model complexity is affordable. 

The goal of our work is to balance these two goals, by proposing a self-attention based sequential model (SASRec) that allows us to capture long-term semantics (like an RNN), but, using an attention mechanism,
makes its predictions based on relatively few actions (like an MC). 

At each time step, SASRec seeks to identify which items are ‘relevant’ from a user’s action history, and use them to
predict the next item. 

Extensive empirical studies show that our method outperforms various state-of-the-art sequential models (including MC/CNN/RNN-based approaches) on both sparse and dense datasets. 

Moreover, the model is an order of magnitude more efficient than comparable CNN/RNN-based models. 

Visualizations on attention weights also show how our model adaptively handles datasets with various density, and uncovers meaningful
patterns in activity sequences.
