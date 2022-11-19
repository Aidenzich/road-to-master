# CARCA: Context and Attribute-Aware Next-Item Recommendation via Cross-Attention
| Title | Venue | Year | Code |
|-|-|-|-|
| [CARCA: Context and Attribute-Aware Next-Item Recommendation via Cross-Attention](https://arxiv.org/abs/2204.06519) | RecSys | ['22](https://recsys.acm.org/recsys22/accepted-contributions/) | [code](https://github.com/ahmedrashed-ml/CARCA) |

## Promblem Defined
### What's attribute-aware and context-aware model?
- `attribute-aware` 是應用影像AI的技巧，透過 ResNet 從影像中萃取出 attrubute 作為 Input 參數

## Motivation
- User's context and item attributes play a crucial role in deciding which items to recommend next. Despite that, recent works in sequential and time-aware recsys usually ignore them.
- This paper proposed a context and attribute-aware recommender model that can capture the dynamic nature of the user profiles in terms of `contextual features` and `item attributes` via `multi-head self-attention blocks` that extract profile-level features and predicting item scores.

## Model Structure
![](https://i.imgur.com/2PPQmIy.png)
- **User profiles (User's Transaction history)** 
    $$p^u_t:=\{ i^P_1, i^P_2, ..., i^P_{|P^u_t|}\}$$
    - $|P^u_t|$ User's sequence length
    