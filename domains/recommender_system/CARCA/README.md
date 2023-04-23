| Property  | Data |
|-|-|
| Created | 2023-02-21 |
| Updated | 2023-04-23 |
| Author | [@Aiden](https://github.com/Aidenzich) |
| Tags | #study |

# CARCA: Context and Attribute-Aware Next-Item Recommendation via Cross-Attention
| Title | Venue | Year | Code |
|-|-|-|-|
| [CARCA: Context and Attribute-Aware Next-Item Recommendation via Cross-Attention](https://arxiv.org/abs/2204.06519) | RecSys | ['22](https://recsys.acm.org/recsys22/accepted-contributions/) | [code](https://github.com/ahmedrashed-ml/CARCA) |

## Promblem Defined
### What's attribute-aware and context-aware model?
| Item | Description |
|-|-|
| `attribute` | Image features extracted by Resnet or the category data of item |
| `context` | Time features |

## Motivation
- User's context and item attributes play a crucial role in deciding which items to recommend next. Despite that, recent works in sequential and time-aware recsys usually ignore them.
- This paper proposed a context and attribute-aware recommender model that can capture the dynamic nature of the user profiles in terms of `contextual features` and `item attributes` via `multi-head self-attention blocks` that extract profile-level features and predicting item scores.

## Model Structure
![](https://i.imgur.com/2PPQmIy.png)
- **User profiles (User's Transaction history)** 
    $$p^u_t:=\{ i^P_1, i^P_2, ..., i^P_{|P^u_t|}\}$$
    - $|P^u_t|$ User's sequence length

## Experiments
### 5.4  Impact of item attributes and contextual features (RQ3)
![attribute&contextual](./assets/attribute%26contextual.png)

- `Item attributes` such as their image features have a significant impact on the performance compared to the interactions contexts that had a lower impact on the `Men` and `Fashion` datasets. 
- `Contextual features` have a higher impact on CARCA’s performance on the `Games` dataset than item attributes because video games are much more volatile than clothes and fashion-based products as they are susceptible to critics and the satisfaction of their player-bases.