# AsRep, Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer
| Title | Venue | Year | Code |
|-|-|-|-|
| [AsRep, Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer](https://dl.acm.org/doi/pdf/10.1145/3404835.3463036) | SIGIR | '21 | [code](https://github.com/DyGRec/ASReP) |

## Abstract
- Sequential Recommendation characterizes the evolving patterns by modeling item sequences chronologically. 
- The essential target of it is to capture the <font color='orange'>item transition correlations</font>. 
- The recent developments of transformer inspire the community to design effective sequence encoders
    - e.g., SASRec and BERT4Rec. 
- However, we observe that these transformer-based models suffer from the cold-start issue
    - i.e., performing poorly for short sequences.
- Therefore, we propose to augment short sequences while still preserving original sequential correlations. 
    - a new framework for Augmenting Sequential Recommendation with Pseudo-prior items (ASReP). 
        1. We firstly pre-train a transformer with sequences in a reverse direction to predict prior items. 
        2. Then, we use this transformer to generate <font color='red'>fabricated(捏造的)</font> historical items at the beginning of short sequences. 
        3. Finally, we fine-tune the transformer using these augmented sequences from the time order to predict the next item. 
- Experiments on two real-world datasets verify the effectiveness of ASReP. 


## Structure
![](https://i.imgur.com/bBWv9so.png)

