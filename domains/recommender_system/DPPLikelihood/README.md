| Property  | Data |
|-|-|
| Created | 2023-02-21 |
| Updated | 2023-03-03 |
| Author | [@Aiden](https://github.com/Aidenzich) |
| Tags | #study |

# Determinantal Point Process Likelihoods for Sequential Recommendation
| Title | Venue | Year | Code |
|-|-|-|-|
| [Determinantal Point Process Likelihoods for Sequential Recommendation](https://arxiv.org/pdf/2204.11562.pdf) | SIGIR | ['22](https://sigir.org/sigir2022/program/accepted/) | [✓](https://github.com/l-lyl/DPPLikelihoods4SeqRec) |


## Abstract
| Component | Definition | Example |
|-|-|-|
| Problem Definition | Existing sequential recommendation techniques lack tailored loss functions that fit naturally into the practical application scenario of sequential recommender systems. | Such as `cross-entropy` and `Bayesian Personalized Ranking (BPR)` are widely used in the sequential recommendation area, but suffer from 2 inherent drawbacks: <br> 1.  the dependencies among elements of a sequence are overlooked in these loss formulations;  <br> 2.  Instead of balancing accuracy (quality) and diversity, only generating accurate results has been over emphasized.   |
| Proposed Solution | Two new loss functions based on the `Determinantal Point Process (DPP) likelihood`, which captures natural dependencies among temporal actions, and a `quality` vs. `diversity` decomposition of the DPP kernel to push beyond accuracy-oriented loss functions. | `DPP Set Likelihood-Based Loss`, `Conditional DPP Set Likelihood-Based Loss` |
| Experiment Result | The proposed loss functions show marked improvements over state-of-the-art sequential recommendation methods in both quality and diversity metrics in experiments on three real-world datasets. | - |
