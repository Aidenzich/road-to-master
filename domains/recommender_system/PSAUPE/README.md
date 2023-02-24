| Property  | Data |
|-|-|
| Created | 2023-02-21 |
| Updated | 2023-02-21 |
| Author | [@Aiden](https://github.com/Aidenzich) |
| Tags | #study |

# Progressive Self-Attention Network with Unsymmetrical Positional Encoding for Sequential Recommendation
In real-world recommendation systems, the preferences of users are often affected by `long-term constant interests` and `short-term temporal needs`. 
All equivalent item-item interactions in original self-attention are cumbersome, failing to capture the drifting of users' local preferences, which contain abundant short-term patterns. 
This paper proposes a novel `interpretable convolutional self-attention`, which efficiently captures both short- and long-term patterns with a progressive attention distribution. 
- A `down-sampling convolution module` is proposed to segment the overall long behavior sequence into a series of local subsequences. 
- Accordingly, the segments are interacted with each item in the self-attention layer to produce locality-aware contextual representations, during which the quadratic complexity in original self-attention is reduced to nearly linear complexity. 
- Moreover, to further enhance the robust feature learning in the context of Transformers, an `unsymmetrical positional encoding strategy` is carefully designed.