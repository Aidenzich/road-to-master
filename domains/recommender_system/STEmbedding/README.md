| Property  | Data |
|-|-|
| Created | 2023-02-21 |
| Updated | 2023-02-21 |
| Author | [@Aiden](https://github.com/Aidenzich) |
| Tags | #study |

# A New Sequential Prediction Framework with Spatial-temporal Embedding 
Sequential prediction is one of the key components in recommendation. 
In online e-commerce recommendation system, user behavior consists of the sequential visiting logs and item behavior contains the interacted user list in order. 
Most of the existing state-of-the-art sequential prediction methods **only consider the user behavior while ignoring the item behavior**. 
In addition, the paper finds that user behavior varies greatly at different time, and most existing models fail to characterize the rich temporal information. To address the above problems, proposing a **transformer-based spatial-temporal recommendation framework (STEM)**. 
In the STEM framework
- This paper first utilizes attention mechanisms to model user behavior and item behavior
- then exploit spatial and temporal information through a transformer-based model. 

The STEM framework, as a plug-in, is able to be incorporated into many neural network-based sequential recommendation methods to improve performance. This paper conduct extensive experiments on three real-world Amazon datasets. The results demonstrate the effectiveness of our proposed framework.
