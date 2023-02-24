| Property  | Data |
|-|-|
| Created | 2023-02-21 |
| Updated | 2023-02-21 |
| Author | [@Aiden](https://github.com/Aidenzich) |
| Tags | #study |

# UltraGCN
The core of GCNs lies in its **Message Passing Mechanism** to aggregate neighborhood information. 

message passing largely slows down the convergence of GCNs during training, 
especially for large-scale recommender systems, which hinders their wide adoption. 

LightGCN makes an early attempt to simplify GCNs for collaborative filtering by **omitting feature transformations and nonlinear activations**. 

This paper propose an **ultra-simplified formulation** of GCNs (dubbed UltraGCN), which skips infinite layers of message passing for efficient recommendation. 

Instead of explicit message passing, UltraGCN resorts to directly approximate
the limit of infinite-layer graph convolutions via a **constraint loss**. 

Meanwhile, UltraGCN allows for **more appropriate edge weight assignments** and **flexible adjustment** of the relative importances among different types of relationships.

This finally yields a simple yet effective UltraGCN model, which is easy to implement and efficient to train. Experimental results on four benchmark datasets show that UltraGCN not only outperforms the state-of-the-art GCN models but also achieves more than 10x speedup over LightGCN.