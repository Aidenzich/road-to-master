# Determinantal Point Process Likelihoods for Sequential Recommendation
In the training process of recommender systems, the loss function plays an essential role in guiding the optimization of recommendation models to generate accurate suggestions for users. 
However, Most existing sequential recommendation techniques focus on designing algorithms or neural network architectures, and **few efforts have been made to tailor loss functions** that fit naturally into the practical application scenario of sequential recommender systems.
Ranking-based losses, such as `cross-entropy` and `Bayesian Personalized Ranking (BPR)` are widely used in the sequential recommendation area. 
This paper argues that such objective functions suffer from two inherent(固有) drawbacks(缺陷): 
1.  the dependencies among elements of a sequence are overlooked in these loss formulations; 
2.  Instead of balancing accuracy (quality) and diversity, only generating accurate results has been over emphasized. 


This paper proposes two new loss functions based on the `Determinantal Point Process (DPP) likelihood`, that can be adaptively applied to estimate the subsequent item or items. The DPP-distributed item set captures natural dependencies among temporal actions, and a `quality` vs. `diversity decomposition` of the DPP kernel pushes us to go beyond accuracy-oriented loss functions. 