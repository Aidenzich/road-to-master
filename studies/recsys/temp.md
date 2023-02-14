### 👍 Progressive Self-Attention Network with Unsymmetrical Positional Encoding for Sequential Recommendation
In real-world recommendation systems, the preferences of users are often affected by `long-term constant interests` and `short-term temporal needs`. 
All equivalent item-item interactions in original self-attention are cumbersome, failing to capture the drifting of users' local preferences, which contain abundant short-term patterns. 
This paper proposes a novel `interpretable convolutional self-attention`, which efficiently captures both short- and long-term patterns with a progressive attention distribution. 
- A `down-sampling convolution module` is proposed to segment the overall long behavior sequence into a series of local subsequences. 
- Accordingly, the segments are interacted with each item in the self-attention layer to produce locality-aware contextual representations, during which the quadratic complexity in original self-attention is reduced to nearly linear complexity. 
- Moreover, to further enhance the robust feature learning in the context of Transformers, an `unsymmetrical positional encoding strategy` is carefully designed. 


### 👍 RESETBERT4Rec: A Pre-training Model Integrating Time And User Historical Behavior for Sequential Recommendation
Despite the great success of existing sequential recommendation-based methods, they **focus too much on item-level modeling of users' click history** and **lack information about the user's entire click history (such as click order, click time, etc.)**. 
To tackle this problem, inspired by recent advances in pre-training techniques in the field of `natural language processing`, This paper builds a new pre-training task based on the original BERT pre-training framework and incorporate temporal information. 
Specifically, This paper proposes a new model called the **RE**arrange **S**equence pr**E**-training and **T**ime embedding model via BERT for sequential **R**ecommendation (RESETBERT4Rec ) 
It further captures the information of the `user's whole click history` by adding a `rearrange sequence prediction task` to the original BERT pre-training framework, while it integrates different views of time information. 


### 👍 CaFe: Coarse-to-Fine Sparse Sequential Recommendation
Self-attentive methods still struggle to `model sparse data`, on which they struggle to learn high-quality item representations. 
This paper proposes to model user dynamics from shopping intents and interacted items simultaneously. The learned intents are `coarse-grained` and work as `prior knowledge` for item recommendation. 
This paper presents a coarse-to-fine self-attention framework, namely `CaFe`, which explicitly learns `coarse-grained` and fine-grained sequential dynamics. Specifically, CaFe first learns intents from coarse-grained sequences which are `dense` and hence provide high-quality user intent representations. Then, CaFe fuses intent representations into item encoder outputs to obtain improved item representations. 
Finally, infer recommended items based on representations of items and corresponding intents.

### 👍 A New Sequential Prediction Framework with Spatial-temporal Embedding 
Sequential prediction is one of the key components in recommendation. 
In online e-commerce recommendation system, user behavior consists of the sequential visiting logs and item behavior contains the interacted user list in order. 
Most of the existing state-of-the-art sequential prediction methods **only consider the user behavior while ignoring the item behavior**. 
In addition, the paper finds that user behavior varies greatly at different time, and most existing models fail to characterize the rich temporal information. To address the above problems, proposing a **transformer-based spatial-temporal recommendation framework (STEM)**. 
In the STEM framework
- This paper first utilizes attention mechanisms to model user behavior and item behavior
- then exploit spatial and temporal information through a transformer-based model. 

The STEM framework, as a plug-in, is able to be incorporated into many neural network-based sequential recommendation methods to improve performance. This paper conduct extensive experiments on three real-world Amazon datasets. The results demonstrate the effectiveness of our proposed framework.

### 👍 ELECRec: Training Sequential Recommenders as Discriminators
Despite their prevalence(普遍), these methods usually require training with more
meaningful samples to be effective, which otherwise will lead to a poorly trained model. 
In this work, this paper proposes to train the sequential recommenders as `discriminators` rather than generators.
Instead of predicting the next item, our method trains a discriminator to distinguish if a sampled item is a real target item or not. 
A generator, as an auxiliary model, is trained jointly with the discriminator to sample plausible(似是而非) alternative next items and will be thrown out after training.

### 👍 Determinantal Point Process Likelihoods for Sequential Recommendation
In the training process of recommender systems, the loss function plays an essential role in guiding the optimization of recommendation models to generate accurate suggestions for users. 
However, Most existing sequential recommendation techniques focus on designing algorithms or neural network architectures, and **few efforts have been made to tailor loss functions** that fit naturally into the practical application scenario of sequential recommender systems.
Ranking-based losses, such as `cross-entropy` and `Bayesian Personalized Ranking (BPR)` are widely used in the sequential recommendation area. 
This paper argues that such objective functions suffer from two inherent(固有) drawbacks(缺陷): 
1.  the dependencies among elements of a sequence are overlooked in these loss formulations; 
2.  Instead of balancing accuracy (quality) and diversity, only generating accurate results has been over emphasized. 


This paper proposes two new loss functions based on the `Determinantal Point Process (DPP) likelihood`, that can be adaptively applied to estimate the subsequent item or items. The DPP-distributed item set captures natural dependencies among temporal actions, and a `quality` vs. `diversity decomposition` of the DPP kernel pushes us to go beyond accuracy-oriented loss functions. 

---



 
## Interesting
### Personalized Showcases: Generating Multi-Modal Explanations for Recommendations
Existing explanation models generate only text for recommendations but still struggle to produce diverse contents. 
In this paper, to further enrich explanations, this paper proposes a new task named personalized showcases, in which provides both textual and visual information to explain our recommendations. 
Specifically, the paper first selects **a personalized image set that is the most relevant to a user’s interest toward a recommended item**. Then, natural language explanations are generated accordingly given our selected images.
For this new task, we collect a large-scale dataset from Google Local (i.e., maps) and construct a high-quality subset for generating multi-modal explanations. 
We propose a personalized multi-modal framework which can generate diverse and visually-aligned explanations via contrastive learning. 
Experiments show that our framework benefits from different modalities as inputs, and is able to produce more diverse and expressive explanations compared to previous methods on a variety of evaluation metrics.

### Is News Recommendation a Sequential Recommendation Task?
News recommendation is often modeled as a sequential recommendation task, which assumes that there are rich short-term dependencies over historical clicked news. 
However, in news recommendation scenarios users usually have strong preferences on the temporal diversity of news information and may not tend to click similar news successively, which is very different from many sequential recommendation scenarios such as e-commerce recommendation. 
In this paper, we study whether news recommendation can be regarded as a
standard sequential recommendation problem. Through extensive experiments on two realworld datasets, we find that modeling news recommendation as a sequential recommendation problem is **suboptimal**. 
To handle this challenge, we further propose a temporal diversityaware news recommendation method that can promote candidate news that are diverse from
recently clicked news, which can help predict future clicks more accurately. 
Experiments show that our approach can consistently improve various news recommendation methods.

### Improving Conversational Recommender Systems via Transformer-based Sequential Modelling
In `Conversational Recommender Systems (CRSs)`, conversations usually involve a set of related items and entities e.g., attributes of items. 
These items and entities are mentioned in order following the development of a dialogue.
In other words, potential sequential dependencies exist in conversations. 
However, most of the existing CRSs neglect(忽視) these potential sequential dependencies. 
In this paper, we propose a **Transformer-based sequential conversational recommendation method (TSCR)**, which models the sequential dependencies in the conversations to improve CRS. 
We represent conversations by items and entities, and construct user sequences to discover user preferences by considering both mentioned items and entities. Based on the constructed sequences, we deploy a `Cloze task` to predict the recommended items along a sequence. 

### Bias Mitigation for Toxicity Detection via Sequential Decisions
Increased social media use has contributed to the greater prevalence of `abusive`, `rude`, and `offensive` textual comments. Machine learning models have been developed to detect toxic comments online, yet these models tend to show biases against users with marginalized or minority identities (e.g., females and African Americans). 
Established research in debiasing toxicity classifiers often:
- (1) Takes a static or batch approach, assuming that all information is available and then making a one-time decision
- (2) Uses a generic strategy to mitigate different biases (e.g., gender and racial biases) that assumes the biases are independent of one another.

However, in real scenarios :
- the input typically arrives as a sequence of comments/words over time instead of all at once. Thus, decisions based on partial information must be made while additional input is arriving. 
- Moreover, social bias is complex by nature. Each type of bias is defined within its unique context, which consistent with `intersectionality theory` within the social sciences, might be correlated with the contexts of other forms of bias. 

In this work, we consider `debiasing toxicity detection` as a sequential decision-making process where different biases can be interdependent. 
In particular, we study debiasing toxicity detection with two aims: 
- (1) to examine whether different biases tend to correlate with each other; 
- (2) to investigate how to jointly mitigate these correlated biases in an interactive manner to minimize the total amount of bias. 

At the core of our approach is a framework built upon theories of sequential `Markov Decision Processes` that seeks to maximize the prediction accuracy and minimize the bias measures tailored to individual biases. 
Evaluations on two benchmark datasets empirically validate the hypothesis that biases tend to be correlated and corroborate the effectiveness of the proposed sequential `debiasing strategy`


### Who to follow
- [Julian McAuley](https://cseweb.ucsd.edu/~jmcauley/)
- [Jiacheng Li](https://jiachengli1995.github.io/)
