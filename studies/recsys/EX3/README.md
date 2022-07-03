# EX3: Explainable Attribute-aware Item-set Recommendations
| Title | Venue | Year | Code |
|-|-|-|-|
| [EX3: Explainable Attribute-aware Item-set Recommendations](https://dl.acm.org/doi/pdf/10.1145/3460231.3474240) | RecSys | ['21](https://recsys.acm.org/recsys21/accepted-contributions/) | X |

## Purpose
- Generate K sets of items (recommendations) each of which is associated with an important attribute (explanation) to justify why the items are recommended to users.
    - based on important item attributes whose value changes ***will affect*** user purchase decisions.
- attempt to help users broaden their consideration set by presenting them with differentiated options by an attribute type.

## Contribution
The contributions of this paper are three-fold:
- Highlight the importance of jointly considering important attributes and relevant items in achieving the optimal user experience in explainable recommendations.
    - e.g. in matrix-factorization, only consider relevant items, which is without considering the importance of attributes.
- Propose a novel three-step framework, EX3, to approach the explainable attribute-aware item-set recommendation problem along with a couple of novel components.
    - The whole framework is carefully designed towards large-scale real-world scenarios.
- Extensively conduct experiments on the real-world benchmark for item-set recommendations. The results show that EX3 achieves 11.35% better NDCG than state-of-the-art baselines, as well as better explainability in terms of important attribute ranking.

### Notes

- Modeling behavior-oriented attributes from users’ historical actions rather than manual identification is a critical component to conduct explainable recommendations.
- Symbolic Interpretation
    - $\mathcal{P}$ :  the universal set of items.
    - $\mathcal{A}$ :  the set of all available attributes.
    - $\mathcal{B}$ :  the items with relationship
        - $\mathcal{B}_{cp}$ : co-purchase
        - $\mathcal{B}_{cv}$  : co-view
        - $\mathcal{B}_{pv}$ : view-then-purchased
    - $\mathcal{N}_p$ : the related items for an item p $\in \mathcal{P}$
        - $\mathcal{N}_p = \{p_i|(p, p_i)\} \in \mathcal{B}$
- Proposed method
    
    ![Screen Shot 2022-02-15 at 8.12.43 AM.png](EX3%20Explai%20a8db2/Screen_Shot_2022-02-15_at_8.12.43_AM.png)
    
    - Extract-Step
        - An attention-based item embedding learning framework, which is scalable to generating embeddings for billions of items, and can be leveraged to refine coarse-grained candidate items for a given pivot item.
            - coarse-grained candidate (?)
            - pivot item (?)
        
        $$
        f(p, N_p) = \lambda \ - \parallel \phi(p) - h(N_p) \parallel_2^2
        $$
        
        - Define a metric function $f(p, N_p)$ to measure the distance between the item and its related items.
        - $\lambda$ : the base distance to distinguish $p$ and $N_p$.
        - $\phi$ : item encoder, which is modeled as an MLP with non-linear activation function.
        - $h(\cdot)$: denotes an aggregation function over item set $N_p$, encodes $N_p$ into the same $d_p$-dimensional space as $\phi(p)$ .
            - a weighted sum over item embeddings via dot-product attention.
            
            $$
            h(N_p) = \sum_{pi \in N_p} \alpha_i\phi(p_i) \\
            \alpha_i = \frac{exp(\phi(p)^\intercal \phi(p_i))}{\sum_{p_j \in N_p}exp(\phi(p)^{\intercal}\phi(p_j))}
            $$
            
        - The encoder $\phi$ can be trained by minimizing a hinge loss
            - with the following objective function:
                
                $$
                \ell_{extract} = \sum_{p \in \mathcal{P}} max(0, \epsilon - y^+ f(p, N_p)) + max(0, \epsilon-y^-f(p, N_p^-))
                $$
                
                - Assign a positive label $y^+$ = 1 for each pair of $(p, N_p)$.
                - For ***non-trivial learning(?)*** to distinguish item relatedness, randomly sample $|N_p|$ items from $B_{pv}$ as negative samples denoted by $N_p^-$ with assigned label $y^- = -1$.
                - $\epsilon$ is the margin distance.
        - For each pivot item $q \in \mathcal{P}$, we can retrieve a set of $m$ coarse-grained related items as q’s candidate set $C_q$.
            - |$N_p$| << $m$ << $|\mathcal{P}|$
            - $C_q =$  {
            $p_i$| rank$($ $f(q, {q})$$)$ = $i$, 
            $p_i$ $\in$  $\mathcal{P}$ \ {q},
            $i$ $\in$   [$m$]
            }
    - Expect-Step
        - Learn the utility function $u(q, p, a)$ to estimate how likely a candidate item $p$ will be clicked or purchased by users after being compared with pivot item $q$ on attribute $a$.
        - Assume the utility function can be decomposed into 2 parts:
            
            $$
            u(q, p,a) = g(u_{rel}(q,p), u_{att}(a | q, p))
            $$
            
            - $u_{rel}$: item relevance
                - reveals the fine-grained item relevance
                - the likelihood of item $p$ being clicked by users after compared with pivot $q$, no matter which attributes are considered.
            - $u_{att}$: Attribute importance
                - the importance of displaying attribute $a$ to users when they compare items $q$ and $p$
            - $g$: [0, 1] $\times$ [0, 1] → [0, 1], a binary operation.
            - Practically, the ground truth of important attributes still remains unknown, which leads to the challenge of how to infer the attribute importance without supervision.
                - each item may contain arbitrary number of attributes and their values may contain arbitrary content and data types.
                - This paper proposes a novel neural model named Attribute Differentiating Network to solve this issue.
        - $\text{Attribute Differentiating Network}$$, ADN$
            
            ![Screen Shot 2022-02-15 at 8.16.23 AM.png](EX3%20Explai%20a8db2/Screen_Shot_2022-02-15_at_8.16.23_AM.png)
            
            - Input
                - a pivot item $q$
                - a candidate item $p$
                - along with the corresponding $n$ attribute-value pairs $A_q$, $A_p$
                    - e.g. $A_q$ = {$($$a_1$, $v(q, a_1)$$)$, ...}
                    - in Attribute Brand, q’s value is “Orgain” and p’s value is “Bulletproof”
            - Output
                - item relevance score $\hat{Y_p}   \in [0, 1]$
                - attribute importance scores $\hat{y_{p,j}} \in [0, 1]$ for attribute $a_j  (j = 1, ..., n)$
            - $\text{3 Components}$
                - $\text{Value-difference module}$
                    - capture the different levels of attribute values of two items.
                    - Each attribute $a_j$ as a one-hot vector, embed it into $d_a$-dimensional space via linear transformation.
                        
                        $$
                        ⁍
                        $$
                        
                        - $W_a$ learnable parameters
                    - Treat value $v(p, a_j)$  of item $p$ as a sequence of characters, which can represent as a matrix:
                        
                        $$
                        v_{pj} \in \mathbb{R}^{n_c \times d_c}
                        $$
                        
                        - $d_c$: Each character is embedded into a $d_c$-dimensional vector.
                        - $n_c$: Suppose the length of character sequence is at most $n_c$.
                    - Inspired by character-level CNN, adopt convolutional layers to encode the character sequence as follows:
                        
                        $$
                        x_{ij} = maxpool(ReLU(conv(ReLU(conv(v_{pj})))))
                        $$
                        
                        - $conv(\cdot)$: the 1D convolution layer
                        - $maxpool(\cdot)$: the 1D pooling layer
                        - $x_{ij} \in \mathbb{R}^{d_C}$: output is the latent representation of arbitrary value $v_{ij}$
                    - Encode the attribute vector $a_j$ and the value vectors $x_{qj}$ and $x_{pj}$ via an MLP:
                        
                        $$
                        x_{vj} = MLP_{v}([a_j;x_{qj};x_{ij}])
                        $$
                        
                - $\text{Attention-based attribute scorer}$
                    - implicitly predict the attribute contribution.
                    - entangle each value-difference vector $x_{vj}$ of attribute $a_j$ conditioned on item vector $x_{qp}$ as follows:
                        
                        $$
                        w_{pj}=MLP_p([x_{qp};x_{v_j};x_{qp}\odot x_{v_j};\parallel x_{qp}-x_{v_j} \parallel])
                        $$
                        
                        - $w_{pj}$: item-conditioned value-difference vector.
                    - Use attention mechanism to aggregate n item-conditioned attribute vectors $w_{p1}, ..., w_{pn}$ for representation and automatic detection of important attributes.
                    - $\text{Ramdom-masking Attention Block (RAB)}$
                        - To alleviate issues when directly applying existing attention mechanisms.
                            - The learned attention weights may have a bias on frequent attributes.
                                - That is higher weights may not necessarily indicate attribute importance, but only because they are easy to acquire and hence occur frequently in datasets.
                            - Attribute cardinality varies from item to item due to the issue of missing attribute values.
                                - so model performance is not supposed to only rely on a single attribute. i.e. distributing large weight on one attribute.
                        - Define
                            
                            $$
                            \begin{align}
                            Q &= W_Q x_{qp},K_j \\ 
                            &= W_Kw_{pj},V_j   \\
                            &= W_Vw_{pj}, j\in [n]
                            \end{align}
                            $$
                            
                            $$
                            \hat{y}_{p,j} = \frac{exp(\frac{Q^{\intercal}K_j}{\sqrt{d}\tau_j}) \cdot \eta_j}{\sum_{i\in[n]}exp(\frac{Q^{\intercal}K_i}{\sqrt{d}\tau_j}) \cdot \eta_i}
                            $$
                            
                            $$
                            z_v = ln(MLP_o(o)+o), \\
                            o = ln(Q+\sum_j{\hat{y}_{p,j}V_j})
                            $$
                            
                            - $\eta_j$:(\eta) a random mask that
                                - has value $r$ with probability frequency of atrribute $a_j$ ($freq_j$)
                                - value 1 otherwise.
                                - used to alleviate the influence by imbalanced attribute frequencies.
                            - $\tau_j$:(\tau) The temperature in softmax.
                                - set as $(1+freq_j)$ by default.
                                - Used to shrink the attention on the attribute assigned with large weight.
                            - $\hat{y}_{p,j}$: used to approximate attribute importance $u_{att}(a_j|q,p)$
                            - $z_v$: encodes the aggregated information contributed by all attributes.
                - $\text{Relevance predictor}$
                    - Adopt a linear binary classifier that estimates the fine-grained relevance of two items based on the item vector as well as the encoded attribute-value vector.
                        
                        $$
                        \hat{Y}_p = \sigma(W_y[x_{qp};z_v])
                        $$
                        
                        - The objective function defined as follows:
                        
                        $$
                        \ell_{expect} = - \sum_{(q,p,Y)} Ylog{\hat{Y}_p} - (1-Y)log(1-\hat{Y}_p)
                        $$
                        
                - Estimate the utility value $u(q,p,a_j)$
                    - The relevance score $u_{rel}(q,p) \approx \hat{Y}_p$
                    - The attribute importance score $u_{att}(a_j|q,p) \approx \hat{y}_{p,j}(j=1,...,n)$
                    - The estimated utility value:
                        
                        $$
                        g(u_{rel}(q,p), u_{att}(a_j|q,p)) \approx \hat{Y_p} \cdot \hat{y}_{p,j} \approx u(q,p, a_j)
                        $$
                        
    - Explain-Step

## My Summary

- Item-Based Recommender
    - Compare The **similarity** Between pivot items & candidate items
- Proposed Framework Structure
    - **Attention-based Item Embedding** with an encoder (MLP + non-linear activation function)
    - Learning utility function from item embedding and items attributes.
    - Use utility function in step2, to group items by attributes to j groups. Then, get top k groups as the final result.
- Measure Method
    - Compare NDCG, Recall, Precision with competitors in overall and subdomain.
    - Still research...