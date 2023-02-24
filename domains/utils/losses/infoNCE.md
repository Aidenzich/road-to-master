# InfoNCE
The InfoNCE loss is proposed from [Contrastive Predictive Coding (CPC)](https://arxiv.org/abs/1807.03748), which uses `categorical cross-entropy loss` to identify the positive sample amongst a set of `unrelated noise samples`.
The formula mentioned in [Moco; He, Kaiming](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) as below:

$$
\mathcal{L}_\text{InfoNCE} = - \log \frac{\exp(q \cdot k_{+} / \mathcal{T})}{\Sigma_{i=0}^K \exp(q \cdot k_i / \mathcal{T})}
$$

| Property | Description |
|-|-|
| $\mathcal{T}$ | A temperature hyper-parameter |
| $q$ | An encoded query |
| $k_i = \{k_0, k_1, k_2, ...\}$ | A set of encoded sample |
| $k_+$ | Positive sample |
| ${\sum_{i=0}^K \exp(q \cdot k_i / \mathcal{T})}$ | The sum is over 1 postive and $K$ negative samples |

## The relationship between Cross entropy and InfoNCE
The cross entropy Formula is:

$$
\mathcal{L}_{CE}\big( p(y_i) \big) = - \sum_{i \in K} y_i \log \big(p(y_i) \big)
$$

The softmax formula is:

$$
\text{softmax}(x_i) = \frac{\exp(x_i)}{\Sigma_{j=0}^K \exp(x_j)}
$$

To calculate the probability of $y_i$, we have to use softmax on the model ouput logits $x_i$, so:

$$
\begin{aligned}
p(y_+) &= \text{softmax}(x_+) = \frac{\exp(x_+)}{\Sigma_{j=0}^K \exp(x_j)} \\
\mathcal{L}_{CE}(p(y_+)) &= - \Sigma_{i \in K} y_i \log(p(y_+)) \\
&= - \sum_{i \in K} y_i \log(\frac{\exp(x_+)}{\Sigma_{j=0}^K \exp(x_j)}) \\
&= - \log(\frac{\exp(x_+)}{\Sigma_{j=0}^K \exp(x_j)})
\end{aligned}
$$

In above formula, $K$ is the total classes in the dataset.
- Let's take the `ImageNet dataset` in the CV field as an example. There are **1.28 million* images in the dataset. We use data augmentation techniques (such as `random cropping`, `random color distortion`, and `random Gaussian blur`) to generate `positive sample` pairs for contrastive learning. Each image is a separate category, so $K$ is **1.28 million categories**. The more images there are, the more categories there are. 
But calculating with softmax on such a large number of categories is very time-consuming, especially with `exponential operations`. When the dimension of the vector is several million, the computation complexity is quite high. So the $\mathcal{L}_{CE}$ is not suitable for use on the Contrastive learning.

Then we look back to $\mathcal{L}_{\text{InfoNCE}}$:

$$
\mathcal{L}_\text{InfoNCE} = - \log \frac{\exp(q \cdot k_{+} / \mathcal{T})}{\Sigma_{i=0}^K \exp(q \cdot k_i / \mathcal{T})}
$$

If we ignore the temperature hyper-parameter $\mathcal{T}$, the loss function became:

$$
\mathcal{L}_\text{InfoNCE} = - \log \frac{\exp(q \cdot k_{+})}{\Sigma_{i=0}^K \exp(q \cdot k_i )}
$$

As we can see, The `InfoNCE` loss is actually a `cross entropy loss`, and it performs a classification task with $k+1$ classes.

## The relationship between InfoNCE and Mutual information
First we need to derive the probability of positive sample when given a context vector $c$:

$$
\begin{aligned}
p(x_+ | X, c) &= \frac{p(x_+) \prod_{i=1,...,N;i \neq +}p(x_i)}{\Sigma^N_{j=1} [ p(x_j|c) \prod_{i=1,...,N;i \neq j}p(x_j)] } \\
&= \frac{\color{blue} \frac{p(x_+|c)}{p(x_+)}}{\Sigma_{j=1}^N \frac{p(x_j|c)}{p(x_j)}} \\
&= \frac{f(x_+, c)}{\Sigma^N_{j=1}f(x_j, c)}
\end{aligned}
$$
- where the scoring function $f(x, c) \propto \frac{p(x|c)}{p(x)}$

For brevity, let us write the loss of InfoNCE as:

$$
\mathcal{L}_{\text{InfoNCE}} = - \mathbb{E} \big[ \log \frac{f(x,c)}{\Sigma_{x' \in X} f(x', c)} \big]
$$

And then we dervie the mutual information with [In terms of PMFs for discrete distributions](https://en.wikipedia.org/wiki/Mutual_information):

$$
\begin{aligned}
I(x; c) &= \sum_{x,c}p(x,c) \log \frac{p(x, c)}{p(x) p(c)} \\
&= \sum_{x,c} p(x,c) \log {\color{blue} \frac{ p(x | c)}{p(x)}}
\end{aligned}
$$

Where the term in blue is estimated by $f$, which means when we maximize the $f$ between input $x_+$ and context vector $c$, we maximize the`mutual information` between input $x_+$ and context vector $c$

