# Cross Entropy
$$
\mathcal{L}_{\text{CE}} = - \sum_{i \in K} y_i \log(p(y))
$$
| Property | Description |
|-|-|
| $y_i$ | Indicator variable |
| $p(y_i)$ | The probability of class $i$  |
| $\sum_{i \in K}$ | Sum over classes |
- [torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- input is `logit`, which is an unnormalized scores



## Binary Cross Entropy
Binary cross entropy is a type of cross entropy loss used in binary classification problems.
- It measures the `dissimilarity` between the `predicted probability distribution` and the `actual distribution` (which is either 0 or 1) for each example in the dataset.
- The `binary cross entropy loss` is computed by taking the `negative logarithm of the predicted probability for the positive class`, if the actual class is positive, or for the negative class, if the actual class is negative. 
The resulting loss values are then `summed` over all examples and averaged to produce the final binary cross entropy loss. 
- Minimizing this loss will result in the model having higher accuracy in its predictions for binary classification problems.

$$
\frac{- 1}{N} \sum_1^N {y_i \times \log(p({y}_i)) + (1-y_i) \times \log(1- p({y}_i))}
$$
- What's the value of $y_i$
    | label | $y_i$ |
    |-|-|
    | Positive | 1 |
    | Negative | 0 |

#### Information Entropy
- Ref: https://www.ycc.idv.tw/deep-dl_2.html
#### Maximum Likelihood Estimation
- Ref: https://www.ycc.idv.tw/deep-dl_3.html
- Use Model to infer Data's distribution
    - We know the x, and we calculate the Probilities of Data distribution
    - Use $ln$ 
- Probability is not Likelihood
    ![](./assets/prob_likelihood.png)
    - Distribution -> Probability
    - Likelihood is probability's reverse

