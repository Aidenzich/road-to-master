# Loss Functions
> **English** | [繁體中文](./README.zh-TW.md)

> The core goal of a loss function is to "minimize the model's degree of surprise at reality". When the information cost is driven to its minimum, the distribution predicted by the model achieves perfect alignment with the actual distribution, and the uncertainty gap (information loss) disappears.
> The physical meaning of "information cost" (also called self-information or Surprise) is very intuitive: it measures how much "new information" or "reduction of uncertainty" an event brings when it occurs. Training a model is the process of turning the "unexpected" into the "matter of course".

## Log and Loss Functions: The Metric of Information Cost

In machine learning, the core role of the $\log$ function is to convert **probability** (the confidence value the model predicts) into **information cost** (the penalty score of the loss function). This conversion makes errors additive, makes gradient computation more stable, and is mathematically equivalent to maximum likelihood estimation (MLE).

---

## 1. Why can't loss functions live without $\log$?

$\log$ plays the role of the "scorekeeper", mapping the probability space onto the cost space:
* **Physical intuition**: $\log$ turns the "multiplication of independent probabilities" into the "addition of information content".
* **Optimization advantage**: it solves the numerical underflow problem, and provides more stable gradients (avoiding the vanishing gradient caused by the saturation region of Sigmoid/Softmax).
* **Penalty logic**:
    - When the predicted probability $Q \to 1$, $-\log Q \to 0$ (no penalty)
    - When $Q \to 0$, $-\log Q \to \infty$ (extremely large penalty).

---

## 2. Dissecting the Core Loss-Function Formulas

### A. Cross Entropy — Weighted Surprise
Used for classification tasks, measuring the difference between the model's predicted distribution and the true label.
$$H(P, Q) = -\sum P(x) \log Q(x)$$
* **$\log Q(x)$**: the information cost (surprise) predicted by the model.
* **$P(x)$**: the true label serving as a weight, so that only the cost of the "class that actually occurs" is counted.
* **Physical meaning of $\log Q(x)$**: when $P(x)$ occurs, the "information cost" (or "degree of surprise") you must pay to describe this event using the predicted distribution $Q(x)$ (i.e. the model's probability distribution).

### B. Binary Cross Entropy / NCE — The Cost of a True/False Question
Used for binary classification or large-scale negative sampling.
$$\mathcal{L} = - [ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) ]$$
* **$\log(\hat{y})$**: the cost of the model predicting "the real thing is real".
* **$\log(1-\hat{y})$**: the cost of the model predicting "the fake thing is fake".
* **Both take the $\log$ of the model's prediction.**

### C. InfoNCE — The Feature-Alignment Cost of Positive Sample Pairs
Used for contrastive learning (such as CLIP, SimCLR); the core logic is to "pull related pairs closer and push unrelated items apart" in the feature space.

$$ \mathcal{L}_{InfoNCE} = -\log \frac{\exp(sim(q, k_+) / \tau)}{\sum_{i=0}^{K} \exp(sim(q, k_i) / \tau)} $$

* **$q$**: the query vector (Anchor).
* **$k_+$**: the positive sample (positive key), corresponding to $q$.
* **$k_i$**: the sample set, containing 1 positive sample and $K$ negative samples.
* **$\tau$ (temperature parameter)**: adjusts the smoothness of the distribution. The smaller $\tau$ is, the more extreme the amplification effect of `exp`, and the more the model focuses on those hard negatives that "most resemble the positive sample".

#### The Physical Meaning of the Exponential Function `exp`
Why not use the similarity score directly, but take `exp` instead?
* **Non-negative weight conversion**: similarity (such as Cosine) may be negative, but a probability must be positive. `exp` maps the score to $(0, \infty)$, ensuring the denominator sum has physical meaning.
* **Feature magnifier**: `exp` grows nonlinearly. This means the model **disproportionately** rewards high-similarity items, and exerts a strong pull on positive samples that are "not similar enough".
* **Boltzmann Distribution**: this formula is consistent with statistical mechanics. Similarity can be viewed as "negative energy", and the system tends to place positive samples in a low-energy (high-probability) state.

| Function of exp | Problem solved | Advantage brought |
| :--- | :--- | :--- |
| Map to positive numbers | Similarity may be negative | Satisfies probability axioms, stable computation |
| Exponential amplification | Insufficient discrimination of negatives | Automatically focuses on Hard Negatives, stronger training |
| Cancels with log | Derivative formula too complex | Produces a Q−P residual, clear optimization path |


#### The Core of InfoNCE
* **Essence**: this is a **$K+1$-class Cross Entropy**.
* **Mnemonic**: `exp` is responsible for "widening the gap", and `log` is responsible for "measuring the widened gap back".
* **Final goal**: to precisely identify "our own ($k_+$)" out of "the bystanders ($k_i$)".

## 3. Quick Comparison Table

| Loss Function | Target of $\log$ | Physical Meaning | Applicable Scenario |
| :--- | :--- | :--- | :--- |
| **Cross Entropy** | predicted probability $Q$ | average communication cost under the true label | general classification (Softmax) |
| **NCE** | true/false discrimination probability | binary cost of distinguishing data from noise | large-scale classes (e.g. Word2Vec) |
| **InfoNCE** | normalized similarity value | cost of recognizing the positive sample amid negative-sample interference | self-supervised learning, feature representation |
| **MSE** | no $\log$ (implicit in the exponent of $e$) | physical distance in Euclidean space | regression, numerical prediction |

---

## 4. Key Takeaway: How to Remember It?
> **Every $\log$ is scoring the "model prediction ($Q$)". The label ($P$) then decides "which part of the score" is recorded on the final bill.**