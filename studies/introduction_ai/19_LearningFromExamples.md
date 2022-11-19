# Chapter 19 Learning from Examples
- [course material](https://moodle.ncku.edu.tw/pluginfile.php/864906/mod_resource/content/1/Chapter%2019%20Learning%20from%20Examples.pdf)
![](https://i.imgur.com/2zDrj7c.png)

## Supervised Learning
### Learning Decision Trees
#### Decision Tree
![](https://i.imgur.com/iJyubdt.png)
- **Input**: a vector of attribute value
- **Output**: a decision
- With some simple heuristics, we can find a good approximate solution. 
- Adopts a greedy **divide and conquer strategy**:
    - Always test the most important attribute first.
        - important attribute mean the one that makes the most difference to the classification of an example.
#### Choosing Attribute tests (for better classification)
-  Decision tree learning is designed to approximately **minimize the depth** of the final tree.
-  Pick the attribute that goes as far as possible toward providing an **exact classification** of the examples.
#### Entropy
![](https://i.imgur.com/2cClWg0.jpg)
- Entropy is a measure of **the uncertainty of a random variable**.
- Acquisition of information corresponds to a reduction in entropy.
    $$
        Entropy \uparrow, \quad Information_{acq} \uparrow
    $$
- Formula:
    \begin{align}
        H(V) &= \sum_{k}P(v_k)log_2{\frac{1}{P(v_k)}} \\ 
        &= - \sum_{k}P(v_k)log_2P(V_k)
    \end{align}

#### Generalization and Overfitting
- Decision Tree Pruning
    ![](https://i.imgur.com/J9oMc7j.png)
    - If the test appears to be **irrelevant** (detecting only noise) then we eliminate the test, replacing it with a leaf node.
        - The information gain is a good clue to irrelevent.

    
---
### Regression and Classification with Linear Models
#### Univariate linear regression
$$
    h_w(x) = w_1x +w0
$$
- finding the $h_w$ that best fits the data.
- Use squared loss function $L_2$, and minimized the loss:
    $$
        Loss(h_w) = \sum^{N}_{j=1} L_2(y_j, h_w(x_j))
    $$
    - **weight space**:
        ![](https://i.imgur.com/hcFAQhU.png)        
        - All possible settings of the weights.
    - The loss function is **convex**
        - There are **no local minima**.
        - Use hill-climbing algorithm or gradient-descent algorithm to optimize.
    - **batch gradient descent**
        $$
            w_0 \leftarrow w_0 + \alpha \sum_{j}(y_j - h_w(x_j)); \\
            w_1 \leftarrow w_1 + \alpha \sum_{j} (y_j - h_w(x_j)) \times x_j
        $$
    - **Stochastic gradient descent (SGD)**
        - It randomly selects a small number of training examples at each time step.
#### Multivariate linear regression
$$
    h_{sw}(x_j) = w_0 + \sum_{i}w_ix_{j,i}
$$
- $h_{sw}$ is simply the dot product of the weights and the input vector.
    $$
        h_{sw}(x_j) = w^\top x_j = \sum_i w_i x_{j,i}
    $$
- Minimizes loss:
    $$
        w^* = argmin_{w} \sum_{j} L_2(y_j, w^{\top} x_j)\\
        w^* = (X^{\top}X)^{-1}X^{\top}y
    $$
    - Common to use **regularization** on multivariate linear functions to avoid overfitting, counting both the empirical loss and the complexity of the hypothesis.
        $$
            Cost(h) = EmpLoss(h)+\lambda Complexity(h)
        $$
    - Complexity can be specified as a function:
        $$
            Complexity(h_w) = L_q(w) = \sum_{i}|w_i|^q
        $$
        - q = 1, $L_1$ regularization, minimizes the sum of the absolute values.
            - Advantage is that it tendst to producte a **Sparse Model**
                - which means **the weight of some parameters will be calculatd as 0**.
        - q = 2, $L_2$ regularization, minimizes the sum of squares.
#### Linear classifiers with a hard threshold
![](https://i.imgur.com/NKKQ08i.png)
- Find a **decision boundary**
    - It's a **linear separator**.
    - The data that admit linear separator are called **linearly separable**.
- formula:
    $$
        h_w(x) = Threshold(w^{\top}x)    
    $$
    - if $z \geq 0$ , $Threshold(z) = 1$
    - if $z < 0$, $Threshold(z) = 0$
    - Not differentiable.
    - Discrete distribution.
#### Linear classification with logistic regression
- **Logistic function**
    ![](https://i.imgur.com/87Rx47D.png)
    - a. $Threshold(z)$
    - b. $logistic \, function$
- **Formula**
    $$
        Logistic(z) = \frac{1}{1+e^{-z}}
    $$
- **Benefit**
    $$
        \frac{\partial}{\partial w_i} Loss(w)
    $$
    - It's differentiable.
    - Contunuous distribution.
    - The derivative $g'$ satisfies that:
        $$
            g'(z) = g(z)(1-g(z))
        $$
    
---
### Nonparametric Models
- Paramters is unbounded.
- **Table lookup**:
    - Take all the training examples put them in to lookup table.

#### Nearest neighbor models
 - **KNN** find the $k$ examples that are nearest to $x_q$
     ![](https://i.imgur.com/PoA6LwB.png)
- How to measure the distance?
    - **Minkowski distance($L^p$ norm)** 
        $$
            L^p(x_j, x_q) = (\sum_{i} |x_{j,i}- x_{q,i}|^p)^{1/p}
        $$
        - $p = 2$ is **Euclidean distance (L2 norm)**
            - [weight_decay](https://towardsdatascience.com/weight-decay-l2-regularization-90a9e17713cd)
        - $p = 1$ is **Manhattan distance (L1 norm)**
    - **Mahalanobis distance**
        - A more complex metric.
        - Takes into account hte cvariance between dimensions.
- **Benefit**
    - Nearest Neighbors works very well in low-dimensional spaces with plenty of data.
- **Weakness**
    - It's not work well in high-dimensional spaces.
    - **Curse of Dimensionality**
        $$
            \ell = (k/N)^{1/n}
        $$
        - $\ell$  average side length of a neighborhood.
        - $n$ dimensional unit hypercube.
        - $N$ points uniformly distributed.
        - When $n$ is bigger, $\ell$ is very bigger, e.g. k = 10, N=1,000,000
            | $n$ dimension | $\ell$ (% of the edge length of the unit cube)|
            |---|---|
            | n = 2 | 0.3% of the edge length of the unit cube |
            | n = 3 | 2% of the edge length of the unit cube |
            | n = 17 | 50% of the edge length of the unit cube |
            | n = 200 | 94% of the edge length of the unit cube |

#### k-d Tree
![](https://i.imgur.com/qYGRWH2.png)

- k-dimensional tree.
    - split the dimensions along the ith dimension whether $x_i \leq m$.
        - $m$ to be the median of the examples along the $i$th dimension.
        - recursively **make a tree for the left and right**.
        - stopping when there are fewer than 2 examples left.
- examples must many more than dimensions.
#### Locality-sensitive hashing (LSH)
- randomly distribute values among the bins
- near points grouped together in the same bin.
- **Approximate near-neighbors problem**
    - find an example point is near $x_q$ with high probability.
    - $x_j$ is in the radius $r$ of $x_q$.
        - if not, the algorithm report failure.
        - if true, the algorithm will have high probability to find a point $x_{j'}$ within distance cr of $x_q$.
    - hash function $g(x)$ that
        - The distance between $x_j$ and $x_{j'}$ 
            - if distance < cr, high probability have the same hash code.
            - if distance > cr, small probability have the same hash code.
### Support Vector Machines, SVMs
![](https://i.imgur.com/1YUrFVu.png)
- Construct a **maximum margin separator**.
    - The margin is the width of the area bounded by dashed lines.
- Create a **linear separating hyperplane**.
    - **Kernel Trick** the ability which  to embed the data into a higher-dimensional space.
    ![](https://i.imgur.com/dGMpbY2.png)
    - Some examples are **more important** than others, and that paying attention to them can lead to better generalization.
- SVMs attempt to minimize expected  **generalization loss**.
    - Instead of minimizing expected *empirical loss*.
---
### Ensemble Learning
- Select a collection(ensemble) of hypotheses  and combine their predictions.
![](https://i.imgur.com/rZHV3ID.png)
    - Bound the positive values in the ensemble hypothesis space.

#### Bagging
- Generate $K$ distinct training sets by sampling from the origin training set.
- It's a training strategy, not an algorithm.
- Repeat the process K times, getting K different Training Result(Hypotheses) then calculate the average:
    $$
        h(x) = \frac{1}{k} \sum_{i=1}^{K} h_i(x)
    $$
- Benefit
    - Reduce variance.
    - Standard approach when there is limited data.
    - May be able to reduce overfitting.
#### Random Forests
![](https://i.imgur.com/DsSX0u1.png)
- A form of decision tree bagging.
- Make the ensemble of K trees more diverse.
- At each split point in constructing the tree, it random sampling of attributes then compute which of those trees get the highest information gain.
- Extremely randomized trees (ExtraTrees)
    - Randomly sample from a uniform distribution then select the value with the highest information gain.
    - Every tree in the forest will be different.
- All the hyperparameters be trained by cross-validation.
- **Resistant to overfitting.**
- Breiman(2001), mathematical proof that more trees, the error converges.

#### Stacking
![](https://i.imgur.com/LKNJW6G.png)
- **Stacked generalization**:
    - Combine **differnt models classes** trained on the same data.
- e.g. 
    - Original Features:
        | Param 1 | Param 2 | Param 3 | Param 4 | y |
        |---|---|---|---|---|
        | Yes | No | No | No | Yes | Yes|
    - Add differnt models classed result as new Parameters:
        | Param 1 | Param 2 | Param 3 | Param 4 | SVMs | Logistic R | ...| y |
        |---|---|---|---|---|---|---|---|
        | Yes | No | No | No | `Yes` | `No` |...| Yes|
    - Use validation set with **new param to train a new esemble model**.
- **Benefit**
    - Reduce bias
    - Usually leads to performance that is better than others.

#### Boosting
![](https://i.imgur.com/mBtDkkY.png)
- The most widely used ensemble method.
- **ADAboost**
    - input: weak learing algorithm $L$
    - ADAboost return a hypothesis $K$ which perfectly classifies the training data. (K*L)
        ![](https://i.imgur.com/Rms4X78.png)
        - hypothesis $K$ is **NOT bigger the better**.
            
#### Online Learning
- An agent receives an input $x_j$ from nature, predicts the corresponding y , and then is told the correct answer.
- Online learning is helpful when the data may be changing rapidly over time. 
- Useful for applications that involve a large collection of data that is **constantly growing**, even if changes are gradual. 
- Minimize **Regret**(Loss)
- Boosting + **Chronological**
## Developing Machine Learning Systems
### Problem Formulation
- Need to specify a loss function which is correlated with your target.
- May find that there are multiple componential problems that can be handled by software engineering, not machine learning. 
- Part of problem formulation is deciding.
### 1. Data collection, assessment, and management
- **Data augmentation** can be help when data is limited.
    - e.g. Linear interpolation, SMOTE..., etc.
- **Unbalanced classes problem**
- You should carefully consider **outliers** in your data. 
### 2. Feature Engineering
- Preprocess the data to make it easier to digest. 
- Quantization, normalization, one-hot encoding.
- Introduce new attributes based on your domain knowledge.
### 3. Exploratory data analysis and visualization
- Cluster your data and then visualize a **prototype data** point at the center of each cluster. 
- It is also helpful to **detect outliers** that are far from the prototypes.


### 4. Model selection and training
| Model | Strength |
|---|---|
| Random forests | Have a lot of categorical features and you believe that they are irrelevant. |
| Nonparmeteric methods | Have a lot of data and no prior knowledge. |
| Logistic regression | The data are linearly separable. |
| SVMs | The data set is not too large. |
| Problems dealing with pattern recognition |  Most often approached with deep neural networks.|
- Choosing hyperparamters can be done with a combination of experience and search.

## Trust, Interpretability, Explainability
### Interpretability
- Understand why the model got a particular answer for a given input.
- You don't need to know why.
### Explainablility
- The model can help you understand **why was this output produced for this input?**