$$
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
$$
# A Literature Survey on Domain Adaptation of Statistical Classifiers
## Domain Adaptation
- The problem that arises when the data distribution in **our test domain is different from that in our training domain**. 
- The need for domain adaptation is prevalent in many real-world classification problems. 
    - For example, spam filters can be trained on some public collection of spam and ham emails. 
    - But when applied to an individual person’s inbox, we may want to “personalize” the spam filter, i.e. to adapt the spam filter to fit the person’s own distribution of emails in order to achieve better performance.
- Special kinds of domain adaptation problems have been studied before (under diffrent names)
    - Class imbalance
        - (Japkowicz and Stephen, 2002)
    - Covariate shift
        - (Shimodaira, 2000)
    - smaple selection bias
        - (Heckman, 1979; Zadrozny, 2004)
- Also some closely-related but not equivalent ML problems that have benn studied extensively
    - multi-task learning 
        - (Caruana, 1997) 
    - semi-supervised learning
        - (Zhu, 2005; Chapelle et al., 2006)
## The Goal of this survey
1. There have been a number of methods proposed to address domain adaptation, but it is not clear how these methods are related to each other. This survey tries to organize the existing work and lay out an overall picture of the problem with it's possible solutions.
2. Second, a systematic literature survey naturally reveals the limitations of current work and points out promising directions that should be explored in the future

## Notations 
- **Source domain**
    >We refer to the training domain where labeled data is abundant(豐富) as the source domain.
- **Target domain**
    >The test domain where labeled data is not available or very little as the target domain.

- $X$
    > Let $X$ denote the input variable, i.e. an observation.
- $Y$
    > Let $Y$ denote the output variable, i.e. a class label.

- $P(X,Y)$
    > Use $P(X, Y)$ to denote the true underlying joint distribution of X and Y, which is **unknown**.
    - In domain adaptation, this joint distribution in the target ddoimain differs from that in the source domain.

        - $P_t(X, Y)$
            > Therefore, we use $P_t(X, Y)$ to denote the true underlying joint distribution in the target domain.

        - $P_s(X, Y)$
            > $P_s(X, Y)$ to denote the ture underlying joint distribution in the source domain.

- $P_t(Y), \ P_s(Y), \ P_t(X),  \ P_s(X)$
    > Denote the true marginal distributions of $Y$ and $X$ in the target and the source domains, respectively.

- $P_t(X|Y), P_s(X|Y), P_t(Y|X), P_t(Y|X)$
    > denote the true conditional distributions in the 2 domains.

- $x$
    > use lowercase x to denote a specific value of X.
    > A specific x is also referred to as an observation, an unlabeled instance or simply an instance.

- $y$
    > use lowercase y to denote a specific class label.

- $(x, y)$
    > A pair (x, y) is referred to as a labeled instance.
    - $x \in X$, where $X$ is the input space, i.e. the set of all possible observations.
    - $y \in Y$, where $Y$ is the class label set.

- $P(x, y)$
    > $P(X=x, Y=y) = P(x, y)$ should refer to the joint probability of X=x and Y=y.
    > Similarly, $P(X=x) = P(x)$, $P(Y=y) = P(y)$, $P(X=x|Y=y) = P(x|y)$ , etc also refer to probabilities rather than distributions.

- $D_s = \{(x^s_i, y^s_i)\}^{N_s}_{i=1}$
    > $D_s$ denote the set of **labeled** instances in the source domain.

- $D_{t,u}= \{x^{t,u}_i\}^{N_{t,u}}_{i=1}$
    > $D_{t,u}$ denote the set of **unlabeled** instances.

- $D_{t, l} = \{(x_i^{t,l}, y_i^{t,l})\}_{i=1}^{N_{t,l}}$
    > Sometimes, we may also have a small amount of labeled data from the target domain.
    - In the case when $D_{t,l}$ is not availble, we refer to the problem as **unsupervised domain adaptation**.
    - While when $D_t,l$ is available, we refer to the problem as **supervised domain adapatation**.

## Instance Weighting
> One general approach to addressing(解決) the domain adaptation problem is to **assign instance-dependent weights to the loss function when minimizing the expected loss over the distribution of data**.
> 解決域適應問題的一種通用方法是在最小化數據分佈的預期損失時為損失函數分配與實例相關的權重
### Why instance weighting may help?
- Review the **empirical risk mininmization framework (Vapnik, 1999)** for standard supervised learning.
- Then informally derive an instance weighting solution to domain adaptation.
- $\Theta$: $\Theta$ is a model family
- $\theta^{*}$: We want select an optimal model $\theta^{*}$ for classification task.
- $l(x, y, \theta)$: Let $l(x, y, \theta)$ be a loss function.
- We want to minimize the following objective function in order to obtain the optimal model $\theta^{*}$ for the distribution $P(X, Y)$:

$$
\theta^{*} = \argmin_{\theta \in \Theta} \sum_{(x,y) \in X \times Y} P(x, y)l(x,y,\theta)
$$
- Due to $P(X,Y)$ is unknown, we use the empirical distribution $\tilde{P}(X,Y)$ to approximate P(X,Y)
    - Let $\{(x_i, y_i)\}^N_{i=1}$ be a set of training instances randomly sampled from P(X,Y).
$$
\tilde{\theta} = \argmin_{\theta \in \Theta} \sum_{(x,y) \in X \times Y} \tilde{P}(x, y)l(x,y,\theta)
$$