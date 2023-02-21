| Property  | Data |
|-|-|
| Created | 2022-12-19 |
| Updated | 2022-12-19 |
| Author | @Aiden |
| Tags | #study |

# A Literature Survey on Domain Adaptation of Statistical Classifiers
[paper](http://www.mysmu.edu/faculty/jingjiang/papers/da_survey.pdf)
## The Goal of this survey
1. There have been a number of methods proposed to address domain adaptation, but it is not clear how these methods are related to each other. This survey tries to organize the existing work and lay out an overall picture of the problem with it's possible solutions.
2. Second, a systematic literature survey naturally reveals the limitations of current work and points out promising directions that should be explored in the future

## 1. Domain Adaptation
- The problem that arises when the data distribution in <font color=orange>our test domain is different from that in our training domain</font>. 
- The need for domain adaptation is prevalent in many real-world classification problems. 
    - For example, spam filters can be trained on some public collection of spam and ham emails. 
    - But when applied to an individual person’s inbox, we may want to “personalize” the spam filter, i.e. to adapt the spam filter to fit the person’s own distribution of emails in order to achieve better performance.
- Special kinds of domain adaptation problems have been studied before (under diffrent names)
    - Class imbalance <font color=blue>(Japkowicz and Stephen, 2002)</font>
    - Covariate shift <font color=blue>(Shimodaira, 2000)</font>
    - smaple selection bias <font color=blue>(Heckman, 1979; Zadrozny, 2004)</font>
- Also some closely-related but not equivalent ML problems that have benn studied extensively
    - multi-task learning <font color=blue>(Caruana, 1997)</font>
    - semi-supervised learning <font color=blue>(Zhu, 2005; Chapelle et al., 2006)</font>

## 2. Notations 
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

- $D_{t,u}= \{ x^{t,u}_i\ \}^{N_{t,u}}_{i=1}$
    > $D_{t,u}$ denote the set of **unlabeled** instances.

- $D_{t, l} = \{(x_i^{t,l}, y_i^{t,l})\}_{i=1}^{N_{t,l}}$
    > Sometimes, we may also have a small amount of labeled data from the target domain.
    - In the case when $D_{t,l}$ is not availble, we refer to the problem as <font color=green>unsupervised domain adaptation.</font>
    - While when $D_t,l$ is available, we refer to the problem as <font color=green>supervised domain adapatation</font>.

## 3. Instance Weighting
> One general approach to addressing(解決) the domain adaptation problem is to **assign instance-dependent weights to the loss function when minimizing the expected loss over the distribution of data**.

### Why instance weighting may help?
- Review the [empirical risk mininmization](https://zh.wikipedia.org/wiki/%E7%BB%8F%E9%AA%8C%E9%A3%8E%E9%99%A9%E6%9C%80%E5%B0%8F%E5%8C%96) framework (Vapnik, 1999)** for standard supervised learning.
- Then informally derive an instance weighting solution to domain adaptation.
- $\Theta$: $\Theta$ is a model family
- $\theta^{*}$: We want select an optimal model $\theta^{*}$ for classification task.
- $l(x, y, \theta)$: Let $l(x, y, \theta)$ be a loss function.
- We want to minimize the following objective function in order to obtain the optimal model $\theta^{*}$ for the distribution $P(X, Y)$:

$$
\theta^{*} = \argmin_{\theta \in \Theta} \sum_{(x,y) \in X \times Y} P(x, y)l(x,y,\theta)
$$
- Due to $P(X,Y)$ is unknown, we use the [empirical distribution]() $\tilde{P}(X,Y)$ to approximate $P(X,Y)$
    :::info
    An empirical distribution is one for which each possible event is assigned a probability derived from experimental observation. It is assumed that the events are independent and the sum of the probabilities is 1.
    :::
    - Let $\{(x_i, y_i)\}^N_{i=1}$ be a set of training instances randomly sampled from P(X,Y).
$$
\tilde{\theta} = \argmin_{\theta \in \Theta} \sum_{(x,y) \in X \times Y} \tilde{P}(x, y)l(x,y,\theta)
$$

- Ideally, we want:

$$
\theta^{*}_t = \argmin_{\theta \in \Theta} \sum_{(x,y) \in X \times Y} P_t(x, y) l(x,y,\theta)
$$

- However, the training instance $D_s=\{ (x_i^s, y_i^s)\}$ are randomly sampled from the source distribution $P_s(X,Y)$, so we write the above equation:
    
$$
\begin{align}
\theta^{*}_t &= \argmin_{\theta \in \Theta} \sum_{(x,y) \in X \times Y} \frac{P_t(x, y)}{P_s(x,y)} P_s(x, y) l(x,y,\theta) \\
&\approx \argmin_{\theta \in \Theta} \sum_{(x,y) \in X \times Y} \frac{P_t(x, y)}{P_s(x,y)} \tilde{P_s}(x, y) l(x,y,\theta) \\
&= \argmin_{\theta \in \Theta} \sum_{i=1}^{N_s} \frac{P_t(x_i^s, y_i^s)}{P_s(x_i^s, y_i^s)} l(x_i^s, y_i^s, \theta) \quad \text{(by erm)} \quad (1)
\end{align}
$$

- use [empirical risk mininmization](https://zh.wikipedia.org/wiki/%E7%BB%8F%E9%AA%8C%E9%A3%8E%E9%99%A9%E6%9C%80%E5%B0%8F%E5%8C%96)
- weighting the loss for the instance provides a well-justified solution to the problem.
- It's not possible to computer the exact value of $\frac{P_t(x,y)}{P_s(x,y)}$ for a pair $(x, y)$ (because we don't have enough labeled instances in the target domain.)

## 3.1 Class imbalance
### Assumption
- can make about the connection between the distributions of the source and the target domains is that <font color=orange>given the same class label, the conditional distributions of X are the same in the two domains</font>.

$$
P_t(X|Y=y) =  P_s(X|Y=y), \forall \ y \in Y
$$

- However, the class distributions may be different in the source and target domains.

$$
P_t(Y) \neq P_s(Y)
$$

- This difference is referred to as the class imbalance problem in some work (Japkowicz and Stephen, 2002).
---
### When above assumption is made, equation (1) can be derived as follow:

$$
\begin{align}
\frac{P_t(x, y)}{P_s(x, y)} &= \frac{P_t(y)}{P_s(y)} \frac{P_t(x|y)}{P_s(x|y)} \\
&= \frac{P_t(y)}{P_s(y)}
\end{align}
$$

- Therefore, we only use $\frac{P_t(y)}{P_s(y)}$ to weight the instances. 
    - we can <font color=orange>re-sample the training instances from the source domain</font> so that the re-sampled data roughly has the same class distribution as the target domain. 
    - In re-sampling methods, under-represented classes are over-sampled, and over-represented classes are under-sampled <font color=blue>(Kubat and Matwin, 1997; Chawla et al., 2002; Zhu and Hovy, 2007)</font>.
- This approach has been explored in <font color=blue>Lin et al., 2002</font>
---
### For classification algorithm that: 
- directly model the probability distribution $P(Y|X)$:
    -  e.g. logistic regression
    -  it can be shown theoretically that the estimated probability $P_s(y|x)$ can be transformed into $P_t(y|x)$ in the following way:
    
$$
P_t(y|x) = \frac{r(y)P_s(y|x)}{\sum_{y'\in Y}r(y')P_s(y'|x)}
$$

        where $r(y)$ is defined as:

$$
r(y) = \frac{P_t(y)}{P_s(y)}
$$

- 1. estimate $P_s(y|x)$ from the source domain.
- 2. derive $P_t(y|x)$ using $P_s(Y)$ and $P_t(Y)$
- not directly model $P(Y|X)$:
    - e.g. naive Bayes classifiers and support vector machines
    - If $P(Y|X)$ can be obtained through <font color=red>careful calibration</font>(校驗) the same trick can be applied.
    - Chan and Ng (2006) applied this method to the domain adaptation problem in word sense disambiguation (WSD) using naive Bayes classifiers.
- In Practice:
    - <font color=orange>needs to know the class distribution in the target domain</font> in order to apply the methods described above
        - In some studies, it is assumed that this distribution is known a priori <font color=blue>(Lin et al., 2002)</font>. 
        - However, in reality, we may not have this information. <font color=blue>Chan and Ng (2005)</font> proposed to use the <font color=green>EM algorithm</font> to estimate the class distribution in the target domain.

## 3.2 Covariate Shift
- Assumption one can make about about the connection between the source and the target domains is that given the same observatioin $X=x$, the conditional distributions of $Y$ are the same in the two domains, which can be write as follow:

$$
\forall \ x \in X, \ P_s(Y|X=x) = P_t(Y|X=x)
$$

    However, the marginal distrributions of X may be different in the source and the target domains:
    
$$
P_s(X) \neq P_t(X)
$$

    This difference between the two domains is called <font color=green> $covariate \ shift$ </font> <font color=blue>(Shimodaira, 2000) </font>
    
### Why would the classifier learned from the source domain not perform well on the target domain under covariate shift ?
- (At first glance, it may appear that covariate shift is not a problem.)
- The covariate shift becomes a problem when misspecified models are used.
- Suppose we consider a parameteric model family $\{P(Y| X, \theta) \in \Theta \}$ <font color=red>from which a model </font> $P(Y|X, \theta^{*})$ is selected to minimize the expected classificatioin error.
- If none of the models in the model family can exactly match the true relation between X and Y, there <font color=orange>doesn't exist  any $\theta \in \Theta$</font> that 
    $\forall x \in X, \ P(Y|X=x, \theta) = P(Y|X=x)$, we say that we have a <font color=green>misspecified model family</font>.
    - With a misspecified model family, the optimal model we select depends on $P(X)$ <font color=red>(ntp)</font> and <font color=orange>$P_s(X) \neq P_t(X)$, so the optimal model for the target domain will differ from that for the source domain</font>.
    -  the optimal model performs better in dense regions of $X$ than in sparse regions of $X$, because the dense regions dominate the average classification error, which is what we want to minimize. 
    -  If the <font color=orange>dense regions of X are different in the source and the target domains</font>, the optimal model for the source domain will no longer be optimal for the target domain.


$$
\argmin_{\theta \in \Theta} \sum_{i=1}^{N_s} \frac{P_t(x_i^s, y_i^s)}{P_s(x_i^s, y_i^s)} l(x_i^s, y_i^s, \theta) \quad (1)
$$


-  Under covariate shift, the ratio $\frac{P_t(x,y)}{P_s(x,y)}$ that we derived in Equation (1) can be rewritten as follow:

$$
\begin{align}
\frac{P_t(x, y)}{P_s(x,y)} &= \frac{P_t(x)}{P_s(x)} \frac{P_t(y|x)}{P_s(y|x)} \\
&= \frac{P_t(x)}{P_s(x)} 
\end{align}
$$
    
- therefore, we want to weight each training instance with $\frac{P_t(x)}{P_s(x)}$
- Shimodaira (2000) first propsed to reweight the log-liklihood of each training instance (x, y) using $\frac{P_t(x)}{P_s(x)}$ in maximum likelihood estimation for covariate shift.
    - If the [support](https://en.wikipedia.org/wiki/Support_(mathematics)) of $P_t(X)$ is contained in the support of $P_s(X)$, then the optimal model that maximizes this re-weighted log-liklihood function <font color=red>asymptotically</font> <font color=orange> converges to the optimal model for the target domain</font>.
    - A major challenge is how to estimate the ratio $\frac{P_t(x)}{P_s(x)}$ for each $x$ in the training set.
        - A principled method of using non-parametric kernel density estimation is explored <font color=blue>(Shimodaira, 2000; Sugiyama and Muller, 2005)</font>.
        - <font color=blue>It's proposed to transform the density atio estimation into a "problem of predicting whether an instance is from the source domain or from the target domain"  (Zadrozny, 2004; Bickel and Scheffer, 2007). </font>
        - <font color=blue> Huang et al. (2007) transformed the problem into a kernel mean matching problem in a reproducing kernel Hilbert space. </font>
        - <font color=blue> Bickel et al. (2007) proposed to learn this ratio together with the classification model parameters.</font>

## 3.3 Change of Functional Relations
- Both class imbalance and covariate shift <font color=orange>simplify the difference between $P_s(x,y)$ and $P_t(x,y)$</font>.
- It's still possible that:
    - $P_t(X|Y)$ differs from $P_s(X|Y)$
    - $P_t(Y|X)$ differs from $P_s(Y|X)$
- <font color=blue>Jiang and Zhai (2007a)</font> considered the case when $P_t(Y|X)$ differs from $P_s(Y|X)$, and <font color=orange>proposed a heuristic method to remove “misleading” training instances from the source domain</font>, where $P_s(y|x)$ is very different from $P_t(y|x)$.
    - To discover these''misleading'' training instances, some labeled data from the target domain is needed. 
    - This method therefore is <font color=orange>only suitable for supervised domain adaptation</font>.

## 4 Semi-Supervised Learning
### Semi-Supervised Learning (SSL)
Conditions:
- Ignore the domain difference.
- Treat the labeled source domain instances as labeled data.
- Treat the unlabeled target domain instances as unlabeled data.

We can then apply any SSL algorithms <font color=blue> (Zhu, 2005; Chapelle et al., 2006) </font> to the domain adaptation problem. 

### The subtile difference between SSL and domain adaptation
(1) the amount of labeled data in SSL is small but large in domain adaptation
(2) the labeled data may be noisy in domain adaptation if we do not assume $P_s(Y|X=x) = P_t(Y|X=x)$ for all x, whereas in SSL the labeled data is all reliable.

### Extending Researches
- <font color=blue>Dai et al. (2007a)</font> proposed an EM-based algorithm for domain adaptation
    - which can be shown to be equivalent to a <font color=blue>semi-supervised EM algorithm (Nigam et al., 2000) </font> 
    - except that Dai et al. proposed to estimate the trade-off parameter between the labeled and the unlabeled data <font color=orange>using the KL-divergence between the two domains</font>. 
- <font color=blue>Jiang and Zhai (2007a) </font> proposed to not only include weighted source domain instances <font color=orange>but also weighted unlabeled target domain instances in training</font>, which essentially combines instance weighting with <font color=green>bootstrapping</font>. 
- Xing et al. (2007) proposed a <font color=green>bridged refinement method</font> for domain adaptation using label propagation on a nearest neighbor graph, which has resemblance to graph-based semi-supervised learning algorithms (Zhu, 2005; Chapelle et al., 2006).


## 5 Change of Representation
- The cause of the domain adaptation problem is the difference between Pt(X, Y ) and Ps(X, Y ). 
- Note that <font color=orange>while the representation of Y is fixed, the representation of X can change if we use different features. </font> 
    - Such a change of representation of X can affect both the marginal distribution $P(X)$ and the conditional distribution $P(Y|X)$. 
    - One can assume that under some change of representation of $X$, $P_t(X, Y)$ and $P_s(X, Y )$ will become the same.
- Formally:
    - let $g : X → Z$ denote a transformation function that transforms an observation $x$ represented in the original form into another form $z = g(x) \in Z$. 
    - Define variable $Z$ and an induced distribution of Z that satisfies $P(z) = \sum_{x \in X, g(x)=z}P(x)$. 
    - The joint distribution of $Z$ and $Y$ is then:
    
$$
    P(z, y) = \sum_{x\in X, g(x)=z} P(x,y)
$$

- If we find a transformation function $g$, then $P_t(Z, Y) = P_s(Z, Y)$ and the optimal model $P(Y|Z, \theta*)$, for $P_s(Y|Z)$ is still optimal for $P_t(Y|Z)$
    - The entropy of $Y|Z$ is <font color=orange>likely to increase from the entropy of $Y|X$ </font>
        - $Z$ is usually a simpler representation of the observation than $X$
        - encodes less information, uncertainty rises
        - The Bayes error rate usually increases under a change of representation.
        
### Researches
- <font color=blue>Ben-David et al. (2007)</font> first formally analyzed the effect of representation change for domain adaptation. 
    - They proved a generalization bound for domain adaptation that is dependent on the distance between the induced $P_s(Z, Y)$ and $P_t(Z, Y)$.
- <font color=green>feature subset selection</font> is a special and simple kind of transformation, <font color=blue>Satpal and Sarawagi (2007)</font> proposed a method that <font color=orange>the criterion for selecting features is to minimize an approximated distance function between the distributions in the two domains</font>.
    - still need class labels in the target domain, they used predicted labels for the target domain instance.
- <font color=blue> Blitzer et al. (2006) </font> proposed a <font color=green>Structural Correspondence Learning (SCL)</font> algorithm that <font color=orange>makes use of the unlabeled data from the target domain to find a low-rank representation</font> that is suitable for domain adaptation. 
    - <font color=blue>Ben-David et al., 2007</font> show that <font color=orange>the low-rank representation found by SCL indeed decreases the distance</font> between the distributions in the two domains.
    - SCL doesn't directly try to find a representation Z that minimizes the distance between $P_s(Z, Y)$ and $P_t(Z, Y)$.
    - SCL tries to find a representation that works well <font color=orange>for many related classification tasks for which labels are available</font> in both the source and the target domains. 
    - The Assumption is that:
        - if $Z$ gives good performance for the many related classification tasks in both domains, then $Z$ is also good for the main classification task we are interested in both domains.
    - The core algorithm in SCL is from <font color=blue>Ando and Zhang, 2005</font>

## 6 Bayesian Priors
Review two kinds of methods that work for supervised domain
adaptation, i.e. when a small amount of labeled data from the target domain is available.
- Use the <font color=green>Maximum a Posterior (MAP)</font> estimation approach for supervised learning
    - encode some prior knowledge about the classification model into a Bayesian prior distributon $P(\theta)$:
    
$$
    \prod^N_{i=1} P(y_i | x_i; \theta) \quad (2)
$$

- [Factorial](https://zh.m.wikipedia.org/zh-tw/%E9%9A%8E%E4%B9%98) $\prod^{n}_{i=1}k = n!$
- $\theta$ : model parameter
- Instead of maximizing *equation(2)*, we maximize the following equation, specifically: 

$$
    P(\theta) \prod^N_{i=1} P(y_i | x_i; \theta)
$$

- In *domain adaptation*, we then maximize the following objective function:

$$
    P(\theta | D_s) P(D_{t,l}|\theta) = P(\theta | D_s) \prod^{N_{t,l}}_{i=1} P(y_i^t | x_i^t; \theta)
$$

- $P(\theta | D_s)$ : A Bayesian prior which is dependent on the labeled instances from the source domain

### Researches
- <font color=blue>Li and Bilmes (2007)</font> proposed a general Bayesian divergence prior framework for domain adaptation.
    - showed <font color=orange>how the general prior can be instantiated for Generator and Discriminator</font>. 
- <font color=blue>Chelba and Acero (2004)</font> applied this kind of a Bayesian prior for the task of adapting a maximum entropy <font color=red>capitalizer</font>(?) across domains.

## 7 Multi-Task Learning
- Also known as <font color=green>Transfer Learning</font>.
- The original definition of multi-task learning considers a different setting than domain adaptation. 
    - There is a single distribution of the observation, i.e. a single P(X).
    - However, a number of different variables $Y_1, Y_2, ..., Y_M$, corresponding to $M$ different tasks.
        - Which means, there are different joint distributions $\{P(X, Y_k)\}^M_{k=1}$.
            - The class label sets are for $M$ different tasks.
        - Assume that these different tasks are related.
        - Impose a common component shared by $\{\theta_k\}^{M}_{k=1}$, the $M$ conditional models are:
        
            $$
            \{ P(Y_k|X, \theta_k) \}^M_{k=1}
            $$
            
        - studies:  
            - <font color=blue>(Caruana, 1997; Ben-David and Schuller, 2003; Micchelli and Pontil, 2005; Xue et al., 2007)</font>    
- Domain adaptaion can be treated as a special case of multi-task learning.
    - Domain adaptaion have only a single task but different domain.
    - Which can be seen as one task on the source domain and the other on the target domain.
    - If we have some labeled data from the target domain, we <font color=orange>can apply some existing multi-task learning algorithm.</font>

### Researches
- <font color=blue>Daume III (2007)</font> proposed a simple method for domain adaptation based on <font color=green>feature duplications</font>.
    - The idea is to make a domain-specific copy of the original features for each domain. (?)
    - An instance from domain $k$ is then represented by <font color=orange>both the original features and the features specific to domain $k$</font>. 
    - When linear classification algorithms are used, this feature duplication based method is equivalent to <font color=orange>decomposing the model parameter for domain k $\theta_k$ into $\theta_c + \theta_k^{'}$ </font>
        - $θ_c$ is shared by all domains. 
        - Similar to the regularized multi-task learning method proposed by <font color=blue>Evgeniou and Pontil (2004)</font>.
- <font color=blue>Jiang and Zhai (2007b)</font> proposed a two-stage domain adaptation method (?)
    - The first generalization stage
        - labeled instances from $K$ different source training domains are used together to train $K$ different models
        - These models share a common component, which only applies to a subset of features that are considered generalizable across domains.

## 8 Ensemble Methods
Ensemble Methods combine a set of models to construct a complex classifier for a classification problem. Include bagging, boosting, mixture of experts, etc.
- Assume that there are a number of <font color=orange>different component distributions $\{ P^{(k)}(X,Y)\}^K_{k=1}$, each of which modeled by a simple model</font>.
    -  The distribution of X and Y in either the source domain or the target domain is then a mixture of these component distributions. 
        -  The source and the target domains <font color=orange>are related because they share some of these component distributions</font>
        -  The <font color=orange>mixture coefficients are different in the two domains </font>, making the overall distributions different.
### Researches
-  <font color=blue>Daume III and Marcu (2006)</font> proposed a mixture model for domain adaptation in which
    -  <font color=orange>three mixture components are assumed:</font>
        - one shared by both the source and the target domains
        - one specific to the source domain.
        - one specific to the target domain. 
    - Labeled data from both the source and the target domains is needed to learn this three-component mixture model using the <font color=green>conditional expectation maximization (CEM)</font> algorithm.
-  <font color=blue>Storkey and Sugiyama (2007)</font> considered a more general mixture model in which
    -  the source and the target domains <font color=orange>share more than one mixture components</font>. 
    -  However, they did not assume any target domain specific component, and as a result, <font color=orange>no labeled data from the target domain is needed</font>. 
    -  The mixture model is learned using the <font color=green>expectation maximization (EM)</font> algorithm.
-  Boosting is a general ensemble method that combines multiple weak learners to form a complex and effective classifier. 
    -  <font color=blue>Dai et al. (2007b)</font> proposed to modify the widely-used <font color=green>AdaBoost</font> algorithm to address the domain adaptation problem. 
        -  With some labeled data from the target domain, the idea here is to :
            - <font color=orange>(more weight on target)</font> put more weight on mistakenly classified target domain instances 
            - <font color=orange>(less weight on source)</font> but less weight on mistakenly classified source domain instances in each iteration 
        -  because the goal is to improve the performance of the final classifier on the target domain only.
