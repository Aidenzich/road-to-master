# Chapter 13. Probabilistic Reasoning
https://moodle.ncku.edu.tw/pluginfile.php/842985/mod_resource/content/1/Chapter%2013%20Probabilistic%20Reasoning.pdf
## Bayesian Networks
- A simple Bayesian network
![](https://i.imgur.com/7XnSiDb.png)
- A Bayesian network is a graph or data structure that can effectively represent a [full joint probability distribution](https://hackmd.io/G18dRKjTQ9Kr1JbvwMZf5g?view#Bayes%E2%80%99-rule).
    - It's a ***Directed Acyclic Graph(DAG)***
- **What can a Bayesian Network do?**
    - Independence and Conditional independence relationships among variables can **greatly reduce the number of probabilities** that needed to define the full joint probability distribution.
- The full specification:
    - Each node corresponds to a random variale.
    - A set of directed links or arrows connects pairs of nodes.
    - Each node X, has a conditional probability distribution that **quantifies the effect of the it's parent nodes**.

## Approximate Inference in Bayesian Nets
## Monte Carlo algorithms
![](https://i.imgur.com/EgwNvVF.gif)
>Monte Carlo method applied to approximating the value of x
- **Sampling to Approximate**
- It's a randomized sampling algorithms. 
    - whose output may be incorrect with a certain (typically small) probability.
    - rely on repeated random sampling to obtain numerical results.
- Provide approximate answers 
    - whose **accuracy depends on the number of samples generated**.

## Direct Sampling
- Samples each variable in turn (in topological order)
    - [review](https://alrightchiu.github.io/SecondRound/graph-li-yong-dfsxun-zhao-dagde-topological-sorttuo-pu-pai-xu.html)
- The probability distribution is conditioned on the values already assigned to the variable's parents

## Markov chain Monte Carlo Algorithm (MCMC)
- Work quite differently from rejection sampling and likelihood weighting.
- Generate each sample by making a random change **to the preceding sample** instead of generate sample from scratch.
### Gibbs Sampling
![](https://i.imgur.com/EY2Mc78.png)
- A particular form of [MCMC](#Markov-chain-Monte-Carlo-Algorithm-MCMC)
- Start with an arbitrary state
- And generates a next state by randomly sampling a value for one of the nonevidence variables.
- [wiki](https://en.wikipedia.org/wiki/Gibbs_sampling)
### Addition. Markov Chain
- [Markov Chain](https://hackmd.io/VOZ1AMKRSMSFPNiRD2Fr6w?view)

    

