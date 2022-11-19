# Chapter 12. Quantifying Uncertainty
https://moodle.ncku.edu.tw/pluginfile.php/834242/mod_resource/content/1/Chapter%2012%20Quantifying%20Uncertainty.pdf
## Joint Distributions
![](https://i.imgur.com/gbGcItg.png)

### Full Joint Distributions 
![](https://i.imgur.com/Fj0TqlG.png)
- Probability of all possible worlds can be described using a table called **full joint probability distribution**.

### How is joint probability diffirent from conditional probability? 
- Conditional Probability Table CPT
- **Joint probability**
    $$
        P(A \, \cap \, B) = P(A) \times P(B)
    $$
    - is the probability of **multi things** happening together. (A, B happened at the same time.)
    
- **Conditional probability**
    $$
        P(A\,|\,B) = \frac{P(A \cap B)}{P(B)}
    $$
    - is the probability of one thing happening. (Only one of Possibility happened)



## Bayes' rule
$$
    P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$
- It allows us to compute the single term P(b|a) in terams of 3 terms.
- Bayes' rule is useful in practice **because there are many cases where we do have good probability estimates for these 3 numbers and need to compute the fourth**.
- The effect of unknown case:
$$
    P(cause|effect) = \frac{P(effect|cause)P(cause)}{P(effect)}
$$

### Naïve Bayes Models
- A single cause directly influences a number of effects.
- All of which are conditionally independent given the cause.
- The full joint distribution:
    $$
        P(Cause, Effect_1, ..., Effect_n) = P(Cause)\prod_i P(Effect_i| Cause)
    $$
    - $\prod$ is the pi-product
        - like the summation symbol $\Sigma$ 
        - rather than addition its operation is multiplication.
- Why "naive"?
    - Cause it's often used in case that the "effect" variables are **not strictly independent** given the cause variable.
    - In practice, Naive Bayes syetems often **work very well, even when the conditional independence assumption is not strictly true**.
- Sometimes called a **Bayesian Classifier**.

## WIP
### Apply Bayes' rule
wip...
### Using Bayes' rule - Combining evidence 
wip...