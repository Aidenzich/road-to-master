# Sigmoid
## Sigmoid Activation and Softmax normalization
The reason you need softmax is because your last layer of outputs has more than one neuron.
Suppose, you have a sigmoid activation in the last layer but it has two neurons. How are you going to plug this into a loss function? What is your scalar prediction?
So, with more than neuron you need a way to aggregate the outputs of the neurons into one number, a scalar, your prediction for an input sample. You can use softmax to do it. 

- Conceptually, you can think of a softmax as **an ultimate true last layer with a sigmoid activation**, it accepts outputs of your last layer as inputs, and produces one number on the output (activation). So, the softmax is a sigmoid you want.

- Not only it is a sigmoid, it's also a [multinomial logit](https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_log-linear_model).
- In other words,using softmax is exactly what you suggested: use a sigmoid and loss function.

## Softmax
Applies the `Softmax` function to an `n-dimensional` input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie **in the range [0,1] and sum to 1**.

softmax function = softargmax = normalized exponential function

## Formula

$$
\begin{aligned}
\text{Softmax}(x_i) &= \frac{\text{exp}(x_i)}{\sum_j\text{exp}(x_j)} \\
\end{aligned}
$$



```math
\sigma(x)_i = \frac{e^{x_i}}{\sum^{K}_{j=1} e^{x_j}}
\begin{aligned}
&\text{for} \ i=1, ..., K \ and  \ x = (x_1, ..., x_k) \in \mathbb{R}_k
\end{aligned}
```

## Logistic function
| Property | Description |
|-|-|
| Function | Logistic Function |
| Formula | $f(x) = 1 / (1 + e^-x)$ |
| Range | $0 <= f(x) <= 1$ |
| Domain | All real numbers |
| Characteristics | Sigmoid curve, maps any real-valued number to a value between 0 and 1, commonly used as activation function in machine learning |

The `logistic function` is a type of `sigmoid function`. The sigmoid functions have an "S" shaped curve, and the logistic function is one of the most commonly used types of sigmoid functions.

## Reference
https://stats.stackexchange.com/questions/449510/with-sigmoid-activation-and-softmax-normalization-with-cross-entropy-are-we-fit
