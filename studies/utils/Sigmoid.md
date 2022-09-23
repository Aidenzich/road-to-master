# Sigmoid
## Sigmoid Activation and Softmax normalization
The reason you need softmax is because your last layer of outputs has more than one neuron.
Suppose, you have a sigmoid activation in the last layer but it has two neurons. How are you going to plug this into a loss function? What is your scalar prediction?
So, with more than neuron you need a way to aggregate the outputs of the neurons into one number, a scalar, your prediction for an input sample. You can use softmax to do it. 

- Conceptually, you can think of a softmax as **an ultimate true last layer with a sigmoid activation**, it accepts outputs of your last layer as inputs, and produces one number on the output (activation). So, the softmax is a sigmoid you want.

- Not only it is a sigmoid, it's also a [multinomial logit](https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_log-linear_model).
- In other words,using softmax is exactly what you suggested: use a sigmoid and loss function.

## Softmax
Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie **in the range [0,1] and sum to 1**.
$$
\text{Softmax}(x_i) = \frac{\text{exp}(x_i)}{\sum_j\text{exp}(x_j)}
$$
## Reference
https://stats.stackexchange.com/questions/449510/with-sigmoid-activation-and-softmax-normalization-with-cross-entropy-are-we-fit