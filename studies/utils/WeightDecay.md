# Weight decay (ridge regression)
- Reference
    - [pytorch Adam](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    - [Weight Decay == L2 Regularization?](https://towardsdatascience.com/weight-decay-l2-regularization-90a9e17713cd)

## Why we need weight decay?
 - We need to minimize the parameter in $W$ and let the optimization process become much smoothly.
     - By adding another loss which is $L_n$ norm.
 - Prevent our model become overfitting. 
     - This concept is from ridge regression. 
## How to do weight decay?
Suppose our model looks like $f\left(Wx+b\right)$
- The formula of updating our parameter in our model is showed in the next line.
$W_{t+1}\ =\ W_t-\alpha\frac{\partial L\left(y,\bar{y} \right)}{\partial W_t}$ 
$W_t$ : the current matrix
$W_{t+1}$ : the next matrix
$\partial L\left(y,\bar{y} \right)$ : the loss funtion
$y$ : output of your model
$\bar{y}$ : the labeled answer

- While we are doing weight decay we change the loss function into $L\left(y,\bar{y} \right) + L_r$
- $L_r = \lambda \left \|W_t\right \|_n$
$\lambda$ : hyperparameter ( the value of weight decay )
$\left \|W_t\right \|_n$ : $L_n$ norm of $W_t$

## Why this method works?
- By adding $L_r$ , there are two way to optimize. The first way is to minimize $L\left(y,\bar{y} \right)$, the other way is to minimize $L_r$
- Minimizing $L_r$ can minimize the model output $y$. 
- Minimizing $L\left(y,\bar{y} \right)$ can let the model fit the dataset's distribution.
- Because $L_r$ is added, the model can never match the dataset's distribution. So we can also solve the overfitting problem. 
## Some other quesiton.
- Why we don't let $L_r\ =\ \lambda\left( \left \|W_t\right \|_n + \left \|b\right \|_n \right)$ ?
    - If we let the bias also be small, our distribution of model will be near to the original point. This output is not we want.
    
## Weight Decay $\neq$ L2 Regularization
Hence we conclude that though weight decay and L2 regularization may reach equivalence under some conditions still are slightly different concepts and should be treated differently otherwise can lead to unexplained performance degradation or other practical problems.

