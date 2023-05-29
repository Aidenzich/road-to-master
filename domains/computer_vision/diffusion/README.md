# DDPM Report
| Title | Venue | Year | Code | Review |
|-|-|-|-|-|
| [DDPM, Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | NIPS | '20 | [code](./diffusion/diffusion.ipynb) | [review](./diffusion/) |
- [Math of DDPM](./proof.md)
- [⭐️ Slide](./assets/DDPM_v3.pdf)

## Contribution
- presented high quality image samples using diffusion models
- found connections **among diffusion models and variational inference** for 
    - training Markov chains
    - denoising score matching
    - annealed Langevin dynamics 
        - (and energy-based models by extension)
    -  autoregressive models
    -  progressive lossy compression 
## What is diffusion model?
![](https://i.imgur.com/P7Ei3ZD.png)
Imagine we take an image and add a bit of gaussian noise to it and repeat this many times, eventually we'll have an unrecognizable image of static a sample of pure noise.

- Diffusion model is trained to undo this process.
- Diffusion models are inspired by non-equilibrium thermodynamics. 
- Define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. 

## The components of Diffusion Model
![](https://i.imgur.com/LHj21z2.png)



### What is Forward Process?
![](https://i.imgur.com/d8x4vWN.png)
![](https://i.imgur.com/U7tzLPF.png)
- Simply, it's the process of gradually adding noise to the original image.
- The image will become complete Gaussian noise in the finally step.
- This process is unrelated to the Model and is fixed throughout the entire process.
- $x_0$ target image
- $x_T$ random gaussian noise
### What is Reverse Process?
![](https://i.imgur.com/VXVKa7m.png)
- The task of the model is to remove the noise so that it can restore the original image.
- The goal of each step in restoring the image from Gaussian noise is to turn $x_t$ back into $x_{t-1}$.
### Define $x_t$
- Initially labeled as $x_0$, the final noise is labeled as $x_T$.
- From $x_1$ to $x_T$, it is labeled as $x_{1:T}$.

## What is ELBO?
### Evidence lower bound, ELBO
<Details>
    <summary><strong>Introduction of ELBO</strong></summary>

![](https://i.imgur.com/PZaVd54.png)    
We hope that the agent `normal distribution q(Z)` and the `sampling distribution P(Z|X)` are as close as possible.
![](https://i.imgur.com/4WeO6Bh.png)
- Note that $P(Z|X)$ can be an abstract distribution
- We want that the Kullback-Leibler Divergence (KLD) between q and P(Z|X) is as small as possible.
</Details>


#### ELBO in VAE
```math
\begin{aligned}
\mathbb{E}_{q_{\phi}} \Bigg[ \log p_{\theta}(x|z)-D_{KL} \Big( q_{\phi}(z|x) \; || \; p_\theta(z) \Big) \Bigg]
&\leq \log p_\theta(x)
\end{aligned}
```

- In VAE (Variational Autoencoder), $q_\phi$ is our encoder, and $p_\theta$ is the decoder.
- We use the ELBO (Evidence Lower Bound) as the loss function to train the model so that the output gets as close as possible to the input.

#### ELBO in Diffusion Model
- Combining the concept of the Markov chain, we replace the latent factor $Z$ with $x_{1:T}$, and we get:

```math
\mathbb{E}_{q} \Big[ \log{p_{\theta}(x_0 | x_{1:T})} - D_{\text{KL}} \Big( q(x_{1:T}|x_0)|| p_{\theta}(x_{1:T}) \Big) \Big] \leq \log p_\theta(x)
```
- $x_1, ..., x_T$ are latents of the same dimensionality as the data
- $x_0 \sim q(x_0)$
- Schematic diagram:
    ![](https://i.imgur.com/whPD2HO.png)
    ![](https://i.imgur.com/U7BXD0E.png)
- Note that the encoding part of the diffusion model is fixed.

## Details of Forward Process
```math
\begin{aligned}
q(x_{1:T}|x_0) &:= \prod^T_{t=1} q(x_t|x_{t-1}), \\ 
q(x|x_{t-1}) &:= N(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
\end{aligned}
```
- The process of adding noise to the original image according to a variance schedule.
- It can be seen as the encoder of VAE (Variational Autoencoder), but it does not contain any model parameters, and it only outputs sample results according to a pre-set schedule.
- **Forward Process** is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a **variance schedule** $\beta_1, ..., \beta_T$
- Control the noise added by the forward process through a fixed variance schedule.




## Details of Reverse Process
### $ELBO = L = L_T + L_{1:T} + L_0$
- Here, after the derivation of mathematical formulas (see appendix):
```math
\begin{aligned}
\text{ELBO} &= \mathbb{E_q}\biggl [ \log{\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}}  \biggr] \\ 
&= \mathbb{E}_{q} \Big[ \log \frac{p(x_T)}{q(x_T|x_0)} + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t)} + \log p_{\theta}(x_0|x_1) \Big] \\
&= L_T + L_{1:T} + L_{0}
\end{aligned}
```

### $L_{1:T}$
When calculating $L_{1:T}$, since both $p_\theta(x_{t-1}|x_t)$ and $q(x_{t-1}|x_t)$ are normal distributions, we can directly apply the formula of KLD (Kullback-Leibler divergence) on two normal distributions to calculate $L_{t-1}$.

    
#### $L_{t-1}$
$$
L_{t-1} = D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))
$$
- Since $q(x_{t-1}|x_t, x_0)$ is the posterior distribution of the forward process, and because the variance schedule is fixed, it has tractable properties, following a normal distribution with mean $\tilde\mu$ and $\tilde\beta_t$. ([proof](https://hackmd.io/-OkX9N67Q32PwKvndoyq6A?view#Prove-span-idMathJax-Element-1-Frame-classmjx-chtml-MathJax_CHTML-tabindex0-data-mathmlqxtampx22121xtx0-rolepresentation-stylefont-size-115-position-relativeqxt%E2%88%921xtx0qxt%E2%88%921xtx0qx_t-1x_t-x_0))：

```math
\begin{aligned}
& q(x_{t-1}|x_t, x_0) = N(x_{t-1};\tilde{u_t}(x_t,x_0), \tilde\beta_t I) \\
& \tilde\mu_t(x_t, x_0) := \frac{\sqrt{\alpha_{t-1}}\beta_t{}}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\
& \tilde\beta_t := \frac{1-\tilde\alpha_{t-1}}{1-\tilde\alpha_{t}}\beta_t
\end{aligned}
```
- Substitute $\tilde\mu$ into the KLD formula:
```math
L_{t-1} = \mathbb{E}_q \Big[ \frac{1}{2\sigma^2_t}||\tilde\mu(x_t, x_0) - \mu_\theta(x_t, t)||^2 \Big]
```

#### Predict $\epsilon$ instead of predict $\mu_\theta$
- Through the nice property derived from reparameterization, we find that we can express both $\mu_\theta$ and $\tilde\mu$ in the form of $\epsilon$, $x_t$, and $\alpha$.
- Using [Reparameter trick and epsilon](https://hackmd.io/-OkX9N67Q32PwKvndoyq6A?both#Reparameter-trick) we can derive $x_t$:
```math
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
\mathbb E_{x_0, \epsilon} \Big [ \frac{\beta^2_t}{2\sigma_t^2 \alpha_t(1-\bar{\alpha_t})} ||\epsilon - \epsilon_\theta(x_t, t) ||^2 \Big]
```
- Simplify the weight to be denoted as $w_t$.
```math
\mathbb E_{x_0, \epsilon} \Big[ w_t ||\epsilon - \epsilon_\theta(x_t, t) ||^2 \Big]
```
- This paper propose that **ignore $w_t$** allowing training to focus on more challenging great noise $\epsilon$
    - Can be understood as the model predicting $\epsilon_{\theta, t}$ at time t through $x_t$ and $t$.
    - The loss function calculates the gap between the random variable $\epsilon$ and $epsilon_{\theta, t}$.
- Ignore $w_t$ to derive the training algorithm.
    ![](https://i.imgur.com/JMgdqvx.png)


## Reference
- [Diffusion Model](https://medium.com/ai-blog-tw/%E9%82%8A%E5%AF%A6%E4%BD%9C%E9%82%8A%E5%AD%B8%E7%BF%92diffusion-model-%E5%BE%9Eddpm%E7%9A%84%E7%B0%A1%E5%8C%96%E6%A6%82%E5%BF%B5%E7%90%86%E8%A7%A3-4c565a1c09c)
- [Paper](https://arxiv.org/pdf/2006.11239.pdf)
- [Github Code](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py)
- [Mathematical Symbols](https://zh-yue.wikipedia.org/wiki/%E6%95%B8%E5%AD%B8%E7%AC%A6%E8%99%9F)
- [Excellent Introduction on Youtube](https://www.youtube.com/watch?v=fbLgFrlTnGU&ab_channel=AriSeff)
- [Discussion on reddit](https://www.reddit.com/r/MachineLearning/comments/wvnnvb/d_loss_function_in_diffusion_models/)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [What is diffusion model](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)


