# ELBO
Please take a look at my [DDPM proof]((../../cv/diffusion/prove.md)) fully ELBO derivation.

## Variational Inference
![](./assets/vi1.png)
![](./assets/vi2.png)
![](./assets/vi3.png)
![](./assets/vi4.png)
![](./assets/vi5.png)

## The relationship between VAE and GMM
![](./assets/vae_gmm.png)

## ELBO in VAE
$$ 
\begin{aligned}  
D_\text{KL}( q_\phi({z}|{x}) \| p_\theta({z}|{x}) )  &=\int q_\phi({z}|{x}) \log\frac{q_\phi({z} | {x})}{\color{teal}{p_\theta(z|x)}} d{z}  \\ 
&=\int q_\phi({z} | {x}) \log\frac{q_\phi({z} | {x})\color{teal}{p_\theta(x)}}{\color{teal}{p_\theta(z, x)}} d{z}  \\ 
&=\int q_\phi({z} | {x}) \big( \log p_\theta({x}) + \log\frac{q_\phi({z} | {x})}{p_\theta({z}, {x})} \big) d{z}  \\ 
&=\log p_\theta({x}) + \int q_\phi({z} | {x})\log\frac{q_\phi({z} | {x})}{\color{teal}{p_\theta(z, x)}} d{z} \\ 
&=\log p_\theta({x}) + \int q_\phi({z} | {x})\log\frac{q_\phi({z} | {x})}{\color{teal}{p_\theta(x|z)p_\theta(z)}} d{z} \\ 
&=\log p_\theta({x}) + \mathbb{E}_{{z}\sim q_\phi({z} | {x})}\Big[\log \frac{q_\phi({z} | {x})}{p_\theta({z})} - \log p_\theta({x} | {z})\Big] \\ 
&=\log p_\theta({x}) + D_\text{KL}(q_\phi({z}|{x}) \| p_\theta({z})) - \mathbb{E}_{{z}\sim q_\phi({z}|{x})}\log p_\theta({x}|{z})
\end{aligned} 
$$

- Because $D_{KL} \geq 0$:
    $$
        \begin{aligned}
        \log p_\theta({x}) &+ D_\text{KL}(q_\phi({z}|{x}) \| p_\theta({z})) - \mathbb{E}_{{z}\sim q_\phi({z}|{x})}\log p_\theta({x}|{z}) \geq 0 \\
        \log p_\theta(x) &\geq \color{orange}{\mathbb{E}_{{z}\sim q_\phi({z}|{x})}\log p_\theta({x}|{z}) - D_\text{KL}(q_\phi({z}|{x}) \| p_\theta({z}))} \\
        &= \mathbb{E}_{q_{\phi}} \Big [\log p_{\theta}(x|z)-D_{KL} \Big( q_{\phi}(z|x) \; || \; p_\theta(z) \Big) \Big ] 
        \end{aligned}
    $$

## Reparameterization trick
$$
\begin{aligned}
z \sim q_\phi(z|x) &= N(z; \mu, \sigma^2 I) \\
z &= \mu + \sigma \odot \epsilon \\
\epsilon &\sim N(0, I)
\end{aligned}
$$
![reparameterization_trick](./assets/reparameterization_trick.png)
- In VAE, the reparameter trick is used to sample the latent variable z through the generated $\mu$ and $\sigma$ from the encoder.