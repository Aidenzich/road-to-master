# DDPM Prove
https://github.com/lilianweng/lilianweng.github.io/blob/master/posts/2021-07-11-diffusion-models/index.html
https://www.overleaf.com/learn/latex/Using_colours_in_LaTeX
## Groud Truth
$$
    \begin{aligned}
    {\color{blue}{p_{\theta}(x_{0:T})}} &:= p(x_T) \prod_{t=1}^{T} p_{\theta}(x_{t-1}|x_t), \quad p_\theta(x_{t-1}|x_t) := N(x_{t-1};\mu_{\theta}(x_t,t), \Sigma_{\theta}(x_t, t)) \\
    \color{red}{q(x_{1:T}|x_0)} &:= \prod_{t=1}^T q(x_t|x_{t-1}), \quad q(x_t|x_{t-1}) := N(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
    \end{aligned}
$$
### Variational Schedule
- notation $\alpha_t$
    $$
        \alpha_t := 1 - \beta_t, \quad
        \bar{\alpha}_{t} = \prod^t_{s=1} \alpha_s
    $$
- Using $\bar\alpha_t$, we have:
    $$
        q(x_t|x_0) = N(x_t; \sqrt{\bar\alpha_t}x_{0}, (1-\bar{\alpha_t})I)
    $$


## $\text{ELBO}$
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

- 因為$D_{KL}$ 大於等於 0
    $$
        \begin{aligned}
        \log p_\theta({x}) &+ D_\text{KL}(q_\phi({z}|{x}) \| p_\theta({z})) - \mathbb{E}_{{z}\sim q_\phi({z}|{x})}\log p_\theta({x}|{z}) \geq 0 \\
        \log p_\theta(x) &\geq \color{orange}{\mathbb{E}_{{z}\sim q_\phi({z}|{x})}\log p_\theta({x}|{z}) - D_\text{KL}(q_\phi({z}|{x}) \| p_\theta({z}))} \\
        &= \mathbb{E}_{q_{\phi}} \Big [\log p_{\theta}(x|z)-D_{KL} \Big( q_{\phi}(z|x) \; || \; p_\theta(z) \Big) \Big ] 
        \end{aligned}
    $$

## ELBO in Diffusion
$$
        \begin{aligned}
        & \mathbb{E}_{q} \Big [ \log{p_{\theta}(x_0 | x_{1:T})} - D_{\text{KL}} \Big ( q(x_{1:T}|x_0)|| p_{\theta}(x_{1:T}) \Big ) \Big ] \\
        &= \mathbb{E}_{q} \Big [ \log{p_{\theta}(x_0|x_{1:T})} -  \log{\frac{q(x_{1:T}|x_0)}{p_{\theta}(x_{1:T})}} \Big ] \\
        &= \mathbb{E}_{q} \Big [ \log{\color{teal}{p_{\theta}(x_0|x_{1:T})}} + \log \frac{\color{teal}{p_{\theta}(x_{1:T})}}{q(x_{1:T}|x_0)} \Big ] \\
        &= \mathbb{E_q}\Big [ \log{\frac{\color{blue}{p_{\theta}(x_{0:T})}}{\color{red}{q(x_{1:T}|x_0)}}}  \Big{]} \\
    &= \mathbb{E_q}\Big[ \log \frac{p_\theta(x_T) p_\theta(x_0|x_1)...p_\theta(x_{T-1}|x_T)}{q(x_1|x_0)q(x_2|x_1)...q(x_T|x_{T-1})} \Big] \\
&= \color{orange}{\mathbb{E}_q \Big[ \log p_{\theta}(x_T) + \sum_{t \geq 1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} \Big]}
\end{aligned}
$$

## Split $\text{ELBO}$  to $L_T + L_{1:T} + L_0$
$$
\begin{aligned}
L &= - \color{orange}{\mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t \geq 1} \log{\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}} \Big{]}} \\
&= - \mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t > 1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} + \log \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} \Big{]} \\
&= - \mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_{t})}{q(x_{t-1}| x_t) } \cdot \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} + \log \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} \Big{]} \\
&= - \mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t)} + \log \frac{q(x_{1}|x_{0})}{\color{teal}{q(x_2|x_0)}} \cdot \frac{\color{teal}{q(x_2|x_0)}}{\color{teal}{q(x_3|x0)}} \cdot \text{...} \cdot \frac{\color{teal}{q(x_{T-1}|x_0)}}{q(x_T|x_0)}  + \log \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} \Big{]} \\
&= - \mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t)} + \log \frac{q(x_{1}|x_{0})}{q(x_T|x_0)}  + \log \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} \Big{]} \\
&= - \mathbb{E}_{q} \Big{[} \log \frac{p(x_T)}{q(x_T|x_0)} + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t)} + \log p_{\theta}(x_0|x_1) \Big{]} \\
&= \mathbb{E}_{q} \Big[ D_{\text{KL}}(q(x_T|x_0) || p(x_T)) + \sum_{t>1} D_{\text{KL}}(q(x_{t-1}|x_t, x_0) || p_{\theta}(x_{t-1}|x_t)) - \log p_{\theta}(x_0|x_1) \Big] \\
&= L_T + L_{1:T} + L_0
\end{aligned}
$$


## Reduce training objective
### $L_T$ can be ignore
$$
D_{KL}(q(x_T|x_0) || p(x_T))
$$
- $q(x_T|x_0)$ & $p(x_T)$ is fixed, and no parameters $\theta$ need to be trained, so $L_T$ is a const and can be ignore.

### $L_0$ seen as transform
...
### $L_{t-1}$
$$
\begin{aligned}
L_{t-1} &= \mathbb{E}_q \Big[ \frac{1}{2\sigma^2_t}||\tilde\mu(x_t, x_0) - \mu_\theta(x_t, t)||^2 \Big] \\
L_{t-1} &= \mathbb{E}_q \Big[ \frac{1}{2||\Sigma_{\theta}(x_t, t)||^2}||\tilde\mu(x_t, x_0) - \mu_\theta(x_t, t)||^2 \Big] 
\end{aligned}
$$
## Prove $q(x_{t-1}|x_t, x_0)$
The reverse conditional probability is tractable when conditioned on $x_0$:
$$ 
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I}) 
$$  

- Using Bayes' rule and Gaussian Density Function, we have:
$$ 
\begin{aligned} 
q(x_{t-1} | x_t, x_0) &= \color{orange}{q(x_t|x_{t-1}, x_0)} \times
    \frac{ \color{green}{q(x_{t-1}|x_0) }}{\color{teal}{q(x_t|x_0)}} \\ 
&\color{green}{\propto} \exp \Big(-\frac{1}{2} \big(\frac{(x_t - \color{orange}{ \sqrt{\alpha_t} x_{t-1} })^2}{\color{orange}{\beta_t}} + \frac{(x_{t-1} - \color{green}{\sqrt{\bar{\alpha}_{t-1}} x_0})^2}{\color{green}{ 1-\bar{\alpha}_{t-1} } } - \frac{(x_t - \color{teal}{\sqrt{\bar{\alpha}_t} x_0 } )^2}{\color{teal}{1-\bar{\alpha}_t}} \big) \Big) \\ 
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2} }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\ 
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)} \end{aligned} 
$$
- **Left term** of PDF:
    $$
    \begin{aligned}
    & \color{orange}{\frac{1}{\sigma_{q(x_t|x_{t-1}, x_0)}\sqrt{2\pi}}} \color{green}{\frac{1}{\sigma_{q(x_{t-1}|x_0)}\sqrt{2\pi}}} \color{teal}{\frac{1}{\sigma_{q(x_t|x_0)}\sqrt{2\pi}}} \\
    &= \color{orange}{\frac{1}{\beta_t\sqrt{2\pi}}} \color{green}{\frac{1}{(1 - \bar\alpha_{t-1})\sqrt{2\pi}}} \color{teal}{\frac{1}{ (1- \bar\alpha_t)\sqrt{2\pi}}}
    \end{aligned}
    $$
- $C$ 是常數可以忽略
- 未知的 $q(x_{t-1}|x_0, x_0)$ 轉換成可用已知 $q(x_t|x_{t-1})$ 與 $q(x_t|x_0)$ 的型式
    ![](https://i.imgur.com/VbCDgBu.png)
- The $\color{blue}{\tilde\mu(x_t, x_0)}, \color{red}{\tilde\beta_t}$ can be parameterized  to $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)'s$ PDF as below:
$$
\begin{aligned} 
    \color{red}{\tilde{\beta}_t} &= 1/\color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} = 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})}) = \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\ 
    \color{blue}{\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)} &= \color{blue}{(\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)}/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\ 
    &= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\ 
    &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\ 
\end{aligned} 
$$

<Details>
<summary>Appendix</summary>

- $q(x_t|x_{t-1})$
$$
    q(x_t|x_{t-1}) := N(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

- $q(x_t|x_0)$
$$
\alpha_t := 1 - \beta_t, \quad
\bar{\alpha}_{t} = \prod^t_{s=1} \alpha_s \\
q(x_t|x_0) = N(x_t; \sqrt{\bar\alpha_t}x_{0}, (1-\bar{\alpha_t})I)
$$
- PDF
    $$
    \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
    $$
    - $exp(x) = e^x$
- $\propto$ 正比於

</Details>

<Details>
<summary>Appendix. KL Divergence between 2 Gaussians</summary>
    
$$
\text{log} \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma^2_2} - \frac{1}{2}
$$
- when $\sigma_1 = \sigma_2$
    $$
    \log \frac{\sigma_1}{\sigma_1}  + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_1^2} - \frac{1}{2} = \frac{1}{2\sigma_1^2}(\mu_1 - \mu_2)^2
    $$
</Details>
<Details>
<summary>Appendix. Read NormalDistribution Symbol</summary>

$$
N(x; \mu, \sigma^2)
$$
- First term x is the input
- Second term is the mean of the distribution
- Second term is the variance of the distribution
</Details>

## Reparameterize trick
$$
\begin{aligned}
z \sim q_\phi(z|x) &= N(z; \mu, \sigma^2 I) \\
z &= \mu + \sigma \odot \epsilon \\
\epsilon &\sim N(0, I)
\end{aligned}
$$
- 在 VAE 中，reparameter trick 是透過encoder 生成的 $\mu$ 與 $\sigma$ 來 sample 出 latent variable z 
- 但在 diffusion model 中，因為 forward(encoding) 的結果固定，且每個時間點的 $\sigma$ 取決於該時點固定的 $\beta$ 值，可得$x_t$ 與 $x_{t-1}$ 的關係為($x_{t}$的$\mu$ 是 $x_{t-1}$)：
    $$
    \begin{aligned}
    x_t &= \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1} \\
    &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1} \\
    &= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\bar\epsilon_{t-2} \\
    &= ... \\
    &= \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon \\
    \color{orange}{x_0} &= \frac{1}{\sqrt{\bar\alpha_t}}(x_t - \sqrt{1-\bar\alpha_t}\epsilon)
    \end{aligned}
    $$
- DDPM **suggest a reparameterization** that aims to have the network predict the noise that was added rather than the gaussian mean $\mu$

## Convert $\tilde\mu_t$, $\mu_{\theta}$ in $\epsilon$ form
${\tilde{\boldsymbol{\mu}}_t (x_t, x_0)}$
$$
\begin{aligned}
\color{blue}{\tilde{\boldsymbol{\mu}}_t (x_t, x_0)} &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \color{orange}{x_0} \\ 
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \color{orange}{\frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)} \\
&= \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \Big)}
\end{aligned}
$$

${{\boldsymbol{\mu}}_{\theta} (x_t, t)}$
- 因為 $x_t$, $t$ 都為已知項，可在訓練時當輸入，因此 DDPM 將模型改成以 $x_t$, $t$ 去預測 $\epsilon_t$
- 使用與 ${\tilde{\boldsymbol{\mu}}_t (x_t, x_0)}$ 相同的運算式，但將$\epsilon_t$ 更改為 $\epsilon_\theta$：

$$
\color{violet}{\mu_\theta(x_t, t)} = \frac{1}{\sqrt{\alpha_t}} \Big( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}} \color{violet}{\epsilon_\theta(x_t, t)}  \Big)
$$

## Rewrite $L_{t-1}$ with $\epsilon$
$$ 
\begin{aligned} 
L_{t-1} &= \mathbb{E}_q \Big[ \frac{1}{2\sigma^2_t}||\color{blue}{\tilde\mu(x_t, x_0)} - \color{violet}{\mu_\theta(x_t, t)} ||^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \sigma^2_t } \| \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \Big)} - \color{violet}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\ 
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \sigma^2_t \alpha_t (1 - \bar{\alpha}_t)} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\ 
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\color{orange}{\frac{ (1 - \alpha_t)^2 }{2 \sigma^2_t }\alpha_t (1 - \bar{\alpha}_t) } \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, t)\|^2 \Big] \end{aligned} 
$$
### Simplified step-specific weighting from ELBO
$$        
        \text{loss} = \mathbb{E}_{x_0, \epsilon, t}\Big[ \color{orange}{w_t} || \epsilon - \epsilon_{\theta}(x_t, t) ||^2 \Big]
$$
- DDPM  find that a simpler version of the variational bound that discards the term weights that appear in the original bound led to better sample quality.
    $$
    \text{loss}_{\text{simple}} = \mathbb{E}_{{x_0},{\epsilon},t}\big[||\epsilon - \epsilon_\theta(x_t, t)||^2 \big]
    $$