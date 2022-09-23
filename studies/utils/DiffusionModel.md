# Denoising Diffusion Probabilistic Models
![Generator Model Compare](./imgs/generator_compare.png)
- [Diffusion Model](https://medium.com/ai-blog-tw/%E9%82%8A%E5%AF%A6%E4%BD%9C%E9%82%8A%E5%AD%B8%E7%BF%92diffusion-model-%E5%BE%9Eddpm%E7%9A%84%E7%B0%A1%E5%8C%96%E6%A6%82%E5%BF%B5%E7%90%86%E8%A7%A3-4c565a1c09c)
- [Paper](https://arxiv.org/pdf/2006.11239.pdf)
- [Github Code](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py)
- [Mathematical Symbols](https://zh-yue.wikipedia.org/wiki/%E6%95%B8%E5%AD%B8%E7%AC%A6%E8%99%9F)
- [Excellent Introduction on Youtube](https://www.youtube.com/watch?v=fbLgFrlTnGU&ab_channel=AriSeff)
- [Discussion on reddit](https://www.reddit.com/r/MachineLearning/comments/wvnnvb/d_loss_function_in_diffusion_models/)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
## Model Structure
- presented high quality image samples using diffusion models
- found connections **among diffusion models and variational inference** for 
    - training Markov chains
    - denoising score matching
    - annealed Langevin dynamics 
        - (and energy-based models by extension)
    -  autoregressive models
    -  progressive lossy compression 
## Derivation
- **ELBO 推導**
    $$
    \begin{aligned}
    & \mathbb{E}_{q} \Big [ \log{p_{\theta}(x_0 | x_{1:T})} - D_{\text{KL}} \Big ( q(x_{1:T}|x_0)|| p_{\theta}(x_{1:T}) \Big ) \Big ] \\
    &= \mathbb{E}_{q} \Big [ \log{p_{\theta}(x_0|x_{1:T})} \Big ] - \mathbb{E}_{q} \Big [ \log{\frac{q(x_{1:T}|x_0)}{p_{\theta}(x_{1:T})}} \Big ] \\
    &= \mathbb{E}_{q} \Big [ \log{p_{\theta}(x_0|x_{1:T})} + \log \frac{p_{\theta}(x_{1:T})}{q(x_{1:T}|x_0)} \Big ] \\
    &= \mathbb{E_q}\Big [ \log{\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}}  \Big{]} 
    \end{aligned} 
    $$
    - **註**：可以推導出最後一項是因為 $p(x_{1:T})$ 展開為 $p(x_1 | x_{2:T}) \times ... \times p(x_{T-1} | x_{{T-1}:T})$ 呈上 $p(x_0 | x_{1:T})$
    $$p_{\theta}(x_{0:T}) := p(x_T) \prod_{t=1}^{T} p_{\theta}(x_{t-1}|x_t)$$
- **展開 x:T**
    $$
    \begin{aligned}
    L &= - \mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t \geq 1} \log{\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}} \Big{]} \\
    &= - \mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t > 1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} + \log \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} \Big{]} \\
    &= - \mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_{t})}{q(x_{t-1}| x_t) } \cdot \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} + \log \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} \Big{]} \\
    &= - \mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t)} + \log \frac{q(x_{1}|x_{0})}{q(x_2|x_0)} \cdot \frac{q(x_2|x_0)}{q(x_3|x0)} \cdot \text{...} \cdot \frac{q(x_{T-1}|x_0)}{q(x_T|x_0)}  + \log \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} \Big{]} \\
    &= - \mathbb{E}_{q} \Big{[} \log p(x_T) + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t)} + \log \frac{q(x_{1}|x_{0})}{q(x_T|x_0)}  + \log \frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)} \Big{]} \\
    &= - \mathbb{E}_{q} \Big{[} \log \frac{p(x_T)}{q(x_T|x_0)} + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t)} + \log p_{\theta}(x_0|x_1) \Big{]} \\
    &= L_T + L_{1:T} + L_0
    \end{aligned}
    $$
- **$L_T + L_{1:T} + L_0$ in KLD**
    $$
    \mathbb{E}_{q} \big{[} -D_{\text{KL}}(q(x_T|x_0) || p(x_T)) - \sum_{t>1} D_{\text{KL}}(q(x_{t-1}|x_t, x_0) || p_{\theta}(x_{t-1}|x_t)) + \log p_{\theta}(x_0|x_1) \big{]}
    $$
### Reduce variance objective
#### $L_T$ can be ignore
$$
D_{KL}(q(x_T|x_0) || p(x_T))
$$
- $q(x_T|x_0)$ & $p(x_T)$ is fixed, and no parameters $\theta$ need to be trained, so $L_T$ is a const and can be ignore.


#### $L_0$ seen as tranform
#### $L_{1:T}$
- reverse
    $$
    p_\theta(x_{t-1}|x_t) := N(x_{t-1};\mu_{\theta}(x_t,t), \Sigma_{\theta}(x_t, t))
    $$
    - $\Sigma_{\theta}$ is fixed
    - $\theta$ only use to learn predict $\mu$
- forward (gradully adds Gaussian noise)
    $$
    \begin{aligned}
        q(x_t|x_{t-1}) &:= N(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
    \end{aligned}
    $$
    - notation $\alpha_t$
        $$
        \alpha_t := 1 - \beta_t
        $$
        - $\bar{\alpha}$
            $$
            \bar{\alpha}_{t} = \prod^t_{s=1} \alpha_s
            $$
    - Using notation $\bar\alpha_t$, we have:
        $$
        q(x_t|x_0) = N(x_t; \sqrt{\bar\alpha_t}x_{0}, (1-\bar{\alpha_t})I)
        $$
    - they then **suggest a reparameterization** that aims to have the network predict the noise that was added rather tahn the gaussian mean $\mu$
        $$
        x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \ \epsilon,\ \epsilon \sim N(0, 1)
       $$
    - reverse model's loss
        $$        
        \text{loss} = \mathbb{E}_{x_0, \epsilon, t}\big[ w_t || \epsilon - \epsilon_{\theta}(x_t, t) ||^2 \big]        
        $$
        - $w_t$ step-specific weighting from variational lower bound.
        - $\theta$ model is simply use to predict the epsilon
    - A simpler version of the variational bound that discards the term weights that appear in the original bound led to better sample quality.
        $$
            \text{loss}_{\text{simple}} = \mathbb{E}_{{x_0},{\epsilon},t}\big[||\epsilon - \epsilon_\theta(x_t, t)||^2 \big]
        $$

## Model
![](https://i.imgur.com/UwUAoKA.png)
- $x_0$ target image
- $x_T$ random gaussian noise
- forward process $q$
    - fixed
    - 將影像變成高斯雜訊
- reverse process $p_{\theta}$
    - 將影像從高斯雜訊還原，目標也就是將 $X_t$ 還原成 $X_{t-1}$
    - 模型是學習如何讓雜訊還原回去

## Report structure
- What are diffusion Models?
- Model Structure
    - Forward process
        - Posterior of forward process
    - Reverse process
    - Variational lower bound
        - Reduced variance objective
        - Reverse step implementation
        
## 實作
### p sample
$$
P_\theta
$$
- reverse
- p_losses
    - model 接 x_noisy 與 time torch
    - x_noisy 來自 q_sample 自 batch image
    - 算 Model 預測的 noisy 與 x_noisy 間的誤差

每一張預測的


### q sample
- forward
- fixed
#### schedule
- 設定好 timestep 總數
- linear_beta_schedule
#### sample from noise
- noise 與 input 大小相同

### Inference
- p_sample_loop 回傳 timestep 長度的影像陣列
    - 每個元素都是該 step 下的圖片
    - 在 inference 也是跑跟訓練時相同的timestep (實際上也可以跳過部分step來訓練)
    