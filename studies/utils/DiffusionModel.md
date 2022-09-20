# Diffustion Model
![Generator Model Compare](./imgs/generator_compare.png)
- [Diffusion Model](https://medium.com/ai-blog-tw/%E9%82%8A%E5%AF%A6%E4%BD%9C%E9%82%8A%E5%AD%B8%E7%BF%92diffusion-model-%E5%BE%9Eddpm%E7%9A%84%E7%B0%A1%E5%8C%96%E6%A6%82%E5%BF%B5%E7%90%86%E8%A7%A3-4c565a1c09c)

# Denoising Diffusion Probabilistic Models
- https://arxiv.org/pdf/2006.11239.pdf
- [code](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py)
- https://huggingface.co/blog/annotated-diffusion
- https://zh-yue.wikipedia.org/wiki/%E6%95%B8%E5%AD%B8%E7%AC%A6%E8%99%9F
- [Excellent Introduction on Youtube](https://www.youtube.com/watch?v=fbLgFrlTnGU&ab_channel=AriSeff)
- [Discussion on reddit](https://www.reddit.com/r/MachineLearning/comments/wvnnvb/d_loss_function_in_diffusion_models/)
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
- ELBO 推導
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
- 展開 x:T
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
- $L = L_T + L_{1:T} + L_0$ with KLD 
    $$
    \mathbb{E}_{q} [-D_{\text{KL}}(q(x_T|x_0) || p(x_T)) - \sum_{t>1} D_{\text{KL}}(q(x_{t-1}|x_t, x_0) || p_{\theta}(x_{t-1}|x_t)) + \log p_{\theta}(x_0|x_1)]
    $$
- Reduce variance objective
    - fixed $q(x_T|x_0)$ and $p(x_T)$, which make 