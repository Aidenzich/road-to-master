# DDPM Report
## TODO
- [x] $L_{t-1}$ 證明補齊
- [x] $q(x_{t−1}|x_t,x_0)$ 證明補齊
- [x] reparameter trick 證明與介紹
- [x] 合併 first look 跟 proves
- [ ] 優化 Report
## Report Structure
- What is diffusion model
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
- 簡單來講，就是在原始圖片中逐步添加雜訊的過程
- 最終會將影像變成完整的高斯雜訊
- 這個過程跟 Model 無關，在整個過程裡是固定的
- $x_0$ target image
- $x_T$ random gaussian noise
### What is Reverse Process?
![](https://i.imgur.com/VXVKa7m.png)
- 模型的任務即將雜訊移除，使其能夠還原回原始的圖像
- 將影像從高斯雜訊還原，每一個step的目標是將 $x_t$ 還原成 $x_{t-1}$
### Define $x_t$
- 最初標為 $x_0$, 最後的雜訊標為 $x_T$
- 由 $x_1$ ~ $x_T$ 就標為 $x_{1:T}$

## What is ELBO?
### Evidence lower bound, ELBO
<Details>
    <summary><strong>Introduction of ELBO</strong></summary>
    
我們期望一代理常態分佈 q(Z) 與抽樣分佈 P(Z|X) 越接近越好
![](https://i.imgur.com/4WeO6Bh.png)
- 註 $P(Z|X)$ 可以是抽象分佈，與
- 我們希望 q 與 P(Z|X) 的 KLD 越小越好
![](https://i.imgur.com/PZaVd54.png)
</Details>


#### ELBO in VAE
$$
\begin{align}
\mathbb{E}_{q_{\phi}} \Big [\log p_{\theta}(x|z)-D_{KL} \Big( q_{\phi}(z|x) \; || \; p_\theta(z) \Big) \Big ] 
&\leq \log p_\theta(x)
\end{align}
$$

- 在VAE中 $q_\phi$ 即我們的encoder, $p_\theta$ 即 decoder
- 我們透過 ELBO 作為loss function 來絢練模型使 output 與 input 越接近越好

#### ELBO in Diffusion Model
- 結合 markov chain 的概念，將latent factor $Z$ 替換成 $x_{1:T}$，得
    $$
    \mathbb{E}_{q} \Big [ \log{p_{\theta}(x_0 | x_{1:T})} - D_{\text{KL}} \Big ( q(x_{1:T}|x_0)|| p_{\theta}(x_{1:T}) \Big ) \Big ] \leq \log p_\theta(x)
    $$
    - $x_1, ..., x_T$ are latents of the same dimensionality as the data
    - $x_0 \sim q(x_0)$
- 示意圖：
    ![](https://i.imgur.com/whPD2HO.png)
    ![](https://i.imgur.com/U7BXD0E.png)
- 要注意的是 diffusion model 的 encoding 部分是固定的

    
## Details of Forward Process
![](https://i.imgur.com/d8x4vWN.png)
![](https://i.imgur.com/U7tzLPF.png)
- 在原始圖片中依據 variance schedule 添加雜訊的過程
- 可視為 VAE 的 encoder, 但不含任何模型參數， 只是根據設定好的 schedule 輸出 sample 結果
- **Forward Process** is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a **variance schedule** $\beta_1, ..., \beta_T$
- 透過固定的 variance schedule 來控制 forward process 加上的雜訊
    $$
    \begin{aligned}
    q(x_{1:T}|x_0) &:= \prod^T_{t=1} q(x_t|x_{t-1}), \\ 
    q(x|x_{t-1}) &:= N(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
    \end{aligned}
    $$



## Details of Reverse Process
### $ELBO = L = L_T + L_{1:T} + L_0$

- 這邊經過數學式的推導(見附錄)：
$$
\begin{aligned}
\text{ELBO} &= \mathbb{E_q}\Big [ \log{\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}}  \Big{]} \\ 
&= \mathbb{E}_{q} \Big{[} \log \frac{p(x_T)}{q(x_T|x_0)} + \sum_{t>1} \log \frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t)} + \log p_{\theta}(x_0|x_1) \Big{]} \\
&= L_T + L_{1:T} + L_{0}
\end{aligned}
$$

### $L_{1:T}$
在計算 $L_{1:T}$ 時，因為 $p_\theta(x_{t-1}|x_t)$ 與 $q(x_{t-1}|x_t)$ 都是常態分佈，我們可以直接套用KLD在兩個常態分佈上的公式來計算 $L_{t-1}$：

    
#### $L_{t-1}$
$$
L_{t-1} = D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))
$$
- 由於 $q(x_{t-1}|x_t, x_0)$ 即 forward process 的後驗分佈，又因為固定 variance schedule，其具有 tractable 的性質，服從平均數$\tilde\mu$ 與$\tilde\beta_t$的常態分佈(證明見[link](https://hackmd.io/-OkX9N67Q32PwKvndoyq6A?view#Prove-span-idMathJax-Element-1-Frame-classmjx-chtml-MathJax_CHTML-tabindex0-data-mathmlqxtampx22121xtx0-rolepresentation-stylefont-size-115-position-relativeqxt%E2%88%921xtx0qxt%E2%88%921xtx0qx_t-1x_t-x_0))：
    $$
    \begin{aligned}
    & q(x_{t-1}|x_t, x_0) = N(x_{t-1};\tilde{u_t}(x_t,x_0), \tilde\beta_t I) \\
    & \tilde\mu_t(x_t, x_0) := \frac{\sqrt{\alpha_{t-1}}\beta_t{}}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\
    & \tilde\beta_t := \frac{1-\tilde\alpha_{t-1}}{1-\tilde\alpha_{t}}\beta_t
    \end{aligned}
    $$

- 將 $\tilde\mu$ 帶入 KLD 公式中：
    $$
    L_{t-1} = \mathbb{E}_q \Big[ \frac{1}{2\sigma^2_t}||\tilde\mu(x_t, x_0) - \mu_\theta(x_t, t)||^2 \Big]
    $$

#### Predict $\epsilon$ instead of predict $\mu_\theta$
- 透過由 reparameter 推出的 nice property, 得到我們可以將 $\mu_\theta$ 與 $\tilde\mu$ 都以 $\epsilon$ , $x_t$, $\alpha$ 的形式表示
- 利用 [Reparameter trick and epsilon](https://hackmd.io/-OkX9N67Q32PwKvndoyq6A?both#Reparameter-trick) 我們推導出 $x_t$:
    $$
    x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
    $$
- 最終推得 
    $$
    \mathbb E_{x_0, \epsilon} \Big [ \frac{\beta^2_t}{2\sigma_t^2 \alpha_t(1-\bar{\alpha_t})} ||\epsilon - \epsilon_\theta(x_t, t) ||^2 \Big]
    $$
- 權重簡化記成$w_t$
    $$
    \mathbb E_{x_0, \epsilon} \Big [ w_t ||\epsilon - \epsilon_\theta(x_t, t) ||^2 \Big]
    $$    
    - This paper propose that **ignore $w_t$** allowing training to focus on more challenging great noise $\epsilon$
    - 可以理解為模型透過吃 $x_t$ 與 $t$ 來預測該t時點的$\epsilon_{\theta, t}$
    - loss function 計算 random variable $\epsilon$ 與 $epsilon_{\theta, t}$ 間的差距
- 忽略 $w_t$ 得出 training algorithm
- ![](https://i.imgur.com/JMgdqvx.png)


## Reference
- [Diffusion Model](https://medium.com/ai-blog-tw/%E9%82%8A%E5%AF%A6%E4%BD%9C%E9%82%8A%E5%AD%B8%E7%BF%92diffusion-model-%E5%BE%9Eddpm%E7%9A%84%E7%B0%A1%E5%8C%96%E6%A6%82%E5%BF%B5%E7%90%86%E8%A7%A3-4c565a1c09c)
- [Paper](https://arxiv.org/pdf/2006.11239.pdf)
- [Github Code](https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py)
- [Mathematical Symbols](https://zh-yue.wikipedia.org/wiki/%E6%95%B8%E5%AD%B8%E7%AC%A6%E8%99%9F)
- [Excellent Introduction on Youtube](https://www.youtube.com/watch?v=fbLgFrlTnGU&ab_channel=AriSeff)
- [Discussion on reddit](https://www.reddit.com/r/MachineLearning/comments/wvnnvb/d_loss_function_in_diffusion_models/)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [What is diffusion model](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)


## 實作紀錄
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