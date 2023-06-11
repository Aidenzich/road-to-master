# Proof
- First, we use a variational distribution $q( z, z' | x, x' )$ to approximate the posterior distribution $p(x, x' | z, z')$, which could be factorized as:

$$
q(z, z'|x, x') = q(z|x)q(z'|x')
$$

- Then, with [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality):

```math
\begin{aligned}
\log p(x, x') &= \log p(x) p(x') \\
&= \log \int p(x, x', z, z') \cdot dz \cdot dz' \\
&= \log \mathbb{E}_{q(z,z'|x,x')} \Bigg[ \frac{p(x, x', z, z')}{q(z, z' | x, x')} \Bigg] \\
&\geq \mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(x, x', z, z')}{q(z, z' | x, x')} \Bigg]
\end{aligned}
```

- And the probability distribution of $x$ and $x'$ is only related to $z$ and $z'$, $x$ and $x'$ are independent of each other, so we can derive that:

```math
\begin{aligned}            
&\geq \mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{\color{orange}p(x, x', z, z')}{\color{blue}q(z, z' | x, x')} \Bigg] \\
&= \mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{\color{orange}p(x|z)p(x'|z')p(z, z')}{\color{blue}{q(z|x) q(z'|x')}} \Bigg] \\
&= \mathbb{E_{q(z|x)}} \log [p(x|z)] + \mathbb{E_{q(z'|x')}} \log [p(x'|z')] + \color{red}{\mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(z,z')}{q(z|x)q(z'|x')} \Bigg]}
\end{aligned}
```

- The red term can be expand as:

```math
\begin{aligned}
&\color{red}{\mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(z,z')}{q(z|x)q(z'|x')} \Bigg]}  \\
&= \mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(z, z'){\color{orange}p(z)p(z')}}{{\color{orange}p(z)p(z')}q(z|x)q(z'|x')} \Bigg] \\
&= \mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(z,z')}{p(z) p(z')} \Bigg] + {\color{green} \mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(z)p(z')}{q(z|x)q(z'|x')} \Bigg]} \\
&= \mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(z,z')}{p(z) p(z')} \Bigg]  \color{green}{ - \Bigg(  D_{KL} [q(z|x)||p(z)] + D_{KL}[q(z'|x')||p(z'))] \Bigg)} 
\end{aligned}
```
- The green term can be derived by the [definition](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) of KL divergence

- Finally, expand the red term, we get:

```math
\begin{aligned}
&= \mathbb{E_{q(z|x)}} \log [p(x|z)] + \mathbb{E_{q(z'|x')}} \log [p(x'|z')] + \color{red}{\mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(z,z')}{q(z|x)q(z'|x')} \Bigg]} \\
&=  {\mathbb{E_{q(z|x)}} \log [p(x|z)]} + {\mathbb{E_{q(z'|x')}} \log [p(x'|z')]} + \color{red}{ \mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(z,z')}{p(z) p(z')} \Bigg]  - \Bigg(  D_{KL} [q(z|x)||p(z)] + D_{KL}[q(z'|x')||p(z'))] \Bigg)}  \\
&=  \mathbb{E_{q(z|x)}} \log [p(x|z)] - D_{KL} [q(z|x)||p(z)] \\ 
&+ \mathbb{E}_{q(z'|x')} \log [p(x'|z')] - D_{KL}[q(z'|x')||p(z'))] \\ 
&+ \mathbb{E}_{q(z,z'|x,x')} \log \Bigg[ \frac{p(z,z')}{p(z) p(z')} \Bigg] 
\end{aligned}
```
