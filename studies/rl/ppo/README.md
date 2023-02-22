| Property  | Data |
|-|-|
| Created | 2022-02-21 |
| Updated | 2022-02-22 |
| Author | [@MasterYee](https://github.com/Destiny0504), @Aiden |
| Tags | #study |

# PPO
| Title | Venue | Year | Code |
|-|-|-|-|
| [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf?fbclid=IwAR0JBy3rk97TCdlrTEM4ocp7wJPcytP9nbc6VVqBmoHyCkGocv6GIQkjwUs) | OpenAI | '17 | [✓](https://github.com/nikhilbarhate99/PPO-PyTorch) |
## Notation Table
| Property | Definition | 
|-|-|
| ${\color{orange}\pi_\theta}$ | A stochastic policy. (The probability that $a_t$  when give $s_t$ ) |
| ${\color{red}\pi_{\theta_\text{old}}}$ | The old stochastic policy. ${\theta}_{\text{old}}$ is the verctor of policy parameters before the update |
| $\hat{A}_t$  | An estimator of the advantage function at timestep $t$  |
| $\epsilon$ | The hyperparameter, $\epsilon$ = 0.2 |
| $\color{cyan}r_t(\theta)$ | The probability ratio $r_t(\theta) = \frac{\pi_{\theta}}{\pi_{\theta_{\text{old}}}}$, so $r_t(\theta_{\text{old}})=1$ |

### Conservative Policy Iteration 

$$
L^{CPI}(\theta) = \hat{\mathbb{E}}_t \bigg[ \frac{\color{orange}\pi_{\theta}(a_t | s_t)}{\color{red}\pi_{\theta_{old}} (a_t | s_t)} \hat{A}_t \bigg] = \hat{\mathbb{E}}_t \big[ {\color{cyan} r_t (\theta)} \hat{A_t}  \big]
$$
- conservative (保守)
- Without a constraint, maximization of LCP I would lead to an excessively large policy update.
### Clipped Surrogate Objective

![l_clip](./assets/l_clip.png)
$$
L^{CLIP} (\theta) = \hat{\mathbb{E}}_t \bigg[ min( {\color{cyan} r_t(\theta)} \hat{A}_t, {\color{orange} \text{clip}(r_t (\theta), 1 - \epsilon, 1 + \epsilon )}  \bigg] \\
\text{clip} ({\color{green}\text{Between upper and lower bounds}} , {\color{red}\text{Lower bound}}, {\color{blue}\text{Upper bound}})
$$

- The probability ratio $r$ is clipped at $1 − \epsilon $  or $1 + \epsilon$ depending on whether the advantage is positive or negative.