# BiVAE
| Title | Venue | Year | Code |
|-|-|-|-|
| [BiVAECF: Bilateral Variational Autoencoder for Collaborative Filtering](https://dl.acm.org/doi/abs/10.1145/3437963.3441759?casa_token=kx0j7ylafLoAAAAA:SMKCK053ya5QKekElJG3ScXWbJqjMp_VH0twFbLEmIOaHJiKBUHmrLJdmpqHNUQlIM6Awl84dYtXE7I) | WSDM | '21 | [✓](https://github.com/PreferredAI/cornac/tree/master/cornac/models/bivaecf) |

## Pain point in vallina VAE
- `over-regularized latent space`, also known as `posterior collapse` in the VAE, may appear due to assuming an over-simplified prior (isotropic Gaussian) over the latent space.

## Innovation
| Change | Result |
|-|-|
| `Bilateral VAE` | `Bilateral VAE` can take the form of a `Bayesian variational autoencoder` either on the `user` or `item` side. As opposed to the `vanilla VAE model`, BiVAE treated user and item similarly, making it more apt for two-way or dyadic data. |
| `Constrained Adaptive Priors (CAP)` | A mitigation of `posterior collapse` by learning user and item-dependent prior distributions. |

### Constrained Adaptive Priors (CAP)
- While the ELBO objective is theoretically sound, optimizing it in practice may result in `over-simplified representations` for users and items. This is due to the *KL terms encouraging the posteriors to
forget observations $R$ by matching them to the same simple prior distribution*.
- CAP is proposed to lower the effect of the KL regularization by adopting `user- and item-dependant priors`, which can adapt during learning. 
- Original KL Divergence:
    $$\text{KL}(q(\beta_i | r_{*i}) || p(\beta_i)) = - \log (\sigma) - \frac{1}{2} + \sigma^2 + \mu^2$$
- After CAP:
    $$\text{KL}(q(\beta_i | r_{*i}) || p(\beta_i)) = - \log (\sigma) - \frac{1}{2} + \sigma^2 + (\mu - {\color{red}\mu_{\text{prior}}} )^2$$
- CAP can seen as a form of Empirical Bayes:
    >Empirical Bayes is a statistical method that involves using observed data to estimate the parameters of a prior distribution, which is then used in a Bayesian analysis. The main idea behind empirical Bayes is to use data to estimate the prior distribution rather than relying on subjective expert knowledge or subjective assumptions. This allows for more flexible and data-driven analysis, as the prior distribution can be updated as more data becomes available. Empirical Bayes methods are often used in problems where the prior information is limited or uncertain, such as in genomic studies or in sparse data situations. In such cases, the empirical Bayes approach can lead to more accurate results compared to using a fixed prior or a prior based on subjective expert knowledge.
- NOTICE: In particular the paper build such prior using **external features**, extracted from available user and item side information. but the external features source is unknown.
