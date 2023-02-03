# ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation
| Title | Venue | Year | Code |
|-|-|-|-|
| [ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation](https://arxiv.org/pdf/2209.00456.pdf) | CIKM | '22 | [✓](https://github.com/YuWang-1024/ContrastVAE) |

## Abstract

# ContrastVAE Introduction 
![](https://i.imgur.com/AxR9ntR.png)

Using the original sequence's **`origin view`** and **`augmented view`** to VAE, and get the `latent factor` $z$ and $z'$ to maximize agreement contrastively.
- **Argumented view:**
    ![strategies_of_augmentations](./assets/strategies_of_augmentations.png)

## First. Introduction of the Background 
`Sequential Recommendation (SR)` has attracted increasing attention due to its ability to **model the temporal dependencies** in `users’ clicking histories`, which can help better understand user behaviors and intentions. 
Recent research *justifies(證明)* the *promising(有希望)* ability of self-attention models [SASRec](../SASRec/), [TiSASRec](../TiSASRec/), [BERT4Rec](../Bert4Rec/) in characterizing the temporal dependencies on real-world sequential recommendation tasks.
These methods encode a sequence as an embedding via an attentionbased weighted sum of items’ hidden representations. To name a few, [SASRec](../SASRec/) is a *pioneering(開創性)* work adopting the self-attention mechanism to learn transition patterns in item sequences, and [Bert4Rec](../Bert4Rec/) extends it as a bi-directional encoder to predict the next item.

## Second. The first Supporting paragraphs of background 
> "**Identifying problems** encountered in the existing branch and proposing technologies to solve them."

Despite their great representation power, both the **`Uncertainty problem`** and the **`Sparsity Issue`** impair their performance.

| Problem | Description | example |
|-|-|-|
| **`Uncertainty Problem`** | Due to the rigorous assumption of sequential dependencies, which may be destroyed by unobserved factors in real-world scenarios. | For music recommendations, the genre of music that a user listens may vary according to different circumstances. Nevertheless, those factors are unknown and cannot be fully revealed in sequential patterns. |
| **`Sparsity Issue`** | Sparsity Issue is a long-existing and not yet a wellsolved problem in recommender systems | Supposing that a user only interacts with a few items, current methods are unable to learn high-quality representations of the sequences, thus failing to characterize sequential dependencies.  |

- Moreover, the `sparsity issue` **increases** the deficiency(不足) of uncertainty in sequential recommendation. More concretely, if a user has fewer historical interactions, those uncertain factors are of higher dominance over sequential patterns. However, these 2 issues are seldom studied simultaneously.

## Third. The second Supporting paragraphs of background
> Proposing method which can solve the identified problem mentioned in the first supporting paragraphs



Therefore, we investigate the potential of adopting **`Variational AutoEncoder (VAE)`** into sequential recommendation. The reasons are threefold. 
1. VAE **can estimate the uncertainty of the input data** (really?). More specifically, it characterizes the distributions of those `hidden representations` via an encoder-decoder learning paradigm, which assumes that those representations follow a `Gaussian distribution`. Hence, the `variances` in `Gaussian Distribution` can well characterize the uncertainty of the input data. 
    - Moreover, the decoder maximizes the expected likelihood of input data conditioned on such latent variables, which can thus **reduce the deficiency from unexpected uncertainty**. 
2. The **posterior distribution estimation in VAE decreases the vulnerability to the sparsity issue**.
    - Though a sequence contains few items, we can still characterize its distribution from `learned prior` knowledge and thus generate the next item. 
3. Probabilistic modeling of those hidden representations also **enhance the robustness of sparse data against uncertainty**. 
    - Specifically, if we can ensure the `estimated posterior` of `perturbed input` still being in distribution, the decoder in `VAE` will tolerate such perturbations and yield correct next-item prediction

| Identified Problem | Method | Reason |
|-|-|-|
| `Uncertainty Problem` | VAE characterizes the distributions of the `hidden representations` via an encoder-decoder learning paradigm.  | The `variances` in `Gaussian Distribution` can well characterize the uncertainty of the input data. |
| | | The `decoder` maximizes the expected likelihood of input data **conditioned on** the latent variables. Can reduce the deficiency from unexpected uncertainty |
| `Sparsity Issue` | The posterior distribution estimation in VAE decreases the vulnerability to the sparsity issue. | Though a sequence contains few items, we can still characterize its distribution from learned prior knowledge and thus generate the next item. |