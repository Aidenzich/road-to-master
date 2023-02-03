

## 第四節
However, `conventional VAE` suffers from posterior collapse issues [21, 46]. 
Concretely, if the decoder is sufficiently expressive, the estimated posterior distributions of latent factors tend to resemble the standard Gaussian distributions, i.e., these estimations are indistinguishable from each other as they follow the same distribution [2, 21]. 

Furthermore, VAE might collapse to point estimation for rare classes that simply memorize the locations in latent space. The highly skewed distribution of user behaviors will exaggerate these problems. 

Specifically, the sequential input data consists of long-tail items [20], which refer to the infrequent items that rarely appear in the users’ historical records. 

Such items account for a large portion of all items. These limitations prevent VAE from achieving
satisfactory performance for SR tasks.

## 第五節
Recent advances in adopting contrastive learning (CL) for alleviating representation degeneration problem [29] motivate us to resort to contrastive learning to mitigate the above issues. Concretely, contrastive learning encourages the uniform distribution of latent representations of different inputs [38], thus enforcing them distinguishable in latent space. 
Besides, augmentations in CL encourage perturbed sequences to share similar representations, thus being robust to a large variance of the estimated posterior distribution. 
To incorporate contrastive learning into the framework of VAE, we first extend the conventional single-variable ELBO to the two-view case and propose ContrastELBO. 
We theoretically prove that optimizing ContrastELBO induces a mutual information maximization term, which could be effectively optimized with
CL [28, 37].


## 第六節
To `instantiate(實作)` `ContrastELBO` for `Sequential Recommender`, we propose `ContrastVAE`, a **two-branched(?)** VAE model that naturally incorporates CL. `ContrastVAE` takes two augmented views of a sequence as input and follows the conventional(傳統的) encoder-sampling-decoder architecture to generate the next predicted item. 
- The model is learned through optimizing an additional contrastive loss between the latent representations of two views in addition to the **vanilla(having no special or extra features; ordinary or standard.)** reconstruction losses and KL-divergence terms. 
- To deal with the potential inconsistency problem led by uninformative data augmentations, we further propose two novel augmentation strategies: 
    - model augmentation and variational augmentation: which introduce perturbations in the latent space instead of the input space. 
- We conduct comprehensive experiments on four benchmark datasets, which verify the effectiveness of the proposed model for sequential recommendation tasks, especially on recommending long-tail items. 

The contributions of this paper are summarized as follows:
- We derive `ContrastELBO`, which is an extension of conventional single-view ELBO to two-view case and naturally incorporates contrastive learning into the framework of VAE.
- We propose `ContrastVAE`, a two-branched VAE framework guided by `ContrastELBO` for sequential recommendation.
- We introduce model augmentation and variational augmentation to avoid the `semantic inconsistency problem` led by conventional data augmentation.
- We conduct comprehensive experiments to evaluate our method. The results show that our model achieves stateof-the-art performance on 4 SR benchmarks. 
- Extensive ablation studies and empirical analysis verify the effectiveness of the proposed components.