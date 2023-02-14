# Personalized Showcases: Generating Multi-Modal Explanations for Recommendations
| Title | Venue | Year | Code |
|-|-|-|-|
| [Personalized Showcases: Generating Multi-Modal Explanations for Recommendations](https://arxiv.org/pdf/2207.00422.pdf) | pre | '22 | x |

Existing explanation models generate only text for recommendations but still struggle to produce diverse contents. 
In this paper, to further enrich explanations, this paper proposes a new task named personalized showcases, in which provides both textual and visual information to explain our recommendations. 
Specifically, the paper first selects **a personalized image set that is the most relevant to a user’s interest toward a recommended item**. Then, natural language explanations are generated accordingly given our selected images.
For this new task, we collect a large-scale dataset from Google Local (i.e., maps) and construct a high-quality subset for generating multi-modal explanations. 
We propose a personalized multi-modal framework which can generate diverse and visually-aligned explanations via contrastive learning. 
Experiments show that our framework benefits from different modalities as inputs, and is able to produce more diverse and expressive explanations compared to previous methods on a variety of evaluation metrics.
## Method
![method](./assets/method.png)

| Component | Description |
|-|-|
| `Multi-Model Encoder`  | `CLIP`, a state-fo-the-art pre-trained cross-modeal model as both `textual- and visual-encoders`  |
| `Image Selection Model` | Use [DPP]() to select the image subset |
| `Visually-Aware Explanation Generation` | Generating personalized explanations given a set of images and a user's historical reviews, with the extracted explanation dataset `GEST-s2`. build with `GPT-2` as the backbone  |

### Visually-Aware Explanation Generation
#### Multi-Modal Encoder

$$
\begin{aligned}
X_u &= \{ x_1, x_2, ..., x_K \} \quad {\color{orange}\text{User's historical reviews}} \\
R &= \{ r_1, r_2, ..., r_K \} \quad {\color{orange}\text{Review features extracted from the text encoder of CLIP}} \\
I &= \{ i_1, i_2, ..., i_n \} \quad {\color{cyan}\text{Personalized Images}} \\
V &= \{v_1, v_2, ..., v_n \} \quad {\color{cyan}\text{Visual features extracted from the visual encoder of CLIP}} \\
Z^V_i &= W^V v_i \quad {\color{magenta}\text{Latent space with } V  \text{ and learnable projection matrices } W^V} \\
Z^R_i &= W^R r_i \quad {\color{magenta}\text{Latent space with } R  \text{ and learnable projection matrices } W^R} \\
Y &= \{ y_1, ..., y_{t-1} \} \quad {\color{yellow}\text{Target explanation with decoding process at each time step } t} \\ 
\end{aligned}
$$

Then use a [multi-modal attention (MMA) module with stacked self attention layers]() to encode the input features:

$$
\big[ H^V ; H^R \big] = \text{MMA}([Z^V; Z^R])
$$

| Property | Definition |
|-|-|
| `;` | denotes concatenation |
| $H^V$, $H^R$ | aggregate features from 2 modalities |

#### Multi-Model Decoder
Use `GPT-2` as the decoder for generating explanations.

$$
\hat{y}_t = \text{Decoder}({\color{magenta}[H^V; H^R]}, {\color{yellow}Y})
$$

- The loss function is : 

$$
\mathcal{L}_\text{CE} = \sum^N_{i=1} \log p_\theta (Y^i|X^i, I^i)
$$

| Property | Definition |
|-|-|
| $N$ | Number of training samples $(X^i, I^i, Y^i)^N_{i=1}$ |
| $I$ | Use `ground truth images from the user` for training <br> Use image from the `image-selection model` for inference|


## Dataset
![](./assets/datasets.png)

| Dataset | Description | Details |
|-|-|-|
| *GEST-raw* | Collected reviews with images from *Google Local*. |
| *GEST-s1* | Subset of *GEST-raw*. For *personalized image set selection*. | Remove users with only one review for building a personalized dataset, then filter out reviews whose image urls are expired. |
| *GEST-s2* | Subset of *GEST-raw*. For *visually-aware explanation generation*. |