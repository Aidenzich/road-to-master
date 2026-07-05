# xLSTM — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | xLSTM: Extended Long Short-Term Memory |
| Venue | Neural Information Processing Systems (NeurIPS) |
| Year | 2024 |
| Authors | Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter |
| Official Code | https://github.com/NX-AI/xlstm |
| Venue Kind | paper |

> This is a research note based on the arXiv full text (`2405.04517`, first version 2024-05-07). The official NeurIPS 2024 version is cross-confirmed via Semantic Scholar's publication venue field; as of 2026-07-03, that record reports about 632 citations (Semantic Scholar, subject to change over time).

## First Principles

### From the constant error carousel to three fatal limitations

The core of the LSTM is the constant error carousel and gating proposed in the 1990s: the cell state is updated additively, with sigmoid gates controlling writing and forgetting, thereby circumventing the vanishing-gradient problem of RNNs. The authors argue that the LSTM has three main limitations: (i) it cannot revise a stored decision (storage decision), because a sigmoid forget gate has difficulty overwriting an old value when a more similar vector appears; (ii) its storage capacity is limited, with information compressed into a scalar cell state; (iii) it cannot be parallelized because of memory mixing (hidden-hidden connections). The entire xLSTM paper aims to relax these three points one by one while preserving the spirit of the LSTM.

The cell state and hidden state updates of the original LSTM can be written as below, where $f_t, i_t, o_t$ are sigmoid gates, $z_t$ is the cell input, and $\psi$ is a squashing function:

$$
c_t = f_t \, c_{t-1} + i_t \, z_t, \qquad h_t = o_t \, \psi(c_t)
$$

### sLSTM: exponential gating and scalar memory

The first change in sLSTM is to swap the activation function of the input gate (and optionally the forget gate) from sigmoid to exponential, so that the gate value is no longer squeezed into $[0,1]$, and the model can therefore greatly amplify a new input and effectively "overwrite" old memory—this exactly addresses limitation (i). The exponential can explode, so the authors additionally introduce a normalizer state $n_t$ (accumulating the input gate multiplied by all subsequent forget gates) to normalize the output, and use a stabilizer state $m_t$ (taking the max in the log domain) for numerical stability; the paper proves that this stabilization does not change the network's output or gradients.

The scalar forward recurrence and normalization of sLSTM are as follows:

$$
c_t = f_t \, c_{t-1} + i_t \, z_t, \quad n_t = f_t \, n_{t-1} + i_t, \quad \tilde{h}_t = c_t / n_t, \quad h_t = o_t \, \tilde{h}_t
$$

sLSTM also retains multiple memory cells and can be split into multiple heads: within the same head, memory mixing is done via a recurrent matrix $R$, but heads do not mix with each other. The authors emphasize that combining heads with exponential gating constitutes a new form of memory mixing, and this is precisely the key differentiating point of sLSTM relative to SSMs and linear attention—the latter two have no memory mixing and therefore cannot do state tracking.

### mLSTM: matrix memory and the covariance update rule

mLSTM targets limitation (ii): it upgrades the scalar cell state $c \in \mathbb{R}$ to a matrix memory $C \in \mathbb{R}^{d \times d}$, so that "retrieval" becomes a single matrix multiplication. Borrowing Transformer terminology, at each time step a pair of key $k_t$ and value $v_t$ is stored, to be later retrieved by a query $q_{t+\tau}$; this is in fact the setup of Bidirectional Associative Memory. Storing uses the covariance (outer-product) update rule $C_t = C_{t-1} + v_t k_t^\top$; after fitting it into the LSTM framework, the forget gate corresponds to the decay rate, the input gate to the learning rate, and the output gate scales the retrieved vector.

The update and retrieval of the matrix memory (with the normalizer state $n_t$ and a denominator lower-bounded by 1) can be written as:

$$
C_t = f_t \, C_{t-1} + i_t \, v_t k_t^\top, \quad n_t = f_t \, n_{t-1} + i_t \, k_t, \quad \tilde{h}_t = \frac{C_t \, q_t}{\max\{\,|n_t^\top q_t|,\; 1\,\}}
$$

Because mLSTM removes memory mixing (no hidden-hidden connections), this recurrence can be rewritten into a parallel version, computed in parallel along the sequence during training like attention; the cost is that a $d \times d$ matrix must be processed at every step, which is computationally heavy, but these matrix operations contain no parameters and can be parallelized on the GPU, so the extra wall-clock overhead is limited.

### The xLSTM block and the xLSTM[a:b] architecture

The two kinds of cells are each wrapped into different residual blocks: sLSTM uses a post up-projection block (like a Transformer: first nonlinearly summarize the past in the original space, then project to a higher dimension, apply nonlinearity, and project back), while mLSTM uses a pre up-projection block (like an SSM: first project to a higher dimension and then summarize, because the matrix memory has larger capacity in a high-dimensional space); the design motivation cites Cover's theorem—patterns nonlinearly embedded in a high-dimensional space are more easily separated linearly. The overall architecture simply stacks these blocks in a pre-LayerNorm residual fashion. The paper uses xLSTM[$a$:$b$] to denote the ratio of mLSTM to sLSTM blocks: for example, xLSTM[7:1] means seven mLSTM and one sLSTM per eight blocks; for a common total block number of 48, this translates to 6 sLSTM blocks and 42 mLSTM blocks.

### A concrete walkthrough of "LSTM → xLSTM"

The best way to understand the contribution of each xLSTM component is to look at the authors' ablation on 15B-token SlimPajama, where they progressively morph a vanilla LSTM into xLSTM: starting from the most primitive multi-layer LSTM, they successively add a ResNet backbone, an up-projection backbone, exponential gating, and matrix memory, observing how the validation perplexity (lower is better) drops all the way down.

| Model stage | Exponential gating | Matrix memory | #Params (M) | SlimPajama (15B) ppl ↓ |
|-|-|-|-|-|
| Vanilla multi-layer LSTM | ✗ | ✗ | 607.8 | 2417.86 |
| + ResNet backbone | ✗ | ✗ | 506.1 | 35.46 |
| + up-projection backbone | ✗ | ✗ | 505.9 | 26.01 |
| xLSTM[0:1] (add exponential gating) | ✓ | ✗ | 427.3 | 17.70 |
| xLSTM[7:1] (also add matrix memory) | ✓ | ✓ | 408.4 | 13.48 |

This trajectory tells the story very clearly: merely placing a bare LSTM of about 600M parameters into a residual backbone drops the perplexity from 2417.86 to 35.46 (a bare LSTM of this scale is essentially untrainable); the up-projection further lowers it to 26.01; switching to exponential gating (at this point it is already the pure-sLSTM xLSTM[0:1]) jumps it to 17.70; and finally adding mLSTM's matrix memory gives 13.48. On this basis the authors attribute the main gains to both exponential gating and matrix memory, rather than to a single component—this is also the most persuasive table in the whole paper.

### Main language modeling results

Under the same 15B-token setting, the authors pit xLSTM against various 350M-class models (all aligned to the dimensions of GPT-3 350M). The table below excerpts the validation perplexity of representatives from each category:

| Model (category) | #Params (M) | SlimPajama (15B) ppl ↓ |
|-|-|-|
| Llama (Transformer) | 407 | 14.25 |
| Mamba (SSM) | 423 | 13.70 |
| RWKV-5 (RNN) | 456 | 14.25 |
| HGRN2 (RNN) | 411 | 14.32 |
| **xLSTM[1:0]** | 409 | **13.43** |
| xLSTM[7:1] | 408 | 13.48 |

In this table xLSTM outperforms all existing methods in validation perplexity, with the pure-mLSTM xLSTM[1:0] taking the overall best at 13.43, slightly beating Mamba's 13.70 and Llama's 14.25. The authors then scale the data up 20-fold, training 300B tokens, and compare xLSTM, Llama, Mamba, and RWKV-4 across four scales from 125M to 1.3B. The following excerpts the validation perplexity and downstream tasks at the 1.3B scale:

| Model (1.3B) | #Params (M) | SlimPajama ppl ↓ | LAMBADA acc ↑ | HellaSwag acc ↑ | Avg. acc ↑ |
|-|-|-|-|-|-|
| RWKV-4 | 1515.2 | 9.83 | 49.78 | 56.20 | 54.78 |
| Llama | 1420.4 | 9.44 | 57.44 | 57.81 | 56.99 |
| Mamba | 1475.3 | 9.14 | 55.64 | 60.45 | 58.41 |
| **xLSTM[1:0]** | 1422.6 | **8.89** | 57.83 | 60.91 | 58.48 |

At 1.3B, xLSTM[1:0]'s validation perplexity of 8.89 is still the lowest. Finer-grained evidence comes from PALOMA's 571 text domains: the authors report that xLSTM[1:0] has lower perplexity than Mamba on 568 out of 571 (99.5%) domains, lower than Llama on 85.1% of domains, and lower than RWKV-4 on 99.8% of domains. In addition, in length-extrapolation experiments trained at context 2048 and tested up to 16384, xLSTM's perplexity stays stable, and its recurrent nature makes generation time grow linearly with the sequence, allowing a larger batch than Llama and thus higher throughput.

## 🧪 Critical Assessment

### Linear-time sequence models catching up with Transformers is a real need

The problem the paper poses is itself real: at a time when the quadratic complexity of Transformers has become a bottleneck for long sequences and inference cost, the question of "how far do we get in language modeling" with linear-time, constant-memory sequence models that can catch up with Transformers shares the same lineage as the whole SSM/RWKV research line, and is not a manufactured need. The reasonable reading is that this is a problem with genuine engineering value, and that xLSTM offers an angle of "modernizing the LSTM" rather than starting from scratch, which has its own independent significance.

### Under broad baselines, a single corpus and a 1.3B ceiling weaken external validity

The baseline coverage is quite broad (Transformer, SSM, multiple RWKV generations, GLA, HGRN2, RetNet, etc.), and the step-by-step LSTM→xLSTM ablation is nicely done—this is a strength. What is questionable is the external validity: all language modeling results are built on a single corpus, SlimPajama (125M, 350M, 760M, 1.3B), the maximum reaches only 1.3B, and the main metric is heavily concentrated on validation perplexity. With perplexity as the main axis and scale stopping at 1.3B, the claim of "being able to rival the most advanced Transformers" remains an extrapolation rather than empirical evidence at true LLM scale.

### mLSTM shares a lineage with linear attention; sLSTM's memory mixing is the differentiator

Two parts need to be honestly distinguished. mLSTM's matrix memory and outer-product update are not entirely new: the paper itself states that the covariance update rule is equivalent to Fast Weight Programmers, and it shares the same mathematical skeleton with linear attention, Retention, and RWKV-5/6; this part is closer to rearranging existing mechanisms and adding exponential gating and normalization. What is genuinely harder to find a counterpart for in other linear models is the state-tracking capability claimed by sLSTM's head-based memory mixing—this is the most differentiating contribution in the xLSTM narrative, and where it opens a conceptual gap from SSMs.

### The formal-language experiments are designed around sLSTM's strengths

The formal language experiments call for some caution: the authors show that Transformers and SSMs cannot solve, e.g. regular grammars like the parity task, whereas the sLSTM with memory mixing can. Although the direction of this conclusion is consistent with existing theory, the benchmark itself is picked precisely on the family of tasks that "require memory mixing," which amounts to defining the problem around their own sLSTM's strengths; what it proves is that "having memory mixing is better than not," which is not necessarily equivalent to a general advantage on mainstream downstream tasks.

### Holds below 1.3B, but unoptimized kernels leave large-scale deployment unproven

Within the range below 1.3B measured by perplexity, "extending the LSTM to rival Transformers/SSMs" roughly holds; but the paper does not claim the problem is fully solved. In the Limitations the authors state plainly that the CUDA kernel is not yet optimized and mLSTM is about 4 times slower than FlashAttention, and admit that we did neither fully optimize the architecture nor the hyperparameters, expecting that an extensive optimization process is needed for xLSTM to reach its full potential. Therefore, as for real-world deployment, xLSTM currently looks more like a strong existence proof, and whether it can win simultaneously on downstream quality and actual throughput at scales above 7B remains unproven.

## 🔗 Related notes

- [Attention Is All You Need](../AttentionIsAllYouNeed/) — the Transformer baseline and self-attention starting point that xLSTM benchmarks against throughout.
