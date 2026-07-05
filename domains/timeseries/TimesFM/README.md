# TimesFM — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | A decoder-only foundation model for time-series forecasting |
| Venue | ICML |
| Year | 2024 |
| Authors | Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou (Google Research) |
| Official Code | https://github.com/google-research/timesfm |
| Venue Kind | paper |

## First Principles

TimesFM sets out to answer a very direct question: since NLP can do zero-shot inference on unseen tasks with a single large pre-trained model, can time series also have an "out-of-the-box" foundation model that, facing a never-seen dataset, gives predictions close to a specially trained model without any fine-tuning? The authors' answer is yes, and they achieve it with only about 200M parameters and about O(100B) time points of pre-training data — far smaller than the scale of contemporary LLMs.

Formally, the task is to learn a function that maps a history context $\mathbf{y}_{1:L}$ of length $L$ to a forecast $\hat{\mathbf{y}}_{L+1:L+H}$ of the next $H$ steps. Because a "single" general model is to be trained, training cannot rely on any dataset-specific static or dynamic covariate, and the model can only consume the time series' own past values:

$$
f:\ \mathbf{y}_{1:L}\ \longrightarrow\ \hat{\mathbf{y}}_{L+1:L+H}
$$

### Why "patched decoder-only"

The entire architecture is held up by four design principles. First is patching: cut the time series into non-overlapping segments (patches), where each patch is like a token in a language model — a practice inherited from the long-term forecasting work PatchTST. The benefit is that the number of tokens fed into the transformer is reduced by a factor of the patch length, making inference faster. Second is decoder-only: unlike PatchTST, which uses an encoder-decoder, TimesFM, given a sequence of input patches, is trained to predict the next patch from "all past patches," which lets the model keep forecasting after seeing any number of input patches, naturally supporting variable context lengths.

The third, and the point that differs most from LLMs: the output patch can be longer than the input patch. Experience in long-term forecasting is that "emitting the whole horizon at once" is more accurate than step-by-step auto-regressive decoding, but in the zero-shot setting the horizon is unknown and cannot be emitted all at once. The authors' compromise is to let one output token directly predict a relatively long stretch of the future (for example, input patch length 32, output patch length 128), which greatly reduces the number of auto-regressive steps. Fourth is patch masking: if fixed patches were used naively, the model would only learn well on contexts that are "integer multiples of the patch length," so during training a leading segment of each series is randomly masked, letting the model see every context length from 1 up to the maximum.

![TimesFM architecture: input patches are projected into tokens by a Residual Block, and after adding positional encoding are fed into a num_layers-layer causal self-attention transformer; each output token is then mapped by a Residual Block into a prediction of length output_patch_len.](imgs/architecture.png)

### Three Residual Blocks and the causal transformer

The input layer first cuts $\mathbf{y}_{1:L}$ into patches according to `input_patch_len` (denoted $p$); the $j$-th patch is $\tilde{\mathbf{y}}_j=\mathbf{y}_{p(j-1)+1:pj}$, accompanied by a binary padding mask $\tilde{\mathbf{m}}_j$ (1 means the point should be ignored). Each patch is projected by an MLP with a single hidden layer and a skip connection (a Residual Block) into a `model_dim`-dimensional vector, and after adding a sinusoidal positional encoding, gives the $j$-th token:

$$
\mathbf{t}_j=\mathrm{InputResidualBlock}\big(\tilde{\mathbf{y}}_j\odot(1-\tilde{\mathbf{m}}_j)\big)+\mathrm{PE}_j
$$

These tokens are fed into a `num_layers`-layer standard transformer, where each layer is multi-head causal self-attention followed by an FFN, and the $j$-th output token can only attend to the tokens before it (including itself) in the sequence. The output layer then uses another Residual Block to map each output token $\mathbf{o}_j$ into the prediction that immediately follows it, of length `output_patch_len` (denoted $h$); that is, all of $\mathbf{y}_{1:pj}$ is encoded into $\mathbf{o}_j$, which is used to predict the next $h$ points:

$$
\hat{\mathbf{y}}_{pj+1:pj+h}=\mathrm{OutputResidualBlock}(\mathbf{o}_j)
$$

Because this paper only does point forecasting, the training loss is the average MSE over all patch positions (where $N=\lfloor L/p\rfloor$ in the equation below is the number of tokens); for probabilistic forecasting, one only needs to switch to multiple quantile heads or a maximum-likelihood loss, which the authors leave to future work:

$$
\mathrm{TrainLoss}=\frac{1}{N}\sum_{j=1}^{N}\mathrm{MSE}\big(\hat{\mathbf{y}}_{pj+1:pj+h},\ \mathbf{y}_{pj+1:pj+h}\big)
$$

The sampling of the mask during training is clever: for each series draw an $r\in[0,p-1]$ and set the first $r$ points as masked. Using the paper's own example, with a maximum context of 512, $p=32$, and $r=4$, the first output token is optimized to predict "after seeing $28=32-4$ points," the second token "after seeing $28+32$ points," and so on; sweeping over all possible $r$, the model covers every context length from 1 to 512.

### Walking through a real forward pass: ETT zero-shot

Running one ETT zero-shot task with the paper's main setting (`model_dim=1280`, 20 layers, 16 heads, $p=32$, $h=128$), the shape changes are as follows:

```text
context L = 512, input_patch_len p = 32
  → split into N = floor(512/32) = 16 input patches, each holding 32 time points
  → each patch through InputResidualBlock → 1280-dim token, + positional encoding
  → 16 tokens fed into 20-layer causal transformer (16 heads, FFN hidden dim = 1280)
  → obtain 16 output tokens o_1..o_16
  → o_16 through OutputResidualBlock → predict y_513..y_640 (output_patch_len = 128 points)

horizon = 96 task: directly take the first 96 points output by o_16, one forward pass, zero auto-regression.
horizon = 512 task: first emit 513..640, then feed the prediction back into the input, needing ceil(512/128)=4 auto-regression steps in total;
  if output_patch_len were only 32, the same task would need 16 auto-regression steps.
```

It is worth noting that normalization uses the "standardization" part of reversible instance normalization: the entire context is scaled by the mean and standard deviation of the first input patch in the context, because in the zero-shot setting the affine parameters cannot be learned as in the original RevIN. The table below lists the hyperparameters for the three sizes; note that the 200M model has `model_dim` 1280 and 20 layers, and `output_patch_len` (128) is deliberately longer than `input_patch_len` (32):

| Size | num_layers | model_dims | output_patch_len | input_patch_len | num_heads | dropout |
|-|-|-|-|-|-|-|
| 200M | 20 | 1280 | 128 | 32 | 16 | 0.2 |
| 70M | 10 | 1024 | 128 | 32 | 16 | 0.2 |
| 17M | 10 | 512 | 128 | 32 | 16 | 0.2 |

### The data engine: the real contribution is here

The model is not large, but the pre-training corpus is the key to the whole work. The authors piece together volume and diversity from three major sources: Google Trends (about 22k popular queries, search interest from 2007–2022, at hourly/daily/weekly/monthly granularity), Wiki Pageviews (page views from 2012–2023, about 300B time points after cleaning and aggregation, the bulk of the corpus), and synthetic data (an additive combination of ARMA processes, sine-cosine seasonality, trends with change points, and step functions; 3M series in total, each of length 2048). The training sampling ratio is 80% real data and 20% synthetic data, and the real data is given equal weight across the four groups "hourly/sub-hourly, daily, weekly, monthly," to keep the high-frequency granularities from drowning out the low-frequency ones. The table below excerpts the corpus composition, showing that Wiki alone accounts for the vast majority of time points:

| Dataset | Granularity | # Time series | # Time points |
|-|-|-|-|
| Synthetic | - | 3,000,000 | 6,144,000,000 |
| Wiki hourly | Hourly | 5,608,693 | 239,110,787,496 |
| Wiki daily | Daily | 68,448,204 | 115,143,501,240 |
| Trends daily | Daily | 22,435 | 122,921,365 |
| M4 monthly | Monthly | 48,000 | 10,382,411 |

The maximum context length depends on the granularity: 512 in general, 256 for weekly data because the series are not long enough, and 64 for monthly or coarser granularities. Training the entire 200M model for 1.5M iterations (global batch 4096) on 16 TPUv5e cores takes about 2 days.

### Results and ablations

Zero-shot evaluation is done on three groups of public benchmarks deliberately excluded from pre-training. In terms of the average MAE on ETT long-term forecasting (horizons 96 and 192, context 512, 8 tasks in total), TimesFM(ZS) is 0.36, nearly tied with the strongest supervised method PatchTST (0.37), while the rest of the specially trained methods are all clearly worse:

| Method | llmtime(ZS) | PatchTST | PatchTST(ZS) | FEDFormer | AutoFormer | Informer | TimesFM(ZS) |
|-|-|-|-|-|-|-|-|
| ETT Avg MAE | 0.45 | 0.37 | 0.35 | 0.53 | 0.53 | 0.99 | 0.36 |

On the Monash archive (18 datasets), the paper aggregates with a scaled MAE that "divides by a naive baseline and takes the geometric mean," and TimesFM(ZS) scores 0.6846, slightly ahead of N-BEATS's 0.7005 to top the ranking, and over 25% better than zero-shot llmtime (0.9715). On the ablation side, three experiments support the core design: as the model is scaled from 17M to 70M and 200M, the scaled MAE on Monash decreases monotonically with FLOPS (lower-left figure); as `output_patch_len` is increased from 8 to 128 (on the 512-step ETT forecasting task), the average MAE decreases monotonically (lower-right figure), confirming that "a long output patch reducing the number of auto-regression steps" does help; `input_patch_len` is best around 16 and 32 and worsens if too large or too small, and $p=32$ trains about twice as fast as $p=16$, so 32 is chosen.

![Left: Monash scaled MAE(GM) decreases monotonically with FLOPS across the three sizes 17M/70M/200M.](imgs/scaling_flops.png)

![Right: on the 512-step ETT forecasting task, the average MAE decreases monotonically as output_patch_len increases from 8 to 128.](imgs/output_patch_len.png)

The ablation on synthetic data is also persuasive: after removing synthetic data, performance on Monash worsens because it contains more granularities under-represented by the real corpus (quarterly, yearly, 10-minute), while on ETT there is almost no difference for the hourly-level ETTh where granularity is plentiful, but a clear regression for the 15-minute ETTm — showing that synthetic data mainly fills in "under-represented frequencies." The authors also mention that such a from-scratch foundation model trained only on time series can obtain better zero-shot performance at a cost far below GPT-3 / LLaMA-2.

## 🧪 Critical Assessment

### The problem is a genuine need, but the definition of "zero-shot" is relaxed

"A single pre-trained model doing out-of-the-box cross-domain forecasting" is indeed a real industry pain point: it saves the burden of retraining and tuning for every dataset. This motivation is impeccable, and the model is indeed only 200M, open-sourceable, with a low barrier to deployment. But one must be careful that "zero-shot" in the paper is not entirely clean: on some Monash datasets, the authors admit to doing "inference-time tuning of the context length" (choosing the best among 32, 64, and the maximum length using the validation metric on the tail of training). Although this is defended as "most Monash deep-learning baselines already use different context lengths," it is already a validation-set-based hyperparameter choice, and counting it fully as zero-shot overstates the true out-of-the-box quality.

### Both the baselines and the aggregation favor the authors

Two points in the evaluation design are questionable. First, the choice of aggregation metric: the main figure uses the geometric mean (GM) to claim TimesFM tops Monash, but in the arithmetic mean (AM) version in the paper's appendix, TimesFM is merely "close to the top-ranked N-BEATS within the error range" rather than the best — switch to an equally reasonable aggregation and the lead vanishes, which is exactly the typical risk of defining the benchmark where it is favorable to oneself. Second, the currency of the baselines: the llmtime zero-shot comparison switched to GPT-3.5-Turbo because OpenAI deprecated GPT-3, so the model is already different; and on Darts the authors themselves admit llmtime may suffer data contamination. These all discount the force of "beating zero-shot rivals."

### The architectural novelty is limited; the contribution really lies in data and scale

Breaking it down item by item, patching comes from PatchTST, decoder-only from LLMs, and the residual block from TiDE; what is truly original is mainly the "output patch longer than input patch" compromise, together with the data engine that underpins training. The paper's own PatchTST(ZS) ablation in fact reveals an awkward fact: on ETT with context fixed at 512, PatchTST(ZS) pre-trained on the same corpus has an average MAE of 0.35, still slightly beating TimesFM's 0.36 — suggesting that when context is plentiful the architectural difference is actually small, and TimesFM's real advantage is "being able to adapt to variable, shorter contexts" rather than the attention stack itself being stronger. It is more apt to understand it as "a data and training recipe born for zero-shot forecasting" than as "a stronger network."

### "Is it solved?": approaching rather than surpassing, with clear blind spots

As for the claimed goal (zero-shot approaching supervised SOTA), the evidence largely holds: tied with PatchTST on ETT and on par with N-BEATS on Monash. But "approaching" is not "surpassing," and on Darts with only 8 single series, TimesFM's GM (0.5767) actually loses to ARIMA (0.5219) and llmtime (0.4882), showing that when data is scarce and seasonality is simple, classical statistical methods remain competitive. A more fundamental blind spot is that the model only does univariate point forecasting: no covariates, no probabilistic output, no uncertainty intervals — precisely what real forecasting scenarios like retail and energy need most. The authors list both probabilistic modeling and covariates as future work, so it can be said to have proven that "a time-series foundation model is feasible," but it is still a fair distance from "directly replacing a production-grade forecasting pipeline" — this is a solid and honest feasibility proof, not an endpoint.

## 🔗 Related notes

- [Autoformer](../Autoformer/)
- [Informer](../informer/)
- [TimesNet](../TimesNet/)
