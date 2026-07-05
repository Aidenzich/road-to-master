# LTSF-Linear — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Are Transformers Effective for Time Series Forecasting? |
| Venue | AAAI 2023 |
| Year | 2023 |
| Authors | Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu |
| Official Code | https://github.com/cure-lab/LTSF-Linear |
| Venue Kind | paper |

> This note is written based on the arXiv preprint `2205.13504v3` (the full OpenReview text returns HTTP 403 and cannot be retrieved anonymously, so the author-identity-verified arXiv version is used instead). The officially published version is AAAI 2023, and the camera-ready content may differ slightly from the numbers cited here.

This paper takes a deliberately provocative stance: instead of proposing a new method, it questions whether the flood of Transformer variants that have emerged in recent years for long-term time series forecasting (LTSF) are really effective. The authors use an "embarrassingly simple" single-layer linear model, LTSF-Linear, as a control, and on nine commonly used benchmarks it beats LogTrans, Informer, Autoformer, Pyraformer, and FEDformer across the board — arguing that this entire line of research may have been taking its temperature in the wrong place.

## First Principles

### Why "permutation invariance" is a fatal flaw for time series

The core of the Transformer is multi-head self-attention, which excels at extracting the "semantic correlations between elements" in a long sequence — for example, words in a sentence or 2D patches in an image. But self-attention is inherently permutation-invariant: shuffle the order of the input tokens and the pairwise correlations it computes remain unchanged. NLP can tolerate this because text is itself semantically rich, so rearranging some words largely preserves the meaning. Time series are the opposite — raw values (stock prices, electricity readings) carry almost no point-wise semantic correlation, and what we truly care about is "the temporal change across a set of consecutive points," where the order itself is the most critical information. The authors argue that even with positional encoding added and sub-series embedded into tokens, applying self-attention still inevitably causes a loss of temporal information.

The paper breaks the common pipeline of existing LTSF-Transformers into four stages (figure below), which are also the targets the later ablation experiments dismantle one by one: pre-processing (normalization, timestamp preparation, seasonal-trend decomposition), input embedding (channel projection, fixed positional encoding, local/global timestamps), the encoder (each method's own sparse or low-rank attention variant), and the decoder (mostly changed to one-shot direct multi-step decoding).

![The common pipeline of existing Transformer LTSF approaches](imgs/pipeline.png)

### A "deliberately over-simple" control: LTSF-Linear

The basic model the authors propose is just a single linear layer along the temporal axis, directly regressing the historical series into the future series via a weighted sum:

$\hat{X}_{i} = W X_i$

where $W \in \mathbb{R}^{T \times L}$, $L$ is the look-back window length, $T$ is the forecast length, and $X_i$, $\hat{X}_i$ are the input and prediction of the $i$-th variate. The key design choices are: the weights are shared across all variates, and the spatial correlation between variates is not modeled at all. This equation also embeds an often-overlooked difference in experimental setup — it is direct multi-step (DMS) forecasting that emits all $T$ steps at once, whereas the traditional baselines beaten in existing papers are mostly iterated multi-step (IMS), which accumulate error step by step. The authors therefore suspect that a large part of the gains claimed by Transformer papers actually comes from the DMS strategy rather than the architecture itself.

![The basic linear model: weighted regression of L steps of history into T steps of the future](imgs/linear-model.png)

On top of this, two pre-processing variants targeting data characteristics are added:

- **DLinear**: first uses a moving average kernel (kernel size 25, the same as Autoformer) to split the input into a trend component and a remaining (seasonal) component, attaches one linear layer to each component, and finally sums them. This is especially helpful when the data has a clear trend.
- **NLinear**: first subtracts the last value of the series from the entire input, and after passing through the linear layer adds this value back. This amounts to a simple normalization of the input, used to counter the distribution shift between training and test.

### A forward pass and cost walk-through with real numbers

Take the Electricity dataset, look-back window $L=96$, forecast $T=720$ as an example, and look at what one DLinear forward pass does: take the 96-dimensional history vector $x$ of some variate, first compute the moving average to get the trend $x_t$, then take the residual to get the seasonal component $x_s = x - x_t$; each of the two branches has a $720 \times 96$ weight matrix $W_t$, $W_s$, and the output is $\hat{y} = W_t x_t + W_s x_s \in \mathbb{R}^{720}$. Because the weights are shared across all 321 variates, the parameter count of the entire Electricity model is just these two matrices: $2 \times T \times L = 2 \times 720 \times 96 = 138{,}240$, about 138.2K (the 139.7K measured in Table 8 additionally includes the bias terms of the two branches, about 1.4K more). Compared against the actual cost measured in the paper's Table 8 ($L=96, T=720$, Electricity):

| Model | MACs | Params | Inference time | Memory |
|-|-|-|-|-|
| DLinear | 0.04G | 139.7K | 0.4ms | 687MiB |
| Informer | 3.93G | 14.39M | 49.3ms | 3869MiB |
| Autoformer | 4.41G | 14.91M | 164.1ms | 7607MiB |
| FEDformer | 4.41G | 20.68M | 40.5ms | 4143MiB |

DLinear's parameter count is about two orders of magnitude smaller than Autoformer's, yet its inference is tens to hundreds of times faster. For accuracy, one setup difference must be watched: to let each method run at its best, the paper's main benchmark table enlarges DLinear's look-back window to $L=336$ while keeping the Transformers at $L=96$ (supp.tex:63 explicitly states "report L=336 for DLinear and L=96 for Transformers by default"). In the Electricity, $T=720$ cell, DLinear ($L=336$) has an MSE of 0.203, better than the strongest Transformer (FEDformer 0.246), corresponding to the 17.47% relative improvement noted in the paper. Note that the cost table above measures the $L=96$ DLinear (138.2K parameters, 0.4ms), which is not the same configuration as the 0.203 accuracy here — but even putting DLinear at $L=336$, the parameter count is only $2\times720\times336\approx484$K, and the inference cost is still far below any Transformer. In other words, the conclusion of smaller, faster, and more accurate holds under both look-back windows.

### Progressively reducing Informer to a linear layer

The authors run a very persuasive "destructive" ablation: simplifying Informer step by step. The first step replaces each self-attention layer with a linear layer (Att.-Linear, since attention can be viewed as a fully connected layer with dynamically changing weights), while the embedding, FFN, and other auxiliary designs are all still present; the second step drops the FFN and other auxiliary designs, keeping only the embedding layer and the linear layer (Embed+Linear, embedding retained); finally the embedding is also removed, reducing to a single linear layer. Taking Exchange-Rate, $T=96$ as an example, this path is not monotonically improving: the MSE first "rises" from Informer's 0.847 to Att.-Linear's 1.003, then drops sharply to Embed+Linear's 0.173 after dropping the FFN and other auxiliary designs, and finally to pure Linear's 0.084 after removing the embedding. Notably, simply replacing attention with a linear layer (0.847→1.003) does not help by itself and even makes things worse; the decisive step is removing the complex auxiliary modules such as the FFN (1.003→0.173, with the embedding still present), and removing the embedding is only a small finishing improvement (0.173→0.084). Overall, what this chain of ablations proves to be "non-essential" is attention's dynamic weighting plus those complex auxiliary designs, rather than pinning the blame on any single component; at least for existing LTSF benchmarks, neither self-attention nor these complex modules are the key.

### Two pieces of evidence pointing directly at "overstated temporal-modeling ability"

First, **lengthening the look-back window**. A model that truly extracts temporal relationships should predict more accurately the more it sees. The authors sweep $L \in \{24, 48, ..., 720\}$ and find that the error of existing Transformers mostly stays flat or even worsens as the window grows longer, whereas LTSF-Linear steadily improves (figure below, Traffic, $T=720$). This suggests that on long inputs the Transformer is overfitting temporal noise rather than extracting more temporal information.

![MSE under different look-back window lengths (Traffic, T=720)](imgs/lookback-traffic.png)

Second, **shuffling the input order**. If a model truly relies on temporal order, shuffling the input should badly hurt it. The authors test with Shuf. (randomly shuffling the whole segment) and Half-Ex. (swapping the first and second halves): on Exchange-Rate, the error of all Transformers barely moves (FEDformer's average change is −0.09%), while LTSF-Linear drops by 27.26%. This conversely shows that the Transformer does not take order seriously at all, whereas the linear model does. Interestingly, there is also an extremely naive Repeat baseline (simply repeating the last value of the look-back window) that actually beats all Transformers by about 45% on Exchange-Rate — because it at least does not wildly guess the trend.

## 🧪 Critical Assessment

### Is the problem real, or amplified by the experimental setup?

The question "is the Transformer effective for LTSF" is itself valuable: the field has indeed stacked up a great deal of complex architecture over a few years, yet few have gone back to do this kind of controlled comparison. The paper's most solid contribution is not that linear model, but the series of ablations that separate "architectural gains" from "DMS-strategy gains" — lengthening the window, shuffling, dismantling Informer, removing the embedding, and comparing measured cost. These pieces of evidence are mutually independent yet point to the same conclusion, far more persuasive than a single accuracy table alone. From first principles, permutation invariance and the lack of semantics in numerical sequences are indeed structural weaknesses of self-attention applied to time series.

However, a symmetric risk must be watched: the paper's conclusion depends heavily on whether the DMS/IMS confounder is fully controlled. It candidly admits that the traditional baselines in existing Transformer papers are mostly IMS and therefore carry an error-accumulation disadvantage; but conversely, the authors' own linear model is DMS, and the Transformers being compared are already DMS too, so this part is fair. What is not fully answered is: if the same DMS + decomposition + normalization pre-processing were transplanted unchanged onto a moderately complex non-linear model, would the gains still hold? The paper does not test this intermediate point, so both readings — "linear is enough" and "these benchmarks are too easy" — are tenable.

### Are the baselines, datasets, and metrics sufficient?

The five Transformers being compared are all representative works of the time, and their implementations directly reuse the original authors' or Autoformer's code and default hyperparameters, which deserves credit. But two asymmetries are worth noting. First, to "compare each at its best," the authors use $L=336$ for the linear model and $L=96$ for the Transformers (stated explicitly in the appendix), on the grounds that a short window underfits the linear model while a long window overfits the Transformer — this setup favors the linear model, and although it is backed by the window-lengthening experiment, the fairness of the headline table is thereby discounted. Second, the metrics are only MSE/MAE, all on nine highly homogeneous benchmarks (the ETT series alone accounts for four). Whether these datasets are themselves sufficient to represent the grand proposition of "long-term forecasting" is questionable — most of them have clear daily/weekly periodicity, which happens to be the home turf of linear trend-seasonal decomposition.

### Is it renaming, or genuinely simpler?

Linear regression is of course nothing new, and DLinear's decomposition is borrowed directly from Autoformer/FEDformer; in terms of "methodological novelty" this paper is close to zero, and the authors themselves state in the conclusion that the contribution is not proposing a linear model. So it should not be evaluated as a "new model," but as an "audit of benchmark validity." There is a point worth being wary of here: the entire evaluation is built around benchmarks defined by the strengths of the authors' own method (clear trend and periodicity, predominantly univariate, measured by MSE). On low signal-to-noise financial data like Exchange-Rate, even Repeat can beat the Transformers, which — rather than proving the linear model is strong — proves that these benchmarks are especially unfriendly to models that "wildly extrapolate trends." The scope of extrapolation of the conclusion should be limited to "these existing nine LTSF benchmarks," and the authors cautiously add this qualifier in the main text.

### Is the problem really solved, and how much does it mean for the real world?

The paper itself is very honest: it repeatedly stresses that LTSF-Linear is only a "competitive simple baseline," that its model capacity is limited, that a single-layer linear model struggles to capture the dynamics caused by change points, and that new designs are still needed in the future. So it does not claim to "solve" long-term forecasting, but rather recalibrates the bar — any subsequent complex model must first beat this nearly zero-cost baseline to count. This contribution is quite concrete for practice: in many operational scenarios with clear periodicity (electricity, traffic), a model with 139.7K parameters and 0.4ms inference may well be the more reasonable default choice. What is truly unresolved is this paper's more open, larger question — whether the nine benchmarks it picked are themselves so easy that they cannot distinguish any model's true temporal-modeling ability. If so, the community should invest more in harder evaluation protocols that better reflect real dynamics, rather than continuing to shave decimals on benchmarks that are saturated by linear models. This is also the most productive thorn this paper leaves for the field.

## 🔗 Related notes

- [Autoformer](../Autoformer/)
- [Informer](../informer/)
- [Non-stationary Transformers](../Non-stationary-Transformers/)
