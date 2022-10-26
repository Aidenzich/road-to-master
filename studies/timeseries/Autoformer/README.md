# Autofomer
## Introduction
Propose an original Autoformer in place of the Transformers for long-term time series forecasting. 

Autoformer still follows residual and encoder-decoder structure but
**renovates(翻新)** Transformer into a **decomposition(分解)** forecasting architecture. 
By embedding our proposed decomposition blocks as the inner operators, Autoformer can progressively(逐步) separate the **long-term trend information** from predicted hidden variables. 
This design allows our model to alternately decompose and refine the intermediate results during the forecasting procedure. 
- Inspired by the stochastic process theory [8, 24], Autoformer introduces an Auto-Correlation mechanism in place of self-attention, which discovers the sub-series similarity based on the series:
    - periodicity and aggregates similar sub-series from underlying periods. 
This series-wise mechanism achieves O(Llog L) complexity for length-L series and breaks the information utilization bottleneck by **expanding the point-wise representation aggregation to sub-series level**.

Autoformer achieves the state-of-the-art accuracy on six benchmarks. The contributions are summarized as follows: