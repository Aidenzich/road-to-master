# Transformer (Attention Is All You Need) — Research Note

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Attention Is All You Need |
| Venue | NIPS |
| Year | 2017 |
| Authors | Vaswani et al. |
| Official Code | unknown |
| Venue Kind | paper |

## First Principles

This note describes the model in plain prose only. The layer relates queries, keys and
values, running several projections in parallel and combining the results. No equations,
algorithm blocks or tables are included in this deliberately thin body.

![scaled dot-product attention](imgs/fig1.png)

## 🧪 Critical Assessment

### Problem realness and importance
Long-range dependency modelling in sequence transduction is a real, well-motivated problem:
RNNs serialise computation along sequence length and struggle to propagate gradients across
long spans. The framing is not manufactured.

### Baseline, ablation, dataset and metric sufficiency
WMT14 EN-DE / EN-FR and BLEU are standard, and the ablations over head count, key dimension
and positional encodings are reasonable, though the reported gains partly ride on
training-budget and regularisation choices the ablations do not fully isolate.

### Novelty vs engineering repackaging
Attention predates this work; the genuine novelty is discarding recurrence entirely and
showing a pure-attention stack is trainable and parallelisable. This is a real architectural
claim, not a rename or a self-defined benchmark (射箭畫靶).

### Is the claimed problem actually solved, and is it real-world relevant?
The narrowed claim (competitive MT quality at lower training cost) is supported; the broader
"attention is all you need" slogan is stronger than the evidence for tasks outside the
studied setting.

## 🔗 Related notes

- BERT and later encoder-only models build directly on this stack.
