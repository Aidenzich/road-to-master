# Positional Embeddings in Self-Attention
## Purpose of Positional Encoding
The purpose of training `positional embedding` is to allow the model to learn the relationships between different positions in a sequence and to better handle variable-length sequences. Because positional embedding is trainable, it can learn relative position information based on the characteristics of the training data and use that information when generating output. Additionally, `positional embedding` can also help the model better capture `long-term dependencies` and handle long sentences in language understanding tasks.

## What do Position Embedding Learn?
|  | Description |
|-|-|
| `Transformer encoders` | learn the `local position information` that can only be effective in masked language modeling.  |
| `Transformer decoders` | actually learn about `absolute positions`.  | 
- The different NLP tasks with different model architectures and different training objectives may **utilize** the position information in different ways. 

## Peformance
Performance on text classification (encoding), language modeling (decoding) and machine translation (encoding and decoding). 
Note that each `chosen task` has its own important property where `position information` may cause different effects in Transformers.


## Reference
- [What Do Position Embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding](https://arxiv.org/abs/2010.04903)
- [How Positional Embeddings work in Self-Attention (code in Pytorch)](https://theaisummer.com/positional-embeddings/)