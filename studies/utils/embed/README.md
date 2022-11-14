# Embedding
## Positional Encoding
Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.

The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed.

TBA...