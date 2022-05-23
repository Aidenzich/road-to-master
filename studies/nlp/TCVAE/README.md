# T-CVAE
https://www.ijcai.org/proceedings/2019/727
## Introduction
This paper present a novel conditional variational autoencoder based on Transformer for missing plot generation. 
- Story completion is a very challenging task of generating the missing plot for an incomplete story, which requires not only understanding but also inference of the given contextual(語境) clues.
- The model uses shared attention layers for encoder and decoder, which make the most of the contextual clues, and a latent variable for learning the distribution of coherent(連貫) story plots(情節).
  - Through drawing samples from the learned distribution, diverse reasonable plots can be generated.
- Both automatic and manual evaluations show that our model generates better story plots than SOTA models in terms of readability, diversity and coherence.