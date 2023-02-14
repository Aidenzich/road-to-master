# CaFe: Coarse-to-Fine Sparse Sequential Recommendation
Self-attentive methods still struggle to `model sparse data`, on which they struggle to learn high-quality item representations. 
This paper proposes to model user dynamics from shopping intents and interacted items simultaneously. The learned intents are `coarse-grained` and work as `prior knowledge` for item recommendation. 
This paper presents a coarse-to-fine self-attention framework, namely `CaFe`, which explicitly learns `coarse-grained` and fine-grained sequential dynamics. Specifically, CaFe first learns intents from coarse-grained sequences which are `dense` and hence provide high-quality user intent representations. Then, CaFe fuses intent representations into item encoder outputs to obtain improved item representations. 
Finally, infer recommended items based on representations of items and corresponding intents.
