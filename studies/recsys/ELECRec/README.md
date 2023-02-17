### 👍 ELECRec: Training Sequential Recommenders as Discriminators
Despite their prevalence(普遍), these methods usually require training with more
meaningful samples to be effective, which otherwise will lead to a poorly trained model. 
In this work, this paper proposes to train the sequential recommenders as `discriminators` rather than generators.
Instead of predicting the next item, our method trains a discriminator to distinguish if a sampled item is a real target item or not. 
A generator, as an auxiliary model, is trained jointly with the discriminator to sample plausible(似是而非/可能) alternative next items and will be thrown out after training.