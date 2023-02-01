# Stable diffusion
- [paper](https://ommer-lab.com/research/latent-diffusion-models/)

## How to train a Stable diffusion from scratch
Stable diffusion is a latent diffusion model. A diffusion model is basically smart denoising guided by a prompt. It's effective enough to slowly hallucinate what you describe a little bit more each step (it assumes the random noise it is seeded with is a super duper noisy version of what you describe, and iteratively tries to make that less noisy). The latent part means that it is "de noising" a compressed version of the image that represents the qualitive aspects a bit better than the raw pixel grid. At the end the decoder converts it into an actual image.

Now, iirc stable diffusion uses clip embeddings, which themselves are based on gpt-2/3. These embeddings are encoded and fed into the attention layers of the u-net. In simpler terms, parts of the neural network are sandwiched by layers that take in a "thing" that is a math remix of the prompt.

Simply put, if you want to isolate the part of it that is natural language processing, you're basically playing with GPT-2/3.

## Inference Progress
1. prompt -> clip encoder -> tensor
2. tensor -> ddpm -> image