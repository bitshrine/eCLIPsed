# StyleGAN 2

> Analyzing and Improving the Image Quality of StyleGAN
>
> Karras et al, 2020
> [<a target="download" href="data/papers/styleGAN2.pdf">PDF</a>]

## 1. Abstract

> The style-based GAN architecture (StyleGAN) yields state-of-the-art results in data-driven unconditional generative image modeling. We expose and analyze several of its characteristic artifacts, and propose changes in both model architecture and training methods to address them. In particular, we redesign the generator normalization, re-visit progressive growing, and regularize the generator to encourage good conditioning in the mapping from latent codes to images. In addition to improving image quality, this path length regularizer yields the additional benefit that the generator becomes significantly easier to invert. This makes it possible to reliably attribute a generated image to a particular network. We furthermore visualize how well the generator utilizes its output resolution, and identify a capacity problem, motivating us to train larger models for additional quality improvements. Overall, our improved model redefines the state of the art in unconditional image modeling, both in terms of existing distribution quality metrics as well as perceived image quality.

<figure>
    <img src="data/imgs/stylegan2.png">
</figure>

## 2. Summary

StyleGAN 2 presents a few improvements and modifications to the architecture proposed in StyleGAN 1.
One notable issue fixed with StyleGAN2 is the appearance of so-called "droplets" in images, which was caused by AdaIN normalization, and betrayed the fact that the image was fake. As a side note, it seems as though the StyleGAN1 discriminator did not pick up on this artifact.

### Weight modulation and demodulation
Firstly, the AdaIN operator is removed and replaced with an explicit normalization and modulation step.

