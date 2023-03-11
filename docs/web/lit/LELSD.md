# LELSD

> **Optimizing Latent Space Directions for GAN-Based Local Image Editing**
> 
> Ehsan Pajouheshgar, Tong Zhang, and Sabine Süsstrunk, 2022 
> [<a href="data/papers/LELSD.pdf">PDF</a>]

## 1. Abstract

> Generative Adversarial Network (GAN) based localized image editing can suffer from ambiguity between semantic attributes. We thus present a novel objective function to evaluate the locality of an image edit. By introducing the supervision from a pre-trained segmentation network and optimizing the objective function, our framework, called <mark>Locally Effective Latent Space Direction (LELSD)</mark>, is applicable to <mark>any dataset and GAN architecture</mark>. Our method is also computationally fast and exhibits a high extent of disentanglement, which allows users to interactively perform a sequence of edits on an image. Our experiments on both GAN-generated and real images qualitatively demonstrate the high quality and advantages of our method.


<figure>
    <img src="data/imgs/lelsd.png" width="700px">
</figure>

## 2. Summary

The framework put forth by the authors of this paper seeks to introduce localized image editing to GANs by finding the latent space directions that yield localized changes in the output image. The method incorporates the layer-wise editing from [GANSpace](/web/lit/ganspace.md#editing-the-output) to allow for both coarse and fine-grained semantic changes, and a binary mask provided by a semantic image segmentation network to ensure the locality of changes.

### Methods

The "Methods" section of the paper is quoted here for quick access, as it is quite concise:

> [...] The generator network $G(.)$ in a GAN generates an image starting from a latent code $\omega \in \Omega$, i.e. $\textbf{x} = G(\omega) = f(h(\omega))$ where $r = h(\omega)$ is a tensor representing the activation of an intermediate layer in the network. The latent space $\Omega$ can be any of $\mathcal{Z}, \mathcal{W}, \mathcal{W}+, \mathcal{S}$ for the StyleGAN generator [...]. Semantic editing of an image is done by moving its latent code along a specific direction
> $$\textbf{x}^\text{edit}(\textbf{u}) = f(r^\text{edit}(\textbf{u})) = G(\omega + \alpha\textbf{u})$$
> where $\alpha$ controls the intensity of the change, and the latent direction $\textbf{u}$ determines the semantic of the edit.
> Our goal is to find an editing direction $\textbf{u}_c$ that mostly changes parts of the generated image corresponding to a binary mask given by a pretrained semantic segmentation model $\textbf{s}_c(\textbf{x})$ where $c$ indicates the desired object to edit in the image. Based on this, we can write the localization score as
> $$
LS(\textbf{u}) = \frac{\sum_{i, j}\overset{\downarrow}{\tilde{\textbf{s}}}_c(\textbf{x}, \textbf{x}^{\text{edit}}) \odot |\textbf{r} - \textbf{r}^{\text{edit}(\textbf{u})}|^2}{\sum_{i, j} | \textbf{r} - \textbf{r}^{\text{edit}}(\textbf{u})|^2} $$
> where $i, j$ iterate over the spatial dimensions and $\overset{\downarrow}{\tilde{\textbf{s}}} (\textbf{x}, \textbf{x}^\text{edit})$ is the average of the two semantic segmentation masks downsampled to the resolution of the corresponding layer. This objective function measures the proportion of the change in the featuremap that happens inside the semantic segmentation mask. Our final objective function is calculated by simply summing up the localization scores for all intermediate layers in the generator network. Unlike [17] that only aims to achieve localized change in the generated image, we also en- courage the intermediate featuremaps to only change locally. This allows us to achieve a larger variety of edits than [17]. For example, we can change both hairstyle and hair color, while [17] cannot manipulate hairstyle.

In keeping with the goal of disentangled image editing, the authors also introduce a regularization term to their objective function, which encourages the editing directions to be mutually perpendicular.

The objective function is given as follows:

$$
\begin{align*}
J(\textbf{u}_1, ..., \textbf{u}_k) &= \sum_k LS(\textbf{u}_k) + cR(\textbf{u}_1, ..., \textbf{u}_k) \\
&= \sum_k LS(\textbf{u}_k) - c\frac{1}{2}||\text{Corr}(\textbf{u}_1, ..., \textbf{u}_k) - \textbf{I}_k||_F
\end{align*}
$$

where $\text{Corr}(.)$ is the correlation matrix, $||.||_F$ is the Frobenius norm, and $\textbf{I}_K$ is the $K\times K$ identity matrix.

### Comparisons

LELSD outperforms first-order Taylor expansion methods when the distance between the generated image and latent code is increased, as the linearity assumption the latter methods rely on no longer holds.

### Usage

<figure>
    <img src="data/imgs/lelsd_edits.png" width="500px">
</figure>

Using a GAN Inversion model to project real face photos onto the latent space of the StyleGAN, localized editing can be achieved. Furthermore, many edits can be sequentially applied by adding up the discovered latent space directions for each semantic.
