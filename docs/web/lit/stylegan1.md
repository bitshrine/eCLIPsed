# StyleGAN 1

> **A Style-Based Generator Architecture for Generative Adversarial Networks**
>
> Karras et al, 2019
> [<a href="data/papers/styleGAN1.pdf" target="download">PDF</a>]


## 0. Preface

### Progressive GAN

ProgressiveGAN (Karras et al, 2017) generates high-resolution images by progressively increasing the size of outputs of sub-units (4x4 $\rightarrow$8x8 $\rightarrow$16x16 $\rightarrow$...). Each such resolution produces an image in latent space which is converted to RGB with a 1x1 convolution. Each computation block upscales the resolution of its input image by a factoro of 2, and adds two new 3x3 convolution layers. A skip-forward connection between resolutions, whose weight is slowly reduced, allows for a smooth transition between blocks. Both the generator and discriminator follow this architecture.

### Latent spaces
The latent space can be understood as a space where each image is represented by an n-dimensional vector. Ideally, each of these dimensions would correspond to a semantically relevant feature, so a picture of an apple could be represented by a vector `(size, color, stem_length, ...)`. Such a space where the features are distinctly represented is referred to as 'disentangled'. In an entangled latent space, modifying one of the attributes of an image leads to the joint modification of another attribute, as they are encoded in the same dimension, with potentially undesirable outcomes.

## 1. Abstract

> We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an <mark>automatically learned, unsupervised separation of high-level attributes</mark> (e.g., pose and identity when trained on human faces) and <mark>stochastic variation in the generated images</mark> (e.g., freckles, hair), and it enables <mark>intuitive, scale-specific control of the synthesis</mark>. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. To quantify interpolation quality and disentanglement, we propose <mark>two new, automated methods that are applicable to any generator architecture.</mark> Finally, we introduce a new, highly varied and high-quality dataset of human faces.


<figure>
    <img src="data/imgs/stylegan.png" width="400px">
</figure>

## 2. Summary

This paper proposes a generator architecture with several improvements, while the discriminator remains untouched. The generator, shown above, is comprised of two networks. 

### Mapping network
The mapping network maps the input $\mathcal{Z}$ space to an intermediary $\mathcal{W}$ space. The elements of this space are then subjected to an affine transformation (the output of which they call **styles**) before being fed into the synthesis network at various levels. The mapped $\mathcal{W}$ space is disentangled to a degree, measured by the two new evaluation metrics proposed by the authors (see below).

### Synthesis network
The synthesis network does not take input, but rather is initialized by a learned constant. At each layer, it receives fresh noise which it applies on a per-pixel basis to the output of each convolution. The purpose of the noise is to help with stochastic variation, which is useful for generating elements whose attributes do not matter as much as their distribution (hair, eyelashes, pores, freckles, etc). It also receives a style vector from the mapping network. The style and noise are combined in an AdaIN unit, which consists of a normalization followed by a modulation/biasing/shifting:

$$
\begin{align*}
\text{AdaIN}(x_i, y_i) = y_{s, i}\frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b, i} && \text{where }y_i = (y_{s, i}, y_{b, i}) = f_{A_i}(w)
\end{align*}
$$

After several such processing units, it produces an output image.

The interaction between both networks is summarized as such:
> We can view the mapping network and affine transformations as a way to draw samples for each style from a learned distribution, and the synthesis network as a way to generate a novel image based on a collection of styles.

### Perceptual Path Length
This metric measures the degree of changes done on the image when performing interpolation, when going from one image vector to another. Namely, the metric measures the distance between embeddings by splitting the interpolation into smaller linear segments, over which the perceptual difference and distance are computed. This distance should be minimized for better results.

### Linear separability
This metric measures how linearly separable two latent space points are, when they represent two different image classes, separated by a hyperplane. This is done by computing the conditional entropy which reveals how much information is required to accurately classify both points.



### StyleMixing regularization

The idea here is to take two different codes $w1$ and $w2$ and feed them to the synthesis network at different levels. This way, $w1$ will be applied from the first layer up to a certain layer in the network, which they call the crossover point, and $w2$ is applied from that point on until the end.
> This regularization technique prevents the network from assuming that adjacent styles are correlated.[1]

Besides the impact of style regularization on the FID score, which decreases when applying it during training, it is also an interesting image manipulation method. The below figure shows the results of style mixing with different crossover points:

<figure>
    <img src="data/imgs/stylegan_regularization.webp">
</figure>

Given that the network changes the dimensions of the image as it travels through it, using styles at different resolutions has a different impact on the image.  Copying (or "switching-over") the styles corresponding to coarse resolutions ($4^2 - 8^2$) means high-level aspects (such as glasses, face shape, pose) are copied from B, whereas copying the finer resolutions ($64^2 - 1024^2$) only takes minor details such as colors into account.


The figure below shows the performance of different techniques applied to various generator designs:
<figure>
    <img src="data/imgs/stylegan_techniques.png" width="400px">
</figure>


### Truncation trick in the W space

Poorly represented images in the dataset are generally very hard to generate by GANs. Since the generator doesn’t see a considerable amount of these images while training, it can not properly learn how to generate them which then affects the quality of the generated images.
To encounter this problem, there is a technique called “the truncation trick” that avoids the low probability density regions to improve the quality of the generated images. But since we are ignoring a part of the distribution, we will have less style variation.
This technique is known to be a good way to improve GANs performance and it has been applied to Z-space.
StyleGAN offers the possibility to perform this trick on W-space as well. This is done by firstly computing the center of mass of W:

$$
\bar{\textbf{w}} = \mathbb{E}_{\textbf{z} \sim P(\textbf{z})}[f(\textbf{z})]
$$


That gives us the average image of our dataset. Then, we have to scale the deviation of a given w from the center:

$$
\begin{align*}
\textbf{w}' = \bar{\textbf{w}} + \psi(\textbf{w} - \bar{\textbf{w}}) && \text{where }\psi < 1
\end{align*}
$$

Interestingly, the truncation trick in w-space allows us to control styles. As shown in the following figure, when we tend the parameter to zero we obtain the average image. On the other hand, when comparing the results obtained with 1 and -1, we can see that they are corresponding opposites (in pose, hair, age, gender..). This highlights, again, the strengths of the W-space.

<figure>
 <img src="data/imgs/stylegan_truncation.webp" width="700px">
</figure>