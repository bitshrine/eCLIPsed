# GANSpace

> **GANSpace: Discovering Interpretable GAN controls**
> 
> Härkönen et al, 2016
>[<a href="data/papers/GANSpace.pdf" target="download">PDF</a>]

## 0. Preface

### Principal Component Analysis
Principal Component Analysis, often shortened to PCA, is a dimensionality reduction method that is often used to reduce the dimensionality of large datasets, by transforming a large set of variables into a smaller one that still contains most of the information of the large set. Intuitively, it can be understood as only keeping data along dimensions where the value is least predictable, i.e. the dimensions along which the data has maximum variance.

1. Standardize values to have zero mean and unit variance
2. Compute the covariance matrix $\mathcal{C}$ of all samples
3. Compute $\textbf{Q}$ and $\Lambda$ from the eigenvalue decomposition $\textbf{A} = \textbf{Q}\Lambda\textbf{Q}^{-1}$ 
4. Compute the feature vector by keeping only the $k$ first eigenvectors (by plotting all the eigenvalues in order, there is usually a "falloff" point after which the eigenvalues suddenly decrease in value; the index of the last "big" eigenvalue is usually chosen as $k$)
5. Recast the data along the principal component axes: $\text{Dataset}_\text{PCA} = \textbf{Q}_{\text{truncated}}^T * \text{Dataset}_\text{standardized}^T$

## 1. Abstract

> This paper describes a simple technique to analyze Generative Adversarial Networks (GANs) and create interpretable controls for image synthesis, such as change of viewpoint, aging, lighting, and time of day. We identify important latent directions based on Principal Component Analysis (PCA) applied either in latent space or feature space. Then, we show that a large number of interpretable controls can be defined by layer-wise perturbation along the principal directions. Moreover, we show that BigGAN can be controlled with layer-wise inputs in a StyleGAN-like manner. We show results on different GANs trained on various datasets, and demonstrate good qualitative matches to edit directions found through earlier supervised approaches.


## 2. Summary


### PCA to understand the learned features

The purpose of this paper is to put forward a way to introduce edits and specifications to a pre-trained GAN's generator network, acting as somewhat of a "black box", to control attributes of the output image. This allows for more general use of the generator without having to re-train it or use supervised methods. In the case of both StyleGAN and BigGAN, using PCA in the latent space enables "browsing" through the concepts that the GAN has learned. The authors also explain the method for applying that knowledge to introduce specifications to the network and control its output.

The paper starts with a short paragraph laying out the notation used, which is reported below:

> We begin with a brief review of GAN representations. The most basic GAN comprises a probability distribution $p(\textbf{z})$, from which a latent vector $\textbf{z}$ is sampled, and a neural network $G(\textbf{z})$ that produces an output image $I: \textbf{z} ∼ p(\textbf{z}), I = G(\textbf{z})$. 
> The network can be further decomposed into a series of L intermediate layers $G_1...G_L$. The first layer takes the latent vector as input and produces a feature tensor $\textbf{y}_1 = G_1(\textbf{z})$ consisting of set of feature maps. The remaining layers each produce features as a function of the previous layer’s output: $\textbf{y}_i = \hat{G}_i(\textbf{z}) ≡ G_i (\textbf{y}_i−1)$. The output of the last layer $I = G_L(\textbf{y}_L−1)$ is an RGB image. [...] In a StyleGAN model, the first layer takes a constant input $\textbf{y}_0$. Instead, the output is controlled by a non-linear function of $\textbf{z}$ as input to intermediate layers:
$$\begin{align*}\textbf{y}_i = G_i(\textbf{y}_i−1,\textbf{w}) && \text{with }\textbf{w} = M(\textbf{z}) \end{align*}$$
where $M$ is an 8-layer multilayer perceptron. In basic usage, the vectors $\textbf{w}$ controlling the synthesis at each layer are all equal; the authors demonstrate that allowing each layer to have its own $\textbf{w}_i$ enables powerful “style mixing,” the combination of features of various abstraction levels across generated images.

### Model-specific analysis

For StyleGAN, the authors give the following procedure to find the principal components:

> Our goal is to identify the principal axes of $p(\textbf{w})$. To do so, we sample $N$ random vectors $\textbf{z}_{1:N}$, and compute the corresponding $\textbf{w}_i = M(\textbf{z}_i)$ values. We then compute PCA of these $w_{1:N}$ values. This gives a basis $\textbf{V} for \mathcal{W}$. Given a new image defined by $\textbf{w}$, we can edit it by varying PCA coordinates $x$ before feeding to the synthesis network:
> $$\textbf{w}' = \textbf{w} + \textbf{V}x $$
> where each entry $x_k$ of $x$ is a separate control parameter. The entries $x_k$ are initially zero until
modified by a user.

### Editing the output

Recall the [architecture of StyleGAN](web/lit/stylegan1.md#1-abstract). The authors of the paper state that StyleGAN's styles can be leveraged to apply layer-wise edits.

> Given an image with latent vector $\textbf{w}$, layerwise edits entail modifying only the $\textbf{w}$ inputs to a range of layers, leaving the other layers’ inputs unchanged. We use notation $E(\textbf{v}_i, j-k)$ to denote edit directions; for example, $E(\textbf{v}_1, 0-3)$ means moving along component $\textbf{v}_1$ at the first four layers only. $E(\textbf{v}_2, all)$ means moving along component $\textbf{v}_2$ globally: in the latent space and to all layer inputs. Edits in the $\mathcal{Z}$ latent space are denoted $E(\textbf{u}_i, j-k)$.

### Properties

The authors make the observation that not all principal components are of equal influence; indeed, in all trained models they explored, "large-scale changes to geometric configuration and viewpoint are limited to the first 20 principal components $(\textbf{v}_0, \textbf{v}_{20})$", whereas the successive components focus less on layout and more on appearance of the object/background. Furthermore, in the case of StyleGANv2, the latent distribution $p(\textbf{z})$ has a "relatively simple structure". Its principal components are "nearly-independent variables with non-Gaussian unimodal distributions". Finally, only the first 100 or so components have large-scale influence on the image's appearance, whereas the other 412 components control subtler aspects.

The authors also observe that some entanglement of the space is linked to the dataset; they give the example of a car image, which has a "more "open road" background" when it is made sportier, but more "urban" or "forest-like" backgrounds when it is made more family-oriented, which they explain by the content of marketing images used in training. Other such effects exist in images of dogs, when rotated, and people, as the gender is changed.

On top of these seemingly linked combinations, the authors also make note of "disallowed" combinations. Indeed, attempting to add "wrinkles" to a child's face yields little result, same as trying to apply makeup to a male-presenting face.
