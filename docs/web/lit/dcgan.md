# DCGAN

> **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks**
> 
> Radford et al, 2016
>[<a href="../../data/papers/DCGAN.pdf" target="download">PDF</a>]


## 0. Preface - Networks
### Convolutional Neural Networks
> A subclass of Artificial Neural Networks, particularly well-suited for image recognition and processing.

| **Layer** | **Description** | **Effect** |
|---|---|---|
|**Convolutional Layer**|Applies a filter matrix to an image by convoluting it over the input. |Extract features such as edges, textures, and shapes|
|**Pooling layer**|Perform an aggregation operation over several neighboring matrix values|Down-sample feature map, reducing spatial dimensions while attempting to only keep the most relevant information|

Since the network is feed-forward and multi-layered, sequential processing of the image by many convolutional layers allows for the network to learn hierarchical attributes. Three or four convolutional layers allow the network to recognize handwritten digits, and 25 layers allow the network to recognize human faces.

<figure>
<img src="../../data/imgs/cnn.png" width="700px" />
<figcaption>Diagram of a sample Convolutional Neural Network</figcaption>
</figure>

The outputs of pooling layers are passed to dense/fully-connected layers which then produce the network output.

### Generative Adversarial Neural Networks
> A type of deep learning network architecture where two neural networks compete against each other in a zero-sum game.

The goal of GANs is to generate new, synthetic data that resembles some known data distribution. For this purpose, they have two components:

|**Component**|**Purpose**|
|---|---|
|**Generator network**|Creates synthetic data|
|**Discriminator network**| Evaluates the synthetic data and tries to determine if it belongs to the training data's distribution|

The generator network produces synthetic data and the discriminator network evaluates it.
The generator is trained to fool the discriminator and the discriminator is trained to correctly identify real and fake data.
This process continues until the generator produces data that is indistinguishable from real data.

<!-- Add network diagram -->

## 1. Abstract
> In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called <mark class="highlight"> deep convolutional generative adversarial networks (DCGANs)</mark>, that have certain architectural constraints, and demonstrate that they are a <mark class="highlight">strong candidate for unsupervised learning.</mark> Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.


<figure>
<img src="../../data/imgs/dcgan.png" width="700px">
</figure>

## 2. Summary
The particular architecture framework outlined by this paper, which they call DCGAN, is in essence a GAN composed of two competing Convolutional Neural Networks. The DCGAN family of architectures "[results] in stable training across a range of datasets and [allows] for training higher resolution and deeper generative networks".

DCGAN creates discriminator networks with performance that is competitive with other unsupervised algorithms when used for image classification, and their generators have "interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples" (more on that later). Furthermore, visualization of the filters learnt by GANs shows that specific filters have learned to draw specific objects.

### Architecture Guidelines
```
- Replace any pooling layers with:
    - strided convolutions (discriminator)
    - fractional-strided convolutions (generator)
- Use batchnorm in both the generator and the discriminator
- Remove fully connected/hidden layers for deeper architectures
- Use ReLU activation in generator for all layers except the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers
```

#### Replacing the pooling layers

Instead of performing a single aggregation operation (i.e. Max Pooling, where several neighboring pixels "collapse" into a single one with their maximum value, leading to fewer pixels in the layer output), the network performs strided convolutions, which are effectively capable of similar operations, with the same output dimension reduction. Replacing the pooling layers with strided convolutions in the discriminator allows the network to learn its own spatial downsampling, and replacing the pooling layers with fractional-strided convolutions in the generator allows it to learn its own spatial upsampling.


#### Batch normalization
> Stabilizing learning by normalizing the input to each unit to have zero mean and unit variance.

This helps with problems which arise from poor weight initialization and helps gradient flow in deeper models. It notably prevents a very common problem in GANs, where the generator collapses all of the samples into a single point (and therefore mostly generates very similar images, which do not capture the full sample space). To prevent sample oscillation and model instability, batchnorm is not applied to the generator output layer nor the discrimination input layer.

#### ReLU/LeakyReLU Activation
> Used in the generator except for the output layer, which uses Tanh.

Bounded activation allows the model to learn more quickly to saturate and cover the color space of the training distribution. For the discriminator, leaky rectified activation worked well for higher resolution modeling.

## Results

Having trained their models, the team underwent several experiments to understand what the networks had learned, and if these representations were otherwise exploitable.

### Visualizing the internals of the network

The team describes "walking in the latent space" to understand the way in which the space is hierarchically collapsed, and whether the network "memorized" training samples, which would be indicated by sharp transitions in the space. By generating similar images in sequence, they found that features smoothly transitioned from image to image, while maintaining a similar level of plausibility. Over the course of the image sequence, some features also smoothly transitioned into others (a window transforming into a TV)

### Visualizing the discriminator features

Results show that an unsupervised DCGAN trained on a large image dataset can actually learn a hierarchy of semantically relevant features (trained on a dataset of pictures of bedrooms, the network learned representations for "bed", "window", "TV", etc) as a linear structure in representation space. This means they obey vector arithmetic on average. One given example is that the linear combination `vector('King') - vector('Man') + vector('Woman')` resulted in a vector whose nearest neighbor was `vector('Queen')`

<figure>
    <img src="../../data/imgs/dcgan_arithmetic.png" width="700px">
</figure>

<figure>
    <img src="../../data/imgs/dcgan_interpolation.png" width="700px">
</figure>