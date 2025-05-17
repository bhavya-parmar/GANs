# GANs
---
## DCGAN

Here I try to implement a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the **MNIST** dataset, written from scratch using PyTorch. This was inspired by this [paper](https://arxiv.org/abs/1511.06434), and [Aladdin Perrson](https://youtu.be/IZtv9s_Wx9I?feature=shared).

DCGANs are a type of GAN where the generator and discriminator both use deep convolutional layers, making them especially suited for image generation tasks.

This project includes:
- Custom **Discriminator** and **Generator** architectures
- Proper **weight initialization**
- Complete **training loop**
- Logging with **TensorBoard**
- MNIST dataset usage for realistic synthetic handwritten digits generation

### Model Architectures

#### Generator

- **Input:** Noise vector of shape `(z_dim, 1, 1)`
- **Architecture:**
  - Series of `ConvTranspose2d` layers
  - `BatchNorm2d` after each convolution
  - `ReLU` activation functions
- **Output:** Grayscale image of shape `(1, 64, 64)`
  - Final activation: `Tanh` (to produce pixel values in range `[-1, 1]`)

#### Discriminator

- **Input:** Image of shape `(1, 64, 64)`
- **Architecture:**
  - Series of `Conv2d` layers
  - `BatchNorm2d` after each convolution (except the first)
  - `LeakyReLU` activation functions
- **Output:** Single scalar value
  - Final activation: `Sigmoid` (probability that input is real)
---
