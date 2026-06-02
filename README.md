# Interactive DCGAN Digit Generator

An interactive application powered by a **Deep Convolutional Generative Adversarial Network (DCGAN)** that generates handwritten digits in real time.

Unlike a typical GAN demo, this project exposes the model's **100-dimensional latent space** through sliders, allowing you to directly manipulate the input vector and observe how changes affect the generated digit. You can also randomize the latent vector with a single button click to instantly generate a new digit.



## Features

- Generate handwritten digits using a trained DCGAN
- Control all 100 latent-space dimensions with sliders
- Real-time image generation and updates
- Randomize the latent vector with a reset button
- Explore how latent-space changes influence generated digits
- Built with PyTorch


## Screenshots

<p align="center">
  <img src="images/img1.png" alt="Latent Space Controls" width="500"/>
</p>

<p align="center">
  <img src="images/img2.png" alt="Generated Digit Example" width="500"/>
</p>

<p align="center">
  <img src="images/img3.png" alt="Interactive Generation" width="500"/>
</p>

## What is Latent Space?

Latent space is a compressed numerical representation learned by the GAN during training. Each of the 100 sliders controls one dimension of this representation.

Small changes to the latent vector can produce smooth changes in the generated digit, making it possible to explore how the model has learned to represent handwritten numbers.



## Installation

### 1. Create a Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch

#### CUDA (NVIDIA GPU)

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### CPU Only

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

## How It Works

A DCGAN generator takes a 100-dimensional latent vector as input and transforms it into a handwritten digit image.

This application allows you to:

1. Adjust individual latent dimensions using sliders.
2. See the generated digit update in real time.
3. Explore how different regions of the latent space correspond to different digit shapes and styles.
4. Generate a completely new latent vector using the reset button.

The project provides an intuitive way to visualize and experiment with the latent space learned by a generative neural network.

