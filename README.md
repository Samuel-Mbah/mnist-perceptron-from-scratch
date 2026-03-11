# Perceptron from Scratch on MNIST

>
> The current name "Perceptron" is generic. A more descriptive name — `mnist-perceptron-from-scratch` — immediately communicates *what* is being implemented (a perceptron), *how* it is built (from scratch, without high-level ML frameworks), and *on what data* it is trained (the MNIST handwritten-digit dataset).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samuel-Mbah/Perceptron/blob/main/Training_Perceptron_with_the_MNIST_Dataset.ipynb)

A hands-on implementation of a single-layer **Perceptron** trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten-digit dataset. The project is designed as a learning resource for understanding the core building blocks of neural networks: gradient descent, the Mean Squared Error (MSE) loss function, forward and backward propagation, and the effect of different activation functions on learning dynamics.

---

## Table of Contents

- [Overview](#overview)
- [Concepts Covered](#concepts-covered)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Activation Functions](#activation-functions)
- [Results](#results)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [License](#license)

---

## Overview

This project builds a Perceptron classifier **entirely from scratch** using only NumPy. Rather than relying on frameworks like scikit-learn or PyTorch, every component — weight initialisation, forward pass, loss calculation, and gradient-descent weight update — is implemented manually. This makes the code an ideal study companion for anyone new to machine learning or wanting a deeper understanding of how neural networks work under the hood.

The binary classification task is to distinguish between the digits **0** and **1** from the MNIST dataset.

---

## Concepts Covered

| Concept | Description |
|---|---|
| **Perceptron** | A single artificial neuron that learns a linear decision boundary |
| **Forward Pass** | Computing the prediction: `ŷ = activation(W · x + b)` |
| **Mean Squared Error (MSE)** | Loss function measuring the average squared difference between predictions and targets |
| **Gradient Descent** | Iterative optimisation algorithm that updates weights in the direction that minimises the loss |
| **Backward Pass** | Applying the chain rule to propagate gradients back through the activation and loss functions to update weights and bias |
| **Activation Functions** | Non-linear functions applied to the pre-activation value (see [Activation Functions](#activation-functions)) |
| **Learning Rate** | Hyperparameter controlling the step size during weight updates |
| **Epochs** | Number of complete passes through the training dataset |

---

## Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a benchmark image classification dataset widely used in machine learning research.

| Property | Value |
|---|---|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image dimensions | 28 × 28 pixels (grayscale) |
| Pixel value range | 0–255 (normalised to 0–1 in this project) |
| Classes (full) | 10 (digits 0–9) |
| Classes (this project) | 2 (digits 0 and 1) |

Images are flattened from a 28 × 28 matrix into a 784-dimensional vector before being fed to the Perceptron.

---

## Project Structure

```
.
├── Training_Perceptron_with_the_MNIST_Dataset.ipynb   # Main notebook
├── README.md                                           # This file
└── LICENSE                                             # MIT License
```

---

## Notebook Walkthrough

The notebook is organised into the following logical sections:

1. **Import & Explore the Dataset**
   - Load MNIST via `keras.datasets.mnist`
   - Inspect shapes of training and test sets
   - Visualise sample images

2. **Prepare the Data**
   - Filter for digits 0 and 1 to create a binary classification problem
   - Flatten 28 × 28 images to 784-dimensional vectors
   - Normalise pixel values to the range [0, 1]

3. **Define the Loss Function**
   - `cost_function(prediction, target)` — MSE loss and its derivative

4. **Define Activation Functions**
   - Heaviside step function, Sigmoid, Tanh, ReLU — each supports a `derivative` flag for use in the backward pass

5. **Build the Perceptron Class**
   - `__init__`: configurable weight initialisation, bias, activation function, and loss function
   - `forward_pass(input_v)`: computes the prediction
   - `backward_pass(inputs, targets, learning_rate, epochs)`: runs gradient descent and records loss, weight, and bias history

6. **Train & Compare Activation Functions**
   - Train separate Perceptron instances using Heaviside, Sigmoid, Tanh, and ReLU
   - Plot weight trajectories over epochs for each activation function

7. **Compare Loss Curves**
   - Side-by-side 2 × 2 subplot comparing the training loss curves for all four activation functions

8. **Analyse Learning Rate Effects**
   - Iterate over a range of learning rates and measure training time to illustrate the impact of this hyperparameter

---

## Activation Functions

Four activation functions are implemented and compared:

| Function | Formula | Notes |
|---|---|---|
| **Heaviside Step** | `1 if x ≥ θ else 0` | Non-differentiable; derivative approximated as 1 everywhere |
| **Sigmoid** | `1 / (1 + e^(-x))` | Smooth, output in (0, 1); gradient can vanish for large `\|x\|` |
| **Tanh** | `(e^x - e^(-x)) / (e^x + e^(-x))` | Output in (-1, 1); zero-centred, often trains faster than sigmoid |
| **ReLU** | `max(0, x)` | Computationally efficient; sparse activation; can suffer from dying neurons |

---

## Results

Training a Perceptron for **100 epochs** at a learning rate of **0.001** on the binary MNIST subset (digits 0 and 1) shows:

- **Heaviside**: Converges quickly but the non-differentiable step introduces instability in weight updates.
- **Sigmoid**: Smooth loss curve; converges steadily.
- **Tanh**: Similar to Sigmoid but zero-centred; typically converges faster.
- **ReLU**: Very fast updates; may exhibit some instability on this simple task.

Weight heatmaps (reshaping the 784-dimensional weight vector back to 28 × 28) reveal which pixel regions the Perceptron focuses on for each activation function.

---

## Prerequisites

- Python 3.7+
- [NumPy](https://numpy.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/) (used only for loading the MNIST dataset)
- [Matplotlib](https://matplotlib.org/)

---

## Getting Started

### Run in Google Colab (recommended — no local setup required)

Click the badge at the top of this README or [open the notebook directly in Colab](https://colab.research.google.com/github/Samuel-Mbah/Perceptron/blob/main/Training_Perceptron_with_the_MNIST_Dataset.ipynb).

### Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/Samuel-Mbah/Perceptron.git
   cd Perceptron
   ```

2. **Install dependencies**
   ```bash
   pip install numpy tensorflow matplotlib jupyter
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook Training_Perceptron_with_the_MNIST_Dataset.ipynb
   ```

4. **Run all cells** sequentially from top to bottom (`Kernel → Restart & Run All`).

---

## License

This project is licensed under the [MIT License](LICENSE).
