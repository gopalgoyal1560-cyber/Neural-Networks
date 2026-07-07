
# Neural Network Implementations

This repository contains implementations of Multi-Layer Perceptrons (MLPs) built in two different ways:

* **From scratch using NumPy**
* **Using PyTorch**

The goal of this project is to understand how feedforward neural networks, backpropagation, and gradient-based optimization work by first implementing them manually and then comparing the implementation with a deep learning framework.

## Repository Structure

├── second.py        # MLP implemented entirely using NumPy
├── third.py      # Equivalent implementation using PyTorch
├── model.pth           # Saved PyTorch model (generated after training)
└── README.md

## Features

### NumPy Implementation

* Fully connected Multi-Layer Perceptron
* Configurable number of hidden layers
* Configurable number of neurons per hidden layer
* He weight initialization
* ReLU activation
* Softmax output layer
* Cross-Entropy loss
* Manual forward propagation
* Manual backpropagation
* Gradient descent parameter updates
* Mini-batch training
* Training loss visualization
* Test accuracy visualization

  
### PyTorch Implementation

* Dynamic MLP architecture
* User-defined hidden layers
* User-defined neurons per layer
* ReLU activation
* Dropout regularization
* CrossEntropyLoss
* Adam optimizer
* Mini-batch training with DataLoader
* Validation accuracy tracking
* Model saving using `torch.save`
* Training loss visualization

## Dataset

Both implementations use the **MNIST handwritten digit dataset** downloaded through `fetch_openml()` from scikit-learn.

The images are:

* 28 × 28 grayscale images
* Flattened into 784-dimensional vectors
* Standardized before training
* 
## Requirements

Install the required packages:

```bash
pip install numpy matplotlib scikit-learn torch

## Running the Code

### NumPy Version

```bash
python second.py

You will be prompted to enter:

* Number of layers
* Learning rate
* Number of neurons in each hidden layer

### PyTorch Version

```bash
python third.py
```
You will be prompted to enter:

* Number of layers
* Number of neurons in each hidden layer

After training, the model parameters are saved as:
model.pth

## Example Architecture

Example input:

```
Number of layers: 3

Hidden Layer 1: 128 neurons

Hidden Layer 2: 64 neurons
```

Resulting network:

```
784
 │
128
 │
64
 │
10

## Learning Objectives

This project was built to gain a deeper understanding of:

* Feedforward neural networks
* Matrix-based forward propagation
* Backpropagation
* Gradient computation
* Weight initialization
* Activation functions
* Cross-Entropy loss
* Mini-batch gradient descent
* Training neural networks with PyTorch

## Future Improvements

* Convolutional Neural Network (CNN)
* Batch Normalization
* Learning rate scheduling
* Early stopping
* Additional optimizers
* Support for GPU training
* Model evaluation metrics (precision, recall, confusion matrix)
* Hyperparameter configuration through command-line arguments

This project is intended for educational and learning purposes.
