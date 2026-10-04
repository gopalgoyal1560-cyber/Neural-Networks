# Neural Network Implementations: NumPy vs PyTorch

> [!IMPORTANT]
> **This is a learning project.** I built it to understand how neural networks work internally, not to produce a production-ready model. The code favors clarity over performance, and some of my understanding is still evolving. If you spot a mistake or a better way to explain something, please open an issue. I would genuinely like to learn from it.

A Multi-Layer Perceptron (MLP) for **MNIST handwritten digit classification**, implemented twice:

1. **From scratch with NumPy**: forward pass, backpropagation and weight updates written by hand.
2. **With PyTorch**: the same idea using the framework's layers, autograd and optimizers.

Building it manually first and then comparing it with a framework is the core of the exercise. It shows what PyTorch is doing behind `loss.backward()` and `optimizer.step()`.

---

## Why I made this

I wanted to build my own language model one day. Before jumping to transformers, I wanted to truly understand the basics: how a network turns inputs into outputs, how the loss is measured, and how gradients flow backward to update weights. This repository is that foundation. A transformer implementation is part of my learning path (see the [roadmap](#roadmap)).

---

## Repository structure

```
.
├── second.py        # MLP implemented entirely with NumPy
├── third.py         # Equivalent MLP implemented with PyTorch
├── model.pth        # Saved PyTorch model (generated after training)
├── requiments.txt   # Python dependencies
└── README.md
```

---

## Features

### NumPy implementation (`second.py`)
- Fully connected MLP with a configurable number of hidden layers and neurons per layer
- He weight initialization
- ReLU activation and a Softmax output layer
- Cross-entropy loss
- Manual forward propagation and manual backpropagation (chain rule)
- Gradient descent parameter updates with mini-batch training
- Training loss and test accuracy plots

### PyTorch implementation (`third.py`)
- Dynamic MLP architecture (user-defined layers and neurons)
- ReLU activation and Dropout regularization
- `CrossEntropyLoss` and the Adam optimizer
- Mini-batch training with `DataLoader`
- Validation accuracy tracking
- Model saving with `torch.save`
- Training loss plot

---

## Dataset

Both scripts use the **MNIST** handwritten digit dataset, downloaded with scikit-learn's `fetch_openml()`.

- 28 × 28 grayscale images
- Flattened into 784-dimensional vectors
- Standardized before training
- 10 classes (digits 0–9)

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/gopalgoyal1560-cyber/Neural-Networks.git
cd Neural-Networks
```

### 2. Install dependencies

```bash
pip install -r requiments.txt
```

### 3. Run an implementation

```bash
# NumPy version
python second.py

# PyTorch version
python third.py
```

Each script asks for:
- the number of hidden layers
- the number of neurons in each hidden layer

The PyTorch script saves its trained parameters to `model.pth`.

### Example architecture

Entering two hidden layers with 128 and 64 neurons gives:

```
Input (784) → Dense (128) + ReLU → Dense (64) + ReLU → Output (10) + Softmax
```

---

## What I learned

- How matrix-based forward propagation works (`z = X @ W + b`)
- Why activation functions are needed, and how ReLU and Softmax behave
- How cross-entropy loss measures prediction error
- How backpropagation applies the chain rule layer by layer
- How gradient descent updates weights and biases using a learning rate
- Why mini-batches and shuffling are used
- Why weight initialization and regularization (dropout) matter
- How a framework like PyTorch automates gradients, optimizers and data loading

---

## Current limitations

Being honest about what this repo is and is not:

- It only covers a plain MLP on MNIST. It does not include CNNs or any other architecture yet.
- There is no command-line configuration beyond the interactive prompts.
- There is no inference or prediction script yet. `model.pth` is only saved, not served.
- There are no automated tests, and hyperparameters are not tuned.

---

## Roadmap

### Learning roadmap (next)
- [ ] Convolutional Neural Network (CNN) for MNIST
- [ ] Batch normalization
- [ ] Learning rate scheduling
- [ ] Early stopping
- [ ] Additional optimizers (SGD with momentum, AdamW)
- [ ] GPU training support
- [ ] Evaluation metrics: precision, recall, confusion matrix
- [ ] Hyperparameters through command-line arguments
- [ ] Train/validation/test split and a cleaner project layout
- [ ] A small decoder-only **transformer language model** (my longer-term goal), added to this repo or a separate one

### Deployment plans (future)

Deployment is **planned, not done yet**. I want to learn it step by step, using this digit classifier as a small first project:

1. **Inference script**: load `model.pth` and predict the digit for a new image.
2. **REST API**: wrap the model in a [FastAPI](https://fastapi.tiangolo.com/) service with a `/predict` endpoint.
3. **Interactive demo**: a [Gradio](https://www.gradio.app/) or Streamlit app where you draw a digit and see the prediction.
4. **Containerization**: package the app with Docker so it runs the same everywhere.
5. **Hosting**: publish the demo on a free platform such as Hugging Face Spaces or Render.
6. **Basic MLOps**: pinned dependencies, simple tests and a GitHub Actions workflow for checks.

**Longer-term vision:** once the transformer work matures, the plan is to train a small language model, fine-tune it on instruction/conversation data to behave like a chatbot, and later explore tool use to build an agent-style project. This is a long journey, and each step above is meant to teach me something before the next one.

---

## Contributing and feedback

This is a personal learning repo, but feedback is welcome. Please open an issue for corrections, explanations or suggestions.

## License

No license has been added yet. If you want to reuse this code, please open an issue and ask.

---

*Built for education and curiosity. Not intended for production use.*
