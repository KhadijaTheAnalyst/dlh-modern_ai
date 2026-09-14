# Keras & TensorFlow: Building, Training, and Evaluating Neural Networks

## 📖 About the Project

This project is a hands-on introduction to building neural networks using **Keras**, the high-level deep learning API built into **TensorFlow**. Rather than implementing forward/backward propagation from scratch, this project focuses on understanding how to leverage a production-grade framework to design, compile, train, evaluate, and deploy neural network models efficiently.

The goal isn't just to write code that works — it's to understand *why* each step exists in the deep learning workflow: why we compile before we train, why the choice of loss function matters, why monitoring validation performance during training prevents costly surprises later, and how to persist a trained model so it can be reused without retraining.

By the end of this project, the objective is to be able to explain every concept below **without relying on Google** — a strong signal that the underlying "why," not just the "how," has been internalized.

---

## 🎯 Learning Objectives

At the end of this project, I should be able to explain:

| # | Concept |
|---|---------|
| 1 | What is Keras? |
| 2 | What is TensorFlow? |
| 3 | What is a model? |
| 4 | What is a shallow neural network? |
| 5 | What defines a deep neural network? |
| 6 | What is the Sequential model in Keras? |
| 7 | What is the functional API in Keras? |
| 8 | When to use a Sequential model vs. a functional model |
| 9 | What compiling a model in Keras does |
| 10 | How to train a model in Keras |
| 11 | How to choose the right loss function and optimizer |
| 12 | How to monitor performance during training |
| 13 | How to assess the performance of a trained model |
| 14 | How to make predictions on new data using a trained model |
| 15 | How to save an entire Keras model |
| 16 | How to save only the weights of a model |
| 17 | What TensorBoard is and what it's used for |

---

## 📊 Dataset

All tasks in this project use the **MNIST handwritten digits dataset** — a classic benchmark dataset of 70,000 grayscale images (28×28 pixels) of handwritten digits (0–9), split into 60,000 training samples and 10,000 test samples. It's small enough to iterate on quickly, yet nuanced enough to meaningfully exercise everything from a simple Sequential model to more advanced architectures.

Keras ships with built-in access to MNIST, so no manual downloading or preprocessing pipeline is needed to get started.

### 📥 Loading the dataset

```python
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

This returns two tuples: `(x_train, y_train)` for training and `(x_test, y_test)` for testing. Each `x` array holds the raw 28×28 pixel-intensity images (values from 0–255), and each `y` array holds the corresponding integer label (0–9).

### 🖼️ Visualizing sample images

To get a feel for the data before modeling it, the snippet below displays the first 10 training images alongside their labels:

```python
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

num_samples = 10
plt.figure(figsize=(10, 2))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### 🔍 Inspecting raw pixel values

Since each MNIST image is really just a 28×28 grid of integers, it can help to look at the raw numbers behind the picture at least once — it makes the "an image is just a matrix" intuition concrete before any preprocessing (normalization, reshaping, flattening) is applied later in the pipeline:

```python
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), _ = keras.datasets.mnist.load_data()

image = x_train[0]
label = y_train[0]

print("Pixel Values (28x28):")
for row in image:
    print(" ".join(f"{pixel:3}" for pixel in row))

plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.colorbar(label="Pixel Intensity")
plt.show()
```

> [!tip] Note
> Pixel values here are unnormalized (0–255). A common early step in most of the tasks that follow is scaling these to a 0–1 range before feeding them into a network, since neural networks tend to train faster and more stably on normalized inputs.

---

## 🧠 Core Concepts

### Frameworks: TensorFlow & Keras
**TensorFlow** is the underlying numerical computation engine — it handles tensors, automatic differentiation, and hardware acceleration (CPU/GPU/TPU). **Keras** is the high-level API layered on top of it, providing an intuitive, readable interface (`Dense`, `compile`, `fit`) so that model architecture can be expressed declaratively instead of manually wiring gradient computations.

### Models & Network Depth
A **model** bundles together an architecture (layers), learned parameters (weights), and a training configuration (loss, optimizer, metrics). A **shallow network** has just one hidden layer, while a **deep network** stacks multiple hidden layers — each learning progressively more abstract representations of the input.

### Two Ways to Build in Keras
- **Sequential API** — layers stacked linearly, one feeding directly into the next. Ideal for simple, single-input/single-output architectures.
- **Functional API** — layers connected explicitly like a graph, enabling multiple inputs/outputs, shared layers, and branching architectures (e.g. residual connections).

### The Training Workflow
1. **Compile** — attach a loss function, optimizer, and evaluation metrics to the model.
2. **Train** — call `.fit()` to run forward passes, compute loss, backpropagate, and update weights over multiple epochs.
3. **Monitor** — use validation data and callbacks (`EarlyStopping`, `ModelCheckpoint`, `TensorBoard`) to track performance and catch overfitting in real time.
4. **Evaluate** — call `.evaluate()` on held-out test data to get an unbiased measure of performance.
5. **Predict** — call `.predict()` to generate outputs on new, unseen data.
6. **Persist** — save either the entire model (architecture + weights + optimizer state) or just the learned weights, depending on the use case.

### Visualizing Training with TensorBoard
**TensorBoard** is TensorFlow's built-in visualization suite. By logging metrics during training via a callback, it becomes possible to inspect loss/accuracy curves, weight distributions, and computation graphs interactively in the browser — an essential tool for debugging training dynamics and comparing experiments.

---

## 🛠️ Technologies Used

- **Python 3**
- **TensorFlow / Keras**
- **NumPy**
- **Google Colab** (development & experimentation environment)

---

## 📂 Project Structure

```
.
├── README.md
├── 0-sequential.py
├── 1-input.py
├── 2-optimize.py
├── 3-one_hot.py
├── 4-train.py
├── ...
└── notebooks/
    └── keras_reference_notebook.ipynb
```

> Task files follow the incremental, checker-driven format standard across the curriculum — each script builds on concepts introduced in the previous one, moving from model architecture definition through compilation, training, evaluation, and persistence.

---

## 🚀 Usage

Clone the repository and run any task file directly:

```bash
git clone <repo-url>
cd <project-directory>
python3 0-sequential.py
```

All scripts are written to be compatible with the Ubuntu 20.04 LTS / Python 3.9+ / TensorFlow environment specified by the curriculum checker.

---

## 📝 Notes

- A self-annotated, first-person "study notes" reference notebook accompanies this project (see `notebooks/`), documenting the reasoning behind each design decision — from architecture choice to optimizer selection — for future review.
- Style compliance follows `pycodestyle` conventions, consistent with prior projects in this curriculum track.

---

## 👤 Author

**Khadija Mustafa**
AI Academy — Digital Learning Hub Luxembourg (Holberton School curriculum)
[GitHub](https://github.com/khadijatheanalyst) · [Portfolio](https://khadijatheanalyst.github.io)