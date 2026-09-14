# Deep Learning Optimization & Regularization Project

A comprehensive guide to mastering optimization techniques and regularization strategies in deep learning using TensorFlow/Keras.

## Overview

This project consists of 10 tasks covering the complete pipeline of building, optimizing, and tuning deep neural networks for the MNIST classification task. Each task builds upon previous concepts to create production-ready models.

## Tasks Summary

### Task 0: Gradient Descent Variants
**Concept:** Understanding different ways to update model weights during training.

Implements three gradient descent variants with different batch sizes:
- **Batch GD**: Updates after entire dataset (stable but slow)
- **Stochastic GD**: Updates after each sample (fast but noisy)
- **Mini-batch GD**: Updates after small batches (best practical balance)

**Key Function:** `train_with_gradient_descent_variant(variant, learning_rate, x_train, batch_size)`

Returns: optimizer and appropriate batch size based on variant.

---

### Task 1: Momentum-Based SGD Variants
**Concept:** Accelerating convergence using momentum.

Implements three momentum variants:
- **SGD**: Standard stochastic gradient descent
- **SGD+Momentum**: Accumulates velocity from past gradients (smoother convergence)
- **SGD+Momentum+Nesterov**: Look-ahead gradient computation (faster convergence)

**Key Function:** `get_optimizer_SGD(name, lr, momentum, nesterov)`

Returns: configured SGD optimizer with specified momentum strategy.

---

### Task 2: Adaptive Optimizers vs. SGD
**Concept:** Automatic learning rate adaptation for each parameter.

Implements three adaptive optimizers:
- **SGD**: Fixed learning rate for all parameters
- **Adam**: Adaptive learning rates per parameter (BEST in practice)
- **RMSprop**: Adapts based on gradient magnitude history

**Key Function:** `get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho)`

Returns: configured optimizer (SGD, Adam, or RMSprop).

---

### Task 3: Learning Rate Schedules
**Concept:** Dynamically changing learning rate during training.

Implements two schedule types:
- **Exponential Decay**: Smooth, continuous decrease (lr = initial_lr × decay_rate^(step/decay_steps))
- **Inverse Time Decay**: Gradual decrease preventing learning rate from dropping too fast

**Key Function:** `get_optimizer_SGD_with_schedule(schedule_type, initial_lr, decay_steps, decay_rate, momentum)`

Returns: SGD optimizer with learning rate schedule.

---

### Task 4: Weight Initialization
**Concept:** Proper initialization to prevent vanishing/exploding gradients.

Implements activation-specific initializers:
- **Sigmoid/Tanh**: Glorot Uniform (Xavier) initializer
- **ReLU/LeakyReLU**: He Normal initializer

**Key Function:** `build_model_initializer_by_activation(input_dim, hidden_units, activation)`

Returns: compiled Keras model with appropriate weight initialization.

---

### Task 5: L2 Regularization
**Concept:** Penalizing large weights to prevent overfitting.

Applies L2 penalty to kernel weights: Loss = Original_Loss + lambda × sum(weights²)

**Key Function:** `build_model_with_L2_regularization(input_dim, hidden_units, n_layers, lambda_l2)`

Returns: compiled model with L2 regularization on hidden layers.

---

### Task 6: Dropout Regularization
**Concept:** Randomly deactivating neurons during training to prevent co-adaptation.

Architecture:
- Input dropout after input layer
- Dropout after each hidden layer (ReLU activated)
- No dropout on output layer

**Key Function:** `build_model_with_dropout(input_dim, hidden_units, n_layers, dropout_rate_input, dropout_rate_hidden)`

Returns: compiled model with dropout layers.

---

### Task 7: Early Stopping
**Concept:** Stopping training when validation metric stops improving.

Prevents overfitting by monitoring validation loss/accuracy and restoring best model weights.

**Key Function:** `get_early_stopping_callback(patience, monitor, verbose)`

Returns: configured EarlyStopping callback for model.fit().

---

### Task 8: Build Model to be Tuned
**Concept:** Creating a model with tunable hyperparameters for automated search.

Tunable parameters:
- `num_layers`: 1-2 hidden layers
- `units`: 4-12 neurons per layer (step of 4)
- `activation`: relu or sigmoid
- `learning_rate`: 1e-2 or 1e-3

**Key Function:** `build_model(hp)`

Returns: compiled Keras model with hyperparameters from HyperParameters object.

---

### Task 9: Initiate the Tuner
**Concept:** Setting up automated hyperparameter optimization.

Supports three tuner types:
- **Hyperband**: Fast, progressive elimination of poor configs
- **RandomSearch**: Random sampling from search space
- **BayesianOptimization**: Smart probabilistic search

**Key Function:** `initiate_tuner(tuner_type, build_model, seed, hyperband_iterations, max_trials, objective)`

Returns: configured Keras Tuner instance.

---

### Task 10: Search and Return Best Model
**Concept:** Running the hyperparameter search and retrieving the best configuration.

Executes the search process and returns optimal hyperparameters found.

**Key Function:** `search_and_return_best_model(tuner, x_train, y_train, epochs, validation_split, verbose)`

Returns: HyperParameters object with best hyperparameter configuration.

---

## Key Concepts

### Optimization
Finding the best weights to minimize loss through iterative updates. Different variants trade off speed vs. stability:
- **Batch**: Stable but slow
- **Stochastic**: Fast but noisy
- **Mini-batch**: Best practical balance
- **Adaptive (Adam)**: Automatic per-parameter learning rates

### Regularization
Preventing overfitting by penalizing model complexity:
- **L2**: Penalizes weight magnitude (smooth, distributed penalty)
- **Dropout**: Randomly deactivates neurons (sharp, implicit ensemble)
- **Early Stopping**: Stops before overfitting begins
- **Weight Initialization**: Prevents gradient issues from the start

### Hyperparameter Tuning
Automatically searching for optimal hyperparameters:
- **Hyperband**: Fastest, eliminates bad configs early
- **RandomSearch**: Simple baseline, random sampling
- **BayesianOptimization**: Smartest, learns from previous trials

## Dataset

All tasks use MNIST handwritten digit dataset:
- 60,000 training images
- 10,000 test images
- 28×28 pixel grayscale images
- 10 classes (digits 0-9)

Data shape after preprocessing: (num_samples, 784)

## Dependencies

```
tensorflow >= 2.0
keras-tuner >= 1.0
numpy
matplotlib
pandas
```

## Quick Start

```python
# Task 0: Choose gradient descent variant
optimizer, batch_size = train_with_gradient_descent_variant(
    'mini_batch', lr=0.01, x_train=x_train, batch_size=32
)

# Task 2: Use adaptive optimizer
optimizer = get_optimizer('adam', lr=0.001, momentum=0.9, 
                          beta_1=0.9, beta_2=0.999, rho=0.9)

# Task 4: Build model with proper initialization
model = build_model_initializer_by_activation(784, 128, 'relu')

# Task 5: Add L2 regularization
model = build_model_with_L2_regularization(784, 128, 2, lambda_l2=1e-6)

# Task 6: Add dropout
model = build_model_with_dropout(784, 128, 2, 
                                 dropout_rate_input=0.2,
                                 dropout_rate_hidden=0.5)

# Task 7: Early stopping
callback = get_early_stopping_callback(patience=3, monitor='val_loss')

# Task 8-10: Hyperparameter tuning
build_model = build_model  # from task 8
tuner = initiate_tuner('Hyperband', build_model, seed=0, 
                       hyperband_iterations=5, max_trials=5)
best_hp = search_and_return_best_model(tuner, x_train, y_train, 
                                       epochs=10, validation_split=0.2)
```

## Learning Path

1. **Start with basics** (Tasks 0-3): Understand optimization strategies
2. **Prevent overfitting** (Tasks 4-7): Learn regularization techniques
3. **Automate tuning** (Tasks 8-10): Master hyperparameter optimization

Each task builds on the previous, creating a complete pipeline for production deep learning.

## Best Practices Summary

| Aspect | Recommendation |
|--------|-----------------|
| **Optimizer** | Start with Adam, fall back to SGD+Momentum for stability |
| **Learning Rate** | Start at 1e-3, use schedule for fine-tuning |
| **Regularization** | Combine L2 + Dropout for best generalization |
| **Early Stopping** | Always monitor validation loss with patience=3-5 |
| **Hyperparameter Tuning** | Use Hyperband for speed, Bayesian for accuracy |
| **Model Architecture** | Start shallow (1-2 layers), increase if needed |

## Performance Comparison

Expected results on MNIST (after tuning):

- Without regularization: 98.5% train, 97.8% validation
- With L2 only: 98.2% train, 98.1% validation
- With Dropout only: 97.8% train, 97.9% validation
- With L2 + Dropout: 97.5% train, 98.2% validation ← Best generalization
- With Early Stopping: Stops at optimal epoch, highest validation accuracy

## Troubleshooting

**High training loss**: Learning rate too low, use schedule or increase
**Oscillating loss**: Learning rate too high, momentum needed
**Overfitting**: Add regularization (L2, Dropout), enable early stopping
**Slow convergence**: Use Adam optimizer, learning rate schedule
**Poor hyperparameter search**: Increase max_trials, use Bayesian optimization

## References

- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Keras Tuner Guide: https://keras.io/keras_tuner/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/

---

**Created**: August 2026
**Project**: Deep Learning Optimization & Regularization
**Tasks**: 10 complete implementations