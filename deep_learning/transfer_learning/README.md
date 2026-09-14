# Transfer Learning with Pretrained CNNs

This project walks through the core building blocks of **transfer learning**
for image classification using Keras/TensorFlow — starting from a frozen,
pretrained feature extractor and ending with a fully fine-tuned classifier
trained on the Caltech101 dataset.

Rather than training a CNN from scratch (which needs huge amounts of data
and compute), each task reuses a CNN (MobileNetV2 / EfficientNetB0)
pretrained on ImageNet, and progressively adapts it to a new task with far
less data.

## Repository

- **GitHub repo:** `dlh-modern_ai`
- **Directory:** `deep_learning/transfer_learning`

## Project Structure

| File | Task | Description |
|---|---|---|
| `0-frozen_extractor.py` | 0 | Loads a frozen, pretrained CNN (MobileNetV2) as a feature extractor |
| `1-classification_head.py` | 1 | Attaches a trainable classification head to a feature extractor |
| `2-unfreeze_top.py` | 2 | Unfreezes the top N layers of a backbone for fine-tuning |
| `3-data_augmentation.py` | 3 | Builds a reproducible image augmentation pipeline |
| `4-transfer_101.py` | 4 | Full end-to-end pipeline: trains and saves a Caltech101 classifier |

## Concepts Covered

### 1. Frozen Feature Extraction
A pretrained CNN (e.g. MobileNetV2) already knows how to recognize general
visual patterns — edges, textures, shapes — from training on 1.4M ImageNet
images. We strip off its classification head (`include_top=False`), freeze
its weights (`trainable = False`), and use it purely as a fixed
`image → feature vector` function via `GlobalAveragePooling2D`.

### 2. Classification Head
A small trainable head (`Dense(128, relu)` → `Dense(num_classes, softmax)`)
is added on top of the frozen extractor. Only this head is trained
initially — it learns to map the general-purpose features to your specific
task's classes.

### 3. Fine-Tuning (Unfreezing Layers)
Once the head has learned something reasonable, the **last N layers** of the
backbone are unfrozen and trained further at a **much lower learning rate**.
Early layers (generic patterns) stay frozen; later layers (higher-level,
task-relevant patterns) adapt slightly to the new dataset. This typically
squeezes out extra accuracy over head-only training.

### 4. Data Augmentation
Random flips, rotations, zooms, and contrast shifts are applied to training
images (never validation/test images) to reduce overfitting, especially
important when the target dataset is much smaller than ImageNet. All
augmentation layers are seeded (`seed=42`) for reproducibility.

### 5. Full Pipeline — Caltech101 Classifier
Combines all of the above into a two-phase training pipeline:

1. **Phase 1:** Train the classification head with the backbone frozen.
2. **Phase 2:** Unfreeze the top layers of the backbone and fine-tune at a
   reduced learning rate.

Includes `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`
callbacks, and saves the best model to `caltech101_model.h5`.

**Target:** ≥ 85% validation accuracy across 102 classes (101 Caltech101
object classes + background).

## Requirements

```
tensorflow>=2.x
tensorflow-datasets
```

Install with:

```bash
pip install tensorflow tensorflow-datasets
```

## Usage

### Individual tasks (0–3)

Each task can be run against its corresponding `*-main.py` test script, e.g.:

```bash
./0-main.py
./1-main.py
./2-main.py
./3-main.py
```

### Full training pipeline (Task 4)

```bash
./4-transfer_101.py
```

This will:
1. Download the Caltech101 dataset via `tensorflow_datasets` (first run only).
2. Train the classification head (Phase 1).
3. Fine-tune the top backbone layers (Phase 2).
4. Save the best model to `caltech101_model.h5`.

### Loading the saved model

```python
from tensorflow import keras

model = keras.models.load_model("caltech101_model.h5")
```

## Key Design Notes

- **`preprocess_input`**: Each pretrained backbone expects inputs normalized
  in a specific way (matching how it was originally trained). Always use the
  matching `keras.applications.<model>.preprocess_input`, not a generic
  `/255.0` scale.
- **`training=False` on the base model call**: Keeps BatchNormalization
  layers using their frozen running statistics rather than batch statistics,
  even if the outer model is later set to trainable.
- **BatchNorm layers stay frozen during fine-tuning**: Even among "unfrozen"
  top layers, BatchNorm layers are kept frozen to avoid destabilizing
  pretrained statistics when fine-tuning on a smaller dataset.
- **Low learning rate for fine-tuning**: Large updates to pretrained weights
  risk "catastrophic forgetting." Fine-tuning uses a learning rate roughly
  10–100x smaller than head-only training.