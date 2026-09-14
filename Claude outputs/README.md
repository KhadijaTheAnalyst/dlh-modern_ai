# Object Detection & Image Segmentation with YOLOv8

A comprehensive computer vision project implementing object detection and image segmentation using YOLOv8, covering data preparation, augmentation, training, hyperparameter tuning, and inference optimization.

## 📚 Learning Objectives

After completing this project, you will be able to explain:

- **Object Detection Fundamentals**
  - What is object detection and how it differs from image classification
  - Single-shot detectors and the YOLO algorithm
  - Intersection over Union (IoU) and how to calculate it
  - Non-maximum suppression (NMS)
  - Anchor boxes and their role in detection

- **Performance Metrics**
  - Mean Average Precision (mAP) and mAP50-95
  - Bounding box mAP vs Segmentation mAP
  - Precision, Recall, and F1-score

- **Image Segmentation**
  - Semantic vs instance segmentation
  - Segmentation masks and their representation
  - Polygon-based annotation
  - Mask IoU calculation
  - NMS application with overlapping masks

- **Practical Skills**
  - Dataset preparation and organization in YOLO format
  - Data augmentation with Albumentations
  - Model training with custom augmentation pipelines
  - Hyperparameter tuning and optimization
  - Inference parameter tuning for production deployment

## 📋 Project Structure

```
dlh-modern_ai/
└── deep_learning/
    └── cv_apps/
        ├── datasets/
        │   ├── detection/
        │   │   ├── images/
        │   │   │   ├── train/
        │   │   │   └── val/
        │   │   ├── labels/
        │   │   │   ├── train/
        │   │   │   └── val/
        │   │   └── data.yaml
        │   └── segmentation/
        │       ├── VOCdevkit/
        │       ├── caltech-101/
        │       └── VOCtrainval_11-May-2012.tar
        ├── 0-prep_data.py
        ├── 1-basic_aug.py
        ├── 1-main.py
        ├── 2-custom_aug.py
        ├── 2-main.py
        ├── 3-train_aug.py
        ├── 3-main.py
        ├── 4-tune_train.py
        ├── 5-tune_inference.py
        ├── 5-main.py
        ├── best_model.pt (output)
        └── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (optional, for GPU acceleration)
- 20GB+ disk space (for datasets and models)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KhadijaTheAnalyst/dlh-modern_ai.git
   cd dlh-modern_ai/deep_learning/cv_apps
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib opencv-python torch albumentations ultralytics
   ```

3. **Download datasets:**
   - Pascal VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
   - Extract to `datasets/segmentation/`

## 📝 Tasks Overview

### Task 0: Data Preparation
**File:** `0-prep_data.py`

Converts Pascal VOC 2012 dataset to YOLOv8 format.

**What it does:**
- Reads Pascal VOC XML annotations
- Filters to keep only: person, car, bicycle classes
- Converts bounding boxes to YOLO format
- Organizes images and labels into train/val splits
- Creates `data.yaml` configuration file

**Usage:**
```bash
python 0-prep_data.py
```

**Output:**
- `datasets/detection/images/train/` - Training images
- `datasets/detection/images/val/` - Validation images
- `datasets/detection/labels/train/` - Training labels (YOLO format)
- `datasets/detection/labels/val/` - Validation labels
- `datasets/detection/data.yaml` - Dataset configuration

---

### Task 1: Basic Augmentation
**File:** `1-basic_aug.py`

Implements basic data augmentation using Albumentations.

**Transformations applied:**
- Horizontal flip (50% probability)
- Brightness/Contrast adjustment (20% probability)
- Affine transformation (50% probability):
  - Translation: ±10%
  - Scaling: ±10%
  - Rotation: -30° to 0°

**Function signature:**
```python
def basic_aug(image, bboxes, labels):
    """Apply basic augmentation to image and bounding boxes."""
    return augmented_image, augmented_bboxes, augmented_labels
```

**Usage:**
```bash
python 1-main.py
```

**Features:**
- ✅ Reproduces with `seed=42`
- ✅ Handles bbox transformations automatically
- ✅ YOLO-compatible output format

---

### Task 2: Custom Albumentations
**File:** `2-custom_aug.py`

Implements advanced Albumentations-exclusive transformations.

**Transformations applied:**
- Motion blur (90% probability, blur_limit=5)
- OneOf distortion (90% probability):
  - Elastic transform (alpha=1, sigma=50)
  - OR Optical distortion (distort_limit=0.05)

**Function signature:**
```python
def custom_aug(image, bboxes, labels):
    """Apply custom Albumentations augmentation."""
    return augmented_image, augmented_bboxes, augmented_labels
```

**Usage:**
```bash
python 2-main.py
```

**Key difference from Task 1:**
- Uses `A.OneOf()` to select one of multiple transforms
- More realistic distortions (motion blur, elastic/optical effects)
- Better for simulating real-world variations

---

### Task 3: Training with Augmentation
**File:** `3-train_aug.py`

Trains YOLOv8 model with custom augmentation pipeline.

**Function signature:**
```python
def train_with_augmentation(
    data_yaml,
    model_path="yolov8n.pt",
    albumentations_transforms=None,
    epochs=50,
    imgsz=640,
    batch=16,
    verbose=False
):
    """Train YOLO model with custom Albumentations."""
    return trained_model, results
```

**Features:**
- Supports custom Albumentations pipeline
- Configurable model size (yolov8n, yolov8s, yolov8m, etc.)
- Automatic checkpoint saving
- Real-time training metrics

**Usage:**
```python
from ultralytics import YOLO
import albumentations as A

custom_transforms = [
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(std_range=(0.2, 0.44), p=0.3),
]

model, results = train_with_augmentation(
    data="datasets/detection/data.yaml",
    model_path="yolov8n.pt",
    albumentations_transforms=custom_transforms,
    epochs=50,
    batch=8
)
```

---

### Task 4: Hyperparameter Tuning & Final Training
**File:** `4-tune_train.py`

Two-phase training pipeline with automatic hyperparameter optimization.

**Phase 1: Lightweight Hyperparameter Search**
- Tests 15-20 configurations
- 10 epochs per trial (for efficiency)
- Explores: learning rates, augmentation, geometric transforms, loss weights
- Saves best hyperparameters and model

**Phase 2: Final Training to Convergence**
- Loads best model from Phase 1
- Trains for ~150 total epochs
- Applies optimal hyperparameters
- Uses early stopping (patience=20)

**Function signature:**
```python
def tune_hyperparameters():
    """Perform two-phase hyperparameter tuning and training."""
    # Returns: trained model, saves best_model.pt
```

**Usage:**
```bash
python 4-tune_train.py
```

**Performance Targets:**
- mAP50 ≥ 65%
- mAP50-95 ≥ 46%

**Output:**
- `runs/detect/tune/` - Phase 1 (tuning) results
- `runs/detect/train/` - Phase 2 (final training) results
- `best_model.pt` - Final trained model

**Tips:**
- Use GPU for faster training: 4-6 hours total
- Monitor `runs/detect/train/` for loss curves
- If targets not met: increase epochs, use larger model, or add more data

---

### Task 5: Inference Parameter Tuning
**File:** `5-tune_inference.py`

Grid search for optimal confidence and IoU thresholds.

**Function signature:**
```python
def inference_tuning(
    data_yaml,
    model,
    conf_list=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    iou_list=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    imgsz=640
):
    """Perform grid search over inference parameters."""
    return results_list  # List of dicts with metrics
```

**What it does:**
- Tests all combinations of confidence × IoU thresholds
- Runs validation for each combination
- Collects mAP50, mAP50-95, Precision, Recall
- Identifies best configuration automatically
- Shows top 5 configurations

**Usage:**
```python
results = inference_tuning(
    data_yaml="datasets/detection/data.yaml",
    model="best_model.pt",
    conf_list=[0.1, 0.15, 0.2, 0.25, 0.3],
    iou_list=[0.3, 0.4, 0.5, 0.6]
)

for result in results:
    print(f"conf={result['conf']}, iou={result['iou']}, "
          f"mAP50={result['map50']:.3f}")
```

**Output:**
```
Best Configuration:
   Confidence: 0.15
   IoU: 0.40
   mAP50: 0.624
   mAP50-95: 0.429
```

**Use Cases:**
- Production deployment optimization
- Balancing precision vs recall
- Meeting specific performance requirements

---

## 📊 Dataset Information

### Pascal VOC 2012 (Detection)
- **Total images:** 17,125
- **Classes:** 20 (filtered to 3: person, car, bicycle)
- **Format:** YOLO (normalized center coordinates)
- **Split:** Train/Val (approximately 11,540 train, 5,585 val)

### Caltech 101 (Segmentation)
- **Total images:** 9,146
- **Classes:** 101 object categories
- **Location:** `datasets/segmentation/caltech-101/`

## 🎯 Expected Results

After completing all tasks:

| Metric | Target | Description |
|--------|--------|-------------|
| mAP50 | ≥ 65% | Mean Average Precision at IoU=0.50 |
| mAP50-95 | ≥ 46% | Mean Average Precision at IoU=0.50:0.95 |
| Precision | > 0.7 | Ratio of correct predictions to total predictions |
| Recall | > 0.7 | Ratio of correct predictions to total ground truth |

## 📦 Dependencies

```
numpy==2.0.2
matplotlib==3.10.0
opencv-python==4.12.0.88
torch==2.8.0
albumentations==2.0.8
ultralytics==8.4.7
```

**Install all:**
```bash
pip install -r requirements.txt
```

## 🔧 Configuration

### YOLOv8 Model Sizes

| Model | Parameters | Speed (GPU) | Use Case |
|-------|-----------|------------|----------|
| yolov8n | 3.2M | Fast | Mobile, Real-time |
| yolov8s | 11.2M | Balanced | Default |
| yolov8m | 25.9M | Accurate | Production |
| yolov8l | 43.7M | Very accurate | High-performance |

### Recommended Hyperparameters

```python
# Training
batch_size = 8-16
epochs = 100-150
learning_rate = 0.001 (starts at lr0)
warmup_epochs = 3
patience = 20  # Early stopping

# Augmentation
mosaic = 1.0
mixup = 0.1
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
degrees = 30
translate = 0.1
scale = 0.1
```

## 📈 Training & Evaluation Workflow

```
1. Task 0: Data Prep
   └─→ Organized detection dataset

2. Task 1-2: Augmentation Testing
   └─→ Understand augmentation effects

3. Task 3: Basic Training
   └─→ Validate training pipeline

4. Task 4: Hyperparameter Tuning
   └─→ Discover optimal settings
   └─→ Train final model

5. Task 5: Inference Optimization
   └─→ Find best thresholds
   └─→ Deploy to production
```

## 💡 Tips & Best Practices

### Data Augmentation
- Start with basic transforms (flip, brightness, rotation)
- Gradually add advanced transforms (elastic, optical distortion)
- Monitor validation metrics to avoid overfitting

### Training
- Use GPU for 4-10x speedup
- Monitor loss curves in `runs/detect/train/`
- Enable early stopping to prevent overfitting
- Save checkpoints regularly

### Hyperparameter Tuning
- Phase 1: Use short epochs (10) for quick exploration
- Phase 2: Use full epochs (150) for convergence
- Test 15-20 configurations for good coverage
- Focus on: learning rate, augmentation, momentum

### Inference
- Lower confidence threshold → Higher recall, lower precision
- Higher IoU threshold → Stricter NMS, fewer detections
- Balance based on use case requirements

## 🚨 Troubleshooting

### GPU Issues
```bash
# Check if GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU-only training
device = 'cpu'
```

### Out of Memory
```python
# Reduce batch size
batch = 4  # or 8

# Reduce image size
imgsz = 416  # or 512
```

### Poor Performance
1. Check dataset is correctly formatted
2. Verify labels are in YOLO format
3. Increase training epochs
4. Use larger model (yolov8m instead of yolov8n)
5. Collect more training data

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [Albumentations Documentation](https://albumentations.ai)
- [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [YOLO: You Only Look Once](https://pjreddie.com/darknet/yolo/)
- [mAP Explanation](https://github.com/rafaelpadilla/Object-Detection-Metrics)

## 👤 Author

Khadi (Khadija) - ML Engineering Student at DLH AI Academy

## 📄 License

This project is part of the DLH AI Academy curriculum.

## 🎓 Skills Gained

After completing this project:
- ✅ Object detection algorithms (YOLO)
- ✅ Data augmentation techniques
- ✅ Model training and optimization
- ✅ Hyperparameter tuning
- ✅ Performance evaluation and metrics
- ✅ Production model deployment
- ✅ Image segmentation fundamentals
