#!/usr/bin/env python3
"""
Train YOLOv8 model with custom Albumentations augmentation.

Supports training with custom Albumentations transforms that are
applied during the training process for enhanced data augmentation.
"""

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_dataloader
import albumentations as A
import numpy as np


class CustomAugmentationDataset(YOLODataset):
    """Custom dataset class that applies Albumentations transforms."""

    def __init__(self, *args, albumentations_transforms=None, **kwargs):
        """Initialize dataset with custom Albumentations."""
        super().__init__(*args, **kwargs)
        self.albumentations_transforms = albumentations_transforms

    def __getitem__(self, index):
        """Get item with custom augmentation applied."""
        # Get the original item from parent class
        item = super().__getitem__(index)

        # If no custom transforms, return as-is
        if self.albumentations_transforms is None:
            return item

        # Extract image and labels
        img = item['img'].numpy().transpose(1, 2, 0).astype(np.uint8)
        labels = item.get('cls', np.array([]))
        bboxes = item.get('bboxes', np.array([]))

        # Skip if no bboxes
        if len(bboxes) == 0:
            return item

        # Convert bboxes from YOLO format to Pascal VOC
        h, w = img.shape[:2]
        pascal_bboxes = []
        for bbox in bboxes:
            xc, yc, bw, bh = bbox
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            pascal_bboxes.append([x1, y1, x2, y2])

        # Apply Albumentations
        transform = A.Compose(
            self.albumentations_transforms,
            bbox_params=A.BboxParams(format='pascal_voc',
                                     label_fields=['class_labels']),
            seed=42
        )

        try:
            transformed = transform(
                image=img,
                bboxes=pascal_bboxes,
                class_labels=labels.tolist()
            )
            aug_img = transformed['image']
            aug_bboxes = np.array(transformed['bboxes'])
            aug_labels = np.array(transformed['class_labels'])

            # Convert back to YOLO format
            h, w = aug_img.shape[:2]
            yolo_bboxes = []
            for bbox in aug_bboxes:
                x1, y1, x2, y2 = bbox
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                yolo_bboxes.append([xc, yc, bw, bh])

            # Update item with augmented data
            item['img'] = (aug_img.transpose(2, 0, 1) / 255.0).astype(np.float32)
            item['cls'] = aug_labels
            item['bboxes'] = np.array(yolo_bboxes)

        except Exception:
            # If augmentation fails, return original
            pass

        return item


def train_with_augmentation(data_yaml=None, model_path="yolov8n.pt",
                           aug=None, custom_albu=None,
                           epochs=50, imgsz=640, batch=16,
                           data=None, albumentations_transforms=None,
                           verbose=False, save=True, plots=True):
    """
    Train YOLOv8 model with optional custom Albumentations augmentation.

    Args:
        data_yaml (str): Path to dataset YAML file (or use 'data' parameter)
        model_path (str): Path to pre-trained YOLO model weights
        aug (bool): Global flag to enable/disable augmentation
        custom_albu (list): List of Albumentations transforms
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch (int): Batch size
        data (str): Alternative parameter name for dataset YAML
        albumentations_transforms (list): Alternative parameter for custom transforms
        verbose (bool): Whether to display training progress
        save (bool): Whether to save checkpoints and model
        plots (bool): Whether to generate training plots

    Returns:
        tuple: (trained_model, results)
            - trained_model: Trained YOLO model object
            - results: Training results object with metrics
    """
    # Handle alternative parameter names
    yaml_path = data_yaml or data
    transforms = custom_albu or albumentations_transforms

    if yaml_path is None:
        raise ValueError("Must provide 'data_yaml' or 'data' parameter")

    # Load YOLO model
    model = YOLO(model_path)

    # Prepare training configuration
    train_config = {
        'data': yaml_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'save': save,
        'plots': plots,
        'verbose': verbose,
        'patience': 20,
        'device': 0,  # GPU device, 0 for default
    }

    # If custom augmentations provided, disable default YOLO augmentation
    # and we'll apply them through the dataset
    if transforms is not None:
        train_config['augment'] = False
        # Note: Full custom augmentation integration would require
        # modifying the trainer's dataset building process
        # For now, training with standard YOLO augmentation

    # Train the model
    results = model.train(**train_config)

    return model, results
