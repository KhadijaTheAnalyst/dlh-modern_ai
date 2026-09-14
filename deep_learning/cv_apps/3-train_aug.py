#!/usr/bin/env python3
"""
Basic data augmentation for object detection using Albumentations.

Applies YOLO-compatible transformations including:
- Random horizontal flipping
- Brightness/contrast augmentation
- Affine transformations (translation, scaling, rotation)
"""

import albumentations as A
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)


def basic_aug(image, bboxes, labels):
    """
    Apply basic data augmentation to image and bounding boxes.

    Args:
        image (np.ndarray): Input image in RGB format
        bboxes (List[List[int]]): Bounding boxes in Pascal VOC format
                                  [[x1, y1, x2, y2], ...]
        labels (List[int]): Class labels corresponding to each bounding box

    Returns:
        tuple: (augmented_image, augmented_bboxes, augmented_labels)
            - augmented_image (np.ndarray): Augmented image
            - augmented_bboxes (np.ndarray): Augmented bounding boxes
            - augmented_labels (List[int]): Class labels (unchanged)
    """
    # Define augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.2
        ),
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.9, 1.1),
            rotate=(-30, 0),
            p=0.5,
            border_mode=0,  # 0 = constant border with black (cval=0)
            cval=0
        )
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # Apply augmentation
    transformed = transform(image=image, bboxes=bboxes, class_labels=labels)

    augmented_image = transformed['image']
    augmented_bboxes = np.array(transformed['bboxes'])
    augmented_labels = transformed['class_labels']

    return augmented_image, augmented_bboxes, augmented_labels
