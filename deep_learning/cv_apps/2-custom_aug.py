#!/usr/bin/env python3
"""
Custom Albumentations data augmentation for object detection.

Applies Albumentations-exclusive transformations including:
- Motion blur
- Elastic distortion OR optical distortion (one of these applied)
"""

import albumentations as A
import numpy as np


def custom_aug(image, bboxes, labels):
    """
    Apply custom Albumentations-exclusive data augmentation.

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
        A.MotionBlur(blur_limit=5, p=0.9),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.OpticalDistortion(distort_limit=0.05, p=0.2)
        ], p=0.9)
    ], bbox_params=A.BboxParams(format='pascal_voc',
                                label_fields=['class_labels']),
                                seed=42)

    # Apply augmentation
    transformed = transform(image=image, bboxes=bboxes, class_labels=labels)

    augmented_image = transformed['image']
    augmented_bboxes = np.array(transformed['bboxes'])
    augmented_labels = transformed['class_labels']

    return augmented_image, augmented_bboxes, augmented_labels
