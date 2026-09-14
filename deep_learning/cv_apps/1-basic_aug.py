#!/usr/bin/env python3
import albumentations as A
import numpy as np

def basic_aug(image, bboxes, labels):
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
            fill_value=0
        )
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
       seed=42)

    transformed = transform(image=image, bboxes=bboxes, class_labels=labels)

    augmented_image = transformed['image']
    augmented_bboxes = np.array(transformed['bboxes'])
    augmented_labels = transformed['class_labels']

    return augmented_image, augmented_bboxes, augmented_labels
