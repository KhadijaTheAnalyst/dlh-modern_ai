#!/usr/bin/env python3
import albumentations as A
import logging
train_with_augmentation = __import__('3-train_aug').train_with_augmentation
model = "yolov8n.pt"
data_yaml = "datasets/detection/data.yaml"
logging.getLogger('ultralytics').setLevel(logging.WARNING)
custom_albu = [
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(std_range=(0.2, 0.44), p=0.3),
]

# train model with custom albumentations aug
model, results = train_with_augmentation(
    data="datasets/detection/data.yaml",
    model_path="yolov8n.pt",
    albumentations_transforms=custom_albu,
    epochs=3,
    imgsz=640,
    batch=8,
    verbose=False
)
metrics = results.results_dict
print(f"Best mAP@0.5 = {metrics['metrics/mAP50(B)']:.3f}")
print(f"Precision = {metrics['metrics/precision(B)']:.2f}")
print(f"Recall = {metrics['metrics/recall(B)']:.2f}\n")
print(model.info())
