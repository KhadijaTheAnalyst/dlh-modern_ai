#!/usr/bin/env python3
"""
Data preparation script for Pascal VOC 2012 detection dataset.

Converts Pascal VOC 2012 dataset to YOLOv8 format.
Filters to keep only 'person', 'car', and 'bicycle' classes.
Organizes dataset into train/val splits with images and labels.
"""

import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml


def get_classes():
    """Return the list of classes to keep."""
    return ["person", "car", "bicycle"]


def parse_voc_annotation(xml_file):
    """
    Parse a Pascal VOC XML annotation file.

    Args:
        xml_file: Path to XML annotation file

    Returns:
        dict: Contains image filename, width, height, and objects list
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image information
    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    classes_list = get_classes()

    # Extract objects
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes_list:
            continue

        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        objects.append({
            "class": class_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects
    }


def convert_to_yolo_format(obj, width, height, class_to_id):
    """
    Convert bounding box to YOLO format.

    YOLO format: class_id center_x center_y width height
    (all normalized to 0-1)

    Args:
        obj: Object dict with xmin, ymin, xmax, ymax
        width: Image width
        height: Image height
        class_to_id: Dict mapping class name to class ID

    Returns:
        str: YOLO format string for the object
    """
    class_id = class_to_id[obj["class"]]

    # Convert to center coordinates
    center_x = (obj["xmin"] + obj["xmax"]) / 2.0 / width
    center_y = (obj["ymin"] + obj["ymax"]) / 2.0 / height
    bbox_width = (obj["xmax"] - obj["xmin"]) / width
    bbox_height = (obj["ymax"] - obj["ymin"]) / height

    # Clip to [0, 1]
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    bbox_width = max(0, min(1, bbox_width))
    bbox_height = max(0, min(1, bbox_height))

    return f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"


def main():
    """Main function to prepare the dataset."""
    # Define paths
    voc_root = Path("C:/dlh-modern_ai/deep_learning/cv_apps/datasets/segmentation/VOCdevkit/VOC2012")
    detection_root = Path("C:/dlh-modern_ai/deep_learning/cv_apps/datasets/detection")

    # Verify VOC dataset exists
    if not voc_root.exists():
        print(f"Error: VOC dataset not found at {voc_root}")
        return

    # Create detection directory structure
    (detection_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (detection_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (detection_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (detection_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print("✅ Created directory structure")

    # Class mapping
    classes_list = get_classes()
    class_to_id = {cls: idx for idx, cls in enumerate(classes_list)}

    print(f"📊 Classes: {classes_list}")
    print(f"📝 Class mapping: {class_to_id}")

    # Read train/val splits
    imageset_main = voc_root / "ImageSets" / "Main"
    train_file = imageset_main / "train.txt"
    val_file = imageset_main / "val.txt"

    train_images = set()
    val_images = set()

    if train_file.exists():
        with open(train_file) as f:
            train_images = {line.strip().split()[0] for line in f}

    if val_file.exists():
        with open(val_file) as f:
            val_images = {line.strip().split()[0] for line in f}

    print(f"📈 Train images: {len(train_images)}, Val images: {len(val_images)}")

    # Process annotations
    annotations_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"

    train_count = 0
    val_count = 0
    skipped_count = 0

    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found at {annotations_dir}")
        return

    for xml_file in sorted(annotations_dir.glob("*.xml")):
        image_id = xml_file.stem

        # Parse annotation
        try:
            ann_data = parse_voc_annotation(xml_file)
        except Exception as e:
            print(f"⚠️  Error parsing {xml_file}: {e}")
            skipped_count += 1
            continue

        # Skip if no relevant objects
        if not ann_data["objects"]:
            skipped_count += 1
            continue

        # Find corresponding image
        image_file = images_dir / ann_data["filename"]
        if not image_file.exists():
            print(f"⚠️  Image not found: {image_file}")
            skipped_count += 1
            continue

        # Determine split
        if image_id in train_images:
            split = "train"
            train_count += 1
        elif image_id in val_images:
            split = "val"
            val_count += 1
        else:
            continue

        # Copy image
        dest_image = detection_root / "images" / split / image_file.name
        shutil.copy2(image_file, dest_image)

        # Create label file
        label_file = detection_root / "labels" / split / f"{image_id}.txt"
        with open(label_file, "w") as f:
            for obj in ann_data["objects"]:
                yolo_line = convert_to_yolo_format(
                    obj,
                    ann_data["width"],
                    ann_data["height"],
                    class_to_id
                )
                f.write(yolo_line + "\n")

    print(f"✅ Processed images:")
    print(f"   - Train: {train_count}")
    print(f"   - Val: {val_count}")
    print(f"   - Skipped: {skipped_count}")

    # Create data.yaml
    data_yaml = {
        "path": str(detection_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes_list),
        "names": classes_list
    }

    yaml_file = detection_root / "data.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✅ Created data.yaml at {yaml_file}")
    print("\n✨ Dataset preparation complete!")


if __name__ == "__main__":
    main()
