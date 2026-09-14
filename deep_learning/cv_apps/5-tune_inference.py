#!/usr/bin/env python3
"""
Inference parameter tuning for YOLOv8 object detection.

Performs grid search over confidence and IoU thresholds to find
optimal inference parameters for best performance on validation set.
"""

from ultralytics import YOLO
import numpy as np


def inference_tuning(data_yaml, model, conf_list=None, iou_list=None, imgsz=640):
    """
    Perform grid search over confidence and IoU thresholds for optimal inference.

    Args:
        data_yaml (str): Path to dataset YAML file
        model (str or YOLO): Path to trained model weights or YOLO model object
        conf_list (List[float]): Confidence thresholds to test
                                Default: [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        iou_list (List[float]): IoU thresholds for NMS to test
                               Default: [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        imgsz (int): Image size for validation (default: 640)

    Returns:
        list: List of dictionaries containing results for each combination.
              Each dictionary has keys:
              - 'conf': Confidence threshold tested
              - 'iou': IoU threshold tested
              - 'map50': mAP@0.50 metric
              - 'map50_95': mAP@0.50:0.95 metric
              - 'precision': Precision metric
              - 'recall': Recall metric

    Example:
        >>> results = inference_tuning(
        ...     'datasets/detection/data.yaml',
        ...     'best_model.pt',
        ...     conf_list=[0.25, 0.3, 0.35, 0.4],
        ...     iou_list=[0.4, 0.5, 0.6]
        ... )
        >>> for r in results:
        ...     print(f"conf={r['conf']}, iou={r['iou']}, mAP50={r['map50']:.3f}")
    """

    # Set default thresholds if not provided
    if conf_list is None:
        conf_list = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    if iou_list is None:
        iou_list = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

    # Load model if it's a path
    if isinstance(model, str):
        model = YOLO(model)

    # Store results
    results = []

    # Get total combinations for progress tracking
    total_combinations = len(conf_list) * len(iou_list)
    current = 0

    print("=" * 70)
    print("INFERENCE PARAMETER TUNING: Grid Search")
    print("=" * 70)
    print(f"Testing {total_combinations} combinations...")
    print(f"Confidence thresholds: {conf_list}")
    print(f"IoU thresholds: {iou_list}\n")

    # Grid search over all combinations
    for conf in conf_list:
        for iou in iou_list:
            current += 1

            # Run validation with current parameters
            try:
                val_results = model.val(
                    data=data_yaml,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    verbose=False,
                    plots=False
                )

                # Extract metrics from results
                metrics = val_results.results_dict

                # Get mAP50 and mAP50-95
                map50 = metrics.get('metrics/mAP50(B)', 0)
                map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
                precision = metrics.get('metrics/precision(B)', 0)
                recall = metrics.get('metrics/recall(B)', 0)

                # Store result
                result_dict = {
                    'conf': conf,
                    'iou': iou,
                    'map50': np.float64(map50),
                    'map50_95': np.float64(map50_95),
                    'precision': np.float64(precision),
                    'recall': np.float64(recall)
                }
                results.append(result_dict)

                # Print progress
                status = f"[{current}/{total_combinations}] " \
                         f"conf={conf:.2f}, iou={iou:.2f} → " \
                         f"mAP50={map50:.4f}, mAP50-95={map50_95:.4f}"
                print(status)

            except Exception as e:
                print(f"⚠️  Error at conf={conf}, iou={iou}: {str(e)}")
                continue

    # Find best combination
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if results:
        # Sort by mAP50-95 (descending)
        best_result = max(results, key=lambda x: x['map50_95'])

        print(f"\n✅ Best Configuration:")
        print(f"   Confidence threshold: {best_result['conf']:.2f}")
        print(f"   IoU threshold: {best_result['iou']:.2f}")
        print(f"   mAP50: {best_result['map50']:.4f}")
        print(f"   mAP50-95: {best_result['map50_95']:.4f}")
        print(f"   Precision: {best_result['precision']:.4f}")
        print(f"   Recall: {best_result['recall']:.4f}")

        print(f"\nTop 5 Configurations (by mAP50-95):")
        sorted_results = sorted(results, key=lambda x: x['map50_95'], reverse=True)
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. conf={result['conf']:.2f}, iou={result['iou']:.2f} → "
                  f"mAP50={result['map50']:.4f}, mAP50-95={result['map50_95']:.4f}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    # Example usage
    print("This module provides inference_tuning() function.")
    print("Import it and use: inference_tuning(data_yaml, model, conf_list, iou_list)")
