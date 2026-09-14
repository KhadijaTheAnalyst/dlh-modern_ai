#!/usr/bin/env python3
"""Inference parameter tuning for YOLOv8 object detection.

Performs grid search over confidence and IoU thresholds to find
optimal inference parameters for best performance on validation set.
"""
from ultralytics import YOLO


def inference_tuning(data_yaml, model, conf_list=None, iou_list=None,
                     imgsz=640):
    """Perform grid search over confidence and IoU thresholds.

    Args:
        data_yaml (str): Path to dataset YAML file.
        model (str or YOLO): Path to trained model or YOLO model object.
        conf_list (list): Confidence thresholds to test.
        iou_list (list): IoU thresholds for NMS to test.
        imgsz (int): Image size for validation.

    Returns:
        list: Results for each confidence/IoU combination.
    """
    if conf_list is None:
        conf_list = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    if iou_list is None:
        iou_list = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

    if isinstance(model, str):
        model = YOLO(model)

    results = []
    total_combinations = len(conf_list) * len(iou_list)
    current = 0

    print("=" * 70)
    print("INFERENCE PARAMETER TUNING: Grid Search")
    print("=" * 70)
    print(f"Testing {total_combinations} combinations...")
    print(f"Confidence thresholds: {conf_list}")
    print(f"IoU thresholds: {iou_list}\n")

    for conf in conf_list:
        for iou in iou_list:
            current += 1

            try:
                val_results = model.val(
                    data=data_yaml,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    verbose=False,
                    plots=False
                )

                metrics = val_results.results_dict

                map50 = metrics.get('metrics/mAP50(B)', 0)
                map50_95 = metrics.get('metrics/mAP50-95(B)', 0)
                precision = metrics.get('metrics/precision(B)', 0)
                recall = metrics.get('metrics/recall(B)', 0)

                result_dict = {
                    'conf': conf,
                    'iou': iou,
                    'map50': float(map50),
                    'map50_95': float(map50_95),
                    'precision': float(precision),
                    'recall': float(recall)
                }
                results.append(result_dict)

                status = f"[{current}/{total_combinations}] " \
                         f"conf={conf:.2f}, iou={iou:.2f} → " \
                         f"mAP50={map50:.4f}, mAP50-95={map50_95:.4f}"
                print(status)

            except Exception as e:
                print(f"Error at conf={conf}, iou={iou}: {str(e)}")
                continue

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if results:
        best_result = max(results, key=lambda x: x['map50_95'])

        print(f"\nBest Configuration:")
        conf_val = best_result['conf']
        iou_val = best_result['iou']
        map50_val = best_result['map50']
        print(f"   Confidence threshold: {conf_val:.2f}")
        print(f"   IoU threshold: {iou_val:.2f}")
        print(f"   mAP50: {map50_val:.4f}")
        map50_95_val = best_result['map50_95']
        prec_val = best_result['precision']
        rec_val = best_result['recall']
        print(f"   mAP50-95: {map50_95_val:.4f}")
        print(f"   Precision: {prec_val:.4f}")
        print(f"   Recall: {rec_val:.4f}")

        print(f"\nTop 5 Configurations (by mAP50-95):")
        sorted_results = sorted(results, key=lambda x: x['map50_95'],
                                reverse=True)
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. conf={result['conf']:.2f}, "
                  f"iou={result['iou']:.2f} → "
                  f"mAP50={result['map50']:.4f}, "
                  f"mAP50-95={result['map50_95']:.4f}")

    print("\n" + "=" * 70)

    return results
