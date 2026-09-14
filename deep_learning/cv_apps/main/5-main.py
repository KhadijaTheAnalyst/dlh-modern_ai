#!/usr/bin/env python3
inference_tuning = __import__('5-tune_inference').inference_tuning

model = 'trained.pt'
data_yaml = "datasets/detection/data.yaml"
conf_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
iou_list = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
inf_results = inference_tuning(data_yaml, model, conf_list=conf_list, iou_list=iou_list)
for r in inf_results:
    print(r)
