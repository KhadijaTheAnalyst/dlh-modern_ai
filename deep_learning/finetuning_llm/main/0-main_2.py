#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
import numpy as np

dataset = load_emotion_dataset()

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

for split_name in dataset.keys():
    print(f"\n{split_name.capitalize()} Set Label Distribution:")
    split_labels = np.array(dataset[split_name]['label'])
    for idx, name in enumerate(label_names):
        count = np.sum(split_labels == idx)
        percentage = 100 * count / len(split_labels)
        print(f"  {name:<10}: {count} samples ({percentage:.2f}%)")

    print(f"\nSample Examples from {split_name.capitalize()} Set:")
    for i in range(min(5, len(dataset[split_name]))):
        text = dataset[split_name]['text'][i]
        label_id = dataset[split_name]['label'][i]
        emotion = label_names[label_id]
        print(f"{i+1}. [{emotion.upper()}] {text}")
