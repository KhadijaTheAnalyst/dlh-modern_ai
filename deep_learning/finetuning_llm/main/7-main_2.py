#!/usr/bin/env python3

inference_mode = __import__('7-inference_mode').inference_mode

model_path = "./hbtn_emotion_classifier_best_model"

texts = [
    "I feel really so disappointed.",
    "I absolutely love spending time with my family.",
    "This is so frustrating! I'm really angry!",
    "I love this new song!",
    "I'm Petrified about what might happen."
]

emotion_classifier = inference_mode(
    model_path,
    top_k=3
)

predictions = emotion_classifier(texts)

for text, pred in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {pred}")
    print("-" * 40)
