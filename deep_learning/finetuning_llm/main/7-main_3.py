#!/usr/bin/env python3

inference_mode = __import__('7-inference_mode').inference_mode

model_path = "./hbtn_emotion_classifier_best_model"

texts = [
    "I am profoundly disenfranchised and disillusioned by the outcome.",
    "I derive immense gratification and elation from familial interactions.",
    "The current state of affairs is entirely exasperating; my indignation is substantial.",
    "This novel musical composition captivates me entirely.",
    "I am utterly paralyzed by an overwhelming apprehension regarding potential eventualities."
]

emotion_classifier = inference_mode(
    model_path,
    top_k=1
)

predictions = emotion_classifier(texts)

for text, pred in zip(texts, predictions):
    print(f"Text: {text}")
    print(f"Prediction: {pred}")
    print("-" * 40)
