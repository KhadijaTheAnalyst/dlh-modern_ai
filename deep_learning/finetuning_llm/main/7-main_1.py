#!/usr/bin/env python3

# load_distilbert = __import__('1-load_distilbert').load_distilbert
import torch
from transformers import logging, set_seed

set_seed(0)
logging.set_verbosity_error()

model_name = "distilbert-base-uncased"
num_classes = 6

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
label2id = {v: k for k, v in id2label.items()}

tokenizer, model = load_distilbert(model_name, num_classes, id2label, label2id)

texts = [
    "I feel really so disappointed.",
    "I absolutely love spending time with my family.",
    "This is so frustrating! I'm really angry!",
    "I love this new song!",
    "I'm Petrified about what might happen."
]


model.eval()
for text in texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()

    predicted_emotion = model.config.id2label[predicted_class]

    print(f"Text: {text}")
    print(f"Prediction: {predicted_emotion}")
    print("-" * 40)
