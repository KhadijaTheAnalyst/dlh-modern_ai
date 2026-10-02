#!/usr/bin/env python3

load_emotion_dataset = __import__('0-load_dataset').load_emotion_dataset
load_distilbert = __import__('1-load_distilbert').load_distilbert
tokenize_and_map = __import__('2-tokenize_and_map').tokenize_and_map
create_data_collator = __import__('3-dynamic_padding').create_data_collator
compute_metrics = __import__('4-compute_metrics').compute_metrics
configure_training_args = __import__('5-configure_training_arguments').configure_training_args
train_model = __import__('6-training_mode').train_model


from transformers import set_seed
set_seed(0)

import os
from huggingface_hub import login
os.environ["HF_TOKEN"] = "YOUR KEY HERE"
login(token=os.environ["HF_TOKEN"])


dataset = load_emotion_dataset()

label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
label_to_id = {name: idx for idx, name in enumerate(label_names)}
id_to_label = {idx: name for idx, name in enumerate(label_names)}
num_classes = len(label_names)
model_name = "distilbert-base-uncased"

tokenizer, model = load_distilbert(model_name, num_classes, id_to_label, label_to_id)

tokenized_train, tokenized_val, tokenized_test = tokenize_and_map(
    dataset,
    tokenizer,
    max_length=128,
    truncation=True,
    batched=True
)

data_collator = create_data_collator(tokenizer)

training_args = configure_training_args(
    output_dir="./hbtn_emotion_classifier",
    epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    seed=0
)

trainer, train_results, test_results = train_model(
    model=model,
    training_args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    test_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    model_save_name="./hbtn_emotion_classifier_best_model"
)

print("\n Training Results:")
print(f"  Total training time: {train_results.metrics['train_runtime']:.2f} seconds")
print(f"  Training loss: {train_results.training_loss:.4f}")
print(f"  Training samples/sec: {train_results.metrics['train_samples_per_second']:.2f}")
print(f"  Training steps/sec: {train_results.metrics['train_steps_per_second']:.2f}")

print("\n Test Set Performance:")
print(f"  Accuracy:  {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.2f}%)")
print(f"  Precision: {test_results['eval_precision']:.4f}")
print(f"  Recall:    {test_results['eval_recall']:.4f}")
print(f"  F1 Score:  {test_results['eval_f1']:.4f}")
print(f"  Loss:      {test_results['eval_loss']:.4f}")
