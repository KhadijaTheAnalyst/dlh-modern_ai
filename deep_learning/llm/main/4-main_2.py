#!/usr/bin/env python3

import torch
load_mlm = __import__('0-load_mlm').load_mlm
tokenize_text = __import__('1-tokenization').tokenize_text
get_mask_index = __import__('2-get_mask_index').get_mask_index
compute_mask_logits = __import__('3-compute_mask_logits').compute_mask_logits
decode_mask_predictions = __import__('4-decode_mask_predictions').decode_mask_predictions

sentences = [
    "AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.",
    "Machine learning models can <mask> patterns in data and <mask> accurate predictions.",
    "Climate change is <mask> the planet and <mask> future generations."
    ]

model = load_mlm("roberta-base")

for i, sentence in enumerate(sentences, 1):
    print(f"\n{'='*70}")
    print(f"Sentence {i}: {sentence}")
    print('='*70)

    tokenizer, inputs = tokenize_text("roberta-base", sentence, padding=True)

    mask_indices = get_mask_index(inputs, tokenizer)

    mask_logits_list = compute_mask_logits(model, inputs, mask_indices)

    decoded_predictions = decode_mask_predictions(mask_logits_list, tokenizer)

    top_predictions = []
    for logits, all_tokens in zip(mask_logits_list, decoded_predictions):
        top_idx = torch.argmax(logits).item()
        top_token = all_tokens[top_idx]
        top_predictions.append(top_token)

    filled_sentence = sentence
    for prediction in top_predictions:
        filled_sentence = filled_sentence.replace(tokenizer.mask_token, prediction, 1)

    print(f"\nPredicted tokens: {top_predictions}")
    print(f"Final sentence: {filled_sentence}")
