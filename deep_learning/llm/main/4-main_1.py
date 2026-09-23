#!/usr/bin/env python3

import torch
load_mlm = __import__('0-load_mlm').load_mlm
tokenize_text = __import__('1-tokenization').tokenize_text
get_mask_index = __import__('2-get_mask_index').get_mask_index
compute_mask_logits = __import__('3-compute_mask_logits').compute_mask_logits
decode_mask_predictions = __import__('4-decode_mask_predictions').decode_mask_predictions

sentence = "AI can <mask> productivity, <mask> creativity, and even <mask> decision-making."

model = load_mlm("roberta-base")

print(f"Original sentence: {sentence}\n")

tokenizer, inputs = tokenize_text("roberta-base", sentence, padding=True)

mask_indices = get_mask_index(inputs, tokenizer)
print(f"Number of masks found: {len(mask_indices)}\n")

mask_logits_list = compute_mask_logits(model, inputs, mask_indices)

decoded_predictions = decode_mask_predictions(mask_logits_list, tokenizer)

for j, (logits, all_tokens) in enumerate(zip(mask_logits_list, decoded_predictions), 1):
    top10_indices = torch.topk(logits, 10).indices
    top10_logits = logits[top10_indices].tolist()
    top10_tokens = [all_tokens[idx] for idx in top10_indices]

    print(f"Mask {j} - Top 10 predicted tokens with logits:")
    for tkn, logit in zip(top10_tokens, top10_logits):
        print(f"  {tkn:20s}: {logit:.4f}")
