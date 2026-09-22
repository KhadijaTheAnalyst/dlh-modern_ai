#!/usr/bin/env python3

load_mlm = __import__('0-load_mlm').load_mlm
tokenize_text = __import__('1-tokenization').tokenize_text
get_mask_index = __import__('2-get_mask_index').get_mask_index
compute_mask_logits = __import__('3-compute_mask_logits').compute_mask_logits

sentences = [
    "AI can <mask> productivity, <mask> creativity, and even <mask> decision-making.",
    "Machine learning models can <mask> patterns in data and <mask> accurate predictions.",
    "Climate change is <mask> the planet and <mask> future generations."
    ]

model = load_mlm("roberta-base")

for i, sentence in enumerate(sentences, 1):
    print(f"\nSentence {i}: {sentence}")

    tokenizer, inputs = tokenize_text("roberta-base", sentence, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_ids = inputs["input_ids"][0]

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    mask_indices = get_mask_index(inputs, tokenizer)
    print("Mask indices:", mask_indices)

    logits_list = compute_mask_logits(model, inputs, mask_indices)
    print(f"Number of masks: {len(logits_list)}")
    print(f"Mask logits: {logits_list}")
    print("Logits shape per mask:", logits_list[0].shape)
    