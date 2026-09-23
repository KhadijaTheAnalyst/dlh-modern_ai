#!/usr/bin/env python3
"""
Decode <mask> token logits into their corresponding vocabulary tokens.
"""


def decode_mask_predictions(mask_logits_list, tokenizer):
    """
    Converts the logits for all <mask> tokens into their corresponding
    vocabulary tokens.

    Args:
        mask_logits_list (list[torch.Tensor]): A list of logits tensors,
            one for each <mask> token in the sentence.
        tokenizer: A RoBERTa tokenizer instance used to decode token IDs
            into readable strings.

    Returns:
        decoded_tokens (list[list[str]]): A list of lists, where each
            inner list contains all vocabulary tokens (as strings)
            corresponding to the logits at each mask position. Each inner
            list contains one token string for every position in the
            model's vocabulary.
    """
    decoded_tokens = []

    for logits in mask_logits_list:
        vocab_size = logits.shape[0]
        vocab_ids = list(range(vocab_size))
        tokens = tokenizer.convert_ids_to_tokens(vocab_ids)
        readable_tokens = [
            tokenizer.convert_tokens_to_string([token]).strip()
            for token in tokens
        ]
        decoded_tokens.append(readable_tokens)

    return decoded_tokens
