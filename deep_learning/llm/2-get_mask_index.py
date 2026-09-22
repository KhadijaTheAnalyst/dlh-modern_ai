#!/usr/bin/env python3
"""
Locate <mask> token positions in tokenized input.
"""
import transformers


def get_mask_index(inputs, tokenizer):
    """
    Identifies and returns the positions of all <mask> tokens in the
    tokenized input sequence.

    Args:
        inputs: Tokenized inputs.
        tokenizer: An instance of RobertaTokenizer.

    Returns:
        mask_indices (list[int]): A list containing the index of every
            <mask> token found in the sequence.
    """
    mask_token_id = tokenizer.mask_token_id
    token_ids = inputs["input_ids"][0]

    mask_indices = [
        i for i, token_id in enumerate(token_ids) if token_id == mask_token_id
    ]

    if not mask_indices:
        raise ValueError("No <mask> token found in the input!")

    return mask_indices
