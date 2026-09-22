#!/usr/bin/env python3
"""
Compute logits for <mask> token positions using a RoBERTa MLM model.
"""
import torch


def compute_mask_logits(model, inputs, mask_indices):
    """
    Computes the raw logits for all <mask> tokens in a tokenized sentence
    using a pre-trained RoBERTa masked language model.

    Args:
        model: A RobertaForMaskedLM model loaded with pre-trained weights.
        inputs: Tokenized inputs.
        mask_indices (list[int]): Positions of all <mask> tokens in the
            input sequence.

    Returns:
        mask_logits_list (list[torch.Tensor]): A list containing logits
            tensors for each <mask> token, representing the model's raw
            predictions at that masked position.
    """
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]

    mask_logits_list = [logits[idx] for idx in mask_indices]

    return mask_logits_list
