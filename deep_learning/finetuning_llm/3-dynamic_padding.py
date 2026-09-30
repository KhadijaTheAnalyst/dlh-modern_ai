#!/usr/bin/env python3
"""Create a data collator for dynamic padding."""
import transformers


def create_data_collator(tokenizer):
    """
    Set up dynamic padding for tokenized inputs.

    Each batch is padded only to the length of its longest
    sequence, which avoids unnecessary padding.

    Args:
        tokenizer: pretrained tokenizer

    Returns:
        DataCollatorWithPadding: pads inputs dynamically per batch
    """
    return transformers.DataCollatorWithPadding(tokenizer=tokenizer)
