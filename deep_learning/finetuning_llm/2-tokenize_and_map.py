#!/usr/bin/env python3
"""Tokenize the Emotion dataset with a pretrained tokenizer."""


def tokenize_and_map(dataset, tokenizer, max_length, truncation, batched):
    """
    Tokenize the text of every split in the dataset.

    Args:
        dataset (DatasetDict): train, validation and test splits
        tokenizer: pretrained tokenizer
        max_length (int): maximum token length for truncation
        truncation (bool): whether to truncate long sequences
        batched (bool): whether to process examples in batches

    Returns:
        tuple: tokenized train, validation and test splits
    """
    def tokenize(examples):
        """Tokenize a batch (or single example) of text."""
        return tokenizer(
            examples["text"],
            truncation=truncation,
            max_length=max_length
        )

    tokenized = dataset.map(tokenize, batched=batched)
    return tokenized["train"], tokenized["validation"], tokenized["test"]
