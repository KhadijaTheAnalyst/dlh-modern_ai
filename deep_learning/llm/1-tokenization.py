#!/usr/bin/env python3
"""
Tokenize input text using a pre-trained RoBERTa tokenizer.
"""
import transformers


def tokenize_text(model_name, sentence, padding=True):
    """
    Loads a pre-trained RoBERTa tokenizer and tokenizes a given input
    sentence.

    Args:
        model_name (str): Name of the pre-trained model to load.
        sentence (str): The text to tokenize.
        padding (bool): Whether to pad the tokenized sequence.
            Defaults to True.

    Returns:
        tokenizer: An instance of RobertaTokenizer.
        inputs: Tokenized representation of the sentence as PyTorch
            tensors.
    """
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    inputs = tokenizer(sentence, padding=padding, return_tensors="pt")

    return tokenizer, inputs
