#!/usr/bin/env python3
"""
Load a pre-trained RoBERTa model for Masked Language Modeling (MLM).
"""
import transformers


def load_mlm(model_name):
    """
    Loads a pre-trained RoBERTa model ready for Masked Language Modeling.

    Args:
        model_name (str): Name of the pre-trained model to load.

    Returns:
        model: An instance of RobertaForMaskedLM ready for inference.
    """
    model = transformers.RobertaForMaskedLM.from_pretrained(model_name)
    model.eval()

    return model
