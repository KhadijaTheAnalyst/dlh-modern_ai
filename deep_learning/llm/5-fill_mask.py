#!/usr/bin/env python3
"""
High-level Masked Language Modeling interface using a Hugging Face pipeline.
"""
import transformers


def fill_mask(model_name, top_k):
    """
    Creates a high-level interface for performing Masked Language Modeling
    using a pre-trained large language model.

    Args:
        model_name (str): Name of the pre-trained model to use.
        top_k (int): Number of top predictions the pipeline will return
            for each mask token.

    Returns:
        fill: A Hugging Face pipeline object.
    """
    fill = transformers.pipeline(
        "fill-mask", model=model_name, top_k=top_k
    )

    return fill
