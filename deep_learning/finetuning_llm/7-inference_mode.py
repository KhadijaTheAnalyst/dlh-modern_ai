#!/usr/bin/env python3
"""Create a text classification pipeline from a fine-tuned model."""
import transformers


def inference_mode(model_path, top_k):
    """
    Initialize a text classification pipeline.

    Args:
        model_path (str): path to the saved model and tokenizer
        top_k (int): number of top predictions to return per input

    Returns:
        transformers.Pipeline: pipeline ready to classify new texts
    """
    return transformers.pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        top_k=top_k
    )
