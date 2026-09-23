#!/usr/bin/env python3
"""
High-level translation interface using a Hugging Face pipeline.
"""
import transformers


def translate_text(model_name, src_lang=None, tgt_lang=None):
    """
    Creates a high-level interface for performing language translation
    using a pre-trained large language model.

    Args:
        model_name (str): Name of the pre-trained model to use.
        src_lang (str): Source language code (e.g., "en" for English).
            Optional — only needed by multilingual models.
        tgt_lang (str): Target language code (e.g., "fr" for French).
            Optional — only needed by multilingual models.

    Returns:
        translator: A Hugging Face pipeline object.
    """
    kwargs = {}
    if src_lang is not None:
        kwargs["src_lang"] = src_lang
    if tgt_lang is not None:
        kwargs["tgt_lang"] = tgt_lang

    translator = transformers.pipeline(
        "translation", model=model_name, **kwargs
    )

    return translator
