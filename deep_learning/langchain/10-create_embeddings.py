#!/usr/bin/env python3
"""Module that loads a HuggingFace embedding model using LangChain."""
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name):
    """Load a HuggingFace embedding model.

    Args:
        model_name: Name of the HuggingFace model used to generate
            embeddings.

    Returns:
        An instance of HuggingFaceEmbeddings to compute embeddings for
        documents.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
