#!/usr/bin/env python3
"""Module that loads a Mistral chat model using LangChain."""
from langchain_mistralai import ChatMistralAI


def load_mistral(model_name, temperature):
    """Initialize a Mistral language model using LangChain.

    The API key is read from the MISTRAL_API_KEY environment variable.

    Args:
        model_name: Name of the Mistral model to use.
        temperature: Float controlling the randomness of the output.

    Returns:
        llm: An instance of ChatMistralAI.
    """
    llm = ChatMistralAI(model=model_name, temperature=temperature)
    return llm
