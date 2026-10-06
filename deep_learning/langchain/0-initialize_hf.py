#!/usr/bin/env python3
"""Module that initializes a Hugging Face LLM for use in LangChain."""
from langchain_community import llms


def initialize_hf_llm(model_name, max_tokens):
    """Set up a Hugging Face text-to-text generation model in LangChain.

    Args:
        model_name: Name of the Hugging Face model to load.
        max_tokens: Maximum number of tokens to generate for the output.

    Returns:
        llm: An instance of HuggingFacePipeline.
    """
    llm = llms.HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text2text-generation",
        model_kwargs={"dtype": "auto"},
        pipeline_kwargs={"max_new_tokens": max_tokens},
    )
    return llm
