#!/usr/bin/env python3
"""Module that creates a reusable prompt template for LangChain."""
from langchain_core.prompts import PromptTemplate


def create_prompt_template(template_str, input_variables):
    """Set up a reusable prompt with placeholders.

    Args:
        template_str: String defining the prompt with placeholders.
        input_variables: List of variable names used in the prompt.

    Returns:
        PromptTemplate: An object for dynamically generating prompts
        from variable inputs.
    """
    template = PromptTemplate(
        template=template_str,
        input_variables=input_variables,
    )
    return template
