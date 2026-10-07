#!/usr/bin/env python3
"""Module that creates a reusable chat-style prompt template."""
from langchain import prompts


def create_chat_prompt_template(system_message, human_message):
    """Set up a reusable chat prompt with separate system and human parts.

    Args:
        system_message: String defining the assistant's instructions.
        human_message: String containing the user's input with placeholders.

    Returns:
        ChatPromptTemplate: An object that can be formatted dynamically
        with variables.
    """
    chat_template = prompts.ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message),
    ])
    return chat_template
