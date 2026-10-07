#!/usr/bin/env python3
"""Module that builds a sequence of chat messages for a LangChain LLM."""
from langchain_core.messages import SystemMessage, HumanMessage


def setup_message_sequence(system_msg, human_msgs):
    """Prepare a system message followed by one or more human messages.

    Args:
        system_msg: A string defining the system's role or behavior.
        human_msgs: A list of strings representing human inputs.

    Returns:
        A list containing the SystemMessage followed by HumanMessage objects.
    """
    messages = [SystemMessage(content=system_msg)]
    for msg in human_msgs:
        messages.append(HumanMessage(content=msg))
    return messages
