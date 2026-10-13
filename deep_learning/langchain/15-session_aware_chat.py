#!/usr/bin/env python3
"""Module that adds session-aware chat history to a retrieval chain."""
from langchain_core.runnables import history


def create_session_aware_chat(retrieval_chain, get_session_history,
                              input_key, history_key, output_key):
    """Wrap a retrieval chain so each session keeps its own chat history.

    Args:
        retrieval_chain: A conversational retrieval chain that fetches
            relevant documents and generates answers.
        get_session_history: A function that returns a chat history object
            for a given session.
        input_key: String naming the key that holds the user input.
        history_key: String naming the key where the chat history is stored.
        output_key: String naming the key where the model output is stored.

    Returns:
        A RunnableWithMessageHistory.
    """
    session_chain = history.RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key=input_key,
        history_messages_key=history_key,
        output_messages_key=output_key,
    )
    return session_chain
