#!/usr/bin/env python3
"""Module that builds a history-aware retriever with LangChain."""
from langchain_core import prompts
from langchain import chains


def build_history_aware_retriever(llm, retriever, human_message):
    """Create a retriever that uses the chat history to build search queries.

    The prompt follows the order: chat_history -> human input.

    Args:
        llm: An instance of a language model used to generate the query.
        retriever: A LangChain retriever object (e.g., FAISS).
        human_message: String with the user's input, containing {input}.

    Returns:
        prompt: The ChatPromptTemplate used to generate search queries.
        history_retriever: A history-aware retriever configured with the
            prompt and the provided retriever.
    """
    prompt = prompts.ChatPromptTemplate.from_messages([
        prompts.MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message),
    ])
    history_retriever = chains.create_history_aware_retriever(
        llm, retriever, prompt
    )
    return prompt, history_retriever


history_aware_retriever = build_history_aware_retriever
