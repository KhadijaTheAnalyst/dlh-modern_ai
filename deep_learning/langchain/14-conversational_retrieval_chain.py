#!/usr/bin/env python3
"""Module that builds a conversational retrieval chain with LangChain."""
from langchain import chains


def create_conversational_retrieval_chain(history_aware_retriever,
                                          document_chain):
    """Build a retrieval chain that answers using retrieved documents.

    Args:
        history_aware_retriever: A retriever-like object that uses the query
            and the chat history to find relevant documents.
        document_chain: A chain that combines the retrieved documents into a
            single context for the language model to generate answers.

    Returns:
        A retrieval chain ready to handle user queries with context-aware
        document retrieval and answer generation.
    """
    retrieval_chain = chains.create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )
    return retrieval_chain
