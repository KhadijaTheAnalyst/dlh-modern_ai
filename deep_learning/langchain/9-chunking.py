#!/usr/bin/env python3
"""Module that splits documents into smaller chunks using LangChain."""
from langchain import text_splitter


def split_into_chunks(docs, chunk_size, chunk_overlap):
    """Split a list of documents into smaller chunks.

    Args:
        docs: List of LangChain Document objects to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        list: A list of Document objects representing the split chunks.
    """
    splitter = text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)
