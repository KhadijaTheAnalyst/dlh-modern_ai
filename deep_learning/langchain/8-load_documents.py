#!/usr/bin/env python3
"""Module that loads documents from a folder using LangChain."""
from langchain_community import document_loaders


def load_documents(folder_path, file_pattern):
    """Load the documents of a folder that match a glob pattern.

    Args:
        folder_path: String path to the folder containing documents.
        file_pattern: String glob pattern to match files.

    Returns:
        list: A list of LangChain document objects loaded from the folder.
    """
    loader = document_loaders.DirectoryLoader(
        folder_path,
        glob=file_pattern,
        loader_cls=document_loaders.TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()
