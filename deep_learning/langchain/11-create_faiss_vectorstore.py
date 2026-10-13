#!/usr/bin/env python3
"""Module that builds a FAISS vector store and retriever with LangChain."""
from langchain_community import vectorstores


def create_faiss_vectorstore(docs, embedding_model, k,
                             search_type="similarity"):
    """Build a FAISS vector store from a list of documents.

    Args:
        docs: A list of LangChain Document objects.
        embedding_model: An instance of a LangChain embeddings model.
        k: Amount of documents to return for each query.
        search_type: Type of search for retrieval (default: "similarity").

    Returns:
        vectorstore: A FAISS vector store containing the document embeddings.
        retriever: A retriever object to query the vector store.
    """
    vectorstore = vectorstores.FAISS.from_documents(docs, embedding_model)
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )
    return vectorstore, retriever
