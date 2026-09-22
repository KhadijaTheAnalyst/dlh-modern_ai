#!/usr/bin/env python3
"""
FastText embeddings for text representation.
"""
import numpy as np
import gensim.models


def fasttext_embeddings(corpus_tokens, vector_size=100, window=5,
                        min_count=1, sg=0, epochs=10, workers=4):
    """
    Train FastText and return per-message embeddings.

    Args:
        corpus_tokens (list[list[str]]): Corpus as a list of token lists.
        vector_size (int): Dimensionality of word vectors. Defaults to 100.
        window (int): Context window size. Defaults to 5.
        min_count (int): Ignore words with frequency < min_count.
                        Defaults to 1 (FastText handles OOV via subwords).
        sg (int): Training algorithm: 0=CBOW, 1=Skip-gram. Defaults to 0.
        epochs (int): Number of training epochs. Defaults to 10.
        workers (int): Number of worker threads. Defaults to 4.

    Returns:
        tuple: (X, model) where:
            X: np.ndarray of shape (n_messages, vector_size) containing
               per-message embeddings (mean of token vectors).
            model: The trained FastText model.
    """
    # Train FastText model
    model = gensim.models.FastText(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers
    )

    # Create message embeddings as mean of token vectors
    embeddings = []
    for tokens in corpus_tokens:
        # Get vectors for all tokens (FastText handles OOV via subwords)
        vectors = [model.wv[token] for token in tokens]

        # Compute mean embedding or zero vector if no tokens
        if vectors:
            embedding = np.mean(vectors, axis=0)
        else:
            embedding = np.zeros(vector_size)

        embeddings.append(embedding)

    # Convert to numpy array
    X = np.array(embeddings, dtype=np.float64)

    return X, model
