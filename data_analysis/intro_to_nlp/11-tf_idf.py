#!/usr/bin/env python3
"""
TF-IDF feature matrix generation.
"""
import sklearn


def tf_idf(corpus_tokens, max_features=5000, ngram_range=(1, 2),
           min_df=2, max_df=0.95, norm='l2'):
    """
    Build a TF-IDF feature matrix from a list of token lists.

    Args:
        corpus_tokens: List of token lists (one list per document)
        max_features: Maximum number of features to keep (default: 5000)
        ngram_range: Range of n-grams to consider (default: (1, 2))
        min_df: Minimum document frequency (default: 2)
        max_df: Maximum document frequency as fraction (default: 0.95)
        norm: Normalization method - 'l2', 'l1', or None (default: 'l2')

    Returns:
        X: Sparse TF-IDF matrix (documents × features)
        vectorizer: The fitted TfidfVectorizer object
    """
    corpus_text = [' '.join(tokens) for tokens in corpus_tokens]

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        norm=norm,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None
    )

    X = vectorizer.fit_transform(corpus_text)

    return X, vectorizer
