#!/usr/bin/env python3
"""
Bag-of-Words feature matrix generation.
"""
import sklearn


def bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1, 2),
                 min_df=2, max_df=0.95, binary=False):
    """
    Build a Bag-of-Words feature matrix from a list of token lists.

    Args:
        corpus_tokens (list[list[str]]): Corpus as a list of token lists.
        max_features (int): Maximum number of features to extract.
                           Defaults to 5000.
        ngram_range (tuple): Range of n-grams (min_n, max_n).
                            Defaults to (1, 2).
        min_df (int): Minimum document frequency. Defaults to 2.
        max_df (float): Maximum document frequency (as proportion).
                       Defaults to 0.95.
        binary (bool): If True, all non-zero counts are set to 1.
                      Defaults to False.

    Returns:
        tuple: (X, vectorizer) where:
            X: Sparse feature matrix of shape (n_samples, n_features).
            vectorizer: The fitted CountVectorizer object.
    """
    # Join each token list into a whitespace-separated string
    corpus_text = [' '.join(tokens) for tokens in corpus_tokens]

    # Create the CountVectorizer
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None
    )

    # Fit and transform the corpus
    X = vectorizer.fit_transform(corpus_text)

    return X, vectorizer
